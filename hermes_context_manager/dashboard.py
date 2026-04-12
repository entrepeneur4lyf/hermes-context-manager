"""Opt-in embedded HTTP+SSE dashboard for live HMC observability.

Launched via the ``hmc_control dashboard action=start`` action, this
module spins up a small localhost HTTP server that serves a single-page
dashboard showing live context-savings metrics.  The page auto-refreshes
via Server-Sent Events: the plugin publishes turn-complete and
session-end events into an in-process bus, and each connected SSE
client drains the bus and streams the events to its browser.

Design constraints
==================
- **Stdlib only.**  ``http.server.ThreadingHTTPServer`` + ``queue.Queue``
  + a hand-rolled SSE writer.  No Flask, no FastAPI, no third-party HTTP
  library.  The HTML page has inline CSS and vanilla JS -- no bundler,
  no external CDN.
- **Localhost only.**  The server binds to ``127.0.0.1`` by default and
  there is no auth layer.  This is explicitly NOT designed for remote
  exposure.  If a user wants that, they should put a real reverse proxy
  in front.
- **Opt-in.**  The dashboard is dormant until the user runs
  ``hmc_control dashboard action=start``.  Most Hermes sessions never
  need it and shouldn't pay for a background thread they don't use.
- **Clean shutdown.**  ``action=stop`` calls ``shutdown()`` +
  ``server_close()``.  The server thread is a daemon so it doesn't
  block Hermes exit, but we still shut down cleanly whenever we can.
- **Never crash the plugin.**  Every public method swallows
  ``OSError`` / ``Exception`` and logs via ``LOGGER.warning``.  A
  dashboard failure must never take down the host plugin.
- **No innerHTML with interpolated data.**  All dynamic values go in
  via ``textContent`` + ``createElement``.  Defensive practice even
  though the data comes from our own backend.

Event schema
============
The SSE channel emits these event types:

- ``hello``: sent once when a client first connects.  Data:
  ``{"time": <epoch>}``.
- ``ping``: keepalive every 15s to prevent proxy buffer timeouts.
  Data: ``{}``.
- ``turn``: emitted by ``Dashboard.publish_turn`` at the end of each
  ``on_pre_llm_call``.  Data: see the JS handler schema.
- ``session_end``: emitted by ``Dashboard.publish_session_end`` at
  the end of each ``on_session_end``.

REST endpoints
==============
- ``GET /``               — HTML dashboard page
- ``GET /api/summary``    — current-session snapshot (JSON)
- ``GET /api/history``    — turn-history ring buffer (JSON)
- ``GET /api/analytics``  — global SQLite analytics summary (JSON)
- ``GET /events``         — SSE stream
"""
from __future__ import annotations

import json
import logging
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:  # pragma: no cover
    from .plugin import HermesContextManagerPlugin

LOGGER = logging.getLogger(__name__)

# SSE keepalive interval.  Browsers and some proxies time out an idle
# event stream after ~30s, so ping every 15s to keep the connection
# warm.
_SSE_KEEPALIVE_SECONDS = 15

# Per-subscriber queue bound.  A full queue indicates a slow/hung
# client; we drop events rather than block the publisher.
_SUBSCRIBER_QUEUE_MAX = 256

# Default base port for the dashboard.  High enough to avoid the
# privileged range and well clear of common service ports.  If this
# port is busy, ``start()`` rotates upward until it finds a free slot
# or exhausts the retry budget.  A fixed base port means the URL is
# bookmarkable across restarts on clean systems; rotation means
# collisions don't break the feature.
_DEFAULT_PORT = 48800

# Maximum port-rotation attempts before giving up.  48800..48899 is
# 100 candidates, comfortably more than any realistic local contention.
_PORT_ROTATION_ATTEMPTS = 100


# ---------------------------------------------------------------------------
# Event bus: publishers write, SSE handlers drain
# ---------------------------------------------------------------------------


class _EventBus:
    """Thread-safe fan-out from plugin hooks to connected SSE clients."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: list[queue.Queue[dict[str, Any]]] = []

    def subscribe(self) -> queue.Queue[dict[str, Any]]:
        q: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=_SUBSCRIBER_QUEUE_MAX)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue[dict[str, Any]]) -> None:
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def publish(self, event: dict[str, Any]) -> None:
        with self._lock:
            subs = list(self._subscribers)
        for q in subs:
            try:
                q.put_nowait(event)
            except queue.Full:
                # Slow subscriber -- drop rather than block the
                # hook call path
                pass

    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)


# ---------------------------------------------------------------------------
# HTTP request handler
# ---------------------------------------------------------------------------


class _DashboardHandler(BaseHTTPRequestHandler):
    """Per-request handler.

    Class attributes ``plugin``, ``event_bus``, and ``html_page`` are
    injected by ``Dashboard.start`` before the server starts listening.
    """

    plugin: Optional["HermesContextManagerPlugin"] = None
    event_bus: Optional[_EventBus] = None
    html_page: str = ""

    # ------------------------- Routing -------------------------------

    def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        try:
            path = self.path.split("?", 1)[0]
            if path == "/":
                self._serve_html()
            elif path == "/api/summary":
                self._serve_json(self._current_summary())
            elif path == "/api/history":
                self._serve_json(self._current_history())
            elif path == "/api/analytics":
                self._serve_json(self._analytics_summary())
            elif path == "/api/sessions":
                self._serve_json(self._recent_sessions())
            elif path == "/events":
                self._serve_sse()
            else:
                self.send_error(404, "not found")
        except (BrokenPipeError, ConnectionResetError):
            # Client disconnected mid-response -- normal, not an error
            pass
        except Exception:  # pragma: no cover - defensive
            LOGGER.exception("HMC dashboard: handler failure")
            try:
                self.send_error(500, "internal error")
            except Exception:
                pass

    # Silence the BaseHTTPRequestHandler default access log -- it
    # writes directly to stderr and spams the Hermes output.
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    # ------------------------- Responses -----------------------------

    def _serve_html(self) -> None:
        body = self.html_page.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _serve_json(self, data: Any) -> None:
        body = json.dumps(data, default=str).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _serve_sse(self) -> None:
        if self.event_bus is None:
            self.send_error(503, "event bus not initialized")
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")  # disable nginx buffering
        self.end_headers()

        q = self.event_bus.subscribe()
        try:
            self._write_sse("hello", {"time": time.time()})
            while True:
                try:
                    event = q.get(timeout=_SSE_KEEPALIVE_SECONDS)
                except queue.Empty:
                    self._write_sse("ping", {})
                    continue
                self._write_sse(event.get("type", "update"), event.get("data", {}))
        except (BrokenPipeError, ConnectionResetError):
            pass  # normal client disconnect
        finally:
            self.event_bus.unsubscribe(q)

    def _write_sse(self, event_type: str, data: dict[str, Any]) -> None:
        payload = f"event: {event_type}\ndata: {json.dumps(data, default=str)}\n\n"
        self.wfile.write(payload.encode("utf-8"))
        self.wfile.flush()

    # ------------------------- Data helpers --------------------------

    def _current_summary(self) -> dict[str, Any]:
        """Current-session metrics snapshot.

        ``session_saved`` is the un-gated ``tokens_kept_out_total`` --
        the real bytes-kept-out-of-the-API-call number -- which is what
        users care about seeing in the dashboard.  The gated counter
        rides along as ``uniq_saved`` so the UI can optionally show
        both numbers.
        """
        if self.plugin is None:
            return {}
        state = self.plugin._current_dashboard_state()
        if state is None:
            return {
                "session_id": None,
                "session_saved": 0,
                "uniq_saved": 0,
                "context_tokens": None,
                "context_percent": None,
                "by_strategy": {},
                "uniq_by_strategy": {},
                "turns": 0,
                "project_path": "",
            }
        return {
            "session_id": state["session_id"],
            # Primary metric (un-gated, real bytes kept out).
            "session_saved": state["tokens_kept_out_total"],
            "by_strategy": dict(state["tokens_kept_out_by_type"]),
            # Diagnostic (gated, unique firings).
            "uniq_saved": state["tokens_saved"],
            "uniq_by_strategy": dict(state["tokens_saved_by_type"]),
            "context_tokens": state["last_context_tokens"],
            "context_percent": state["last_context_percent"],
            "turns": state["current_turn"],
            "project_path": state["project_path"],
        }

    def _current_history(self) -> dict[str, Any]:
        if self.plugin is None:
            return {"turns": []}
        history = self.plugin._current_dashboard_history()
        return {"turns": history}

    def _analytics_summary(self) -> dict[str, Any]:
        if self.plugin is None:
            return {}
        summary = self.plugin.analytics_store.get_summary()
        return {
            "total_saved": summary.total_saved,
            "total_sessions": summary.total_sessions,
            "total_input": summary.total_input,
            "total_output": summary.total_output,
            "savings_pct": round(summary.savings_pct, 2),
            "by_strategy": dict(summary.by_strategy),
        }

    def _recent_sessions(self) -> dict[str, Any]:
        """Return the recent-sessions survey for the Workhorses panel.

        Wraps ``plugin._recent_sessions_for_dashboard`` and packages it
        as ``{"sessions": [...]}`` so future additions (counts,
        pagination, etc.) can ride alongside without changing the
        top-level shape.
        """
        if self.plugin is None:
            return {"sessions": []}
        sessions = self.plugin._recent_sessions_for_dashboard(limit=20)
        return {"sessions": sessions}


# ---------------------------------------------------------------------------
# Dashboard controller
# ---------------------------------------------------------------------------


class Dashboard:
    """Lifecycle wrapper for the embedded HTTP/SSE server.

    One instance lives on the plugin; it's dormant until ``start`` is
    called and can be stopped and restarted at any time.
    """

    def __init__(
        self,
        plugin_ref: Callable[[], Any],
        host: str = "127.0.0.1",
        port: int = _DEFAULT_PORT,
    ) -> None:
        """Construct a dormant dashboard.

        ``port`` is the **base** port to try.  If it's already in use,
        ``start()`` rotates upward up to ``_PORT_ROTATION_ATTEMPTS``
        times before giving up.  Pass ``0`` to let the OS pick a
        random free port (no rotation; URL won't be stable across
        restarts).
        """
        self._plugin_ref = plugin_ref
        self._host = host
        self._requested_port = port
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._bus = _EventBus()
        self._lock = threading.Lock()
        self._started_at: float | None = None

    # ------------------------- Lifecycle -----------------------------

    def start(self) -> dict[str, Any]:
        """Launch the server if not already running.  Returns status dict.

        Port rotation: if the requested port is non-zero and already
        in use, we increment the port and retry, up to
        ``_PORT_ROTATION_ATTEMPTS`` times.  Gives a bookmarkable
        default URL while tolerating local port contention.  Pass
        ``port=0`` at construction time to get OS-picked ports
        (no rotation, no stable URL).
        """
        with self._lock:
            if self._server is not None:
                return self._status_locked()

            try:
                plugin = self._plugin_ref()
                bus = self._bus

                class _BoundHandler(_DashboardHandler):
                    pass

                _BoundHandler.plugin = plugin
                _BoundHandler.event_bus = bus
                _BoundHandler.html_page = _DASHBOARD_HTML

                server = self._bind_with_rotation(_BoundHandler)
                if server is None:
                    LOGGER.warning(
                        "HMC dashboard: no free port found after %d attempts "
                        "starting at %d",
                        _PORT_ROTATION_ATTEMPTS, self._requested_port,
                    )
                    return {
                        "running": False,
                        "error": (
                            f"no free port in "
                            f"{self._requested_port}.."
                            f"{self._requested_port + _PORT_ROTATION_ATTEMPTS - 1}"
                        ),
                    }

                # ThreadingHTTPServer default: each request on a daemon
                # thread.  We also want client threads to not block
                # shutdown.
                server.daemon_threads = True
                self._server = server
                self._started_at = time.time()
                self._thread = threading.Thread(
                    target=server.serve_forever,
                    daemon=True,
                    name="hmc-dashboard",
                )
                self._thread.start()
                LOGGER.info("HMC dashboard started at %s", self.url)
                return self._status_locked()
            except OSError as exc:
                LOGGER.warning("HMC dashboard: failed to start: %s", exc)
                return {"running": False, "error": str(exc)}

    def _bind_with_rotation(
        self, handler_cls: type,
    ) -> ThreadingHTTPServer | None:
        """Bind the server socket, rotating the port if busy.

        Returns the live server, or None if all rotation attempts
        failed.  A requested port of 0 means OS-pick and skips
        rotation entirely.
        """
        # Port 0: OS picks a free port.  No rotation needed -- if this
        # fails the whole system is out of ephemeral ports.
        if self._requested_port == 0:
            return ThreadingHTTPServer((self._host, 0), handler_cls)

        base = self._requested_port
        last_error: OSError | None = None
        for offset in range(_PORT_ROTATION_ATTEMPTS):
            candidate = base + offset
            # Don't roll over into the privileged / ephemeral noise
            # zones.  48800 + 100 = 48900 so this cap doesn't actually
            # kick in at the default base; it's a safety net.
            if candidate > 65535:
                break
            try:
                return ThreadingHTTPServer(
                    (self._host, candidate), handler_cls,
                )
            except OSError as exc:
                last_error = exc
                # errno 98 = EADDRINUSE on Linux, 48 on macOS.  We
                # don't strictly need to check errno -- any OSError
                # during bind is worth rotating past -- but log the
                # cause so operators can diagnose.
                LOGGER.debug(
                    "HMC dashboard: port %d busy (%s), rotating",
                    candidate, exc,
                )
                continue

        if last_error is not None:
            LOGGER.warning(
                "HMC dashboard: rotation exhausted, last error: %s",
                last_error,
            )
        return None

    def stop(self) -> dict[str, Any]:
        with self._lock:
            if self._server is None:
                return {"running": False}
            try:
                self._server.shutdown()
                self._server.server_close()
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.warning("HMC dashboard: shutdown error: %s", exc)
            self._server = None
            self._thread = None
            self._started_at = None
            LOGGER.info("HMC dashboard stopped")
            return {"running": False}

    def status(self) -> dict[str, Any]:
        with self._lock:
            return self._status_locked()

    def _status_locked(self) -> dict[str, Any]:
        """Caller must hold self._lock."""
        if self._server is None:
            return {"running": False}
        addr = self._server.server_address
        return {
            "running": True,
            "url": self.url,
            "host": self._host,
            "port": addr[1],
            "started_at": self._started_at,
            "subscribers": self._bus.subscriber_count(),
        }

    # ------------------------- Publishing ----------------------------

    def publish(self, event_type: str, data: dict[str, Any]) -> None:
        """Fan out an event to all connected SSE clients.  No-op if dormant."""
        if self._server is None:
            return
        try:
            self._bus.publish({"type": event_type, "data": data})
        except Exception:  # pragma: no cover - defensive
            LOGGER.exception("HMC dashboard: publish failed")

    @property
    def url(self) -> str:
        if self._server is None:
            return ""
        # server_address is (host, port) for AF_INET and
        # (host, port, flowinfo, scopeid) for AF_INET6 -- index by
        # position rather than unpacking so both work.
        addr = self._server.server_address
        return f"http://{addr[0]}:{addr[1]}/"

    @property
    def running(self) -> bool:
        return self._server is not None


# ---------------------------------------------------------------------------
# Inlined single-page HTML dashboard
# ---------------------------------------------------------------------------
# Kept as a module-level constant so no runtime template rendering is
# needed.  Vanilla JS, no framework, no build step.  All inline so
# there are no extra HTTP requests for CSS/JS.
#
# SECURITY: all dynamic values are inserted via textContent / createElement,
# never innerHTML with interpolated data.  This is defensive -- the data
# comes from our own backend -- but good practice regardless.

_DASHBOARD_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>HMC Dashboard</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  * { box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: #0b0d10;
    color: #c8c8c8;
    margin: 0;
    padding: 1.5rem;
    line-height: 1.4;
  }
  h1 {
    font-size: 1.25rem;
    color: #fff;
    margin: 0 0 0.25rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }
  h2 {
    font-size: 0.9rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin: 0 0 0.75rem 0;
  }
  .sub { color: #666; font-size: 0.85rem; margin-bottom: 1.5rem; }
  .status-dot {
    display: inline-block;
    width: 0.6rem;
    height: 0.6rem;
    border-radius: 50%;
    background: #00d9a3;
    animation: pulse 2s ease-in-out infinite;
  }
  .status-dot.disconnected { background: #d94a4a; animation: none; }
  @keyframes pulse { 0%,100% { opacity: 1 } 50% { opacity: 0.4 } }

  section { margin: 1.5rem 0; }

  .metrics { display: flex; flex-wrap: wrap; gap: 0.75rem; }
  .metric {
    flex: 1 1 12rem;
    min-width: 10rem;
    padding: 1rem;
    background: #161a1f;
    border-radius: 0.5rem;
    border: 1px solid #23272e;
  }
  .metric-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    color: #777;
    letter-spacing: 0.05em;
  }
  .metric-value {
    font-size: 1.6rem;
    font-weight: 600;
    color: #00d9a3;
    margin-top: 0.25rem;
  }
  .metric-sub { font-size: 0.75rem; color: #666; margin-top: 0.25rem; }

  .strategy-list { display: flex; flex-direction: column; gap: 0.3rem; }
  .strategy-row {
    display: grid;
    grid-template-columns: 10rem 1fr 5rem;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.85rem;
  }
  .strategy-name { color: #aaa; }
  .strategy-bar-wrap {
    background: #1a1d22;
    border-radius: 0.2rem;
    height: 0.5rem;
    overflow: hidden;
  }
  .strategy-bar {
    height: 100%;
    background: linear-gradient(90deg, #00d9a3, #00b888);
    border-radius: 0.2rem;
    transition: width 0.4s ease-out;
  }
  .strategy-value { color: #fff; text-align: right; font-variant-numeric: tabular-nums; }

  .log {
    font-family: "SF Mono", Menlo, Monaco, monospace;
    font-size: 0.8rem;
    max-height: 18rem;
    overflow-y: auto;
    background: #161a1f;
    border: 1px solid #23272e;
    padding: 0.75rem 1rem;
    border-radius: 0.5rem;
  }
  .log-entry {
    padding: 0.2rem 0;
    border-bottom: 1px solid #1a1d22;
    color: #aaa;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .log-entry:last-child { border-bottom: none; }
  .log-entry .ts { color: #555; margin-right: 0.5rem; }
  .log-entry .ok { color: #00d9a3; }
  .log-entry .warn { color: #e0b84a; }

  .sessions-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.8rem;
    font-variant-numeric: tabular-nums;
  }
  .sessions-table th {
    text-align: left;
    font-weight: 500;
    color: #777;
    padding: 0.4rem 0.5rem;
    border-bottom: 1px solid #23272e;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  .sessions-table td {
    padding: 0.4rem 0.5rem;
    border-bottom: 1px solid #1a1d22;
    color: #aaa;
  }
  .sessions-table tbody tr:last-child td { border-bottom: none; }
  .sessions-table tbody tr.live { background: #121a17; }
  .sessions-table tbody tr.live td { color: #c8e8d8; }
  .sessions-table tbody tr.active td:first-child::before {
    content: "\25CF\00a0";  /* filled circle + nbsp */
    color: #00d9a3;
  }
  .sessions-table .num { text-align: right; color: #fff; }
  .sessions-table .breakdown { color: #888; font-size: 0.75rem; }
  .sessions-table .id { color: #666; font-family: "SF Mono", Menlo, Monaco, monospace; }

  .empty { color: #555; font-style: italic; font-size: 0.85rem; padding: 0.5rem 0; }

  footer {
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid #23272e;
    color: #555;
    font-size: 0.75rem;
  }
  footer code { background: #1a1d22; padding: 0.1rem 0.3rem; border-radius: 0.2rem; color: #888; }
</style>
</head>
<body>
<h1>Hermes Context Manager <span class="status-dot" id="status-dot" title="connecting..."></span></h1>
<div class="sub" id="session-sub">connecting...</div>

<section>
  <h2>Current session</h2>
  <div class="metrics">
    <div class="metric">
      <div class="metric-label">Saved this session</div>
      <div class="metric-value" id="m-session-saved">--</div>
      <div class="metric-sub">tokens</div>
    </div>
    <div class="metric">
      <div class="metric-label">Context used</div>
      <div class="metric-value" id="m-context-pct">--</div>
      <div class="metric-sub" id="m-context-tokens">-- tokens</div>
    </div>
    <div class="metric">
      <div class="metric-label">Turns</div>
      <div class="metric-value" id="m-turns">--</div>
    </div>
  </div>
</section>

<section>
  <h2>Lifetime (SQLite analytics)</h2>
  <div class="metrics">
    <div class="metric">
      <div class="metric-label">All-time saved</div>
      <div class="metric-value" id="m-lifetime-saved">--</div>
      <div class="metric-sub" id="m-lifetime-pct">-- % of input</div>
    </div>
    <div class="metric">
      <div class="metric-label">Sessions recorded</div>
      <div class="metric-value" id="m-lifetime-sessions">--</div>
    </div>
  </div>
</section>

<section>
  <h2>Savings by strategy (this session)</h2>
  <div class="strategy-list" id="strategy-list"></div>
</section>

<section>
  <h2>Recent sessions (workhorses)</h2>
  <div id="sessions-container"></div>
</section>

<section>
  <h2>Live events</h2>
  <div class="log" id="event-log"></div>
</section>

<footer>
  HMC dashboard · updates via SSE (<code>/events</code>) · bound to localhost
</footer>

<script>
(function() {
  'use strict';

  const fmt = (n) => {
    if (n == null) return '--';
    if (n >= 1_000_000) return (n / 1_000_000).toFixed(2) + 'M';
    if (n >= 1_000) return (n / 1_000).toFixed(1) + 'K';
    return String(n);
  };
  const fmtPct = (p) => (p == null) ? '--' : (p * 100).toFixed(0) + '%';

  const $ = (id) => document.getElementById(id);

  // Safely create a span with text content
  function span(text, className) {
    const el = document.createElement('span');
    if (className) el.className = className;
    el.textContent = text;
    return el;
  }

  // Replace all children of an element with new children
  function replaceChildren(parent, children) {
    while (parent.firstChild) parent.removeChild(parent.firstChild);
    for (const child of children) parent.appendChild(child);
  }

  // Display an empty-state message
  function showEmpty(parent, message) {
    const el = document.createElement('div');
    el.className = 'empty';
    el.textContent = message;
    replaceChildren(parent, [el]);
  }

  async function fetchJson(path) {
    try {
      const r = await fetch(path, { cache: 'no-store' });
      return await r.json();
    } catch (e) {
      return null;
    }
  }

  async function refreshAll() {
    const [summary, analytics, sessions] = await Promise.all([
      fetchJson('/api/summary'),
      fetchJson('/api/analytics'),
      fetchJson('/api/sessions'),
    ]);

    if (summary) {
      $('m-session-saved').textContent = fmt(summary.session_saved);
      $('m-context-pct').textContent = fmtPct(summary.context_percent);
      $('m-context-tokens').textContent = fmt(summary.context_tokens) + ' tokens';
      $('m-turns').textContent = summary.turns == null ? '--' : String(summary.turns);

      const subLine = summary.session_id
        ? 'session ' + String(summary.session_id).slice(0, 8) + ' · ' +
          (summary.project_path || '(no project)')
        : 'no active session yet — start a conversation';
      $('session-sub').textContent = subLine;

      renderStrategies(summary.by_strategy || {});
    }

    if (analytics) {
      $('m-lifetime-saved').textContent = fmt(analytics.total_saved);
      $('m-lifetime-pct').textContent =
        (analytics.savings_pct != null ? analytics.savings_pct.toFixed(1) : '--') + '% of input';
      $('m-lifetime-sessions').textContent =
        analytics.total_sessions == null ? '--' : String(analytics.total_sessions);
    }

    if (sessions) {
      renderSessions(sessions.sessions || []);
    }
  }

  // HH:MM formatter from epoch seconds (local tz)
  function fmtTime(epoch) {
    if (epoch == null) return '--';
    const d = new Date(epoch * 1000);
    const hh = String(d.getHours()).padStart(2, '0');
    const mm = String(d.getMinutes()).padStart(2, '0');
    return hh + ':' + mm;
  }

  // Build the "strategy_a 4.9K + strategy_b 6.1K" text
  function fmtBreakdown(byStrategy) {
    const entries = Object.entries(byStrategy || {}).filter((e) => e[1] > 0);
    if (entries.length === 0) return 'no strategies fired';
    entries.sort((a, b) => b[1] - a[1]);
    return entries.map(([name, val]) => name + ' ' + fmt(val)).join(' + ');
  }

  function renderSessions(sessions) {
    const container = $('sessions-container');
    if (!sessions || sessions.length === 0) {
      showEmpty(container, 'no sessions recorded yet');
      return;
    }

    const table = document.createElement('table');
    table.className = 'sessions-table';

    // Header
    const thead = document.createElement('thead');
    const headRow = document.createElement('tr');
    for (const label of ['Session', 'Time', 'Ctx', 'Tools', 'Saved', 'Breakdown']) {
      const th = document.createElement('th');
      th.textContent = label;
      headRow.appendChild(th);
    }
    thead.appendChild(headRow);
    table.appendChild(thead);

    // Body
    const tbody = document.createElement('tbody');
    for (const s of sessions) {
      const tr = document.createElement('tr');
      if (s.live) tr.classList.add('live');
      if (s.active) tr.classList.add('active');

      const tdId = document.createElement('td');
      tdId.className = 'id';
      tdId.textContent = s.short_id || '';
      tr.appendChild(tdId);

      const tdTime = document.createElement('td');
      tdTime.textContent = fmtTime(s.mtime);
      tr.appendChild(tdTime);

      const tdCtx = document.createElement('td');
      tdCtx.textContent = fmtPct(s.last_context_percent);
      tr.appendChild(tdCtx);

      const tdTools = document.createElement('td');
      tdTools.className = 'num';
      tdTools.textContent = s.tool_call_count == null ? '--' : String(s.tool_call_count);
      tr.appendChild(tdTools);

      const tdSaved = document.createElement('td');
      tdSaved.className = 'num';
      tdSaved.textContent = fmt(s.tokens_kept_out_total);
      tr.appendChild(tdSaved);

      const tdBreakdown = document.createElement('td');
      tdBreakdown.className = 'breakdown';
      tdBreakdown.textContent = fmtBreakdown(s.tokens_kept_out_by_type);
      tr.appendChild(tdBreakdown);

      tbody.appendChild(tr);
    }
    table.appendChild(tbody);

    replaceChildren(container, [table]);
  }

  function renderStrategies(byStrategy) {
    const list = $('strategy-list');
    const entries = Object.entries(byStrategy || {}).filter((e) => e[1] > 0);
    if (entries.length === 0) {
      showEmpty(list, 'no strategies have fired yet');
      return;
    }
    entries.sort((a, b) => b[1] - a[1]);
    const max = Math.max.apply(null, entries.map((e) => e[1]).concat(1));

    const rows = entries.map(([name, val]) => {
      const row = document.createElement('div');
      row.className = 'strategy-row';

      const nameEl = span(name, 'strategy-name');

      const barWrap = document.createElement('span');
      barWrap.className = 'strategy-bar-wrap';
      const bar = document.createElement('span');
      bar.className = 'strategy-bar';
      bar.style.width = Math.round((val / max) * 100) + '%';
      barWrap.appendChild(bar);

      const valueEl = span(fmt(val), 'strategy-value');

      row.appendChild(nameEl);
      row.appendChild(barWrap);
      row.appendChild(valueEl);
      return row;
    });

    replaceChildren(list, rows);
  }

  function logEntry(parts) {
    // parts is an array of {text, className?}
    const log = $('event-log');
    // Clear empty placeholder on first real entry
    const placeholder = log.querySelector('.empty');
    if (placeholder) placeholder.remove();

    const entry = document.createElement('div');
    entry.className = 'log-entry';

    const ts = new Date().toISOString().slice(11, 19);
    entry.appendChild(span(ts, 'ts'));

    for (let i = 0; i < parts.length; i++) {
      if (i > 0) entry.appendChild(document.createTextNode(' · '));
      entry.appendChild(span(parts[i].text, parts[i].className));
    }

    log.insertBefore(entry, log.firstChild);
    while (log.children.length > 50) log.removeChild(log.lastChild);
  }

  function setupSSE() {
    const es = new EventSource('/events');
    const dot = $('status-dot');

    es.addEventListener('hello', () => {
      dot.classList.remove('disconnected');
      dot.title = 'connected';
      logEntry([{ text: 'connected to HMC event stream', className: 'ok' }]);
    });

    es.addEventListener('ping', () => {
      // keepalive, no UI update
    });

    es.addEventListener('turn', (e) => {
      let d = {};
      try { d = JSON.parse(e.data || '{}'); } catch (_err) { return; }

      const parts = [];
      if (d.turn != null) parts.push({ text: 'turn ' + d.turn });
      if (d.context_tokens != null) parts.push({ text: 'ctx ' + fmt(d.context_tokens) });
      if (d.delta_saved > 0) {
        parts.push({ text: '+' + fmt(d.delta_saved) + ' saved', className: 'ok' });
      }
      if (parts.length > 0) logEntry(parts);
      refreshAll();
    });

    es.addEventListener('tool', (e) => {
      // Per-tool-call heartbeat so the dashboard shows activity
      // during long agent loops between user turns.
      let d = {};
      try { d = JSON.parse(e.data || '{}'); } catch (_err) { return; }

      const parts = [];
      if (d.tool_name) {
        parts.push({
          text: 'tool ' + d.tool_name,
          className: d.is_error ? 'warn' : undefined,
        });
      }
      if (d.context_tokens != null) parts.push({ text: 'ctx ' + fmt(d.context_tokens) });
      if (d.delta_saved > 0) {
        parts.push({ text: '+' + fmt(d.delta_saved) + ' saved', className: 'ok' });
      }
      if (parts.length > 0) logEntry(parts);
      refreshAll();
    });

    es.addEventListener('session_end', (e) => {
      let d = {};
      try { d = JSON.parse(e.data || '{}'); } catch (_err) { return; }
      logEntry([
        { text: 'session ended' },
        { text: 'total ' + fmt(d.total_saved) + ' saved', className: 'ok' },
      ]);
      refreshAll();
    });

    es.onerror = () => {
      dot.classList.add('disconnected');
      dot.title = 'disconnected — retrying';
    };
  }

  // Initial placeholders
  showEmpty($('strategy-list'), 'no strategies have fired yet');
  showEmpty($('event-log'), 'waiting for the next LLM call...');
  showEmpty($('sessions-container'), 'scanning session sidecars...');

  refreshAll();
  setInterval(refreshAll, 10000);  // safety-net poll in case SSE is blocked
  setupSSE();
})();
</script>
</body>
</html>
"""
