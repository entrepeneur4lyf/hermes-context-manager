"""Tests for the opt-in web dashboard.

Covers:
- Dashboard lifecycle (start/stop/status, re-start is idempotent)
- REST endpoints return the expected JSON shape
- SSE endpoint accepts a client, publishes an event, client receives it
- The plugin fires turn events into the bus on on_pre_llm_call
- Session-end event fires on on_session_end
- Event bus fan-out to multiple subscribers
- Slow subscriber doesn't block publisher (queue.Full drop path)
- HTML page renders and contains the expected shell
- handle_hmc_control dashboard action routes correctly without a session
"""
from __future__ import annotations

import json
import queue
import socket
import tempfile
import threading
import time
import unittest
import urllib.parse
import urllib.request
from pathlib import Path
from urllib.error import URLError

from hermes_context_manager.config import HmcConfig
from hermes_context_manager.dashboard import Dashboard, _EventBus
from hermes_context_manager.plugin import HermesContextManagerPlugin
from hermes_context_manager.state import ToolRecord


def _fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=2.0) as r:
        return json.loads(r.read().decode("utf-8"))


def _fetch_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=2.0) as r:
        return r.read().decode("utf-8")


class EventBusTests(unittest.TestCase):
    def test_publish_fans_out_to_all_subscribers(self) -> None:
        bus = _EventBus()
        q1 = bus.subscribe()
        q2 = bus.subscribe()
        self.assertEqual(bus.subscriber_count(), 2)

        bus.publish({"type": "turn", "data": {"v": 1}})

        self.assertEqual(q1.get(timeout=1), {"type": "turn", "data": {"v": 1}})
        self.assertEqual(q2.get(timeout=1), {"type": "turn", "data": {"v": 1}})

    def test_unsubscribe_removes_queue(self) -> None:
        bus = _EventBus()
        q = bus.subscribe()
        self.assertEqual(bus.subscriber_count(), 1)
        bus.unsubscribe(q)
        self.assertEqual(bus.subscriber_count(), 0)

    def test_slow_subscriber_doesnt_block_publisher(self) -> None:
        """If one subscriber's queue is full, publish must not block."""
        bus = _EventBus()
        slow = bus.subscribe()
        fast = bus.subscribe()

        # Saturate the slow subscriber
        for _ in range(260):  # > _SUBSCRIBER_QUEUE_MAX (256)
            bus.publish({"type": "turn", "data": {}})

        # Fast subscriber should still receive events (it gets 256 then drops)
        received = 0
        try:
            while True:
                fast.get_nowait()
                received += 1
        except queue.Empty:
            pass
        self.assertGreater(received, 0)

        # Slow subscriber didn't block -- publisher returned
        self.assertGreater(slow.qsize(), 0)


class DashboardLifecycleTests(unittest.TestCase):
    def _make_plugin(self, tmp: str) -> HermesContextManagerPlugin:
        config = HmcConfig()
        config.analytics.db_path = str(Path(tmp) / "analytics.db")
        return HermesContextManagerPlugin(hermes_home=tmp, config=config)

    def test_start_returns_url_and_status_reports_running(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plugin = self._make_plugin(tmp)
            try:
                started = plugin.dashboard.start()
                self.assertTrue(started["running"])
                self.assertTrue(started["url"].startswith("http://127.0.0.1:"))
                self.assertEqual(started["host"], "127.0.0.1")
                self.assertGreater(started["port"], 0)

                status = plugin.dashboard.status()
                self.assertTrue(status["running"])
                self.assertEqual(status["url"], started["url"])
            finally:
                plugin.dashboard.stop()

    def test_start_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plugin = self._make_plugin(tmp)
            try:
                a = plugin.dashboard.start()
                b = plugin.dashboard.start()
                # Same port, both report running
                self.assertEqual(a["port"], b["port"])
                self.assertTrue(b["running"])
            finally:
                plugin.dashboard.stop()

    def test_stop_when_not_running_is_safe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plugin = self._make_plugin(tmp)
            result = plugin.dashboard.stop()
            self.assertEqual(result, {"running": False})

    def test_stop_then_start_reuses_instance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plugin = self._make_plugin(tmp)
            try:
                plugin.dashboard.start()
                plugin.dashboard.stop()
                self.assertFalse(plugin.dashboard.running)
                plugin.dashboard.start()
                self.assertTrue(plugin.dashboard.running)
            finally:
                plugin.dashboard.stop()

    def test_port_rotation_when_base_port_is_busy(self) -> None:
        """If the default port is taken, the dashboard rotates upward."""
        from hermes_context_manager.dashboard import Dashboard, _DEFAULT_PORT

        with tempfile.TemporaryDirectory() as tmp:
            plugin = self._make_plugin(tmp)
            # Dashboard 1 holds the base port
            d1 = Dashboard(plugin_ref=lambda: plugin, port=_DEFAULT_PORT)
            # Dashboard 2 should rotate to base+1
            d2 = Dashboard(plugin_ref=lambda: plugin, port=_DEFAULT_PORT)
            try:
                s1 = d1.start()
                if not s1.get("running"):
                    self.skipTest(
                        f"base port {_DEFAULT_PORT} unavailable: {s1}",
                    )
                s2 = d2.start()
                self.assertTrue(s2["running"])
                self.assertNotEqual(s1["port"], s2["port"])
                # The rotation should land in the expected window
                self.assertGreaterEqual(s2["port"], _DEFAULT_PORT)
                self.assertLess(s2["port"], _DEFAULT_PORT + 100)
            finally:
                d2.stop()
                d1.stop()

    def test_port_zero_still_uses_os_pick(self) -> None:
        """Port=0 is an opt-out: let the OS pick, no rotation."""
        from hermes_context_manager.dashboard import Dashboard

        with tempfile.TemporaryDirectory() as tmp:
            plugin = self._make_plugin(tmp)
            d = Dashboard(plugin_ref=lambda: plugin, port=0)
            try:
                status = d.start()
                self.assertTrue(status["running"])
                # OS-picked ports are typically in the ephemeral range,
                # definitely not in our default 48800 block
                self.assertGreater(status["port"], 0)
            finally:
                d.stop()


class DashboardHttpTests(unittest.TestCase):
    def _make_plugin(self, tmp: str) -> HermesContextManagerPlugin:
        config = HmcConfig()
        config.analytics.db_path = str(Path(tmp) / "analytics.db")
        return HermesContextManagerPlugin(hermes_home=tmp, config=config)

    def test_root_serves_html_page(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plugin = self._make_plugin(tmp)
            try:
                status = plugin.dashboard.start()
                html = _fetch_text(status["url"])
                self.assertIn("HMC Dashboard", html)
                self.assertIn("/events", html)  # SSE wiring present
                self.assertIn("/api/summary", html)
            finally:
                plugin.dashboard.stop()

    def test_api_summary_reflects_current_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plugin = self._make_plugin(tmp)
            try:
                state = plugin._get_state("s1")
                # Dashboard "session_saved" is now the un-gated accumulator.
                state.tokens_kept_out_total = 500
                state.tokens_kept_out_by_type = {
                    "truncation": 300, "short_circuit": 200,
                }
                state.tokens_saved = 500
                state.tokens_saved_by_type = {"truncation": 300, "short_circuit": 200}
                state.last_context_tokens = 2000
                state.last_context_percent = 0.33
                state.current_turn = 2
                plugin._active_session_id = "s1"

                status = plugin.dashboard.start()
                data = _fetch_json(status["url"] + "api/summary")

                self.assertEqual(data["session_id"], "s1")
                self.assertEqual(data["session_saved"], 500)
                self.assertEqual(data["context_tokens"], 2000)
                self.assertAlmostEqual(data["context_percent"], 0.33)
                self.assertEqual(data["turns"], 2)
                self.assertEqual(data["by_strategy"], {"truncation": 300, "short_circuit": 200})
                # Diagnostic gated counter ships alongside.
                self.assertEqual(data["uniq_saved"], 500)
            finally:
                plugin.dashboard.stop()

    def test_api_summary_handles_no_active_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plugin = self._make_plugin(tmp)
            try:
                status = plugin.dashboard.start()
                data = _fetch_json(status["url"] + "api/summary")
                self.assertIsNone(data["session_id"])
                self.assertEqual(data["session_saved"], 0)
                self.assertEqual(data["by_strategy"], {})
            finally:
                plugin.dashboard.stop()

    def test_api_analytics_returns_store_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plugin = self._make_plugin(tmp)
            plugin.analytics_store.record_session(
                "seed", "/proj", {"truncation": 400}, 1000,
            )
            try:
                status = plugin.dashboard.start()
                data = _fetch_json(status["url"] + "api/analytics")
                self.assertEqual(data["total_saved"], 400)
                self.assertEqual(data["total_sessions"], 1)
                self.assertIn("truncation", data["by_strategy"])
            finally:
                plugin.dashboard.stop()

    def test_api_sessions_returns_recent_sessions_from_sidecars(self) -> None:
        """Regression test for the 0.3.7 workhorses panel.

        Dashboard /api/sessions surveys the hmc_state sidecar
        directory and returns a display-ready list sorted by mtime
        descending, with per-session tool counts, savings, and
        per-strategy breakdowns.
        """
        def _mk_tool(i: int) -> ToolRecord:
            return ToolRecord(
                tool_call_id=f"tc_{i}",
                tool_name="terminal",
                input_args={"i": i},
                input_fingerprint=f"terminal::{i}",
                is_error=False,
                turn_index=0,
                timestamp=1.0,
                token_estimate=10,
            )

        with tempfile.TemporaryDirectory() as tmp:
            plugin = self._make_plugin(tmp)
            try:
                # Seed two sessions and persist them to disk.
                s1 = plugin._get_state("20260411_120000_aaa111")
                s1.tokens_kept_out_total = 11164
                s1.tokens_saved = 4707
                s1.tokens_kept_out_by_type = {"dedup": 4993, "short_circuit": 6171}
                s1.last_context_tokens = 134965
                s1.last_context_percent = 0.67
                for i in range(100):
                    s1.tool_calls[f"tc_{i}"] = _mk_tool(i)
                plugin._save_state("20260411_120000_aaa111")

                s2 = plugin._get_state("20260411_175812_bbb222")
                s2.tokens_kept_out_total = 320
                s2.tokens_saved = 58
                s2.tokens_kept_out_by_type = {"short_circuit": 320}
                s2.last_context_tokens = 82244
                s2.last_context_percent = 0.41
                for i in range(37):
                    s2.tool_calls[f"tc_{i}"] = _mk_tool(i)
                plugin._save_state("20260411_175812_bbb222")
                plugin._active_session_id = "20260411_175812_bbb222"

                status = plugin.dashboard.start()
                data = _fetch_json(status["url"] + "api/sessions")
                sessions = data["sessions"]

                # Both sessions should be present.
                ids = {s["session_id"] for s in sessions}
                self.assertIn("20260411_120000_aaa111", ids)
                self.assertIn("20260411_175812_bbb222", ids)

                # Find each and spot-check fields.
                by_id = {s["session_id"]: s for s in sessions}
                a = by_id["20260411_120000_aaa111"]
                self.assertEqual(a["short_id"], "aaa111")
                self.assertEqual(a["tokens_kept_out_total"], 11164)
                self.assertEqual(a["tool_call_count"], 100)
                self.assertEqual(
                    a["tokens_kept_out_by_type"],
                    {"dedup": 4993, "short_circuit": 6171},
                )
                self.assertAlmostEqual(a["last_context_percent"], 0.67)

                b = by_id["20260411_175812_bbb222"]
                self.assertEqual(b["short_id"], "bbb222")
                self.assertEqual(b["tokens_kept_out_total"], 320)
                self.assertTrue(b["live"])    # still in plugin._states
                self.assertTrue(b["active"])  # matches _active_session_id
                self.assertFalse(a["active"]) # a is not the active session
            finally:
                plugin.dashboard.stop()

    def test_api_sessions_filters_phantom_sidecars(self) -> None:
        """Phantom sessions (tiny ctx, no tools, no savings) must not appear."""
        with tempfile.TemporaryDirectory() as tmp:
            plugin = self._make_plugin(tmp)
            try:
                # Phantom-shaped: tiny context, no tools, no savings
                p = plugin._get_state("phantom_111")
                p.last_context_tokens = 300
                plugin._save_state("phantom_111")

                # Real session
                r = plugin._get_state("real_222")
                r.last_context_tokens = 50000
                r.tokens_kept_out_total = 500
                r.tokens_kept_out_by_type = {"short_circuit": 500}
                plugin._save_state("real_222")

                status = plugin.dashboard.start()
                data = _fetch_json(status["url"] + "api/sessions")
                ids = {s["session_id"] for s in data["sessions"]}
                self.assertIn("real_222", ids)
                self.assertNotIn("phantom_111", ids)
            finally:
                plugin.dashboard.stop()

    def test_api_history_returns_turn_buffer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plugin = self._make_plugin(tmp)
            try:
                state = plugin._get_state("s1")
                state.turn_history.append({
                    "turn": 1,
                    "context_tokens": 1500,
                    "cumulative_saved": 100,
                    "delta_saved": 100,
                    "by_strategy": {"truncation": 100},
                })
                plugin._active_session_id = "s1"

                status = plugin.dashboard.start()
                data = _fetch_json(status["url"] + "api/history")
                self.assertEqual(len(data["turns"]), 1)
                self.assertEqual(data["turns"][0]["turn"], 1)
                self.assertEqual(data["turns"][0]["cumulative_saved"], 100)
            finally:
                plugin.dashboard.stop()

    def test_unknown_route_returns_404(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plugin = self._make_plugin(tmp)
            try:
                status = plugin.dashboard.start()
                with self.assertRaises(URLError):
                    urllib.request.urlopen(status["url"] + "nope", timeout=2.0)
            finally:
                plugin.dashboard.stop()


class DashboardSseTests(unittest.TestCase):
    def _make_plugin(self, tmp: str) -> HermesContextManagerPlugin:
        config = HmcConfig()
        config.analytics.db_path = str(Path(tmp) / "analytics.db")
        return HermesContextManagerPlugin(hermes_home=tmp, config=config)

    def test_sse_stream_delivers_published_event(self) -> None:
        """End-to-end: open /events, publish an event, receive it.

        Uses a raw socket instead of urllib because urllib's HTTPResponse
        buffers reads in a way that doesn't play well with a streaming
        endpoint that's supposed to deliver events byte-for-byte.
        """
        with tempfile.TemporaryDirectory() as tmp:
            plugin = self._make_plugin(tmp)
            try:
                status = plugin.dashboard.start()
                parsed = urllib.parse.urlparse(status["url"])
                host = parsed.hostname or "127.0.0.1"
                port = parsed.port or 80

                received: list[str] = []
                stop_reading = threading.Event()
                ready = threading.Event()

                def sse_reader() -> None:
                    try:
                        sock = socket.create_connection((host, port), timeout=5.0)
                        sock.sendall(
                            b"GET /events HTTP/1.1\r\n"
                            b"Host: " + host.encode() + b"\r\n"
                            b"Accept: text/event-stream\r\n"
                            b"Connection: keep-alive\r\n"
                            b"\r\n"
                        )
                        # Consume response headers (end at \r\n\r\n)
                        header_buf = b""
                        while b"\r\n\r\n" not in header_buf:
                            chunk = sock.recv(4096)
                            if not chunk:
                                return
                            header_buf += chunk
                            if len(header_buf) > 8192:
                                return  # headers too long, give up

                        _, body_start = header_buf.split(b"\r\n\r\n", 1)
                        body = body_start
                        ready.set()

                        # Read body bytes until we have the events we want
                        sock.settimeout(3.0)
                        while not stop_reading.is_set():
                            try:
                                chunk = sock.recv(4096)
                            except socket.timeout:
                                break
                            if not chunk:
                                break
                            body += chunk
                            text = body.decode("utf-8", errors="ignore")
                            while "\n\n" in text:
                                event_block, text = text.split("\n\n", 1)
                                if event_block.strip():
                                    received.append(event_block)
                                body = text.encode("utf-8")
                            if len(received) >= 2:
                                return
                    except Exception:
                        pass
                    finally:
                        try:
                            sock.close()  # type: ignore[name-defined]
                        except Exception:
                            pass

                thread = threading.Thread(target=sse_reader, daemon=True)
                thread.start()

                # Wait for the reader to finish consuming headers +
                # register with the bus (happens inside the handler)
                self.assertTrue(ready.wait(timeout=5.0), "reader never got ready")
                # Small settle so the server's hello event lands
                time.sleep(0.1)

                # Publish a turn event through the plugin's dashboard
                plugin.dashboard.publish("turn", {
                    "turn": 5,
                    "context_tokens": 3000,
                    "delta_saved": 250,
                })

                thread.join(timeout=3.0)
                stop_reading.set()

                # We should have received at least the hello event and
                # our published turn event
                self.assertGreaterEqual(
                    len(received), 2,
                    f"expected >=2 events, got {len(received)}: {received}",
                )
                joined = "\n".join(received)
                self.assertIn("event: hello", joined)
                self.assertIn("event: turn", joined)
                self.assertIn('"turn": 5', joined)
            finally:
                plugin.dashboard.stop()

    def test_publish_on_dormant_dashboard_is_noop(self) -> None:
        """Calling publish when the server isn't running must not raise."""
        with tempfile.TemporaryDirectory() as tmp:
            plugin = self._make_plugin(tmp)
            # Don't start the dashboard
            self.assertFalse(plugin.dashboard.running)
            plugin.dashboard.publish("turn", {"turn": 1})  # must not raise


class DashboardHookIntegrationTests(unittest.TestCase):
    """Verify hooks publish to the dashboard bus during normal flow."""

    def _make_plugin(self, tmp: str) -> HermesContextManagerPlugin:
        config = HmcConfig()
        config.analytics.db_path = str(Path(tmp) / "analytics.db")
        return HermesContextManagerPlugin(hermes_home=tmp, config=config)

    def test_on_pre_llm_call_publishes_turn_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plugin = self._make_plugin(tmp)
            try:
                plugin.dashboard.start()

                # Subscribe directly to the bus to capture published events
                bus = plugin.dashboard._bus
                q = bus.subscribe()

                # Conversation must exceed the phantom-session threshold
                # (_PHANTOM_CONTEXT_THRESHOLD tokens post-0.3.3) so the
                # publish isn't filtered as an auxiliary worker.
                # ``x`` character * 5000 ≈ 1250 tokens at len/4 estimation.
                bulk_payload = "x" * 5000
                messages = [
                    {"role": "user", "content": "hello " + bulk_payload, "timestamp": 1.0},
                    {"role": "assistant", "content": "hi " + bulk_payload, "timestamp": 2.0},
                ]
                plugin.on_pre_llm_call(
                    session_id="s1",
                    user_message="hello",
                    conversation_history=messages,
                    is_first_turn=False,
                    model="test",
                    platform="cli",
                )

                # Drain the queue looking for a turn event
                events: list[dict] = []
                try:
                    while True:
                        events.append(q.get(timeout=0.5))
                except queue.Empty:
                    pass

                turn_events = [e for e in events if e.get("type") == "turn"]
                self.assertEqual(len(turn_events), 1)
                self.assertEqual(turn_events[0]["data"]["turn"], 1)
                bus.unsubscribe(q)
            finally:
                plugin.dashboard.stop()

    def test_on_session_end_publishes_session_end_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plugin = self._make_plugin(tmp)
            try:
                plugin.dashboard.start()
                state = plugin._get_state("s1")
                # Session-end events carry the un-gated totals.
                state.tokens_kept_out_total = 1234
                state.tokens_kept_out_by_type = {
                    "truncation": 1000, "code_filter": 234,
                }
                state.tokens_saved = 1234
                state.tokens_saved_by_type = {"truncation": 1000, "code_filter": 234}

                bus = plugin.dashboard._bus
                q = bus.subscribe()

                plugin.on_session_end(
                    session_id="s1",
                    completed=True,
                    interrupted=False,
                    model="test",
                    platform="cli",
                )

                events: list[dict] = []
                try:
                    while True:
                        events.append(q.get(timeout=0.5))
                except queue.Empty:
                    pass

                end_events = [e for e in events if e.get("type") == "session_end"]
                self.assertEqual(len(end_events), 1)
                self.assertEqual(end_events[0]["data"]["session_id"], "s1")
                self.assertEqual(end_events[0]["data"]["total_saved"], 1234)
                bus.unsubscribe(q)
            finally:
                plugin.dashboard.stop()


class DashboardActionRoutingTests(unittest.TestCase):
    """handle_hmc_control must route `dashboard` without needing a session."""

    def _make_plugin(self, tmp: str) -> HermesContextManagerPlugin:
        config = HmcConfig()
        config.analytics.db_path = str(Path(tmp) / "analytics.db")
        return HermesContextManagerPlugin(hermes_home=tmp, config=config)

    def test_dashboard_action_works_with_no_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plugin = self._make_plugin(tmp)
            try:
                result = json.loads(plugin.handle_hmc_control(
                    {"action": "dashboard", "sub_action": "start"}, task_id=None,
                ))
                self.assertTrue(result["running"])
                self.assertIn("url", result)
            finally:
                plugin.dashboard.stop()

    def test_dashboard_action_works_with_ambiguous_sessions(self) -> None:
        """Regular actions error on ambiguous session; dashboard must succeed."""
        with tempfile.TemporaryDirectory() as tmp:
            plugin = self._make_plugin(tmp)
            plugin._get_state("s1")
            plugin._get_state("s2")
            try:
                result = json.loads(plugin.handle_hmc_control(
                    {"action": "dashboard", "sub_action": "start"}, task_id=None,
                ))
                self.assertTrue(result["running"])
            finally:
                plugin.dashboard.stop()

    def test_dashboard_status_action_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plugin = self._make_plugin(tmp)
            result = json.loads(plugin.handle_hmc_control(
                {"action": "dashboard"}, task_id=None,
            ))
            self.assertIn("running", result)
            self.assertFalse(result["running"])

    def test_dashboard_stop_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plugin = self._make_plugin(tmp)
            plugin.handle_hmc_control(
                {"action": "dashboard", "sub_action": "start"}, task_id=None,
            )
            result = json.loads(plugin.handle_hmc_control(
                {"action": "dashboard", "sub_action": "stop"}, task_id=None,
            ))
            self.assertFalse(result["running"])

    def test_dashboard_stop_does_not_hold_plugin_lock(self) -> None:
        """Regression: dashboard.stop() called via handle_hmc_control
        must NOT hold ``plugin._lock``.

        ``server.shutdown()`` can block for up to the ``serve_forever``
        poll interval (~500ms).  If that blocking happens while
        ``plugin._lock`` is held, every concurrent Hermes hook call
        stalls for the same duration.  This test holds the lock in a
        background thread during the stop call and confirms the stop
        still completes promptly.
        """
        with tempfile.TemporaryDirectory() as tmp:
            plugin = self._make_plugin(tmp)
            plugin.handle_hmc_control(
                {"action": "dashboard", "sub_action": "start"}, task_id=None,
            )

            lock_held = threading.Event()
            release_lock = threading.Event()

            def hold_plugin_lock() -> None:
                with plugin._lock:
                    lock_held.set()
                    release_lock.wait(timeout=5.0)

            holder = threading.Thread(target=hold_plugin_lock, daemon=True)
            holder.start()
            self.assertTrue(lock_held.wait(timeout=2.0),
                            "holder thread never acquired lock")

            # Now call the stop action while the lock is held.  If
            # handle_hmc_control tries to re-acquire the lock (or if
            # dashboard.stop waits on it), this call will block.
            # Give it a generous 2-second budget; anything more means
            # the fix regressed.
            start_time = time.monotonic()
            stop_done = threading.Event()
            stop_result: dict = {}

            def do_stop() -> None:
                try:
                    stop_result["data"] = json.loads(plugin.handle_hmc_control(
                        {"action": "dashboard", "sub_action": "stop"},
                        task_id=None,
                    ))
                finally:
                    stop_done.set()

            stop_thread = threading.Thread(target=do_stop, daemon=True)
            stop_thread.start()

            # The stop call should return within ~1 second despite the
            # lock being held elsewhere -- server.shutdown()'s own
            # poll interval is the only real wait.
            completed = stop_done.wait(timeout=2.0)
            elapsed = time.monotonic() - start_time

            # Release the lock so the holder thread can exit cleanly
            release_lock.set()
            holder.join(timeout=2.0)
            stop_thread.join(timeout=2.0)

            self.assertTrue(
                completed,
                f"dashboard stop took >2s while plugin lock was held "
                f"(elapsed={elapsed:.2f}s) — it is waiting on the lock",
            )
            self.assertFalse(stop_result["data"]["running"])


if __name__ == "__main__":
    unittest.main()
