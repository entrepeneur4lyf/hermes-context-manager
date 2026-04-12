"""Sidecar JSON persistence for Hermes Context Manager state."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import NamedTemporaryFile

from .state import SessionState, session_state_from_dict, session_state_to_dict


class JsonStateStore:
    """Persist HMC session state under $HERMES_HOME."""

    def __init__(self, hermes_home: Path) -> None:
        self.base_dir = Path(hermes_home) / "hmc_state"
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, session_id: str) -> Path:
        safe_id = session_id.replace("/", "_")
        return self.base_dir / f"{safe_id}.json"

    def load(self, session_id: str) -> SessionState | None:
        path = self._path_for(session_id)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        return session_state_from_dict(payload)

    def save(self, session_id: str, state: SessionState) -> None:
        path = self._path_for(session_id)
        payload = session_state_to_dict(state)
        tmp_path: Path | None = None
        try:
            with NamedTemporaryFile(
                "w",
                encoding="utf-8",
                delete=False,
                dir=path.parent,
            ) as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                tmp_path = Path(handle.name)
            tmp_path.replace(path)
            tmp_path = None  # successfully moved into place
        finally:
            # Clean up the temp file if anything above raised.  Without
            # this, a ``json.dump`` failure (non-serializable value,
            # disk full) would leak ``tmp*`` files into the state
            # directory indefinitely.
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass

    def index_path(self, session_id: str) -> Path:
        safe_id = session_id.replace("/", "_")
        return self.base_dir / f"{safe_id}_index.jsonl"

    def append_index(self, session_id: str, entry: dict) -> None:
        import time
        path = self.index_path(session_id)
        entry.setdefault("indexed_at", time.time())
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, sort_keys=True) + "\n")

    def read_index(self, session_id: str) -> list[dict]:
        path = self.index_path(session_id)
        if not path.exists():
            return []
        entries = []
        for line in path.read_text(encoding="utf-8").strip().split("\n"):
            if line.strip():
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return entries

    def list_sessions(self, limit: int = 50) -> list[dict]:
        """Return a display-friendly survey of recent session sidecars.

        Scans ``hmc_state/*.json`` (excluding ``_index.jsonl`` files
        and the phantom ``.json`` file that 0.3.3's empty-session guard
        catches going forward), reads each, extracts just the fields
        the dashboard renders, and returns them sorted by mtime
        descending.  Limited to ``limit`` entries by default so the
        response stays small even on instances with hundreds of
        sessions on disk.

        The returned dicts are JSON-safe and contain:

        - ``session_id`` — full id (mtime-based, e.g. ``20260411_120000_b3df05``)
        - ``short_id`` — last 6 chars for compact display
        - ``mtime`` — epoch seconds of the sidecar's last save
        - ``project_path`` — canonical cwd at session start
        - ``tokens_kept_out_total`` — un-gated real savings
        - ``tokens_saved`` — gated unique-firing count
        - ``tokens_kept_out_by_type`` — per-strategy breakdown
        - ``last_context_tokens`` / ``last_context_percent``
        - ``tool_call_count`` — how many distinct tool_call_ids were tracked
        - ``ended`` — True if the sidecar's a dead record (no longer
          the active session in memory).  Callers can flag live vs
          historical.  We can't know ``ended`` from the sidecar alone
          (state.pop on session_end doesn't leave a marker), so this
          field is always ``False`` here -- the plugin-level caller
          sets it by comparing against ``self._states``.

        Phantoms (tiny ctx, no tools, no savings) are filtered OUT so
        the dashboard's "recent sessions" panel doesn't show noise.
        The caller can pass the raw list to further filter if needed.

        Safe against read errors: any sidecar that fails to parse is
        silently skipped.  This function never raises to callers --
        observability must not crash the dashboard.
        """
        results: list[dict] = []
        try:
            entries = sorted(
                (
                    p for p in self.base_dir.glob("*.json")
                    # Skip the empty-session phantom file (a legacy
                    # artifact from before 0.3.3's guard; we don't
                    # want to show it in the UI).
                    if p.name != ".json"
                ),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
        except OSError:
            return []

        for path in entries[:limit * 2]:  # room to filter phantoms
            try:
                mtime = path.stat().st_mtime
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue

            session_id = path.stem
            short_id = session_id[-6:] if len(session_id) >= 6 else session_id
            tokens_kept_out = int(payload.get("tokens_kept_out_total") or 0)
            tokens_saved = int(payload.get("tokens_saved") or 0)
            tool_count = len(payload.get("tool_calls") or {})
            ctx_tokens = payload.get("last_context_tokens")
            ctx_percent = payload.get("last_context_percent")

            # Phantom heuristic (mirrors plugin._is_phantom_session):
            # skip only if ALL three of (tiny ctx, no tools, no savings)
            # hold.  A real session with zero savings still shows up --
            # users should see those so they know HMC saw the session.
            if (
                (ctx_tokens or 0) < 1000
                and tool_count == 0
                and tokens_kept_out == 0
                and tokens_saved == 0
            ):
                continue

            results.append({
                "session_id": session_id,
                "short_id": short_id,
                "mtime": mtime,
                "project_path": str(payload.get("project_path") or ""),
                "tokens_kept_out_total": tokens_kept_out,
                "tokens_saved": tokens_saved,
                "tokens_kept_out_by_type": dict(
                    payload.get("tokens_kept_out_by_type") or {}
                ),
                "tokens_saved_by_type": dict(
                    payload.get("tokens_saved_by_type") or {}
                ),
                "last_context_tokens": ctx_tokens,
                "last_context_percent": ctx_percent,
                "tool_call_count": tool_count,
                "ended": False,  # caller can override
            })
            if len(results) >= limit:
                break
        return results

