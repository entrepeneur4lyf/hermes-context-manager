"""Persistent token-savings analytics backed by SQLite.

HMC's runtime accumulator (``state.tokens_saved`` / ``tokens_saved_by_type``)
lives in memory and dies with the session.  This module is the cross-session,
cross-project, cross-day historical store.  It runs alongside the existing
JSON sidecars (which still hold the per-session runtime state) and writes
one row per (session, strategy) at session end.

Design notes (RTK port, see docs/superpowers/plans/2026-04-10-rtk-patterns.md):

- Storage: SQLite via stdlib ``sqlite3``.  No third-party dependencies.
- Concurrency: WAL journal mode + 5s busy_timeout so multiple Hermes sessions
  can write concurrently without serializing.  Connections are opened
  per-operation rather than held long-term, which is correct for our
  write-once-per-session pattern and sidesteps the ``check_same_thread``
  rules entirely.
- Retention: on-write 90-day TTL.  Every ``record_session`` call ends with a
  ``DELETE WHERE timestamp < cutoff``.  No background thread, no scheduler.
- Project keying: ``os.path.realpath(os.getcwd())`` captured at session start.
  Queries support both exact-project filtering and subdirectory GLOB
  filtering (sessions launched from a subdir match the project root).
- Failure mode: every public method swallows ``sqlite3.Error`` and
  ``OSError``, logs once via ``LOGGER.warning``, and returns a safe default.
  Analytics is observability; it must never crash the plugin.
"""
from __future__ import annotations

import logging
import os
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

LOGGER = logging.getLogger(__name__)

# All known strategy keys.  Used by tests + the query API to enumerate the
# expected breakdown.  New strategies must be added here when introduced.
KNOWN_STRATEGIES: tuple[str, ...] = (
    "short_circuit",
    "code_filter",
    "truncation",
    "dedup",
    "error_purge",
    "background_compression",
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS hmc_savings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL,
    session_id      TEXT    NOT NULL,
    project_path    TEXT    NOT NULL DEFAULT '',
    strategy        TEXT    NOT NULL,
    saved_tokens    INTEGER NOT NULL,
    session_input   INTEGER NOT NULL,
    session_output  INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_hmc_savings_timestamp
    ON hmc_savings(timestamp);

CREATE INDEX IF NOT EXISTS idx_hmc_savings_project_timestamp
    ON hmc_savings(project_path, timestamp);

CREATE INDEX IF NOT EXISTS idx_hmc_savings_session
    ON hmc_savings(session_id);
"""


@dataclass(slots=True)
class SavingsSummary:
    """Aggregate totals for a query window.

    Returned by ``AnalyticsStore.get_summary``; safe defaults (all zeros)
    are used when the table is empty so callers never need to special-case.
    """

    total_saved: int = 0
    total_sessions: int = 0
    total_input: int = 0
    total_output: int = 0
    by_strategy: dict[str, int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.by_strategy is None:
            self.by_strategy = {}

    @property
    def savings_pct(self) -> float:
        if self.total_input <= 0:
            return 0.0
        return (self.total_saved / self.total_input) * 100.0


class AnalyticsStore:
    """SQLite-backed cross-session token savings store.

    All write operations call ``_cleanup_old`` at the end so the database
    self-prunes without a background scheduler.  All read operations return
    safe defaults (zeros / empty lists) on error so the caller can render
    something instead of crashing.
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        retention_days: int = 90,
    ) -> None:
        self._db_path = Path(db_path) if db_path else self._default_db_path()
        self._retention_days = max(1, int(retention_days))
        self._init_schema()

    @staticmethod
    def _default_db_path() -> Path:
        """Resolve the analytics DB path.

        Order: HMC_DB_PATH env var, then ``~/.hermes/hmc_state/analytics.db``
        next to the existing JSON sidecars.
        """
        env_path = os.environ.get("HMC_DB_PATH")
        if env_path:
            return Path(env_path).expanduser()
        return Path.home() / ".hermes" / "hmc_state" / "analytics.db"

    @property
    def db_path(self) -> Path:
        return self._db_path

    def _ensure_parent(self) -> None:
        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            LOGGER.warning("HMC analytics: cannot create parent directory %s", self._db_path.parent)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Open a fresh connection per operation.

        Per-operation connections sidestep the ``check_same_thread`` rules
        and let multiple Hermes sessions write concurrently under WAL mode.
        SQLite handles file locking; the cost of ``connect()`` is negligible
        compared to the once-per-session write pattern.

        ``timeout=5.0`` is the runtime busy timeout: if a write lock is
        held by another connection, this one waits up to 5 seconds before
        raising ``OperationalError``.  Combined with WAL (which only
        serializes WRITERS -- readers never block) this is enough to keep
        concurrent Hermes sessions from ever hitting "database is locked"
        in practice.
        """
        self._ensure_parent()
        conn = sqlite3.connect(str(self._db_path), timeout=5.0)
        try:
            # busy_timeout and synchronous are PER-CONNECTION, not
            # persisted on disk like journal_mode.  We must set them on
            # every connect.  Without synchronous=NORMAL, every write
            # would fall back to the default FULL fsync, holding the
            # writer slot longer and increasing the risk of busy-timeout
            # collisions under concurrent sessions.
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("PRAGMA synchronous=NORMAL")
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        """Create schema, indexes, and lock in WAL journal mode.

        WAL is a database-level setting that PERSISTS on disk, so we set
        it exactly once here.  We then read back ``PRAGMA journal_mode``
        and log a warning if SQLite refused to enter WAL -- this happens
        on certain network filesystems and would silently downgrade us
        to DELETE journal mode (which serializes ALL access, not just
        writers).  ``busy_timeout`` and ``synchronous`` are per-connection
        and set in ``_connect``.
        """
        try:
            with self._connect() as conn:
                # Result of journal_mode PRAGMA is the mode that
                # ACTUALLY took effect, which may differ from what we
                # asked for (network FS, locked by another process,
                # etc.).
                actual_mode = conn.execute(
                    "PRAGMA journal_mode=WAL"
                ).fetchone()[0]
                if str(actual_mode).lower() != "wal":
                    LOGGER.warning(
                        "HMC analytics: SQLite refused WAL mode (got %r). "
                        "Concurrent sessions may experience 'database is "
                        "locked' errors. DB path: %s",
                        actual_mode, self._db_path,
                    )
                conn.executescript(_SCHEMA)
        except sqlite3.Error as exc:
            LOGGER.warning("HMC analytics: schema init failed: %s", exc)

    def journal_mode(self) -> str:
        """Return the current SQLite journal mode (for diagnostics/tests)."""
        try:
            with self._connect() as conn:
                row = conn.execute("PRAGMA journal_mode").fetchone()
                return str(row[0]).lower() if row else "unknown"
        except sqlite3.Error:
            return "error"

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def record_session(
        self,
        session_id: str,
        project_path: str,
        tokens_saved_by_type: dict[str, int],
        last_context_tokens: int | None,
        timestamp: float | None = None,
    ) -> int:
        """Persist one session's per-strategy savings.

        Writes one row per non-zero strategy in ``tokens_saved_by_type``.
        Sessions with zero total savings produce zero rows -- there is no
        value in noisy "no-op" entries cluttering the historical view.

        Returns the number of rows actually written so callers (and tests)
        can verify writes happened.
        """
        if not tokens_saved_by_type:
            return 0

        non_zero = {k: int(v) for k, v in tokens_saved_by_type.items() if v > 0}
        if not non_zero:
            return 0

        ts = datetime.fromtimestamp(timestamp or time.time(), tz=timezone.utc)
        ts_iso = ts.strftime("%Y-%m-%dT%H:%M:%SZ")

        session_output = int(last_context_tokens or 0)
        total_saved_this_session = sum(non_zero.values())
        session_input = session_output + total_saved_this_session

        rows = [
            (
                ts_iso,
                session_id,
                project_path or "",
                strategy,
                saved,
                session_input,
                session_output,
            )
            for strategy, saved in non_zero.items()
        ]

        try:
            with self._connect() as conn:
                conn.executemany(
                    """
                    INSERT INTO hmc_savings
                        (timestamp, session_id, project_path, strategy,
                         saved_tokens, session_input, session_output)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
                self._cleanup_old(conn)
            return len(rows)
        except sqlite3.Error as exc:
            LOGGER.warning("HMC analytics: record_session failed: %s", exc)
            return 0

    def _cleanup_old(self, conn: sqlite3.Connection) -> None:
        """Delete rows older than ``retention_days`` (on-write TTL)."""
        cutoff = datetime.fromtimestamp(
            time.time() - (self._retention_days * 86400),
            tz=timezone.utc,
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            conn.execute("DELETE FROM hmc_savings WHERE timestamp < ?", (cutoff,))
        except sqlite3.Error as exc:
            LOGGER.warning("HMC analytics: cleanup failed: %s", exc)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _escape_glob(s: str) -> str:
        """Escape SQLite GLOB metacharacters in a literal path string.

        GLOB treats ``*``, ``?``, ``[``, and ``]`` as wildcards.  A
        project path containing any of those would produce wrong
        query results (e.g. ``/home/u/proj[v2]`` would match paths
        like ``/home/u/projv`` or ``/home/u/proj2``).  We escape each
        metacharacter using GLOB's character-class form: ``*`` becomes
        ``[*]`` etc.  The one-element character class matches the
        literal.
        """
        out: list[str] = []
        for ch in s:
            if ch in "*?[":
                out.append(f"[{ch}]")
            elif ch == "]":
                # ']' outside a character class is literal, but inside
                # one it's the terminator.  Wrap in its own class for
                # safety when it occurs after any metacharacter.
                out.append("[]]")
            else:
                out.append(ch)
        return "".join(out)

    @staticmethod
    def _project_filter(project_path: str | None) -> tuple[str, list[Any]]:
        """Build a WHERE clause for project-scoped queries.

        Matches both the exact project root AND any subdirectory via GLOB,
        so a session launched from ``~/proj/src`` correctly attributes to
        project ``~/proj`` queries.  Empty/None means "all projects".

        The GLOB pattern is escaped so project paths containing
        ``*``, ``?``, ``[``, or ``]`` produce correct results.  The
        exact-match arm (``project_path = ?``) is unaffected and works
        regardless of metacharacters.
        """
        if not project_path:
            return "", []
        glob_pattern = (
            f"{AnalyticsStore._escape_glob(project_path)}{os.sep}*"
        )
        return (
            "AND (project_path = ? OR project_path GLOB ?)",
            [project_path, glob_pattern],
        )

    def get_summary(self, project_path: str | None = None) -> SavingsSummary:
        """Return aggregate totals across all stored data."""
        where, params = self._project_filter(project_path)
        try:
            with self._connect() as conn:
                row = conn.execute(
                    f"""
                    SELECT
                        COALESCE(SUM(saved_tokens), 0)        AS total_saved,
                        COUNT(DISTINCT session_id)            AS total_sessions
                    FROM hmc_savings
                    WHERE 1=1 {where}
                    """,
                    params,
                ).fetchone()
                total_saved = int(row[0] or 0)
                total_sessions = int(row[1] or 0)

                # Session input/output: take the max per session and sum.
                # (input/output are denormalized per row but identical
                # across rows of the same session, so MAX is the per-session
                # value.)
                io_row = conn.execute(
                    f"""
                    SELECT
                        COALESCE(SUM(session_input_max), 0),
                        COALESCE(SUM(session_output_max), 0)
                    FROM (
                        SELECT
                            session_id,
                            MAX(session_input)  AS session_input_max,
                            MAX(session_output) AS session_output_max
                        FROM hmc_savings
                        WHERE 1=1 {where}
                        GROUP BY session_id
                    )
                    """,
                    params,
                ).fetchone()
                total_input = int(io_row[0] or 0)
                total_output = int(io_row[1] or 0)

                strategy_rows = conn.execute(
                    f"""
                    SELECT strategy, COALESCE(SUM(saved_tokens), 0) AS s
                    FROM hmc_savings
                    WHERE 1=1 {where}
                    GROUP BY strategy
                    ORDER BY s DESC
                    """,
                    params,
                ).fetchall()
                by_strategy = {row[0]: int(row[1]) for row in strategy_rows}

            return SavingsSummary(
                total_saved=total_saved,
                total_sessions=total_sessions,
                total_input=total_input,
                total_output=total_output,
                by_strategy=by_strategy,
            )
        except sqlite3.Error as exc:
            LOGGER.warning("HMC analytics: get_summary failed: %s", exc)
            return SavingsSummary()

    def get_by_day(
        self,
        days: int = 30,
        project_path: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return per-day savings totals, newest first.

        Each entry: ``{"day": "YYYY-MM-DD", "saved": int, "sessions": int}``
        """
        where, params = self._project_filter(project_path)
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    f"""
                    SELECT
                        DATE(timestamp) AS day,
                        SUM(saved_tokens) AS saved,
                        COUNT(DISTINCT session_id) AS sessions
                    FROM hmc_savings
                    WHERE timestamp >= DATE('now', ?)
                    {where}
                    GROUP BY day
                    ORDER BY day DESC
                    """,
                    [f"-{int(days)} days", *params],
                ).fetchall()
            return [
                {"day": row[0], "saved": int(row[1] or 0), "sessions": int(row[2] or 0)}
                for row in rows
            ]
        except sqlite3.Error as exc:
            LOGGER.warning("HMC analytics: get_by_day failed: %s", exc)
            return []

    def get_by_month(
        self,
        months: int = 12,
        project_path: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return per-month savings totals, newest first.

        Uses SQLite's native ``-N months`` modifier rather than a
        ``months * 31 days`` approximation.  The approximation would
        over-return on a 12-month request by ~6 days, which shows up
        as a 13th partial month in the result set.
        """
        where, params = self._project_filter(project_path)
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    f"""
                    SELECT
                        strftime('%Y-%m', timestamp) AS month,
                        SUM(saved_tokens) AS saved,
                        COUNT(DISTINCT session_id) AS sessions
                    FROM hmc_savings
                    WHERE timestamp >= DATE('now', ?)
                    {where}
                    GROUP BY month
                    ORDER BY month DESC
                    """,
                    [f"-{int(months)} months", *params],
                ).fetchall()
            return [
                {"month": row[0], "saved": int(row[1] or 0), "sessions": int(row[2] or 0)}
                for row in rows
            ]
        except sqlite3.Error as exc:
            LOGGER.warning("HMC analytics: get_by_month failed: %s", exc)
            return []

    def get_recent_sessions(
        self,
        limit: int = 10,
        project_path: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return the most recent N sessions with totals."""
        where, params = self._project_filter(project_path)
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    f"""
                    SELECT
                        session_id,
                        MIN(timestamp)            AS started,
                        MAX(project_path)         AS project_path,
                        SUM(saved_tokens)         AS saved,
                        MAX(session_input)        AS session_input,
                        MAX(session_output)       AS session_output
                    FROM hmc_savings
                    WHERE 1=1 {where}
                    GROUP BY session_id
                    ORDER BY started DESC
                    LIMIT ?
                    """,
                    [*params, int(limit)],
                ).fetchall()
            return [
                {
                    "session_id": row[0],
                    "timestamp": row[1],
                    "project_path": row[2] or "",
                    "saved_tokens": int(row[3] or 0),
                    "session_input": int(row[4] or 0),
                    "session_output": int(row[5] or 0),
                }
                for row in rows
            ]
        except sqlite3.Error as exc:
            LOGGER.warning("HMC analytics: get_recent_sessions failed: %s", exc)
            return []

    def get_by_project(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return savings totals grouped by project_path, biggest savers first."""
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT
                        project_path,
                        SUM(saved_tokens) AS saved,
                        COUNT(DISTINCT session_id) AS sessions
                    FROM hmc_savings
                    GROUP BY project_path
                    ORDER BY saved DESC
                    LIMIT ?
                    """,
                    [int(limit)],
                ).fetchall()
            return [
                {
                    "project_path": row[0] or "",
                    "saved_tokens": int(row[1] or 0),
                    "sessions": int(row[2] or 0),
                }
                for row in rows
            ]
        except sqlite3.Error as exc:
            LOGGER.warning("HMC analytics: get_by_project failed: %s", exc)
            return []
