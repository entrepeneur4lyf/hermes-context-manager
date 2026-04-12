"""Tests for the SQLite-backed analytics layer.

Covers:
- Schema creation (idempotent, indexes present)
- Single-session round trip + multi-strategy writes
- Zero-savings sessions skipped (no noisy no-op rows)
- On-write retention cleanup
- Project filter (exact match + subdirectory GLOB)
- Aggregation queries (summary, by_day, by_month, by_project, recent_sessions)
- WAL pragma + concurrent writes
- HMC_DB_PATH env override
- Plugin integration: on_session_end writes when there are savings;
  no row written when the session had zero savings
- handle_hmc_control "analytics" action: defaults, project scope, periods
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
import threading
import time
import unittest
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from hermes_context_manager.analytics import (
    KNOWN_STRATEGIES,
    AnalyticsStore,
    SavingsSummary,
)
from hermes_context_manager.config import HmcConfig
from hermes_context_manager.plugin import HermesContextManagerPlugin
from hermes_context_manager.state import create_state


class AnalyticsStoreTests(unittest.TestCase):
    def _make_store(self, tmp: str, retention_days: int = 90) -> AnalyticsStore:
        return AnalyticsStore(
            db_path=Path(tmp) / "analytics.db",
            retention_days=retention_days,
        )

    def test_schema_creation_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self._make_store(tmp)
            self._make_store(tmp)  # second init must not raise
            db_path = Path(tmp) / "analytics.db"
            # NB: ``with sqlite3.connect(...) as conn`` only commits/
            # rolls back the transaction on exit, it does NOT close the
            # connection.  Wrap in ``contextlib.closing`` so GC doesn't
            # emit ResourceWarning.
            with closing(sqlite3.connect(db_path)) as conn:
                tables = [row[0] for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()]
                indexes = [row[0] for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='index' "
                    "AND name NOT LIKE 'sqlite_%'"
                ).fetchall()]
            self.assertIn("hmc_savings", tables)
            self.assertIn("idx_hmc_savings_timestamp", indexes)
            self.assertIn("idx_hmc_savings_project_timestamp", indexes)

    def test_record_session_writes_one_row_per_strategy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self._make_store(tmp)
            written = store.record_session(
                session_id="s1",
                project_path="/proj/a",
                tokens_saved_by_type={
                    "truncation": 100,
                    "short_circuit": 200,
                    "dedup": 50,
                },
                last_context_tokens=400,
            )
            self.assertEqual(written, 3)

            summary = store.get_summary()
            self.assertEqual(summary.total_saved, 350)
            self.assertEqual(summary.total_sessions, 1)
            self.assertEqual(summary.total_output, 400)
            self.assertEqual(summary.total_input, 750)  # 400 + 350
            self.assertEqual(summary.by_strategy["truncation"], 100)
            self.assertEqual(summary.by_strategy["short_circuit"], 200)
            self.assertEqual(summary.by_strategy["dedup"], 50)
            self.assertGreater(summary.savings_pct, 0)

    def test_zero_savings_strategies_are_skipped(self) -> None:
        """Strategies with zero savings produce no rows."""
        with tempfile.TemporaryDirectory() as tmp:
            store = self._make_store(tmp)
            written = store.record_session(
                session_id="s1",
                project_path="/proj",
                tokens_saved_by_type={"truncation": 100, "dedup": 0, "error_purge": 0},
                last_context_tokens=200,
            )
            self.assertEqual(written, 1)
            summary = store.get_summary()
            self.assertEqual(summary.by_strategy, {"truncation": 100})

    def test_empty_savings_dict_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self._make_store(tmp)
            self.assertEqual(
                store.record_session("s1", "/p", {}, 100),
                0,
            )
            self.assertEqual(store.get_summary().total_sessions, 0)

    def test_all_zero_savings_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self._make_store(tmp)
            self.assertEqual(
                store.record_session("s1", "/p", {"truncation": 0}, 100),
                0,
            )
            self.assertEqual(store.get_summary().total_sessions, 0)

    def test_retention_cleanup_deletes_old_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self._make_store(tmp, retention_days=30)
            # Backdated 60 days ago
            old_ts = (
                datetime.now(timezone.utc) - timedelta(days=60)
            ).timestamp()
            store.record_session(
                "old", "/p", {"truncation": 999}, 1000,
                timestamp=old_ts,
            )
            # Recent
            store.record_session("new", "/p", {"truncation": 50}, 100)

            summary = store.get_summary()
            # Old row should have been swept by the cleanup at end of write
            self.assertEqual(summary.total_sessions, 1)
            self.assertEqual(summary.total_saved, 50)

    def test_project_filter_exact_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self._make_store(tmp)
            store.record_session("s1", "/proj/a", {"truncation": 100}, 200)
            store.record_session("s2", "/proj/b", {"truncation": 200}, 300)

            scoped = store.get_summary(project_path="/proj/a")
            self.assertEqual(scoped.total_saved, 100)
            self.assertEqual(scoped.total_sessions, 1)

    def test_project_filter_matches_subdirectories_via_glob(self) -> None:
        """Sessions started from a subdir must match project root queries."""
        with tempfile.TemporaryDirectory() as tmp:
            store = self._make_store(tmp)
            store.record_session("s1", "/proj/a", {"truncation": 100}, 200)
            store.record_session("s2", "/proj/a/src", {"truncation": 50}, 100)
            store.record_session("s3", "/proj/a/tests/unit", {"truncation": 25}, 50)
            store.record_session("s4", "/elsewhere", {"truncation": 999}, 1000)

            scoped = store.get_summary(project_path="/proj/a")
            self.assertEqual(scoped.total_saved, 175)  # 100 + 50 + 25
            self.assertEqual(scoped.total_sessions, 3)
            self.assertNotIn("/elsewhere", str(scoped.by_strategy))

    def test_get_by_day_returns_recent_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self._make_store(tmp)
            store.record_session("s1", "/p", {"truncation": 100}, 200)
            rows = store.get_by_day(days=7)
            self.assertEqual(len(rows), 1)
            self.assertIn("day", rows[0])
            self.assertEqual(rows[0]["saved"], 100)
            self.assertEqual(rows[0]["sessions"], 1)

    def test_get_by_month_returns_recent_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self._make_store(tmp)
            store.record_session("s1", "/p", {"truncation": 100}, 200)
            rows = store.get_by_month(months=3)
            self.assertEqual(len(rows), 1)
            self.assertIn("month", rows[0])

    def test_get_by_month_uses_calendar_months_not_31_day_approx(self) -> None:
        """Regression: months=12 must NOT return 13 months of data.

        The old implementation used ``-{months * 31} days`` which is
        ~372 days for months=12, returning an extra partial month.
        """
        with tempfile.TemporaryDirectory() as tmp:
            store = self._make_store(tmp)
            # Seed rows spanning a wide time range.  Include a row
            # from 13 months ago (should be excluded by months=12)
            # and a row from 11 months ago (should be included).
            old_ts = (
                datetime.now(timezone.utc) - timedelta(days=31 * 13)
            ).timestamp()
            medium_ts = (
                datetime.now(timezone.utc) - timedelta(days=31 * 11)
            ).timestamp()
            store.record_session(
                "old", "/p", {"truncation": 1}, 10,
                timestamp=old_ts,
            )
            store.record_session(
                "medium", "/p", {"truncation": 2}, 20,
                timestamp=medium_ts,
            )
            store.record_session("recent", "/p", {"truncation": 3}, 30)

            rows = store.get_by_month(months=12)
            # With calendar months, 13-months-ago is cleanly excluded.
            # With the old 372-day approximation, it would be included.
            months = [r["month"] for r in rows]
            months_ago_13 = (
                datetime.now(timezone.utc) - timedelta(days=31 * 13)
            ).strftime("%Y-%m")
            self.assertNotIn(months_ago_13, months)

    def test_project_filter_escapes_glob_metacharacters(self) -> None:
        """Regression: project paths with ``[``, ``?``, ``*`` must produce
        correct GLOB queries.

        The naive implementation would let SQLite GLOB interpret
        bracket notation in the path as a character class, so a
        session at ``/home/u/project[v2]/src`` would falsely match
        ``/home/u/projectv/src`` or ``/home/u/project2/src``.
        """
        with tempfile.TemporaryDirectory() as tmp:
            store = self._make_store(tmp)
            # Seed sessions in two projects whose paths differ only in
            # a bracket — a naive GLOB would confuse them.
            store.record_session(
                "bracket", "/home/u/project[v2]/src", {"truncation": 100}, 200,
            )
            store.record_session(
                "plain_v", "/home/u/projectv/src", {"truncation": 999}, 1000,
            )
            store.record_session(
                "plain_2", "/home/u/project2/src", {"truncation": 999}, 1000,
            )

            # Query for the bracketed project.  The escaped GLOB must
            # match only the bracketed project's sessions, not the
            # lookalike plain ones.
            summary = store.get_summary(project_path="/home/u/project[v2]")
            self.assertEqual(
                summary.total_saved, 100,
                f"GLOB escape failed -- leaked sessions from lookalike "
                f"paths; got {summary.by_strategy}",
            )
            self.assertEqual(summary.total_sessions, 1)

    def test_escape_glob_helper(self) -> None:
        from hermes_context_manager.analytics import AnalyticsStore as _AS
        self.assertEqual(_AS._escape_glob("plain"), "plain")
        self.assertEqual(_AS._escape_glob("a*b"), "a[*]b")
        self.assertEqual(_AS._escape_glob("a?b"), "a[?]b")
        self.assertEqual(_AS._escape_glob("a[b"), "a[[]b")
        self.assertEqual(_AS._escape_glob("a]b"), "a[]]b")
        self.assertEqual(_AS._escape_glob("a*?[b]"), "a[*][?][[]b[]]")

    def test_get_by_project_orders_by_savings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self._make_store(tmp)
            store.record_session("s1", "/big", {"truncation": 1000}, 100)
            store.record_session("s2", "/small", {"truncation": 10}, 100)
            store.record_session("s3", "/medium", {"truncation": 100}, 100)
            rows = store.get_by_project()
            self.assertEqual([r["project_path"] for r in rows], ["/big", "/medium", "/small"])

    def test_get_recent_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self._make_store(tmp)
            for i in range(5):
                store.record_session(
                    f"sess_{i}",
                    "/p",
                    {"truncation": 10 * (i + 1)},
                    100,
                    timestamp=time.time() + i,  # ensure distinct ts
                )
            rows = store.get_recent_sessions(limit=3)
            self.assertEqual(len(rows), 3)
            # Newest first
            self.assertEqual(rows[0]["session_id"], "sess_4")

    def test_empty_store_summary_returns_zeros(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self._make_store(tmp)
            summary = store.get_summary()
            self.assertEqual(summary.total_saved, 0)
            self.assertEqual(summary.total_sessions, 0)
            self.assertEqual(summary.by_strategy, {})
            self.assertEqual(summary.savings_pct, 0.0)

    def test_known_strategies_constant_is_complete(self) -> None:
        """All HMC strategies that write to tokens_saved_by_type are listed."""
        # If a new strategy ships without being added here, queries that
        # iterate KNOWN_STRATEGIES would silently miss it.
        for s in ("short_circuit", "code_filter", "truncation", "dedup",
                  "error_purge", "background_compression"):
            self.assertIn(s, KNOWN_STRATEGIES)

    def test_wal_mode_is_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self._make_store(tmp)
            store.record_session("s1", "/p", {"truncation": 50}, 100)

            # Verify via the store helper
            self.assertEqual(store.journal_mode(), "wal")

            # Verify via a brand-new raw sqlite3 connection that did NOT
            # go through the AnalyticsStore wrapper.  This proves WAL is
            # persisted on disk, not just set on the connection that
            # opened it.  ``contextlib.closing`` is required because
            # ``with sqlite3.connect()`` only commits/rolls back -- it
            # does not close the connection.
            with closing(sqlite3.connect(store.db_path)) as conn:
                mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
            self.assertEqual(mode.lower(), "wal")

    def test_wal_sidecar_files_created_during_open_writer(self) -> None:
        """SQLite WAL creates -wal/-shm sidecars while a writer is active.

        The files may be checkpointed and removed when all connections
        close cleanly, so we hold a writer connection open during the
        check.
        """
        with tempfile.TemporaryDirectory() as tmp:
            store = self._make_store(tmp)
            store.record_session("s1", "/p", {"truncation": 50}, 100)
            # Hold a connection open so SQLite can't checkpoint+remove
            holder = sqlite3.connect(str(store.db_path), timeout=5.0)
            holder.execute("PRAGMA busy_timeout=5000")
            holder.execute("BEGIN")
            holder.execute(
                "INSERT INTO hmc_savings (timestamp, session_id, project_path, "
                "strategy, saved_tokens, session_input, session_output) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("2026-04-11T00:00:00Z", "tmp", "/p", "truncation", 1, 2, 1),
            )
            try:
                wal_file = store.db_path.with_name(store.db_path.name + "-wal")
                shm_file = store.db_path.with_name(store.db_path.name + "-shm")
                self.assertTrue(wal_file.exists(), f"missing WAL file at {wal_file}")
                self.assertTrue(shm_file.exists(), f"missing SHM file at {shm_file}")
            finally:
                holder.rollback()
                holder.close()

    def test_reader_does_not_block_during_writer_transaction(self) -> None:
        """The headline WAL guarantee: a reader can read while a writer
        holds the writer slot mid-transaction.

        Under the default DELETE journal mode this would deadlock or
        the reader would have to wait.  Under WAL the reader hits the
        committed snapshot and proceeds immediately.
        """
        with tempfile.TemporaryDirectory() as tmp:
            store = self._make_store(tmp)
            # Seed one row so the reader has something to find
            store.record_session("seed", "/p", {"truncation": 100}, 200)

            # Open a writer connection and start a transaction WITHOUT
            # committing.  This holds the writer slot.
            writer = sqlite3.connect(str(store.db_path), timeout=5.0)
            writer.execute("PRAGMA busy_timeout=5000")
            writer.execute("BEGIN IMMEDIATE")
            writer.execute(
                "INSERT INTO hmc_savings (timestamp, session_id, project_path, "
                "strategy, saved_tokens, session_input, session_output) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("2026-04-11T00:00:00Z", "uncommitted", "/p", "truncation", 999, 1000, 1),
            )
            try:
                # While the writer transaction is open, fire a read via
                # the store.  Under WAL this should return immediately
                # with the seed row.
                summary = store.get_summary()
                # The seed row is committed; the uncommitted writer's
                # row should NOT be visible to the reader's snapshot.
                self.assertEqual(summary.total_sessions, 1)
                self.assertEqual(summary.total_saved, 100)
            finally:
                writer.rollback()
                writer.close()

    def test_concurrent_readers_during_writer_burst(self) -> None:
        """End-to-end smoke: 10 readers + 10 writers in parallel, no errors."""
        with tempfile.TemporaryDirectory() as tmp:
            store = self._make_store(tmp)
            store.record_session("seed", "/p", {"truncation": 50}, 100)
            errors: list[BaseException] = []

            def writer(i: int) -> None:
                try:
                    store.record_session(
                        f"w_{i}", f"/p_{i % 3}",
                        {"truncation": 10 + i, "short_circuit": 5 + i},
                        100,
                    )
                except BaseException as e:  # pragma: no cover
                    errors.append(e)

            def reader() -> None:
                try:
                    for _ in range(5):
                        store.get_summary()
                        store.get_recent_sessions(limit=5)
                except BaseException as e:  # pragma: no cover
                    errors.append(e)

            threads: list[threading.Thread] = []
            for i in range(10):
                threads.append(threading.Thread(target=writer, args=(i,)))
                threads.append(threading.Thread(target=reader))
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            self.assertEqual(errors, [], f"concurrent ops produced errors: {errors}")
            # 10 writers + 1 seed
            self.assertEqual(store.get_summary().total_sessions, 11)

    def test_concurrent_writes_do_not_corrupt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self._make_store(tmp)
            errors: list[BaseException] = []

            def writer(i: int) -> None:
                try:
                    store.record_session(
                        f"s_{i}",
                        f"/p_{i % 3}",
                        {"truncation": 10 + i},
                        100,
                    )
                except BaseException as e:  # pragma: no cover
                    errors.append(e)

            threads = [threading.Thread(target=writer, args=(i,)) for i in range(20)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            self.assertEqual(errors, [])
            summary = store.get_summary()
            self.assertEqual(summary.total_sessions, 20)

    def test_db_path_env_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "custom.db"
            with mock.patch.dict(os.environ, {"HMC_DB_PATH": str(target)}):
                store = AnalyticsStore()
            self.assertEqual(store.db_path, target)


class SavingsSummaryTests(unittest.TestCase):
    def test_savings_pct_safe_with_zero_input(self) -> None:
        self.assertEqual(SavingsSummary().savings_pct, 0.0)

    def test_savings_pct_computed_from_input(self) -> None:
        s = SavingsSummary(total_saved=250, total_input=1000)
        self.assertEqual(s.savings_pct, 25.0)


class PluginAnalyticsIntegrationTests(unittest.TestCase):
    """End-to-end: on_session_end writes the session to SQLite."""

    def _make_plugin(self, tmp: str) -> HermesContextManagerPlugin:
        config = HmcConfig()
        # Use a tmp-scoped DB so tests don't pollute the real ~/.hermes path
        config.analytics.db_path = str(Path(tmp) / "analytics.db")
        return HermesContextManagerPlugin(hermes_home=tmp, config=config)

    def test_on_session_end_writes_savings_to_sqlite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plugin = self._make_plugin(tmp)
            state = plugin._get_state("s1")
            # Analytics records the un-gated totals (they're the truer
            # bytes-saved number).  Set both so the gated counter is
            # also realistic.
            state.tokens_kept_out_total = 150
            state.tokens_kept_out_by_type = {
                "truncation": 100, "short_circuit": 50,
            }
            state.tokens_saved = 150
            state.tokens_saved_by_type = {"truncation": 100, "short_circuit": 50}
            state.last_context_tokens = 500

            plugin.on_session_end(
                session_id="s1",
                completed=True,
                interrupted=False,
                model="test",
                platform="cli",
            )

            summary = plugin.analytics_store.get_summary()
            self.assertEqual(summary.total_sessions, 1)
            self.assertEqual(summary.total_saved, 150)
            self.assertEqual(summary.by_strategy["truncation"], 100)
            self.assertEqual(summary.by_strategy["short_circuit"], 50)

    def test_on_session_end_skips_when_no_savings(self) -> None:
        """A session with zero tokens_saved must not produce a phantom row."""
        with tempfile.TemporaryDirectory() as tmp:
            plugin = self._make_plugin(tmp)
            plugin._get_state("s1")  # creates state with zero savings

            plugin.on_session_end(
                session_id="s1",
                completed=True,
                interrupted=False,
                model="test",
                platform="cli",
            )

            self.assertEqual(plugin.analytics_store.get_summary().total_sessions, 0)

    def test_on_session_end_respects_analytics_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plugin = self._make_plugin(tmp)
            plugin.config.analytics.enabled = False
            state = plugin._get_state("s1")
            state.tokens_kept_out_total = 99
            state.tokens_kept_out_by_type = {"truncation": 99}
            state.tokens_saved = 99
            state.tokens_saved_by_type = {"truncation": 99}
            state.last_context_tokens = 100

            plugin.on_session_end(
                session_id="s1",
                completed=True,
                interrupted=False,
                model="test",
                platform="cli",
            )

            self.assertEqual(plugin.analytics_store.get_summary().total_sessions, 0)

    def test_get_state_captures_project_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plugin = self._make_plugin(tmp)
            state = plugin._get_state("s1")
            # Should match os.path.realpath(os.getcwd()) at session start
            self.assertEqual(state.project_path, os.path.realpath(os.getcwd()))


class HmcControlAnalyticsActionTests(unittest.TestCase):
    """The new ``analytics`` action on hmc_control."""

    def _make_plugin_with_data(self, tmp: str) -> HermesContextManagerPlugin:
        config = HmcConfig()
        config.analytics.db_path = str(Path(tmp) / "analytics.db")
        plugin = HermesContextManagerPlugin(hermes_home=tmp, config=config)
        # Seed three sessions across two projects
        plugin.analytics_store.record_session(
            "s_a", "/proj/a", {"truncation": 100, "dedup": 50}, 200
        )
        plugin.analytics_store.record_session(
            "s_b", "/proj/a", {"short_circuit": 75}, 150
        )
        plugin.analytics_store.record_session(
            "s_c", "/proj/b", {"truncation": 300}, 500
        )
        # Pin the current "session" so handle_hmc_control's _session_id_for_task
        # has something to resolve to
        state = create_state()
        state.project_path = "/proj/a"
        plugin._states["current"] = state
        return plugin

    def test_default_action_returns_global_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plugin = self._make_plugin_with_data(tmp)
            import json as _json
            result = _json.loads(
                plugin.handle_hmc_control({"action": "analytics"}, task_id=None)
            )
            self.assertEqual(result["scope"], "global")
            self.assertEqual(result["period"], "all")
            self.assertEqual(result["total_saved"], 525)  # 100+50+75+300
            self.assertEqual(result["total_sessions"], 3)
            self.assertIn("truncation", result["by_strategy"])

    def test_project_scope_filters_by_current_project(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plugin = self._make_plugin_with_data(tmp)
            import json as _json
            result = _json.loads(plugin.handle_hmc_control(
                {"action": "analytics", "scope": "project"}, task_id=None,
            ))
            self.assertEqual(result["project"], "/proj/a")
            self.assertEqual(result["total_saved"], 225)  # 100+50+75
            self.assertEqual(result["total_sessions"], 2)

    def test_period_day_returns_daily_breakdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plugin = self._make_plugin_with_data(tmp)
            import json as _json
            result = _json.loads(plugin.handle_hmc_control(
                {"action": "analytics", "period": "day"}, task_id=None,
            ))
            self.assertEqual(result["period"], "day")
            self.assertGreaterEqual(len(result["rows"]), 1)
            self.assertIn("day", result["rows"][0])

    def test_period_recent_returns_session_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plugin = self._make_plugin_with_data(tmp)
            import json as _json
            result = _json.loads(plugin.handle_hmc_control(
                {"action": "analytics", "period": "recent", "limit": 5},
                task_id=None,
            ))
            self.assertEqual(result["period"], "recent")
            self.assertEqual(len(result["rows"]), 3)

    def test_period_project_returns_per_project_totals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plugin = self._make_plugin_with_data(tmp)
            import json as _json
            result = _json.loads(plugin.handle_hmc_control(
                {"action": "analytics", "period": "project"}, task_id=None,
            ))
            self.assertEqual(result["period"], "project")
            project_paths = [r["project_path"] for r in result["rows"]]
            self.assertIn("/proj/a", project_paths)
            self.assertIn("/proj/b", project_paths)


if __name__ == "__main__":
    unittest.main()
