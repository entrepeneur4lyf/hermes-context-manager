import tempfile
import unittest
from collections import deque
from pathlib import Path

from hermes_context_manager.persistence import JsonStateStore
from hermes_context_manager.state import (
    TURN_HISTORY_MAX_LEN,
    create_state,
    session_state_from_dict,
    session_state_to_dict,
)


class PersistenceTests(unittest.TestCase):
    def test_state_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = JsonStateStore(Path(tmp))
            state = create_state()
            state.manual_mode = True
            state.pruned_tool_ids.add("call_1")
            state.tokens_saved = 42

            store.save("session-1", state)
            loaded = store.load("session-1")

            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertTrue(loaded.manual_mode)
            self.assertEqual(loaded.pruned_tool_ids, {"call_1"})
            self.assertEqual(loaded.tokens_saved, 42)

    def test_turn_history_round_trip(self) -> None:
        """Regression test for 0.3.3 (commit adding turn_history persistence).

        Before the fix the serializer dropped ``state.turn_history``, so
        any state reload would empty the ring buffer.  The dashboard's
        delta-saved computation reads ``prev_cumulative`` from the last
        entry in that buffer; when empty, it fell back to 0 and
        reported the full cumulative as a per-turn delta on every
        reload-then-publish cycle.
        """
        with tempfile.TemporaryDirectory() as tmp:
            store = JsonStateStore(Path(tmp))
            state = create_state()
            state.tokens_kept_out_total = 11100
            state.tokens_saved = 4666
            # Populate a few realistic snapshots.
            for turn in range(1, 4):
                state.turn_history.append(
                    {
                        "turn": turn,
                        "timestamp": 1_000.0 + turn,
                        "context_tokens": 100_000 + turn * 1000,
                        "context_percent": 0.5 + turn * 0.01,
                        "cumulative_saved": turn * 3_700,
                        "delta_saved": 3_700,
                        "by_strategy": {"short_circuit": turn * 2_000},
                        "uniq_saved": turn * 1_500,
                        "uniq_by_strategy": {"short_circuit": turn * 1_500},
                    }
                )

            store.save("session-1", state)
            loaded = store.load("session-1")

            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(len(loaded.turn_history), 3)
            # Deque identity + maxlen preserved.
            self.assertIsInstance(loaded.turn_history, deque)
            self.assertEqual(loaded.turn_history.maxlen, TURN_HISTORY_MAX_LEN)
            # Content round-trips faithfully.
            first = loaded.turn_history[0]
            self.assertEqual(first["turn"], 1)
            self.assertEqual(first["cumulative_saved"], 3_700)
            last = loaded.turn_history[-1]
            self.assertEqual(last["turn"], 3)
            self.assertEqual(last["cumulative_saved"], 11_100)

    def test_turn_history_legacy_sidecar_has_empty_history(self) -> None:
        """Sidecars written before 0.3.3 omit the ``turn_history`` key.

        The deserializer must treat that as an empty ring buffer rather
        than crashing.  The first post-load publish will compute a
        slightly-off delta on the next turn, but the cumulative stays
        correct and the ring buffer fills back up normally.
        """
        legacy_payload = {
            "tool_calls": {},
            "pruned_tool_ids": [],
            "tokens_saved": 100,
            "total_prune_count": 0,
            "tokens_saved_by_type": {"short_circuit": 100},
            "manual_mode": False,
            "last_context_tokens": 1000,
            "last_context_window": 200_000,
            "last_context_percent": 0.005,
            "dedup_group_sizes": {},
            "counted_savings_ids": [],
            "project_path": "/tmp/x",
            # No turn_history key.
        }
        state = session_state_from_dict(legacy_payload)
        self.assertEqual(len(state.turn_history), 0)
        self.assertEqual(state.turn_history.maxlen, TURN_HISTORY_MAX_LEN)
        # Tokens_kept_out_total falls back to tokens_saved on legacy load.
        self.assertEqual(state.tokens_kept_out_total, 100)

    def test_state_to_dict_includes_turn_history(self) -> None:
        """Fast unit test guarding against future serializer regressions."""
        state = create_state()
        state.turn_history.append({"turn": 1, "cumulative_saved": 500})
        payload = session_state_to_dict(state)
        self.assertIn("turn_history", payload)
        self.assertEqual(len(payload["turn_history"]), 1)
        self.assertEqual(payload["turn_history"][0]["cumulative_saved"], 500)


if __name__ == "__main__":
    unittest.main()
