"""Tests for background compressor module."""

import unittest
from hermes_context_manager.background_compressor import identify_stale_ranges, build_index_entry
from hermes_context_manager.state import create_state


class BackgroundCompressorTests(unittest.TestCase):
    def test_identify_stale_finds_old_work(self):
        messages = [
            {"role": "user", "content": "install deps", "timestamp": 1.0},
            {"role": "assistant", "content": "done", "timestamp": 2.0},
            {"role": "user", "content": "build API", "timestamp": 3.0},
            {"role": "assistant", "content": "building", "timestamp": 4.0},
            {"role": "user", "content": "test it", "timestamp": 5.0},
            {"role": "assistant", "content": "testing", "timestamp": 6.0},
            {"role": "user", "content": "deploy", "timestamp": 7.0},
            {"role": "assistant", "content": "deploying", "timestamp": 8.0},
            {"role": "user", "content": "current work", "timestamp": 9.0},
        ]
        state = create_state()
        state.current_turn = 5
        ranges = identify_stale_ranges(messages, state, protect_recent_turns=2)
        self.assertGreater(len(ranges), 0)
        # Should not include the last 2 user messages
        for r in ranges:
            self.assertLess(r["end_idx"], 6)

    def test_not_enough_history(self):
        messages = [
            {"role": "user", "content": "hello", "timestamp": 1.0},
            {"role": "assistant", "content": "hi", "timestamp": 2.0},
        ]
        state = create_state()
        ranges = identify_stale_ranges(messages, state, protect_recent_turns=3)
        self.assertEqual(ranges, [])

    def test_build_index_entry_extracts_topic(self):
        messages = [
            {"role": "user", "content": "Install all dependencies", "timestamp": 1.0},
            {"role": "assistant", "content": "Done", "timestamp": 2.0},
        ]
        entry = build_index_entry(messages, 0, 1, "1-1")
        self.assertEqual(entry["topic"], "Install all dependencies")
        self.assertEqual(entry["turn_range"], "1-1")
        self.assertEqual(entry["message_count"], 2)

    def test_build_index_entry_with_summary(self):
        messages = [{"role": "user", "content": "hello", "timestamp": 1.0}]
        entry = build_index_entry(messages, 0, 0, "1-1", summary="Custom summary")
        self.assertEqual(entry["summary"], "Custom summary")


if __name__ == "__main__":
    unittest.main()
