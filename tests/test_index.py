"""Tests for session index persistence."""

import json
import tempfile
import unittest
from pathlib import Path

from hermes_context_manager.persistence import JsonStateStore


class TestSessionIndex(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = JsonStateStore(Path(self.tmpdir))

    def test_append_index_entry(self):
        """Append one entry, verify file exists and content is correct."""
        entry = {"topic": "Install deps", "turn_range": "1-2", "summary": "Installed dependencies"}
        self.store.append_index("sess1", entry)
        path = self.store.index_path("sess1")
        self.assertTrue(path.exists())
        content = path.read_text(encoding="utf-8").strip()
        parsed = json.loads(content)
        self.assertEqual(parsed["topic"], "Install deps")
        self.assertEqual(parsed["turn_range"], "1-2")
        self.assertIn("indexed_at", parsed)

    def test_multiple_entries_append(self):
        """Append two entries, both should be present."""
        self.store.append_index("sess1", {"topic": "Phase 1", "turn_range": "1-2", "summary": "a"})
        self.store.append_index("sess1", {"topic": "Phase 2", "turn_range": "3-4", "summary": "b"})
        lines = self.store.index_path("sess1").read_text(encoding="utf-8").strip().split("\n")
        self.assertEqual(len(lines), 2)
        self.assertEqual(json.loads(lines[0])["topic"], "Phase 1")
        self.assertEqual(json.loads(lines[1])["topic"], "Phase 2")

    def test_index_path_deterministic(self):
        """Same session_id always returns the same path."""
        p1 = self.store.index_path("my-session")
        p2 = self.store.index_path("my-session")
        self.assertEqual(p1, p2)

    def test_read_index_returns_entries(self):
        """Append then read back entries."""
        self.store.append_index("sess1", {"topic": "A", "turn_range": "1-1", "summary": "x"})
        self.store.append_index("sess1", {"topic": "B", "turn_range": "2-3", "summary": "y"})
        entries = self.store.read_index("sess1")
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]["topic"], "A")
        self.assertEqual(entries[1]["topic"], "B")

    def test_read_index_empty_session(self):
        """Nonexistent session returns empty list."""
        entries = self.store.read_index("nonexistent")
        self.assertEqual(entries, [])


if __name__ == "__main__":
    unittest.main()
