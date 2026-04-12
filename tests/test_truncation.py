"""Tests for head/tail windowing truncation."""

from __future__ import annotations

import unittest

from hermes_context_manager.truncation import head_tail_truncate


class TestHeadTailTruncate(unittest.TestCase):
    """Tests for head_tail_truncate."""

    def test_short_content_unchanged(self):
        """Content within max_lines is returned as-is."""
        content = "\n".join(f"line {i}" for i in range(10))
        result = head_tail_truncate(content, max_lines=50, head=10, tail=10)
        self.assertEqual(result, content)

    def test_long_content_truncated_with_gap(self):
        """Content exceeding max_lines is truncated with head+tail and gap."""
        lines = [f"line {i}" for i in range(100)]
        content = "\n".join(lines) + "\n"
        result = head_tail_truncate(content, max_lines=25, head=10, tail=10)
        result_lines = result.splitlines()
        # First 10 lines preserved
        for i in range(10):
            self.assertEqual(result_lines[i], f"line {i}")
        # Gap marker present
        self.assertIn("... (", result)
        self.assertIn("lines omitted)", result)
        # Last 10 lines preserved
        for i in range(10):
            self.assertEqual(result_lines[-(10 - i)], f"line {90 + i}")

    def test_gap_marker_shows_count(self):
        """Gap marker contains the correct number of omitted lines."""
        lines = [f"line {i}" for i in range(100)]
        content = "\n".join(lines) + "\n"
        result = head_tail_truncate(content, max_lines=25, head=10, tail=10)
        # 100 total - 10 head - 10 tail = 80 omitted
        self.assertIn("(80 lines omitted)", result)

    def test_exact_boundary_no_gap(self):
        """Content exactly at max_lines is returned unchanged."""
        lines = [f"line {i}" for i in range(50)]
        content = "\n".join(lines) + "\n"
        # 50 content lines + trailing newline means 51 entries after splitlines(keepends=True)
        # but let's just set max_lines high enough
        result = head_tail_truncate(content, max_lines=51, head=10, tail=10)
        self.assertEqual(result, content)

    def test_head_only_when_tail_zero(self):
        """With tail=0, only head lines are kept."""
        lines = [f"line {i}" for i in range(100)]
        content = "\n".join(lines) + "\n"
        result = head_tail_truncate(content, max_lines=15, head=10, tail=0)
        result_lines = result.splitlines()
        # First 10 lines + gap
        self.assertEqual(len(result_lines), 11)
        for i in range(10):
            self.assertEqual(result_lines[i], f"line {i}")
        self.assertIn("(90 lines omitted)", result)

    def test_empty_content(self):
        """Empty string is returned unchanged."""
        self.assertEqual(head_tail_truncate(""), "")
        self.assertEqual(head_tail_truncate("", max_lines=10), "")


if __name__ == "__main__":
    unittest.main()
