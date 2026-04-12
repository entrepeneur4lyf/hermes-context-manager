"""Tests for the short-circuit pattern matching module."""

import unittest

from hermes_context_manager.short_circuits import (
    DEFAULT_SHORT_CIRCUIT_RULES,
    ShortCircuitRule,
    apply_short_circuits,
)


class TestShortCircuits(unittest.TestCase):
    """Validate short-circuit replacement logic."""

    def test_exact_success_json_replaced(self) -> None:
        content = '{"status": "ok", "detail": "done"}'
        result = apply_short_circuits(content)
        self.assertEqual(result, "[ok]")

    def test_tests_passed_replaced(self) -> None:
        # pytest style
        content = "collected 12 items\n\n===== 12 passed ====="
        result = apply_short_circuits(content)
        self.assertEqual(result, "[tests: 12 passed]")

        # N passed, 0 failed style
        content2 = "Results: 5 passed, 0 failed"
        result2 = apply_short_circuits(content2)
        self.assertEqual(result2, "[tests: 5 passed]")

        # cargo test style
        content3 = "test result: ok. 42 passed; 0 failed; 0 ignored"
        result3 = apply_short_circuits(content3)
        self.assertEqual(result3, "[tests: 42 passed]")

    def test_file_written_replaced(self) -> None:
        content = '{"bytes_written": 1234}'
        result = apply_short_circuits(content)
        self.assertEqual(result, "[file written]")

    def test_no_match_returns_none(self) -> None:
        content = "Some arbitrary tool output that matches nothing."
        result = apply_short_circuits(content)
        self.assertIsNone(result)

    def test_error_in_content_blocks_short_circuit(self) -> None:
        # Even though the pattern matches, global error indicator blocks it
        content = '{"status": "ok", "error": "something went wrong"}'
        result = apply_short_circuits(content)
        self.assertIsNone(result)

        content2 = "===== 5 passed =====\nTraceback (most recent call last):"
        result2 = apply_short_circuits(content2)
        self.assertIsNone(result2)

    def test_empty_content_returns_none(self) -> None:
        self.assertIsNone(apply_short_circuits(""))
        self.assertIsNone(apply_short_circuits("   "))

    def test_custom_rule(self) -> None:
        custom = [
            ShortCircuitRule(
                pattern=r"BUILD (\w+)",
                replacement=r"[build: \1]",
            )
        ]
        result = apply_short_circuits("BUILD SUCCESS in 3s", custom)
        self.assertEqual(result, "[build: SUCCESS]")

    def test_unless_pattern_blocks_match(self) -> None:
        rule_with_guard = ShortCircuitRule(
            pattern=r"(\d+)\s+passed,\s*0\s+failed",
            replacement=r"[tests: \1 passed]",
            unless=r"warning",
        )
        # Without warning — should match
        result = apply_short_circuits("5 passed, 0 failed", [rule_with_guard])
        self.assertEqual(result, "[tests: 5 passed]")

        # With warning — should be blocked
        result2 = apply_short_circuits("5 passed, 0 failed\n1 warning", [rule_with_guard])
        self.assertIsNone(result2)

    def test_already_up_to_date(self) -> None:
        self.assertEqual(
            apply_short_circuits("Already up to date."),
            "[ok: already up to date]",
        )
        self.assertEqual(
            apply_short_circuits("Package already installed"),
            "[ok: already up to date]",
        )

    def test_clean_working(self) -> None:
        self.assertEqual(
            apply_short_circuits("On branch main\nnothing to commit, working tree clean"),
            "[ok: clean]",
        )


if __name__ == "__main__":
    unittest.main()
