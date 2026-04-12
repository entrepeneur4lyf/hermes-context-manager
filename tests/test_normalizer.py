import unittest

from hermes_context_manager.normalizer import normalize_for_dedup


class NormalizerTests(unittest.TestCase):
    def test_timestamps_replaced(self) -> None:
        text = "Created at 2024-03-15T14:30:00Z and updated 2024-03-15 14:31:05.123"
        result = normalize_for_dedup(text)
        self.assertNotIn("2024-03-15T14:30:00Z", result)
        self.assertNotIn("14:31:05", result)
        self.assertIn("<TIMESTAMP>", result)

    def test_uuids_replaced(self) -> None:
        text = "Session id: 550e8400-e29b-41d4-a716-446655440000 active"
        result = normalize_for_dedup(text)
        self.assertNotIn("550e8400", result)
        self.assertIn("<UUID>", result)

    def test_hex_hashes_replaced(self) -> None:
        text = "Commit abc1234def is on branch main"
        result = normalize_for_dedup(text)
        self.assertNotIn("abc1234def", result)
        self.assertIn("<HEX>", result)

    def test_large_numbers_replaced(self) -> None:
        text = "Process 123456 used 99999 bytes"
        result = normalize_for_dedup(text)
        self.assertNotIn("123456", result)
        self.assertNotIn("99999", result)
        self.assertIn("<NUM>", result)

    def test_small_numbers_preserved(self) -> None:
        text = "Found 3 errors in 42 files"
        result = normalize_for_dedup(text)
        self.assertIn("3", result)
        self.assertIn("42", result)
        self.assertEqual(text, result)

    def test_temp_paths_replaced(self) -> None:
        text = "Wrote output to /tmp/build_abc123/result.json and /var/tmp/cache.db"
        result = normalize_for_dedup(text)
        self.assertNotIn("/tmp/build_abc123/result.json", result)
        self.assertNotIn("/var/tmp/cache.db", result)
        self.assertIn("<TMPPATH>", result)

    def test_identical_after_normalization(self) -> None:
        text_a = "Build at 2024-01-01T10:00:00Z wrote /tmp/out_aaa/result.json with 12345 bytes"
        text_b = "Build at 2025-06-15T22:30:00Z wrote /tmp/out_zzz/result.json with 98765 bytes"
        self.assertEqual(normalize_for_dedup(text_a), normalize_for_dedup(text_b))

    def test_different_content_stays_different(self) -> None:
        text_a = "Compiled the server module successfully"
        text_b = "Ran database migration and seeded tables"
        self.assertNotEqual(normalize_for_dedup(text_a), normalize_for_dedup(text_b))


if __name__ == "__main__":
    unittest.main()
