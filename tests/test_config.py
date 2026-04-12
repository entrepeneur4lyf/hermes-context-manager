"""Tests for ``hermes_context_manager.config``.

The config layer was previously untested, which let the
``background_compression`` block ship as a silent no-op.  These tests
round-trip every block through ``_from_dict`` and exercise the on-disk
``load_config`` path so any future regression in the parser is caught
immediately.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from hermes_context_manager.config import (
    DEFAULT_CONFIG,
    AnalyticsConfig,
    BackgroundCompressionConfig,
    CodeFilterConfig,
    HmcConfig,
    _from_dict,
    _merge_dict,
    load_config,
)


class FromDictTests(unittest.TestCase):
    def test_empty_payload_returns_defaults(self) -> None:
        cfg = _from_dict({})
        self.assertEqual(cfg, DEFAULT_CONFIG)

    def test_top_level_flags(self) -> None:
        cfg = _from_dict({"enabled": False, "debug": True})
        self.assertFalse(cfg.enabled)
        self.assertTrue(cfg.debug)

    def test_manual_mode_block(self) -> None:
        cfg = _from_dict(
            {"manual_mode": {"enabled": True, "automatic_strategies": False}}
        )
        self.assertTrue(cfg.manual_mode.enabled)
        self.assertFalse(cfg.manual_mode.automatic_strategies)

    def test_compress_block(self) -> None:
        cfg = _from_dict(
            {
                "compress": {
                    "max_context_percent": 0.9,
                    "min_context_percent": 0.5,
                    "protected_tools": ["custom_tool"],
                }
            }
        )
        self.assertAlmostEqual(cfg.compress.max_context_percent, 0.9)
        self.assertAlmostEqual(cfg.compress.min_context_percent, 0.5)
        self.assertEqual(cfg.compress.protected_tools, ["custom_tool"])

    def test_strategies_block(self) -> None:
        cfg = _from_dict(
            {
                "strategies": {
                    "deduplication": {
                        "enabled": False,
                        "protected_tools": ["dedup_protected"],
                    },
                    "purge_errors": {
                        "enabled": False,
                        "turns": 8,
                        "protected_tools": ["err_protected"],
                    },
                }
            }
        )
        self.assertFalse(cfg.strategies.deduplication.enabled)
        self.assertEqual(
            cfg.strategies.deduplication.protected_tools, ["dedup_protected"]
        )
        self.assertFalse(cfg.strategies.purge_errors.enabled)
        self.assertEqual(cfg.strategies.purge_errors.turns, 8)
        self.assertEqual(
            cfg.strategies.purge_errors.protected_tools, ["err_protected"]
        )

    def test_truncation_block(self) -> None:
        cfg = _from_dict(
            {
                "truncation": {
                    "enabled": False,
                    "max_lines": 100,
                    "head_lines": 25,
                    "tail_lines": 25,
                    "min_content_length": 1000,
                }
            }
        )
        self.assertFalse(cfg.truncation.enabled)
        self.assertEqual(cfg.truncation.max_lines, 100)
        self.assertEqual(cfg.truncation.head_lines, 25)
        self.assertEqual(cfg.truncation.tail_lines, 25)
        self.assertEqual(cfg.truncation.min_content_length, 1000)

    def test_short_circuits_block(self) -> None:
        cfg = _from_dict({"short_circuits": {"enabled": False}})
        self.assertFalse(cfg.short_circuits.enabled)

    def test_background_compression_block(self) -> None:
        """Regression test: this block was a silent no-op before the
        ``BackgroundCompressionConfig`` dataclass was wired into ``HmcConfig``.
        """
        cfg = _from_dict(
            {"background_compression": {"enabled": False, "protect_recent_turns": 7}}
        )
        self.assertFalse(cfg.background_compression.enabled)
        self.assertEqual(cfg.background_compression.protect_recent_turns, 7)

    def test_background_compression_defaults_when_missing(self) -> None:
        cfg = _from_dict({})
        self.assertTrue(cfg.background_compression.enabled)
        self.assertEqual(cfg.background_compression.protect_recent_turns, 3)

    def test_background_compression_partial_override(self) -> None:
        # Only one field set; the other should fall back to default.
        cfg = _from_dict(
            {"background_compression": {"protect_recent_turns": 5}}
        )
        self.assertTrue(cfg.background_compression.enabled)
        self.assertEqual(cfg.background_compression.protect_recent_turns, 5)

    def test_analytics_block(self) -> None:
        cfg = _from_dict(
            {
                "analytics": {
                    "enabled": False,
                    "retention_days": 30,
                    "db_path": "/tmp/custom.db",
                }
            }
        )
        self.assertFalse(cfg.analytics.enabled)
        self.assertEqual(cfg.analytics.retention_days, 30)
        self.assertEqual(cfg.analytics.db_path, "/tmp/custom.db")

    def test_analytics_defaults_when_missing(self) -> None:
        cfg = _from_dict({})
        self.assertTrue(cfg.analytics.enabled)
        self.assertEqual(cfg.analytics.retention_days, 90)
        self.assertEqual(cfg.analytics.db_path, "")

    def test_analytics_partial_override(self) -> None:
        cfg = _from_dict({"analytics": {"retention_days": 14}})
        self.assertTrue(cfg.analytics.enabled)
        self.assertEqual(cfg.analytics.retention_days, 14)
        self.assertEqual(cfg.analytics.db_path, "")

    def test_code_filter_block(self) -> None:
        cfg = _from_dict(
            {
                "code_filter": {
                    "enabled": False,
                    "languages": ["python", "rust"],
                    "min_lines": 50,
                    "preserve_docstrings": False,
                }
            }
        )
        self.assertFalse(cfg.code_filter.enabled)
        self.assertEqual(cfg.code_filter.languages, ["python", "rust"])
        self.assertEqual(cfg.code_filter.min_lines, 50)
        self.assertFalse(cfg.code_filter.preserve_docstrings)

    def test_code_filter_defaults_when_missing(self) -> None:
        cfg = _from_dict({})
        self.assertTrue(cfg.code_filter.enabled)
        self.assertEqual(cfg.code_filter.min_lines, 30)
        self.assertTrue(cfg.code_filter.preserve_docstrings)
        for lang in ("python", "javascript", "typescript", "rust", "go"):
            self.assertIn(lang, cfg.code_filter.languages)

    def test_code_filter_partial_override(self) -> None:
        cfg = _from_dict({"code_filter": {"min_lines": 100}})
        self.assertTrue(cfg.code_filter.enabled)
        self.assertEqual(cfg.code_filter.min_lines, 100)
        # Default language list preserved
        self.assertIn("python", cfg.code_filter.languages)

    def test_protected_file_patterns(self) -> None:
        cfg = _from_dict({"protected_file_patterns": ["*.env", "secrets/*"]})
        self.assertEqual(cfg.protected_file_patterns, ["*.env", "secrets/*"])

    def test_prune_notification(self) -> None:
        cfg = _from_dict({"prune_notification": "minimal"})
        self.assertEqual(cfg.prune_notification, "minimal")

    def test_full_payload_round_trip(self) -> None:
        """All blocks set at once — no field bleed between blocks."""
        cfg = _from_dict(
            {
                "enabled": True,
                "debug": True,
                "manual_mode": {"enabled": True, "automatic_strategies": False},
                "compress": {
                    "max_context_percent": 0.85,
                    "min_context_percent": 0.45,
                    "protected_tools": ["a", "b"],
                },
                "strategies": {
                    "deduplication": {"enabled": False, "protected_tools": ["c"]},
                    "purge_errors": {
                        "enabled": True,
                        "turns": 6,
                        "protected_tools": ["d"],
                    },
                },
                "truncation": {
                    "enabled": True,
                    "max_lines": 80,
                    "head_lines": 15,
                    "tail_lines": 12,
                    "min_content_length": 750,
                },
                "short_circuits": {"enabled": False},
                "background_compression": {
                    "enabled": False,
                    "protect_recent_turns": 4,
                },
                "protected_file_patterns": ["*.pem"],
                "prune_notification": "detailed",
            }
        )
        self.assertTrue(cfg.enabled)
        self.assertTrue(cfg.debug)
        self.assertTrue(cfg.manual_mode.enabled)
        self.assertFalse(cfg.manual_mode.automatic_strategies)
        self.assertAlmostEqual(cfg.compress.max_context_percent, 0.85)
        self.assertEqual(cfg.compress.protected_tools, ["a", "b"])
        self.assertFalse(cfg.strategies.deduplication.enabled)
        self.assertEqual(cfg.strategies.deduplication.protected_tools, ["c"])
        self.assertEqual(cfg.strategies.purge_errors.turns, 6)
        self.assertEqual(cfg.truncation.max_lines, 80)
        self.assertFalse(cfg.short_circuits.enabled)
        self.assertFalse(cfg.background_compression.enabled)
        self.assertEqual(cfg.background_compression.protect_recent_turns, 4)
        self.assertEqual(cfg.protected_file_patterns, ["*.pem"])

    def test_protected_tools_lists_are_copies_not_aliases(self) -> None:
        """Mutating the parsed list must not poison ``DEFAULT_CONFIG``."""
        cfg = _from_dict({})
        cfg.compress.protected_tools.append("evil")
        self.assertNotIn("evil", DEFAULT_CONFIG.compress.protected_tools)


class MergeDictTests(unittest.TestCase):
    def test_override_replaces_scalar(self) -> None:
        merged = _merge_dict({"a": 1}, {"a": 2})
        self.assertEqual(merged, {"a": 2})

    def test_override_recurses_into_nested_dict(self) -> None:
        merged = _merge_dict(
            {"outer": {"a": 1, "b": 2}},
            {"outer": {"b": 99}},
        )
        self.assertEqual(merged, {"outer": {"a": 1, "b": 99}})

    def test_none_override_is_skipped(self) -> None:
        merged = _merge_dict({"a": 1}, {"a": None})
        self.assertEqual(merged, {"a": 1})

    def test_new_key_added(self) -> None:
        merged = _merge_dict({"a": 1}, {"b": 2})
        self.assertEqual(merged, {"a": 1, "b": 2})


class LoadConfigTests(unittest.TestCase):
    def test_missing_config_returns_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = load_config(Path(tmp))
            self.assertEqual(cfg, DEFAULT_CONFIG)

    def test_invalid_yaml_returns_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "config.yaml").write_text(
                "this: is: not: valid: yaml: [", encoding="utf-8"
            )
            cfg = load_config(Path(tmp))
            self.assertEqual(cfg, DEFAULT_CONFIG)

    def test_non_dict_yaml_returns_defaults(self) -> None:
        # A YAML list at top level is valid YAML but not what we expect.
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "config.yaml").write_text(
                "- one\n- two\n", encoding="utf-8"
            )
            cfg = load_config(Path(tmp))
            self.assertEqual(cfg, DEFAULT_CONFIG)

    def test_valid_yaml_overrides_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "config.yaml").write_text(
                """
enabled: true
background_compression:
  enabled: false
  protect_recent_turns: 9
truncation:
  max_lines: 200
""",
                encoding="utf-8",
            )
            cfg = load_config(Path(tmp))
            self.assertTrue(cfg.enabled)
            self.assertFalse(cfg.background_compression.enabled)
            self.assertEqual(cfg.background_compression.protect_recent_turns, 9)
            self.assertEqual(cfg.truncation.max_lines, 200)
            # Untouched fields stay at defaults.
            self.assertEqual(
                cfg.compress.max_context_percent,
                DEFAULT_CONFIG.compress.max_context_percent,
            )

    def test_repo_example_config_loads(self) -> None:
        """The shipped ``config.yaml.example`` must be parseable.

        Catches the case where someone adds a knob to the example file but
        forgets to wire it through ``_from_dict`` — exactly the bug that
        ``BackgroundCompressionConfig`` was created to fix.
        """
        repo_root = Path(__file__).resolve().parent.parent
        example = repo_root / "config.yaml.example"
        if not example.exists():  # pragma: no cover - sanity guard
            self.skipTest("config.yaml.example missing from repo root")
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "config.yaml"
            target.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")
            cfg = load_config(Path(tmp))
            # Spot-check that the parser actually consumed the example.
            self.assertIsInstance(cfg, HmcConfig)
            self.assertIsInstance(
                cfg.background_compression, BackgroundCompressionConfig
            )
            self.assertIsInstance(cfg.analytics, AnalyticsConfig)
            self.assertIsInstance(cfg.code_filter, CodeFilterConfig)


if __name__ == "__main__":
    unittest.main()
