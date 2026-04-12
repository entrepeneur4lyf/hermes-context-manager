"""Configuration loading for Hermes Context Manager."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_origin, get_type_hints

import yaml


@dataclass(slots=True)
class ManualModeConfig:
    enabled: bool = False
    automatic_strategies: bool = True


@dataclass(slots=True)
class CompressConfig:
    max_context_percent: float = 0.8
    min_context_percent: float = 0.4
    protected_tools: list[str] = field(default_factory=lambda: ["write_file", "patch"])


@dataclass(slots=True)
class DeduplicationConfig:
    enabled: bool = True
    protected_tools: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PurgeErrorsConfig:
    enabled: bool = True
    turns: int = 4
    protected_tools: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ShortCircuitConfig:
    enabled: bool = True


@dataclass(slots=True)
class BackgroundCompressionConfig:
    enabled: bool = True
    protect_recent_turns: int = 3


@dataclass(slots=True)
class AnalyticsConfig:
    """SQLite-backed cumulative savings analytics.

    Disabled-but-still-loaded gives users a way to opt out without
    deleting their existing analytics.db.
    """
    enabled: bool = True
    retention_days: int = 90
    db_path: str = ""  # empty == default location resolved by AnalyticsStore


@dataclass(slots=True)
class CodeFilterConfig:
    """Code-aware compression for tool outputs containing source code.

    Strips function/class bodies, preserving signatures and imports.
    Inspired by RTK's AggressiveFilter, with two improvements: a
    string-aware brace counter (RTK's naive count() corrupts on string
    literals) and a from-scratch indentation-aware Python path (RTK
    has no real Python support).
    """
    enabled: bool = True
    languages: list[str] = field(
        default_factory=lambda: [
            "python", "javascript", "typescript", "rust", "go",
        ]
    )
    min_lines: int = 30
    preserve_docstrings: bool = True


@dataclass(slots=True)
class StrategyConfig:
    deduplication: DeduplicationConfig = field(default_factory=DeduplicationConfig)
    purge_errors: PurgeErrorsConfig = field(default_factory=PurgeErrorsConfig)


@dataclass(slots=True)
class TruncationConfig:
    enabled: bool = True
    max_lines: int = 50
    head_lines: int = 10
    tail_lines: int = 10
    min_content_length: int = 500


@dataclass(slots=True)
class HmcConfig:
    enabled: bool = True
    debug: bool = False
    manual_mode: ManualModeConfig = field(default_factory=ManualModeConfig)
    compress: CompressConfig = field(default_factory=CompressConfig)
    strategies: StrategyConfig = field(default_factory=StrategyConfig)
    truncation: TruncationConfig = field(default_factory=TruncationConfig)
    short_circuits: ShortCircuitConfig = field(default_factory=ShortCircuitConfig)
    background_compression: BackgroundCompressionConfig = field(
        default_factory=BackgroundCompressionConfig
    )
    analytics: AnalyticsConfig = field(default_factory=AnalyticsConfig)
    code_filter: CodeFilterConfig = field(default_factory=CodeFilterConfig)
    protected_file_patterns: list[str] = field(default_factory=list)
    prune_notification: str = "detailed"


DEFAULT_CONFIG = HmcConfig()


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = _merge_dict(result[key], value)
        elif value is not None:
            result[key] = value
    return result


def _hydrate_dataclass(cls: type, payload: dict[str, Any]) -> Any:
    """Recursively build a dataclass instance from a (merged) dict.

    Replaces the 150-line field-by-field ``_from_dict`` with a generic
    pass that walks each dataclass field via ``dataclasses.fields`` +
    ``typing.get_type_hints``.  New config blocks need zero changes
    here -- add the dataclass, reference it from ``HmcConfig``, done.

    Behavior rules per field type (matching the pre-0.3.6 semantics):

    - **Nested dataclass**: recurse with the sub-dict (or ``{}`` if
      the YAML block is missing/malformed).  The sub-dataclass fills
      in its own defaults via this same function.
    - **list[X]**: copy the value into a fresh list (never alias
      the default_factory's list, which would poison DEFAULT_CONFIG
      if a caller mutated it).  None becomes an empty list.
    - **int / float**: defensive ``int(...)`` / ``float(...)`` cast
      so YAML strings like ``"90"`` still land as numbers.  Bool is
      intentionally NOT coerced because ``bool("false") is True``
      is a Python footgun we don't want to propagate.
    - **str**: ``str(...)`` coerce, empty string for None.
    - **anything else** (bool, Any, etc.): pass through.

    Fields absent from ``payload`` use the dataclass default
    (``default`` or ``default_factory``).  This preserves the
    invariant that a YAML config that omits a whole block still
    yields the same object as an empty block would.
    """
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        name = f.name
        if name not in payload:
            continue  # let the dataclass default / default_factory fill it in
        value = payload[name]
        ftype = hints.get(name)
        origin = get_origin(ftype)

        if (
            ftype is not None
            and isinstance(ftype, type)
            and is_dataclass(ftype)
        ):
            sub = value if isinstance(value, dict) else {}
            kwargs[name] = _hydrate_dataclass(ftype, sub)
        elif origin is list:
            kwargs[name] = list(value) if value is not None else []
        elif ftype is int:
            kwargs[name] = int(value)
        elif ftype is float:
            kwargs[name] = float(value)
        elif ftype is str:
            kwargs[name] = str(value) if value is not None else ""
        else:
            # bool, Any, unknown -- pass through without coercion
            kwargs[name] = value
    return cls(**kwargs)


def _from_dict(payload: dict[str, Any]) -> HmcConfig:
    """Build an ``HmcConfig`` from a (possibly merged) payload dict.

    Thin wrapper around ``_hydrate_dataclass`` kept for import-site
    stability -- tests and other modules import this symbol directly.
    """
    return _hydrate_dataclass(HmcConfig, payload)


def load_config(plugin_dir: Path) -> HmcConfig:
    """Load config.yaml from the plugin directory if present.

    No longer pre-merges with ``asdict(DEFAULT_CONFIG)``: the hydrator
    handles missing fields natively via dataclass defaults, so feeding
    it the raw YAML dict produces identical results with less work.
    ``_merge_dict`` is still exported for callers that want deep-merge
    semantics on two arbitrary dicts.
    """
    config_path = plugin_dir / "config.yaml"
    if not config_path.exists():
        return DEFAULT_CONFIG
    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return DEFAULT_CONFIG
    if not isinstance(loaded, dict):
        return DEFAULT_CONFIG
    return _from_dict(loaded)
