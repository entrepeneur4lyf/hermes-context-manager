"""Short-circuit pattern matching for tool output compression.

When a tool result matches a known success pattern, replace the entire content
with a one-liner summary. No LLM call needed — inspired by RTK (Rust Token Killer).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Rule dataclass
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ShortCircuitRule:
    """A single short-circuit replacement rule.

    Attributes:
        pattern: Regex pattern to match against tool output.
        replacement: Replacement string (may use \\1 back-references).
        unless: Optional regex — if this matches, the rule is skipped.
    """

    pattern: str
    replacement: str
    unless: str = ""


# ---------------------------------------------------------------------------
# Global error safety net
# ---------------------------------------------------------------------------

_GLOBAL_ERROR_INDICATORS = re.compile(
    r'"error"\s*:\s*"[^"]+"'
    r"|Traceback"
    r"|FATAL"
    r"|panic"
    r"|Segmentation fault",
    re.MULTILINE,
)

# ---------------------------------------------------------------------------
# Default rules
# ---------------------------------------------------------------------------

DEFAULT_SHORT_CIRCUIT_RULES: list[ShortCircuitRule] = [
    # JSON success  {"status": "ok" ...}
    ShortCircuitRule(
        pattern=r'^\s*\{\s*"status"\s*:\s*"ok".*\}\s*$',
        replacement="[ok]",
    ),
    # {"bytes_written": N}
    ShortCircuitRule(
        pattern=r'^\s*\{\s*"bytes_written"\s*:\s*\d+.*\}\s*$',
        replacement="[file written]",
    ),
    # pytest  === N passed ===
    ShortCircuitRule(
        pattern=r"={3,}\s*(\d+)\s+passed\s*={3,}",
        replacement=r"[tests: \1 passed]",
    ),
    # N passed, 0 failed
    ShortCircuitRule(
        pattern=r"(\d+)\s+passed,\s*0\s+failed",
        replacement=r"[tests: \1 passed]",
    ),
    # cargo test  test result: ok. N passed; 0 failed
    ShortCircuitRule(
        pattern=r"test result:\s*ok\.\s*(\d+)\s+passed;\s*0\s+failed",
        replacement=r"[tests: \1 passed]",
    ),
    # already up to date / installed / exists
    ShortCircuitRule(
        pattern=r"(?i)already\s+(?:up[\s-]to[\s-]date|installed|exists)",
        replacement="[ok: already up to date]",
    ),
    # nothing to commit | clean working | no changes
    ShortCircuitRule(
        pattern=r"(?i)(?:nothing to commit|clean working|no changes)",
        replacement="[ok: clean]",
    ),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_short_circuits(
    content: str,
    rules: list[ShortCircuitRule] | None = None,
) -> str | None:
    """Try to replace *content* with a short summary using pattern rules.

    Returns the replacement string on the first matching rule, or ``None``
    if no rule matched (meaning the content should be left as-is).
    """
    if not content or not content.strip():
        return None

    # Safety: never short-circuit content that contains error indicators.
    if _GLOBAL_ERROR_INDICATORS.search(content):
        return None

    if rules is None:
        rules = DEFAULT_SHORT_CIRCUIT_RULES

    for rule in rules:
        match = re.search(rule.pattern, content, re.MULTILINE | re.DOTALL)
        if match is None:
            continue
        # If an ``unless`` guard is set and it matches, skip this rule.
        if rule.unless and re.search(rule.unless, content, re.MULTILINE | re.DOTALL):
            continue
        return match.expand(rule.replacement)

    return None
