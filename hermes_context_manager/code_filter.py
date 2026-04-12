"""Code-aware compression for tool outputs containing source code.

This is HMC's port of RTK v0.35.0 ``src/core/filter.rs`` AggressiveFilter,
with two material improvements over the original:

1. **String-aware brace counting.**  RTK uses a naive ``count('{')`` /
   ``count('}')`` per line.  That gets wrecked by string literals
   containing braces (``"hello {world}"``), template literals, format
   strings, dict/object literals inline.  HMC's ``_LineScanner`` is a
   character-by-character state machine that tracks single-quoted,
   double-quoted, backtick (template), block-comment, and line-comment
   state, persisting block-comment state across lines.

2. **Indentation-aware Python.**  RTK has Python in its language list
   but the brace-counting algorithm is fundamentally wrong for Python
   (which uses indentation).  HMC has a separate ``_filter_python``
   path that tracks indent levels.

What it does
============
Two-pass pipeline matching RTK:

- **Pass 1 — comment stripping.**  ``_strip_pure_comment_lines`` drops
  lines that are entirely line comments (``//``, ``#``).  Block
  comments (``/* ... */``) are dropped across line boundaries.  Doc
  comments (``///``, ``/** ... */``, Python triple-quote docstrings)
  are preserved when ``preserve_docstrings`` is true.

- **Pass 2 — body elision.**  Function/class/struct/enum/interface
  signatures are emitted; their bodies are replaced with a one-liner
  ``// ... body elided`` (or ``# ... body elided`` for Python).
  Imports, top-level constants, and class field definitions are
  preserved.

What it does NOT do
===================
- **JSX/TSX**: when JSX/TSX components are detected, the filter
  bails out entirely.  React component files have so many ``{}``
  inside JSX expressions that brace counting is hopeless.  We return
  the content unchanged.
- **Regex literals in JS**: not specifically handled.  A regex with
  ``{`` or ``}`` will skew counts.  Acceptable v1 limitation.
- **Rust raw strings (r"...", r#"..."#)**: treated as regular strings.
  Braces inside raw strings will skew counts.
- **Multi-line strings**: only block comments persist across line
  boundaries.  A multi-line raw string in Rust or template literal in
  JS that spans lines will be mis-tracked at line boundaries.
- **Nested template literal interpolation**: ``${...}`` inside a
  template literal is treated as opaque (not re-entering "outside"
  state to count code braces).  This is correct for body elision but
  wrong for "is this brace structural" decisions in pathological
  cases.

Trigger logic
=============
``apply_code_filter`` (in engine.py, the engine integration shim) checks
each tool message and runs the filter when:

1. The content contains a fenced markdown block ```` ```lang ```` , OR
2. The tool that produced the output had a file path argument with a
   code extension (.py, .ts, .tsx, .js, .jsx, .rs, .go).

Plus the content must have at least ``code_filter.min_lines`` lines and
must not be a tool error or already-pruned output.

The detected language must be in ``code_filter.languages`` (defaults
include all five supported languages).
"""
from __future__ import annotations

import re
from typing import Iterable

# Languages this module knows how to filter.  Adding a new language
# requires (1) extending these constants, (2) adding patterns to
# LANGUAGE_PATTERNS below, (3) deciding brace-vs-indent path.
LANGUAGE_EXTENSIONS: dict[str, frozenset[str]] = {
    "python": frozenset({"py", "pyw", "python", "py3"}),
    "javascript": frozenset({"js", "mjs", "cjs", "javascript"}),
    "typescript": frozenset({"ts", "tsx", "typescript"}),
    "rust": frozenset({"rs", "rust"}),
    "go": frozenset({"go", "golang"}),
}

# Languages that use brace-delimited blocks (vs. Python's indentation).
BRACE_LANGUAGES: frozenset[str] = frozenset({"javascript", "typescript", "rust", "go"})

# Patterns for the AggressiveFilter pass.  Each language has:
#   - signature: matches function/class/struct/enum/interface declaration LINES
#   - import: matches import/use lines (always preserved)
#   - keep_outside: top-level declarations preserved outside any body
LANGUAGE_PATTERNS: dict[str, dict[str, re.Pattern[str]]] = {
    "rust": {
        "signature": re.compile(
            r"^\s*(?:pub(?:\s*\([^)]*\))?\s+)?"
            r"(?:async\s+|unsafe\s+|const\s+|extern\s+(?:\"[^\"]*\"\s+)?)*"
            r"(?:fn|struct|enum|trait|impl|type|union)\b"
        ),
        "import": re.compile(r"^\s*(?:use|extern\s+crate|mod)\b"),
        "keep_outside": re.compile(
            r"^\s*(?:pub(?:\s*\([^)]*\))?\s+)?(?:const|static|let)\b"
        ),
    },
    "go": {
        # Match top-level func, receiver methods (`func (r *T) Name`),
        # type declarations, interface declarations, struct declarations.
        "signature": re.compile(
            r"^\s*(?:func\b|type\s+\w+\s+(?:struct|interface)\b|type\s+\w+\s+func\b)"
        ),
        "import": re.compile(r"^\s*(?:import|package)\b"),
        "keep_outside": re.compile(r"^\s*(?:const|var)\b"),
    },
    "javascript": {
        "signature": re.compile(
            r"^\s*(?:export\s+(?:default\s+)?)?"
            r"(?:async\s+)?"
            r"(?:function\s*\*?|class)\s*\w*"
        ),
        "import": re.compile(r"^\s*(?:import|require\s*\(|export\s+\*\s+from)\b"),
        "keep_outside": re.compile(
            r"^\s*(?:export\s+)?(?:const|let|var)\s+\w+\s*="
        ),
    },
    "typescript": {
        "signature": re.compile(
            r"^\s*(?:export\s+(?:default\s+)?)?"
            r"(?:abstract\s+)?"
            r"(?:async\s+)?"
            r"(?:function\s*\*?|class|interface|type\s+\w+\s*=\s*\{|enum\s+\w+)"
        ),
        "import": re.compile(r"^\s*(?:import|require\s*\(|export\s+\*\s+from)\b"),
        "keep_outside": re.compile(
            r"^\s*(?:export\s+)?(?:const|let|var)\s+\w+\s*[:=]"
        ),
    },
}

# Line-comment markers per language.  Used by the comment stripper.
LANG_LINE_COMMENT: dict[str, str] = {
    "python": "#",
    "javascript": "//",
    "typescript": "//",
    "rust": "//",
    "go": "//",
}

# Doc-comment line prefixes that are preserved when preserve_docstrings=True.
LANG_DOC_LINE_PREFIX: dict[str, str | None] = {
    "python": None,  # Python uses triple-quoted strings, not line comments
    "javascript": None,
    "typescript": None,
    "rust": "///",
    "go": None,
}


# ---------------------------------------------------------------------------
# Pass 1: comment stripping (the equivalent of RTK's MinimalFilter)
# ---------------------------------------------------------------------------


def _strip_pure_comment_lines(
    lines: list[str],
    lang: str,
    preserve_docstrings: bool,
) -> list[str]:
    """Drop lines that are entirely comments.

    Block comments are tracked across line boundaries.  Doc comments
    (``/** */``, ``///``) are preserved when ``preserve_docstrings`` is
    True.  Lines containing CODE plus a trailing comment are passed
    through verbatim -- we deliberately do not strip inline comments,
    because the bound between code and comment is fragile inside string
    literals.
    """
    output: list[str] = []
    in_block = False
    block_is_doc = False

    line_marker = LANG_LINE_COMMENT.get(lang, "//")
    doc_line_prefix = LANG_DOC_LINE_PREFIX.get(lang)

    for line in lines:
        stripped = line.strip()

        if in_block:
            # Inside a multi-line block comment
            if block_is_doc and preserve_docstrings:
                output.append(line)
            if "*/" in stripped:
                in_block = False
                block_is_doc = False
            continue

        # Preserve blank lines as code-structure markers
        if not stripped:
            output.append(line)
            continue

        # Block comment start: /** ... */ or /* ... */
        if stripped.startswith("/**"):
            is_doc = True
            single_line = "*/" in stripped[3:]
            if preserve_docstrings:
                output.append(line)
            if not single_line:
                in_block = True
                block_is_doc = is_doc
            continue

        if stripped.startswith("/*"):
            single_line = "*/" in stripped[2:]
            if not single_line:
                in_block = True
                block_is_doc = False
            # Plain block comment: dropped (single-line or multi-line)
            continue

        # Doc line comment (Rust ///)
        if doc_line_prefix and stripped.startswith(doc_line_prefix):
            if preserve_docstrings:
                output.append(line)
            continue

        # Plain line comment
        if stripped.startswith(line_marker):
            continue

        # Code line — keep
        output.append(line)

    return output


# ---------------------------------------------------------------------------
# String-aware brace counting (for the brace-language AggressiveFilter)
# ---------------------------------------------------------------------------


class _LineScanner:
    """Char-by-char scanner that counts braces while respecting string state.

    State persists across lines for block comments only (the most common
    multi-line case).  Multi-line strings (Rust raw strings, JS template
    literals across lines) are not perfectly tracked -- braces inside
    them on intermediate lines may be miscounted.  This is a documented
    v1 limitation.
    """

    __slots__ = ("lang", "_in_block_comment")

    def __init__(self, lang: str) -> None:
        self.lang = lang
        self._in_block_comment = False

    def reset(self) -> None:
        self._in_block_comment = False

    def count_braces(self, line: str) -> tuple[int, int]:
        """Return ``(opens, closes)`` for braces outside strings/comments."""
        opens = 0
        closes = 0
        state = "block_comment" if self._in_block_comment else "outside"
        i = 0
        n = len(line)
        is_js_like = self.lang in ("javascript", "typescript")

        while i < n:
            c = line[i]
            nxt = line[i + 1] if i + 1 < n else ""

            if state == "outside":
                if c == '"':
                    state = "d_quote"
                elif c == "'":
                    state = "s_quote"
                elif c == "`" and is_js_like:
                    state = "template"
                elif c == "/" and nxt == "/":
                    break  # rest of line is line comment
                elif c == "/" and nxt == "*":
                    state = "block_comment"
                    i += 2
                    continue
                elif c == "{":
                    opens += 1
                elif c == "}":
                    closes += 1
            elif state == "d_quote":
                if c == "\\" and i + 1 < n:
                    i += 2
                    continue
                if c == '"':
                    state = "outside"
            elif state == "s_quote":
                if c == "\\" and i + 1 < n:
                    i += 2
                    continue
                if c == "'":
                    state = "outside"
            elif state == "template":
                if c == "\\" and i + 1 < n:
                    i += 2
                    continue
                if c == "`":
                    state = "outside"
                # Note: ${...} interpolations are treated as opaque
            elif state == "block_comment":
                if c == "*" and nxt == "/":
                    state = "outside"
                    i += 2
                    continue
            i += 1

        # Persist block-comment state for the next line
        self._in_block_comment = state == "block_comment"
        return opens, closes


# ---------------------------------------------------------------------------
# Pass 2a: brace-language AggressiveFilter
# ---------------------------------------------------------------------------


def _filter_brace_language(lines: list[str], lang: str) -> list[str]:
    """Strip function/class/struct bodies for brace-delimited languages.

    Rust / Go / JavaScript / TypeScript.  Imports, signatures, and
    top-level declarations are preserved; bodies are replaced with a
    one-liner ``// ... body elided`` followed by the closing brace.
    """
    patterns = LANGUAGE_PATTERNS[lang]
    sig_re = patterns["signature"]
    import_re = patterns["import"]
    keep_outside_re = patterns["keep_outside"]

    scanner = _LineScanner(lang)
    output: list[str] = []
    in_body = False
    in_multiline_sig = False  # sig_re matched but `{` hasn't appeared yet
    brace_depth = 0
    body_indent = ""

    for line in lines:
        stripped = line.strip()

        # Always pass blank lines through (they help readability of
        # the filtered output without costing much)
        if not stripped:
            output.append(line)
            continue

        # Accumulating a multi-line signature: emit continuation lines
        # verbatim until we see the opening brace of the body.  This
        # catches the common Rust/Go/JS/TS pattern of:
        #
        #     fn process_data(
        #         data: &Config,
        #         limit: usize,
        #     ) -> Result<Vec<String>, Error> {
        #         // body
        #     }
        #
        # Without this state, the continuation lines don't match any
        # pattern and would be silently dropped, and body mode would
        # never be entered -- producing a structurally broken filter
        # output.
        if in_multiline_sig:
            output.append(line)
            opens, closes = scanner.count_braces(line)
            brace_depth += opens - closes
            if brace_depth > 0:
                in_multiline_sig = False
                in_body = True
            elif brace_depth == 0 and "{" in stripped and "}" in stripped:
                # Rare: one-liner body that happened to straddle signature
                # continuation (e.g. `... { return x; }` on the same line).
                in_multiline_sig = False
            continue

        if in_body:
            opens, closes = scanner.count_braces(line)
            brace_depth += opens - closes
            if brace_depth <= 0:
                # Body ended.  Emit the elision marker on its own line,
                # then emit the closing-brace line that ended it.
                if not output or not output[-1].lstrip().startswith("// ... body"):
                    output.append(f"{body_indent}    // ... body elided")
                output.append(line)
                in_body = False
                brace_depth = 0
            continue

        # Outside any body
        if import_re.match(line):
            output.append(line)
            continue

        if sig_re.match(line):
            output.append(line)
            opens, closes = scanner.count_braces(line)
            brace_depth = opens - closes
            body_indent = line[: len(line) - len(line.lstrip())]
            if brace_depth > 0:
                in_body = True
            elif brace_depth == 0 and "{" in stripped and "}" in stripped:
                # Single-line body like `fn foo() {}` — fully self-contained,
                # nothing to elide.
                pass
            else:
                # Signature line had no `{` yet -- it's continuing onto
                # subsequent lines.  Enter multi-line-sig accumulation
                # until we find the opening brace.
                in_multiline_sig = True
            continue

        if keep_outside_re.match(line):
            output.append(line)
            continue

        # Otherwise drop (RTK convention: only signatures + imports +
        # top-level constants survive outside body context).

    # If we exit while still in a body (truncated input), emit a marker
    if in_body or in_multiline_sig:
        output.append(f"{body_indent}    // ... body elided (truncated)")

    return output


# ---------------------------------------------------------------------------
# Pass 2b: indentation-aware Python filter
# ---------------------------------------------------------------------------


_PY_DECORATOR_RE = re.compile(r"^\s*@\w")
_PY_DEF_RE = re.compile(r"^(\s*)(?:async\s+)?def\s+\w+")
_PY_CLASS_RE = re.compile(r"^(\s*)class\s+\w+")


def _filter_python(lines: list[str], preserve_docstrings: bool) -> list[str]:
    """Indentation-aware Python body elision.

    Strategy:
    - Decorators pass through.
    - ``def`` signatures are emitted; the body is elided down to the
      docstring (if any) plus a ``# ... body elided`` marker.
    - ``class`` declarations are emitted but do NOT enter body mode --
      class attributes pass through (they are part of the API surface)
      and method ``def`` signatures inside the class are detected
      independently and have their bodies elided.
    - Multi-line docstrings inside elided bodies are tracked via
      triple-quote toggling so an embedded ``def foo():`` in a docstring
      doesn't trigger spurious body detection.
    """
    output: list[str] = []
    body_indent: int | None = None
    docstring_consumed = False
    triple_quote_open: str | None = None  # "\"\"\"" or "'''" if inside multi-line docstring
    body_marker_emitted = False

    def _emit_marker(indent: int) -> None:
        nonlocal body_marker_emitted
        if not body_marker_emitted:
            output.append(" " * (indent + 4) + "# ... body elided")
            body_marker_emitted = True

    for line in lines:
        # Inside a multi-line docstring (which we are preserving)
        if triple_quote_open is not None:
            output.append(line)
            if triple_quote_open in line:
                triple_quote_open = None
                docstring_consumed = True
            continue

        stripped = line.rstrip()
        bare = stripped.lstrip()
        indent = (len(stripped) - len(bare)) if bare else 0

        # Inside an elided function body
        if body_indent is not None:
            if not bare:
                # Blank line inside body — drop (the marker speaks for itself)
                continue

            if indent <= body_indent:
                # Body ended at this dedent.  Reprocess this line at the
                # outer scope.
                _emit_marker(body_indent)
                body_indent = None
                docstring_consumed = False
                body_marker_emitted = False
                # fall through to outer-scope handling
            else:
                # First body line that isn't blank — maybe a docstring
                if preserve_docstrings and not docstring_consumed:
                    if bare.startswith('"""') or bare.startswith("'''"):
                        quote = bare[:3]
                        output.append(line)
                        # Single-line docstring? (opens and closes on same line)
                        if bare.count(quote) >= 2:
                            docstring_consumed = True
                        else:
                            triple_quote_open = quote
                        continue
                    else:
                        # First body line is not a docstring -- mark consumed
                        # so we don't keep checking, then emit the marker.
                        docstring_consumed = True
                        _emit_marker(body_indent)
                        continue
                # Body content after docstring (or with docstrings disabled)
                _emit_marker(body_indent)
                continue

        # Outside any body
        if _PY_DECORATOR_RE.match(line):
            output.append(line)
            continue

        m = _PY_DEF_RE.match(line)
        if m:
            output.append(line)
            body_indent = len(m.group(1))
            docstring_consumed = False
            body_marker_emitted = False
            continue

        if _PY_CLASS_RE.match(line):
            # Don't enter body mode for classes; let inner methods be
            # detected and filtered individually, and let class
            # attributes pass through as part of the API.
            output.append(line)
            continue

        # Top-level statement (import, assignment, etc.)
        output.append(line)

    # If we ran off the end while still in a body, emit the marker
    if body_indent is not None:
        _emit_marker(body_indent)

    return output


# ---------------------------------------------------------------------------
# JSX bailout
# ---------------------------------------------------------------------------


# Match JSX/TSX component usage by requiring an *unambiguous* JSX
# marker, not just a capital letter in angle brackets.  The naive
# ``<[A-Z][\w.]*[\s/>]`` regex false-positives on TypeScript generic
# type arguments like ``Promise<Response>``, ``Array<string>``,
# ``Map<K, V>``, ``useState<UserType>()``, etc. — which show up in
# any nontrivial TS file and would incorrectly trigger the JSX
# bailout.
#
# Real JSX always has one of:
#   1. A closing tag:     ``</Component>``
#   2. A self-closing:    ``<Component />`` or ``<Component.Sub />``
#   3. An attribute:      ``<Component prop=...`` or ``<Card.Title class=...``
# TypeScript generics NEVER have any of the above.
_JSX_RE = re.compile(
    r"</[A-Z]"                      # closing tag
    r"|<[A-Z][\w.]*\s*/>"            # self-closing
    r"|<[A-Z][\w.]*\s+[a-zA-Z_][\w]*\s*="  # attribute
)


def _has_jsx(content: str) -> bool:
    """Detect JSX/TSX components.

    JSX expression braces (``onClick={() => ...}``) catastrophically
    corrupt brace counting.  When detected, the filter bails out and
    returns the content unchanged so we don't ship garbage.

    Crucially, this must NOT false-positive on TypeScript generics:
    ``Promise<Response>``, ``Array<string>``, and ``Map<K, V>`` are
    extremely common in TS code and are NOT JSX.  The regex therefore
    requires an unambiguous JSX marker (closing tag, self-close, or
    attribute) rather than just a capital letter in angle brackets.
    """
    return bool(_JSX_RE.search(content))


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------


def _normalize_lang_hint(hint: str) -> str | None:
    """Map a fenced-block tag or extension to a canonical language name."""
    if not hint:
        return None
    h = hint.lower().lstrip(".")
    for lang, exts in LANGUAGE_EXTENSIONS.items():
        if h == lang or h in exts:
            return lang
    return None


def detect_language(content: str, hint: str | None = None) -> str | None:
    """Detect the language of a code block.

    Detection order:
    1. Explicit ``hint`` (e.g. fenced-block tag or file extension)
    2. Content sniffing on the first 30 lines (regex markers per lang)
    3. Returns ``None`` if no confident match
    """
    if hint:
        canon = _normalize_lang_hint(hint)
        if canon:
            return canon

    head = "\n".join(content.split("\n")[:30])

    # Order matters: TypeScript before JavaScript (TS includes JS-like
    # patterns plus extras), Rust before Go (both have `func`/`fn`
    # collisions, Rust is more idiomatic to detect).
    if re.search(
        r"^\s*(?:def|class|from\s+[\w.]+\s+import|import\s+[\w.]+)\b",
        head,
        re.M,
    ):
        return "python"

    if re.search(
        r"^\s*(?:pub\s+(?:fn|struct)|use\s+\w+::|impl\b|trait\s+\w|fn\s+\w+\s*\(|struct\s+\w)",
        head,
        re.M,
    ):
        return "rust"

    if re.search(
        r"^\s*(?:func\s+\w|package\s+\w|type\s+\w+\s+(?:struct|interface)\b)",
        head,
        re.M,
    ):
        return "go"

    if re.search(
        r"^\s*(?:interface\s+\w|type\s+\w+\s*=\s*\{|enum\s+\w)|:\s*(?:string|number|boolean|void)\b",
        head,
        re.M,
    ):
        return "typescript"

    if re.search(
        r"^\s*(?:function\s+\w|class\s+\w|const\s+\w+\s*=|export\s+(?:default|const|function|class))",
        head,
        re.M,
    ):
        return "javascript"

    return None


# ---------------------------------------------------------------------------
# Top-level filter entry
# ---------------------------------------------------------------------------


def filter_code_block(
    content: str,
    lang: str,
    preserve_docstrings: bool = True,
) -> str:
    """Apply the two-pass filter to a single code block.

    Returns the filtered content.  Returns ``content`` unchanged when:
    - Content is empty or single-line
    - Language is not supported
    - JSX is detected (brace counting would corrupt the output)
    """
    if not content or "\n" not in content:
        return content
    if lang not in LANGUAGE_EXTENSIONS:
        return content

    if lang in ("javascript", "typescript") and _has_jsx(content):
        return content

    lines = content.split("\n")
    lines = _strip_pure_comment_lines(lines, lang, preserve_docstrings)

    if lang == "python":
        lines = _filter_python(lines, preserve_docstrings)
    elif lang in BRACE_LANGUAGES:
        lines = _filter_brace_language(lines, lang)
    else:  # pragma: no cover
        return content

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Fenced markdown block extraction
# ---------------------------------------------------------------------------


_FENCED_RE = re.compile(r"(```(\w+)?\n)(.*?)(```)", re.DOTALL)


def filter_fenced_blocks(content: str, preserve_docstrings: bool = True) -> str:
    """Find ```lang blocks in prose and filter their bodies.

    Each fenced block is processed independently with the language
    determined by its tag.  Untagged or unrecognized-language blocks
    are passed through unchanged.  Prose between blocks is preserved
    verbatim.
    """
    def replacer(match: re.Match[str]) -> str:
        opener, tag, body, closer = match.groups()
        lang = _normalize_lang_hint(tag) if tag else None
        if not lang:
            return match.group(0)
        filtered = filter_code_block(body, lang, preserve_docstrings)
        return opener + filtered + closer

    return _FENCED_RE.sub(replacer, content)


# ---------------------------------------------------------------------------
# Convenience: estimate token savings without actually filtering
# ---------------------------------------------------------------------------


def estimate_savings(content: str, lang: str) -> int:
    """Return the *byte* delta from filtering ``content`` in ``lang``.

    Useful for tests and dry-runs.  Returns 0 for unchanged content.
    """
    filtered = filter_code_block(content, lang)
    return max(0, len(content) - len(filtered))


def supported_languages() -> Iterable[str]:
    """Return the canonical language names this filter understands."""
    return LANGUAGE_EXTENSIONS.keys()
