"""Tests for the code-aware compression filter.

Covers:
- String-aware brace counting (the main improvement over RTK)
- Multi-line block comments (state across lines)
- Comment stripping pre-pass (with doc-comment preservation)
- Per-language signature detection: Python, Rust, Go, JS, TS
- Indentation-aware Python (RTK has no real Python support)
- JSX/TSX bailout
- Fenced markdown block extraction
- Engine integration: code_filter contributes to tokens_saved_by_type
- Cross-strategy stacking with truncation
"""
from __future__ import annotations

import unittest

from hermes_context_manager.code_filter import (
    BRACE_LANGUAGES,
    LANGUAGE_EXTENSIONS,
    _LineScanner,
    _filter_brace_language,
    _filter_python,
    _has_jsx,
    _strip_pure_comment_lines,
    detect_language,
    estimate_savings,
    filter_code_block,
    filter_fenced_blocks,
    supported_languages,
)
from hermes_context_manager.config import HmcConfig
from hermes_context_manager.engine import materialize_view
from hermes_context_manager.state import ToolRecord, create_state


# ---------------------------------------------------------------------------
# String-aware brace scanner (the headline improvement over RTK)
# ---------------------------------------------------------------------------


class LineScannerTests(unittest.TestCase):
    def test_plain_braces_counted(self) -> None:
        s = _LineScanner("rust")
        self.assertEqual(s.count_braces("fn foo() { }"), (1, 1))
        self.assertEqual(s.count_braces("{{{}}}"), (3, 3))

    def test_braces_inside_double_quoted_string_ignored(self) -> None:
        s = _LineScanner("rust")
        self.assertEqual(s.count_braces('let x = "hello { world }";'), (0, 0))

    def test_braces_inside_single_quoted_char_ignored(self) -> None:
        s = _LineScanner("rust")
        self.assertEqual(s.count_braces("let c = '{';"), (0, 0))

    def test_braces_inside_template_literal_ignored_for_js(self) -> None:
        s = _LineScanner("javascript")
        self.assertEqual(s.count_braces("const x = `hello ${name}`;"), (0, 0))

    def test_template_literal_only_for_js_like(self) -> None:
        # In Rust, backtick is not special
        s = _LineScanner("rust")
        self.assertEqual(s.count_braces("let x = `{not a string}`;"), (1, 1))

    def test_escape_sequence_in_string(self) -> None:
        s = _LineScanner("javascript")
        # The \" inside the string should not terminate it
        self.assertEqual(s.count_braces(r'let x = "esc \" still in {string}";'), (0, 0))

    def test_line_comment_terminates_counting(self) -> None:
        s = _LineScanner("rust")
        # The { and } in the comment should not count
        self.assertEqual(s.count_braces("fn foo() { // outer { brace }"), (1, 0))

    def test_block_comment_state_persists_across_lines(self) -> None:
        s = _LineScanner("rust")
        # First line opens block comment with a brace inside
        self.assertEqual(s.count_braces("/* opening with { brace"), (0, 0))
        # Second line still inside the block comment
        self.assertEqual(s.count_braces("still in comment { with } braces"), (0, 0))
        # Third line closes the block comment, then real braces
        self.assertEqual(s.count_braces("end */ fn bar() {"), (1, 0))

    def test_block_comment_single_line(self) -> None:
        s = _LineScanner("javascript")
        self.assertEqual(s.count_braces("/* { } */ const x = {};"), (1, 1))


# ---------------------------------------------------------------------------
# Comment stripper (MinimalFilter equivalent)
# ---------------------------------------------------------------------------


class CommentStripperTests(unittest.TestCase):
    def test_strip_python_line_comments(self) -> None:
        lines = [
            "import os",
            "# this is a comment",
            "x = 1",
            "    # indented comment",
            "y = 2",
        ]
        out = _strip_pure_comment_lines(lines, "python", preserve_docstrings=True)
        self.assertEqual(out, ["import os", "x = 1", "y = 2"])

    def test_strip_js_line_comments_keeps_code(self) -> None:
        lines = ["// comment", "const x = 5;", "// another comment", "function foo() {}"]
        out = _strip_pure_comment_lines(lines, "javascript", preserve_docstrings=True)
        self.assertEqual(out, ["const x = 5;", "function foo() {}"])

    def test_preserve_rust_doc_comments(self) -> None:
        lines = ["/// doc comment", "// regular comment", "pub fn foo() {}"]
        out = _strip_pure_comment_lines(lines, "rust", preserve_docstrings=True)
        self.assertEqual(out, ["/// doc comment", "pub fn foo() {}"])

    def test_drop_rust_doc_comments_when_disabled(self) -> None:
        lines = ["/// doc comment", "pub fn foo() {}"]
        out = _strip_pure_comment_lines(lines, "rust", preserve_docstrings=False)
        self.assertEqual(out, ["pub fn foo() {}"])

    def test_jsdoc_block_preserved(self) -> None:
        lines = [
            "/**",
            " * JSDoc",
            " */",
            "function foo() {}",
        ]
        out = _strip_pure_comment_lines(lines, "javascript", preserve_docstrings=True)
        self.assertEqual(out, lines)

    def test_plain_block_comment_dropped(self) -> None:
        lines = [
            "/* plain block",
            " * not a doc",
            " */",
            "function foo() {}",
        ]
        out = _strip_pure_comment_lines(lines, "javascript", preserve_docstrings=True)
        self.assertEqual(out, ["function foo() {}"])

    def test_blank_lines_preserved(self) -> None:
        lines = ["import x", "", "function foo() {}"]
        out = _strip_pure_comment_lines(lines, "javascript", preserve_docstrings=True)
        self.assertEqual(out, lines)


# ---------------------------------------------------------------------------
# Brace-language filter (Rust / Go / JS / TS)
# ---------------------------------------------------------------------------


class BraceFilterTests(unittest.TestCase):
    def test_rust_pub_fn_body_elided(self) -> None:
        src = """\
use std::io;

pub fn process(data: &str) -> String {
    let result = data.to_uppercase();
    let mut output = String::new();
    output.push_str(&result);
    output
}

pub const VERSION: u32 = 1;
"""
        out = filter_code_block(src, "rust")
        self.assertIn("pub fn process(data: &str) -> String", out)
        self.assertIn("// ... body elided", out)
        self.assertIn("use std::io", out)
        self.assertIn("pub const VERSION: u32 = 1", out)
        self.assertNotIn("output.push_str", out)
        self.assertLess(len(out), len(src))

    def test_rust_struct_passes_through(self) -> None:
        # struct definitions are signatures; bodies (field decls) elided
        src = """\
pub struct Config {
    pub name: String,
    pub version: u32,
    pub enabled: bool,
}
"""
        out = filter_code_block(src, "rust")
        self.assertIn("pub struct Config", out)
        # struct body is treated as a function-like body and elided
        self.assertIn("// ... body elided", out)

    def test_go_top_level_func_body_elided(self) -> None:
        src = """\
package main

import "fmt"

func main() {
    fmt.Println("hello")
    fmt.Println("world")
    fmt.Println("done")
}
"""
        out = filter_code_block(src, "go")
        self.assertIn("func main()", out)
        self.assertIn("import", out)
        self.assertIn("// ... body elided", out)
        self.assertNotIn('fmt.Println("hello")', out)

    def test_go_receiver_method_detected(self) -> None:
        """RTK misses Go receiver methods; HMC must catch them."""
        src = """\
package main

func (r *Receiver) DoThing() error {
    err := r.validate()
    return err
}
"""
        out = filter_code_block(src, "go")
        self.assertIn("func (r *Receiver) DoThing() error", out)
        self.assertIn("// ... body elided", out)
        self.assertNotIn("r.validate()", out)

    def test_javascript_function_body_elided(self) -> None:
        src = """\
import { thing } from 'mod';

export const CONST_VALUE = 42;

function compute(x) {
    const y = x * 2;
    const z = y + 1;
    return z;
}
"""
        out = filter_code_block(src, "javascript")
        self.assertIn("function compute(x)", out)
        self.assertIn("import { thing } from 'mod'", out)
        self.assertIn("export const CONST_VALUE = 42", out)
        self.assertIn("// ... body elided", out)
        self.assertNotIn("const y = x * 2", out)

    def test_typescript_interface_preserved(self) -> None:
        src = """\
export interface User {
    id: string;
    name: string;
    email: string;
}

export function getUser(id: string): User {
    const user = lookup(id);
    return user;
}
"""
        out = filter_code_block(src, "typescript")
        self.assertIn("interface User", out)
        self.assertIn("function getUser", out)
        self.assertIn("// ... body elided", out)

    def test_string_with_braces_does_not_corrupt_brace_count(self) -> None:
        """The whole point of the string-aware scanner."""
        src = """\
fn format_message() -> String {
    let template = "Hello, {name}! Your balance is {amount}.";
    let result = template.replace("{name}", "Alice");
    return result;
}

pub const NEXT_FN: u32 = 99;
"""
        out = filter_code_block(src, "rust")
        # The body should be elided cleanly without bleeding into the
        # const declaration after it
        self.assertIn("fn format_message", out)
        self.assertIn("// ... body elided", out)
        self.assertIn("pub const NEXT_FN: u32 = 99", out)

    def test_template_literal_with_braces(self) -> None:
        """JS template literals with ${...} interpolation."""
        src = """\
function greet(name) {
    return `Hello ${name}, welcome to {our} app`;
}

export const VERSION = "1.0";
"""
        out = filter_code_block(src, "javascript")
        self.assertIn("function greet(name)", out)
        self.assertIn("// ... body elided", out)
        self.assertIn("export const VERSION", out)

    def test_rust_multiline_signature_preserved(self) -> None:
        """Multi-line function signatures must not be silently dropped.

        Regression test for a bug where sig_re matched the first line
        of a multi-line signature, continuation lines didn't match any
        pattern, body mode was never entered, and the body content
        was silently dropped (including the opening-brace line).  The
        resulting filter output was structurally broken.
        """
        src = """\
use std::io;

pub fn process_data(
    data: &Config,
    limit: usize,
) -> Result<Vec<String>, Error> {
    let mut out = Vec::new();
    out.push(data.name.clone());
    Ok(out)
}

pub const VERSION: u32 = 1;
"""
        out = filter_code_block(src, "rust")
        # The full signature across all continuation lines must be preserved
        self.assertIn("pub fn process_data(", out)
        self.assertIn("data: &Config,", out)
        self.assertIn("limit: usize,", out)
        self.assertIn(") -> Result<Vec<String>, Error> {", out)
        # Body must be elided with the marker
        self.assertIn("// ... body elided", out)
        # Body content must NOT appear
        self.assertNotIn("let mut out = Vec::new();", out)
        self.assertNotIn("out.push(data.name.clone());", out)
        # Top-level items after the function must still be present
        self.assertIn("pub const VERSION: u32 = 1;", out)
        self.assertIn("use std::io;", out)

    def test_go_multiline_signature_preserved(self) -> None:
        src = """\
package main

func Process(
    ctx context.Context,
    req *Request,
    opts ...Option,
) (*Response, error) {
    resp := &Response{}
    resp.Data = req.Data
    return resp, nil
}
"""
        out = filter_code_block(src, "go")
        self.assertIn("func Process(", out)
        self.assertIn("ctx context.Context,", out)
        self.assertIn("opts ...Option,", out)
        self.assertIn(") (*Response, error) {", out)
        self.assertIn("// ... body elided", out)
        self.assertNotIn("resp := &Response{}", out)

    def test_typescript_multiline_signature_preserved(self) -> None:
        src = """\
export function processRequest(
    req: Request,
    ctx: Context,
    options?: RequestOptions,
): Promise<Response> {
    const result = validateAndExecute(req, ctx);
    return result;
}
"""
        out = filter_code_block(src, "typescript")
        self.assertIn("export function processRequest(", out)
        self.assertIn("req: Request,", out)
        self.assertIn("options?: RequestOptions,", out)
        self.assertIn("): Promise<Response> {", out)
        self.assertIn("// ... body elided", out)
        self.assertNotIn("validateAndExecute", out)


# ---------------------------------------------------------------------------
# Python indentation-aware filter
# ---------------------------------------------------------------------------


class PythonFilterTests(unittest.TestCase):
    def test_function_body_elided_signature_preserved(self) -> None:
        src = '''\
import os

def compute(x: int, y: int) -> int:
    """Compute the sum of two integers."""
    result = x + y
    return result

CONSTANT = 42
'''
        out = filter_code_block(src, "python")
        self.assertIn("def compute(x: int, y: int) -> int:", out)
        self.assertIn('"""Compute the sum of two integers."""', out)
        self.assertIn("# ... body elided", out)
        self.assertIn("import os", out)
        self.assertIn("CONSTANT = 42", out)
        self.assertNotIn("result = x + y", out)

    def test_decorator_preserved(self) -> None:
        src = '''\
@cached
@retry(max=3)
def fetch(url: str) -> dict:
    response = requests.get(url)
    return response.json()
'''
        out = filter_code_block(src, "python")
        self.assertIn("@cached", out)
        self.assertIn("@retry(max=3)", out)
        self.assertIn("def fetch(url: str) -> dict:", out)
        self.assertIn("# ... body elided", out)

    def test_class_methods_filtered_individually(self) -> None:
        """class declaration does NOT enter body mode; methods are filtered."""
        src = '''\
class Service:
    """A service class."""
    timeout: int = 30
    retries: int = 3

    def __init__(self, name: str):
        self.name = name
        self.connected = False

    def connect(self) -> bool:
        try:
            self._open()
            return True
        except OSError:
            return False
'''
        out = filter_code_block(src, "python")
        # Class declaration preserved
        self.assertIn("class Service:", out)
        # Class attributes preserved
        self.assertIn("timeout: int = 30", out)
        self.assertIn("retries: int = 3", out)
        # Method signatures preserved
        self.assertIn("def __init__(self, name: str):", out)
        self.assertIn("def connect(self) -> bool:", out)
        # Method bodies elided
        self.assertIn("# ... body elided", out)
        self.assertNotIn("self.connected = False", out)
        self.assertNotIn("self._open()", out)

    def test_multiline_docstring_preserved(self) -> None:
        src = '''\
def compute():
    """Compute something.

    This is a multi-line docstring.
    With several lines of explanation.
    """
    x = 1
    y = 2
    return x + y
'''
        out = filter_code_block(src, "python")
        self.assertIn("Compute something.", out)
        self.assertIn("multi-line docstring", out)
        self.assertIn("With several lines", out)
        self.assertIn("# ... body elided", out)
        self.assertNotIn("x = 1", out)

    def test_no_docstring_first_body_line_elided(self) -> None:
        src = '''\
def add(a, b):
    result = a + b
    return result
'''
        out = filter_code_block(src, "python")
        self.assertIn("def add(a, b):", out)
        self.assertIn("# ... body elided", out)
        self.assertNotIn("result = a + b", out)

    def test_async_def_handled(self) -> None:
        src = '''\
async def fetch_data(url):
    """Fetch data from a URL."""
    async with aiohttp.ClientSession() as s:
        async with s.get(url) as r:
            return await r.json()
'''
        out = filter_code_block(src, "python")
        self.assertIn("async def fetch_data(url):", out)
        self.assertIn("Fetch data from a URL.", out)
        self.assertIn("# ... body elided", out)

    def test_top_level_imports_preserved(self) -> None:
        src = '''\
import os
import sys
from typing import Any
from pathlib import Path

def foo():
    return 1
'''
        out = filter_code_block(src, "python")
        self.assertIn("import os", out)
        self.assertIn("import sys", out)
        self.assertIn("from typing import Any", out)
        self.assertIn("from pathlib import Path", out)


# ---------------------------------------------------------------------------
# JSX bailout
# ---------------------------------------------------------------------------


class JsxBailoutTests(unittest.TestCase):
    def test_has_jsx_detects_capital_component(self) -> None:
        self.assertTrue(_has_jsx("return <Button onClick={() => x()}>Click</Button>;"))
        self.assertTrue(_has_jsx("<Component />"))

    def test_has_jsx_ignores_lowercase_html(self) -> None:
        # Plain HTML is OK; we only bail on JSX components which is the
        # case where brace counting actually breaks
        self.assertFalse(_has_jsx("const html = '<div>plain</div>';"))

    def test_jsx_content_returned_unchanged(self) -> None:
        src = """\
import React from 'react';

export function Card({ title, children }) {
    return (
        <Card.Container>
            <Card.Title>{title}</Card.Title>
            <Card.Body>{children}</Card.Body>
        </Card.Container>
    );
}
"""
        out = filter_code_block(src, "javascript")
        self.assertEqual(out, src)  # bailed out — unchanged

    def test_typescript_jsx_also_bails(self) -> None:
        src = """\
export const Greeting: React.FC<Props> = ({ name }) => {
    return <Greeting.Wrapper>Hello, {name}!</Greeting.Wrapper>;
};
"""
        out = filter_code_block(src, "typescript")
        self.assertEqual(out, src)

    def test_typescript_generics_are_not_jsx(self) -> None:
        """Regression: `Promise<Response>`, `Array<T>`, `Map<K, V>` must
        not trigger the JSX bailout.

        Before this fix, any `<CapitalWord>` pattern was treated as JSX,
        which caused the filter to bail out on nearly every real
        TypeScript file (all of them use generics).  The filter would
        return the content unchanged, shipping hundreds of KB of
        function bodies that should have been elided.
        """
        self.assertFalse(_has_jsx("Promise<Response>"))
        self.assertFalse(_has_jsx("const x: Array<string> = [];"))
        self.assertFalse(_has_jsx("Map<K, V>"))
        self.assertFalse(_has_jsx("useState<UserType>()"))
        self.assertFalse(_has_jsx("function foo<T extends Base>(x: T): T"))
        self.assertFalse(_has_jsx("): Promise<Response> {"))

    def test_jsx_still_detected_with_closing_tag(self) -> None:
        self.assertTrue(_has_jsx("return <div>hello</Component>;"))
        self.assertTrue(_has_jsx("</MyComponent>"))
        self.assertTrue(_has_jsx("</Card.Title>"))

    def test_jsx_still_detected_with_self_close(self) -> None:
        self.assertTrue(_has_jsx("<Component />"))
        self.assertTrue(_has_jsx("<Card.Title/>"))
        self.assertTrue(_has_jsx("const x = <Foo.Bar  />;"))

    def test_jsx_still_detected_with_attribute(self) -> None:
        self.assertTrue(_has_jsx("<Button onClick={handler}>"))
        self.assertTrue(_has_jsx("<Input type='text' />"))
        self.assertTrue(_has_jsx("<Card.Title className='big'>"))


# ---------------------------------------------------------------------------
# Fenced markdown blocks
# ---------------------------------------------------------------------------


class FencedBlockTests(unittest.TestCase):
    def test_fenced_block_filtered_prose_preserved(self) -> None:
        content = '''\
Here is some Python code:

```python
def example():
    """An example."""
    x = 1
    y = 2
    return x + y
```

And more prose after.
'''
        out = filter_fenced_blocks(content)
        self.assertIn("Here is some Python code:", out)
        self.assertIn("And more prose after.", out)
        self.assertIn("def example():", out)
        self.assertIn("An example.", out)
        self.assertIn("# ... body elided", out)
        self.assertNotIn("x = 1", out)

    def test_untagged_fenced_block_passes_through(self) -> None:
        content = """\
```
some random text
that has no language tag
```
"""
        out = filter_fenced_blocks(content)
        self.assertEqual(out, content)

    def test_unknown_language_passes_through(self) -> None:
        content = """\
```haskell
main = putStrLn "hello"
```
"""
        out = filter_fenced_blocks(content)
        self.assertEqual(out, content)


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------


class DetectLanguageTests(unittest.TestCase):
    def test_hint_python_extension(self) -> None:
        self.assertEqual(detect_language("", hint="py"), "python")

    def test_hint_typescript_tsx(self) -> None:
        self.assertEqual(detect_language("", hint="tsx"), "typescript")

    def test_content_sniff_python(self) -> None:
        self.assertEqual(
            detect_language("def foo():\n    pass\n"),
            "python",
        )

    def test_content_sniff_rust(self) -> None:
        self.assertEqual(
            detect_language("pub fn foo() -> i32 {\n    42\n}\n"),
            "rust",
        )

    def test_content_sniff_go(self) -> None:
        self.assertEqual(
            detect_language("package main\n\nfunc main() {}\n"),
            "go",
        )

    def test_unknown_returns_none(self) -> None:
        self.assertIsNone(detect_language("just some prose without code markers"))

    def test_supported_languages_listing(self) -> None:
        langs = list(supported_languages())
        for expected in ("python", "javascript", "typescript", "rust", "go"):
            self.assertIn(expected, langs)


# ---------------------------------------------------------------------------
# Engine integration: code filter contributes to tokens_saved_by_type
# ---------------------------------------------------------------------------


class EngineIntegrationTests(unittest.TestCase):
    def _make_messages(self, content: str, fingerprint: str) -> tuple[list, ToolRecord]:
        messages = [
            {"role": "user", "content": "show me the file", "timestamp": 1.0},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "tc1", "function": {"name": "read_file", "arguments": "{}"}}
                ],
                "timestamp": 2.0,
            },
            {"role": "tool", "tool_call_id": "tc1", "content": content, "timestamp": 3.0},
        ]
        record = ToolRecord(
            tool_call_id="tc1",
            tool_name="read_file",
            input_args={"path": "src/main.py"},
            input_fingerprint=fingerprint,
            is_error=False,
            turn_index=0,
            timestamp=1.0,
            token_estimate=200,
        )
        return messages, record

    def test_python_file_credits_code_filter_savings(self) -> None:
        # 30+ lines of Python code with bodies
        body_lines = "\n".join(
            f'    print("verbose body line {i}")' for i in range(40)
        )
        content = f'''def main():
    """Main entry point."""
{body_lines}

def helper():
    """A helper."""
{body_lines}
'''
        messages, record = self._make_messages(
            content, fingerprint='read_file::{"path":"src/main.py"}'
        )
        state = create_state()
        state.tool_calls["tc1"] = record
        config = HmcConfig()

        materialize_view(messages, state, config)

        self.assertIn("code_filter", state.tokens_saved_by_type)
        self.assertGreater(state.tokens_saved_by_type["code_filter"], 0)

    def test_short_content_below_min_lines_not_filtered(self) -> None:
        # Only 5 lines — below default min_lines=30
        content = '''def foo():
    """tiny."""
    x = 1
    return x
'''
        messages, record = self._make_messages(
            content, fingerprint='read_file::{"path":"src/x.py"}'
        )
        state = create_state()
        state.tool_calls["tc1"] = record
        config = HmcConfig()

        materialize_view(messages, state, config)

        self.assertNotIn("code_filter", state.tokens_saved_by_type)

    def test_disabled_code_filter_does_nothing(self) -> None:
        body_lines = "\n".join(
            f'    print("body line {i}")' for i in range(60)
        )
        content = f'''def main():
    """Doc."""
{body_lines}
'''
        messages, record = self._make_messages(
            content, fingerprint='read_file::{"path":"src/main.py"}'
        )
        state = create_state()
        state.tool_calls["tc1"] = record
        config = HmcConfig()
        config.code_filter.enabled = False

        materialize_view(messages, state, config)

        self.assertNotIn("code_filter", state.tokens_saved_by_type)
        # Other strategies may still fire, but code_filter must not

    def test_language_not_in_allow_list_skipped(self) -> None:
        body_lines = "\n".join(f'    println!("body {i}");' for i in range(60))
        content = f"""pub fn main() {{
{body_lines}
}}
"""
        messages, record = self._make_messages(
            content, fingerprint='read_file::{"path":"src/main.rs"}'
        )
        state = create_state()
        state.tool_calls["tc1"] = record
        config = HmcConfig()
        config.code_filter.languages = ["python"]  # rust excluded

        materialize_view(messages, state, config)

        self.assertNotIn("code_filter", state.tokens_saved_by_type)

    def test_error_outputs_skipped(self) -> None:
        body_lines = "\n".join(f'    print("body {i}")' for i in range(60))
        content = f'''def main():
    """Doc."""
{body_lines}
'''
        messages, record = self._make_messages(
            content, fingerprint='read_file::{"path":"src/main.py"}'
        )
        # Mark as error
        record.is_error = True
        state = create_state()
        state.tool_calls["tc1"] = record
        config = HmcConfig()

        materialize_view(messages, state, config)

        self.assertNotIn("code_filter", state.tokens_saved_by_type)

    def test_code_filter_stacks_with_truncation_via_per_strategy_gate(self) -> None:
        """Same tool call can credit both code_filter and truncation."""
        # Long enough that AFTER code_filter, the result is still > min_content_length
        # so truncation can also fire on the elided output's leftover lines.
        body = "\n".join(f'    print("body line {i}" * 20)' for i in range(80))
        content = f'''def main():
    """Doc."""
{body}

def helper():
{body}

def another():
{body}
'''
        messages, record = self._make_messages(
            content, fingerprint='read_file::{"path":"src/main.py"}'
        )
        state = create_state()
        state.tool_calls["tc1"] = record
        config = HmcConfig()

        materialize_view(messages, state, config)

        # code_filter should have credited savings
        self.assertIn("code_filter", state.tokens_saved_by_type)
        self.assertGreater(state.tokens_saved_by_type["code_filter"], 0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class HelpersTests(unittest.TestCase):
    def test_estimate_savings_returns_zero_for_unchanged(self) -> None:
        # Single line — filter returns it unchanged
        self.assertEqual(estimate_savings("x = 1", "python"), 0)

    def test_estimate_savings_positive_for_filtered(self) -> None:
        src = '''def foo():
    """doc."""
    x = 1
    y = 2
    z = 3
    return x + y + z
'''
        self.assertGreater(estimate_savings(src, "python"), 0)

    def test_brace_languages_constant(self) -> None:
        self.assertIn("rust", BRACE_LANGUAGES)
        self.assertIn("go", BRACE_LANGUAGES)
        self.assertIn("javascript", BRACE_LANGUAGES)
        self.assertIn("typescript", BRACE_LANGUAGES)
        self.assertNotIn("python", BRACE_LANGUAGES)

    def test_language_extensions_complete(self) -> None:
        for lang in ("python", "javascript", "typescript", "rust", "go"):
            self.assertIn(lang, LANGUAGE_EXTENSIONS)


if __name__ == "__main__":
    unittest.main()
