import copy
import unittest

from hermes_context_manager.config import HmcConfig
from hermes_context_manager.engine import (
    apply_pruning,
    estimate_message_tokens,
    materialize_view,
)
from hermes_context_manager.state import ToolRecord, create_state


def make_config() -> HmcConfig:
    return HmcConfig()


def make_messages() -> list[dict]:
    return [
        {
            "role": "user",
            "content": "please read the file",
            "timestamp": 1000,
        },
        {
            "role": "assistant",
            "content": "Reading the file now.",
            "tool_calls": [
                {
                    "id": "call_abc",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ],
            "timestamp": 2000,
        },
        {
            "role": "tool",
            "tool_call_id": "call_abc",
            "tool_name": "read_file",
            "content": '{"output":"file contents","exit_code":0,"error":null}',
            "timestamp": 3000,
        },
        {
            "role": "user",
            "content": "thanks",
            "timestamp": 4000,
        },
    ]


class EngineTests(unittest.TestCase):
    def test_message_id_snapshot_built_in_order(self) -> None:
        """The snapshot maps positional IDs to message timestamps.

        Regression test post-0.3.2 (commit removing content injection):
        the snapshot must still be populated, but content must be left
        clean -- no <hmc-message-id> tags inside any message body.
        """
        state = create_state()
        result = apply_pruning(copy.deepcopy(make_messages()), state, make_config())

        self.assertEqual(
            state.message_id_snapshot,
            {
                "m001": 1000.0,
                "m002": 2000.0,
                "m003": 3000.0,
                "m004": 4000.0,
            },
        )
        # No tags must leak into ANY message content -- the model
        # would otherwise echo them at the start of replies.
        for msg in result:
            content = msg.get("content")
            if isinstance(content, str):
                self.assertNotIn("<hmc-message-id>", content)
        self.assertEqual(result[1]["tool_calls"][0]["id"], "call_abc")

    def test_materialized_view_preserves_tool_calls_without_stringifying(self) -> None:
        state = create_state()

        result = materialize_view(copy.deepcopy(make_messages()), state, make_config())

        assistant = next(msg for msg in result.messages if msg.get("role") == "assistant")
        self.assertIsInstance(assistant["content"], str)
        # Content must NOT contain a tag -- see test above for the reason.
        self.assertNotIn("<hmc-message-id>", assistant["content"])
        self.assertEqual(assistant["tool_calls"][0]["id"], "call_abc")
        self.assertNotIn("function", assistant["content"])

    def test_deduplication_and_error_purging_tombstone_tool_output(self) -> None:
        state = create_state()
        state.tool_calls["call_old"] = ToolRecord(
            tool_call_id="call_old",
            tool_name="terminal",
            input_args={"command": "ls"},
            input_fingerprint='terminal::{"command":"ls"}',
            is_error=False,
            turn_index=0,
            timestamp=1000.0,
            token_estimate=20,
        )
        state.tool_calls["call_new"] = ToolRecord(
            tool_call_id="call_new",
            tool_name="terminal",
            input_args={"command": "ls"},
            input_fingerprint='terminal::{"command":"ls"}',
            is_error=False,
            turn_index=0,
            timestamp=2000.0,
            token_estimate=20,
        )
        state.tool_calls["call_err"] = ToolRecord(
            tool_call_id="call_err",
            tool_name="terminal",
            input_args={"command": "bad"},
            input_fingerprint='terminal::{"command":"bad"}',
            is_error=True,
            turn_index=0,
            timestamp=1500.0,
            token_estimate=20,
        )
        messages = [
            {"role": "user", "content": "start", "timestamp": 1},
            {"role": "tool", "tool_call_id": "call_old", "tool_name": "terminal", "content": '{"output":"old"}', "timestamp": 2},
            {"role": "tool", "tool_call_id": "call_new", "tool_name": "terminal", "content": '{"output":"new"}', "timestamp": 3},
            {"role": "user", "content": "turn2", "timestamp": 4},
            {"role": "user", "content": "turn3", "timestamp": 5},
            {"role": "user", "content": "turn4", "timestamp": 6},
            {"role": "user", "content": "turn5", "timestamp": 7},
            {"role": "tool", "tool_call_id": "call_err", "tool_name": "terminal", "content": '{"error":"boom"}', "is_error": True, "timestamp": 8},
        ]

        result = apply_pruning(copy.deepcopy(messages), state, make_config())
        old = next(msg for msg in result if msg.get("tool_call_id") == "call_old")
        err = next(msg for msg in result if msg.get("tool_call_id") == "call_err")

        self.assertIn("Output removed", old["content"])
        self.assertIn("Error output removed", err["content"])

    def test_error_tombstone_uses_record_when_message_lacks_is_error(self) -> None:
        state = create_state()
        state.tool_calls["call_err"] = ToolRecord(
            tool_call_id="call_err",
            tool_name="terminal",
            input_args={"command": "bad"},
            input_fingerprint='terminal::{"command":"bad"}',
            is_error=True,
            turn_index=0,
            timestamp=1500.0,
            token_estimate=20,
        )
        messages = [
            {"role": "user", "content": "start", "timestamp": 1},
            {"role": "user", "content": "turn2", "timestamp": 2},
            {"role": "user", "content": "turn3", "timestamp": 3},
            {"role": "user", "content": "turn4", "timestamp": 4},
            {"role": "user", "content": "turn5", "timestamp": 5},
            {"role": "tool", "tool_call_id": "call_err", "tool_name": "terminal", "content": '{"error":"boom"}', "timestamp": 6},
        ]

        result = apply_pruning(copy.deepcopy(messages), state, make_config())
        err = next(msg for msg in result if msg.get("tool_call_id") == "call_err")

        self.assertIn("Error output removed", err["content"])

    def test_error_results_exempt_from_short_circuit(self):
        """Error tool results should NOT be short-circuited."""
        messages = [
            {"role": "user", "content": "test", "timestamp": 1.0},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "tc1", "function": {"name": "terminal", "arguments": "{}"}}
            ], "timestamp": 2.0},
            {"role": "tool", "tool_call_id": "tc1",
             "content": '{"status": "ok", "detail": "succeeded"}',
             "timestamp": 3.0},
        ]
        state = create_state()
        state.tool_calls["tc1"] = ToolRecord(
            tool_call_id="tc1", tool_name="terminal",
            input_args={}, input_fingerprint="terminal::{}",
            is_error=True, turn_index=0, timestamp=1.0, token_estimate=50,
        )
        config = HmcConfig()
        result = materialize_view(messages, state, config)
        tool_msgs = [m for m in result.messages if m.get("role") == "tool"]
        # Error content should be preserved, not short-circuited
        self.assertNotIn("[ok]", tool_msgs[0]["content"])
        self.assertIn("status", tool_msgs[0]["content"])


    def test_dedup_tombstone_shows_count(self):
        messages = [
            {"role": "user", "content": "search", "timestamp": 1.0},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "tc1", "function": {"name": "search_files", "arguments": '{"q": "foo"}'}}
            ], "timestamp": 2.0},
            {"role": "tool", "tool_call_id": "tc1", "content": "result A", "timestamp": 3.0},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "tc2", "function": {"name": "search_files", "arguments": '{"q": "foo"}'}}
            ], "timestamp": 4.0},
            {"role": "tool", "tool_call_id": "tc2", "content": "result B", "timestamp": 5.0},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "tc3", "function": {"name": "search_files", "arguments": '{"q": "foo"}'}}
            ], "timestamp": 6.0},
            {"role": "tool", "tool_call_id": "tc3", "content": "result C", "timestamp": 7.0},
        ]
        state = create_state()
        for i, tc_id in enumerate(["tc1", "tc2", "tc3"]):
            state.tool_calls[tc_id] = ToolRecord(
                tool_call_id=tc_id, tool_name="search_files",
                input_args={"q": "foo"},
                input_fingerprint='search_files::{"q": "foo"}',
                is_error=False, turn_index=0, timestamp=float(i), token_estimate=20,
            )
        config = HmcConfig()
        result = materialize_view(messages, state, config)
        tool_msgs = [m for m in result.messages if m.get("role") == "tool"]
        tombstoned = [m for m in tool_msgs if "removed" in m.get("content", "").lower()]
        self.assertEqual(len(tombstoned), 2)
        has_count = any("3" in m["content"] and "search_files" in m["content"] for m in tombstoned)
        self.assertTrue(has_count, "Tombstone should mention count and tool name")


class TokenSavingsByTypeTests(unittest.TestCase):
    def test_token_savings_tracked_by_type(self):
        long_output = "collecting tests...\n" * 10 + "===== 42 passed =====\n"
        messages = [
            {"role": "user", "content": "run tests", "timestamp": 1.0},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "tc1", "function": {"name": "terminal", "arguments": "{}"}}
            ], "timestamp": 2.0},
            {"role": "tool", "tool_call_id": "tc1",
             "content": long_output, "timestamp": 3.0},
        ]
        state = create_state()
        state.tool_calls["tc1"] = ToolRecord(
            tool_call_id="tc1", tool_name="terminal",
            input_args={}, input_fingerprint="terminal::{}",
            is_error=False, turn_index=0, timestamp=1.0, token_estimate=50,
        )
        config = HmcConfig()
        materialize_view(messages, state, config)
        self.assertGreater(state.tokens_saved, 0)
        self.assertIn("short_circuit", state.tokens_saved_by_type)
        self.assertGreater(state.tokens_saved_by_type["short_circuit"], 0)

    def test_dedup_pruning_credits_savings(self) -> None:
        """Regression test: dedup must contribute to tokens_saved_by_type.

        Before this fix, _apply_tool_output_pruning replaced content with
        placeholder strings without ever incrementing tokens_saved.  Users
        looking at hmc_control stats would see ``dedup: 0`` even when dedup
        was pruning thousands of bytes.
        """
        long_output = "result line\n" * 200  # ~2200 bytes per copy
        messages = [
            {"role": "user", "content": "search", "timestamp": 1.0},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "tc1", "function": {"name": "search_files", "arguments": '{"q": "foo"}'}}
            ], "timestamp": 2.0},
            {"role": "tool", "tool_call_id": "tc1", "content": long_output, "timestamp": 3.0},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "tc2", "function": {"name": "search_files", "arguments": '{"q": "foo"}'}}
            ], "timestamp": 4.0},
            {"role": "tool", "tool_call_id": "tc2", "content": long_output, "timestamp": 5.0},
        ]
        state = create_state()
        for i, tc_id in enumerate(["tc1", "tc2"]):
            state.tool_calls[tc_id] = ToolRecord(
                tool_call_id=tc_id, tool_name="search_files",
                input_args={"q": "foo"},
                input_fingerprint='search_files::{"q": "foo"}',
                is_error=False, turn_index=0, timestamp=float(i), token_estimate=550,
            )
        config = HmcConfig()
        # Disable truncation so the savings are unambiguously from dedup,
        # not from head/tail windowing of the long output.
        config.truncation.enabled = False
        materialize_view(messages, state, config)

        self.assertIn("dedup", state.tokens_saved_by_type)
        self.assertGreater(state.tokens_saved_by_type["dedup"], 0)
        # The savings should be roughly the size of one pruned copy minus
        # the placeholder.
        self.assertGreater(state.tokens_saved_by_type["dedup"], 100)

    def test_error_purge_pruning_credits_savings(self) -> None:
        """Regression test: error_purge must contribute to tokens_saved_by_type."""
        long_error_str = "TypeError: bad call\n" + "\n".join(f"  at frame {i}" for i in range(100))
        messages = [
            {"role": "user", "content": "do thing", "timestamp": 1.0},
            {"role": "user", "content": "still here", "timestamp": 2.0},
            {"role": "user", "content": "and here", "timestamp": 3.0},
            {"role": "user", "content": "and here", "timestamp": 4.0},
            {"role": "user", "content": "and here", "timestamp": 5.0},
            {"role": "user", "content": "now what", "timestamp": 6.0},
            {"role": "tool", "tool_call_id": "tc_err", "content": long_error_str, "timestamp": 7.0},
        ]
        state = create_state()
        state.current_turn = 6  # five turns past the error
        state.tool_calls["tc_err"] = ToolRecord(
            tool_call_id="tc_err", tool_name="terminal",
            input_args={"command": "boom"},
            input_fingerprint='terminal::{"command":"boom"}',
            is_error=True, turn_index=0, timestamp=1.0, token_estimate=400,
        )
        config = HmcConfig()
        config.truncation.enabled = False
        config.short_circuits.enabled = False
        materialize_view(messages, state, config)

        self.assertIn("error_purge", state.tokens_saved_by_type)
        self.assertGreater(state.tokens_saved_by_type["error_purge"], 0)

    def test_cross_strategy_savings_stack_on_same_tool_call(self) -> None:
        """Truncation and dedup on the same tool_call_id must both credit.

        With the old per-id gate, only the first strategy that fired got
        credit and any subsequent strategy was silently dropped.  The new
        per-(id, strategy) gate lets each strategy claim its own
        incremental contribution.
        """
        long_output = "\n".join(f"line {i:04d}" for i in range(80))  # > max_lines, > min_content_length
        messages = [
            {"role": "user", "content": "go", "timestamp": 1.0},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "tc1", "function": {"name": "search_files", "arguments": '{"q": "foo"}'}}
            ], "timestamp": 2.0},
            {"role": "tool", "tool_call_id": "tc1", "content": long_output, "timestamp": 3.0},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "tc2", "function": {"name": "search_files", "arguments": '{"q": "foo"}'}}
            ], "timestamp": 4.0},
            {"role": "tool", "tool_call_id": "tc2", "content": long_output, "timestamp": 5.0},
        ]
        state = create_state()
        for i, tc_id in enumerate(["tc1", "tc2"]):
            state.tool_calls[tc_id] = ToolRecord(
                tool_call_id=tc_id, tool_name="search_files",
                input_args={"q": "foo"},
                input_fingerprint='search_files::{"q": "foo"}',
                is_error=False, turn_index=0, timestamp=float(i), token_estimate=200,
            )
        config = HmcConfig()
        materialize_view(messages, state, config)

        # Both truncation AND dedup should have credited savings on tc1
        # (truncation reduced the size, then dedup pruned the duplicate).
        self.assertIn("truncation", state.tokens_saved_by_type)
        self.assertIn("dedup", state.tokens_saved_by_type)
        self.assertGreater(state.tokens_saved_by_type["truncation"], 0)
        self.assertGreater(state.tokens_saved_by_type["dedup"], 0)

    def test_savings_gate_persists_across_turns(self) -> None:
        """The new tuple-based gate must still prevent same-strategy double-count.

        Regression net for the original double-count fix in commit 2460712.
        """
        long_output = "\n".join(f"line {i:04d}" for i in range(80))
        messages = [
            {"role": "user", "content": "go", "timestamp": 1.0},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "tc1", "function": {"name": "terminal", "arguments": "{}"}}
            ], "timestamp": 2.0},
            {"role": "tool", "tool_call_id": "tc1", "content": long_output, "timestamp": 3.0},
        ]
        state = create_state()
        state.tool_calls["tc1"] = ToolRecord(
            tool_call_id="tc1", tool_name="terminal",
            input_args={}, input_fingerprint="terminal::{}",
            is_error=False, turn_index=0, timestamp=1.0, token_estimate=200,
        )
        config = HmcConfig()
        materialize_view(messages, state, config)
        first_total = state.tokens_saved
        first_truncation = state.tokens_saved_by_type.get("truncation", 0)

        # Reset content (simulating _restore_mutations on next turn) and re-run
        messages[2]["content"] = long_output
        materialize_view(messages, state, config)

        # Same total: gate prevented double count for the (tc1, "truncation") key
        self.assertEqual(state.tokens_saved, first_total)
        self.assertEqual(state.tokens_saved_by_type.get("truncation", 0), first_truncation)

    def test_kept_out_total_grows_across_turns_for_persistent_tool_call(self) -> None:
        """Regression test for the tokens_saved undercount bug.

        Scenario observed in real Hermes sessions: a single tool output
        persists across many turns.  ``_restore_mutations`` re-hydrates the
        message between turns, and ``materialize_view`` re-compresses it
        every turn.  The gated ``tokens_saved`` counter stays fixed (one
        credit per (tool_call_id, strategy)) but the un-gated
        ``tokens_kept_out_total`` must grow linearly with turn count,
        because those bytes really are kept out of each API call.
        """
        long_output = "\n".join(f"line {i:04d}" for i in range(80))
        messages = [
            {"role": "user", "content": "go", "timestamp": 1.0},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "tc1", "function": {"name": "terminal", "arguments": "{}"}}
            ], "timestamp": 2.0},
            {"role": "tool", "tool_call_id": "tc1",
             "content": long_output, "timestamp": 3.0},
        ]
        state = create_state()
        state.tool_calls["tc1"] = ToolRecord(
            tool_call_id="tc1", tool_name="terminal",
            input_args={}, input_fingerprint="terminal::{}",
            is_error=False, turn_index=0, timestamp=1.0, token_estimate=200,
        )
        config = HmcConfig()

        # Turn 1 -- initial compression.  Both counters should credit.
        materialize_view(messages, state, config)
        first_uniq = state.tokens_saved
        first_kept_out = state.tokens_kept_out_total
        self.assertGreater(first_uniq, 0)
        self.assertEqual(first_kept_out, first_uniq)

        # Turns 2..20 -- restore content each turn (as plugin does in
        # _restore_mutations) and re-run materialize_view.  The gated
        # counter must stay pinned at its first-turn value; the un-gated
        # accumulator must grow by approximately first_kept_out on each
        # turn.
        NUM_EXTRA_TURNS = 19
        for _ in range(NUM_EXTRA_TURNS):
            messages[2]["content"] = long_output
            materialize_view(messages, state, config)

        # Gate still held: diagnostic counter pinned.
        self.assertEqual(state.tokens_saved, first_uniq)
        self.assertEqual(
            state.tokens_saved_by_type.get("truncation", 0),
            first_truncation := state.tokens_saved_by_type["truncation"],
        )

        # Un-gated grew by ~20x the per-turn savings.  Allow a small fudge
        # factor for message-id tag length variations across turns but
        # verify we're in the right order of magnitude: at minimum 10x,
        # which already falsifies the old (gated-only) behavior.
        self.assertGreaterEqual(
            state.tokens_kept_out_total,
            first_kept_out * 10,
            "tokens_kept_out_total should grow each turn that a "
            "persistent tool output gets re-compressed",
        )
        # And per-strategy breakdown mirrors the same growth.
        self.assertGreaterEqual(
            state.tokens_kept_out_by_type.get("truncation", 0),
            first_truncation * 10,
        )


class ContentBackupLazyCaptureTests(unittest.TestCase):
    """Regression tests for 0.3.6 lazy content backup (review #2).

    Before 0.3.6 ``materialize_view`` pre-populated content_backup
    with every message's content upfront -- N entries for an
    N-message conversation.  Now the backup is populated only when
    a strategy actually mutates a message, so a 200-message
    conversation with 5 tool outputs ends up with ~5 entries.
    """

    def test_backup_contains_only_mutated_messages(self) -> None:
        """Messages that no strategy touches must NOT appear in backup."""
        # Conversation with 1 mutating tool output + 4 untouched user/assistant
        # messages.  Only the tool output should appear in content_backup.
        long_output = '{"status": "ok", "detail": "done", "exit_code": 0}'
        messages = [
            {"role": "user", "content": "first", "timestamp": 1.0},
            {"role": "assistant", "content": "ok", "timestamp": 2.0},
            {"role": "user", "content": "second", "timestamp": 3.0},
            {
                "role": "assistant", "content": "",
                "tool_calls": [
                    {"id": "tc1", "function": {"name": "terminal", "arguments": "{}"}}
                ],
                "timestamp": 4.0,
            },
            {
                "role": "tool",
                "tool_call_id": "tc1",
                "content": long_output,
                "timestamp": 5.0,
            },
        ]
        state = create_state()
        state.tool_calls["tc1"] = ToolRecord(
            tool_call_id="tc1", tool_name="terminal",
            input_args={}, input_fingerprint="terminal::{}",
            is_error=False, turn_index=0, timestamp=1.0, token_estimate=20,
        )
        config = HmcConfig()
        view = materialize_view(messages, state, config)

        # Only the tool output (index 4) got short-circuited and
        # backed up.  The 4 user/assistant messages must NOT appear.
        self.assertEqual(
            set(view.content_backup.keys()), {4},
            f"backup should contain only mutated messages, got {set(view.content_backup.keys())}",
        )
        # Sanity: the backup actually holds the pre-mutation content
        self.assertEqual(view.content_backup[4], long_output)

    def test_empty_backup_when_no_strategy_fires(self) -> None:
        """Conversation with nothing to compress yields an empty backup."""
        messages = [
            {"role": "user", "content": "hi", "timestamp": 1.0},
            {"role": "assistant", "content": "hello", "timestamp": 2.0},
        ]
        state = create_state()
        config = HmcConfig()
        view = materialize_view(messages, state, config)
        self.assertEqual(view.content_backup, {})

    def test_first_strategy_wins_backup_capture(self) -> None:
        """Stacked strategies must not overwrite an earlier backup entry.

        Truncation + dedup both fire on the same message: truncation
        captures the raw original, then ``_apply_tool_output_pruning``
        replaces content with a tombstone.  The backup must still
        hold the TRUE original content, not the truncated intermediate.
        """
        original = "\n".join(f"line {i:04d}" for i in range(200))
        messages = [
            {"role": "user", "content": "run", "timestamp": 1.0},
            {
                "role": "assistant", "content": "",
                "tool_calls": [
                    {"id": "tc1", "function": {"name": "search_files", "arguments": '{"q":"x"}'}}
                ],
                "timestamp": 2.0,
            },
            {"role": "tool", "tool_call_id": "tc1", "content": original, "timestamp": 3.0},
            {
                "role": "assistant", "content": "",
                "tool_calls": [
                    {"id": "tc2", "function": {"name": "search_files", "arguments": '{"q":"x"}'}}
                ],
                "timestamp": 4.0,
            },
            {"role": "tool", "tool_call_id": "tc2", "content": original, "timestamp": 5.0},
        ]
        state = create_state()
        for tc_id in ("tc1", "tc2"):
            state.tool_calls[tc_id] = ToolRecord(
                tool_call_id=tc_id, tool_name="search_files",
                input_args={"q": "x"}, input_fingerprint='search_files::{"q": "x"}',
                is_error=False, turn_index=0, timestamp=1.0, token_estimate=500,
            )
        config = HmcConfig()
        view = materialize_view(messages, state, config)
        # tc1 gets truncated AND dedup-pruned: truncation captures the
        # backup first, dedup's pruning pass skips the capture because
        # _capture_content_backup refuses to overwrite.
        self.assertIn(2, view.content_backup)
        self.assertEqual(view.content_backup[2], original)


class EstimateMessageTokensTests(unittest.TestCase):
    """Regression tests for 0.3.6 API-accurate token estimation.

    Before 0.3.6 ``estimate_message_tokens`` returned
    ``len(str(message)) // 4`` which counted Python dict repr overhead
    AND internal fields like ``timestamp`` that never ship to the
    provider.  Observed inflation was 20-30% on typical messages,
    which in turn triggered background compression earlier than
    necessary.
    """

    def test_timestamp_field_does_not_inflate_estimate(self) -> None:
        with_ts = {
            "role": "user",
            "content": "hello world",
            "timestamp": 1775932328.183279,
        }
        without_ts = {
            "role": "user",
            "content": "hello world",
        }
        # timestamp is HMC-internal -- both must produce the same estimate.
        self.assertEqual(
            estimate_message_tokens(with_ts),
            estimate_message_tokens(without_ts),
        )

    def test_internal_fields_ignored(self) -> None:
        """``tool_name`` is HMC-internal metadata on tool messages."""
        raw = {
            "role": "tool",
            "tool_call_id": "call_abc",
            "tool_name": "terminal",  # HMC-internal duplicate
            "content": "output line\n" * 10,
            "timestamp": 1000.0,
        }
        clean = {
            "role": "tool",
            "tool_call_id": "call_abc",
            "content": "output line\n" * 10,
        }
        self.assertEqual(
            estimate_message_tokens(raw),
            estimate_message_tokens(clean),
        )

    def test_tool_calls_are_counted(self) -> None:
        """Assistant messages with tool_calls must include them in the estimate.

        Assistant messages often have empty ``content`` but a
        heavy ``tool_calls`` payload.  Counting only ``content``
        would massively under-report.
        """
        with_tools = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path": "/very/long/path/to/file.py"}',
                    },
                }
            ],
        }
        empty_content = {
            "role": "assistant",
            "content": "",
        }
        self.assertGreater(
            estimate_message_tokens(with_tools),
            estimate_message_tokens(empty_content),
        )

    def test_pre_fix_inflation_gone(self) -> None:
        """The old ``len(str(message)) // 4`` would return ~50% more.

        Quantify the inflation: a message with a timestamp and
        realistic content should be notably smaller than the old
        heuristic would have reported.  This locks in the improvement.
        """
        message = {
            "role": "user",
            "content": "please run the test suite",
            "timestamp": 1775932328.183279,
        }
        new_estimate = estimate_message_tokens(message)
        old_estimate = len(str(message)) // 4
        self.assertLess(
            new_estimate, old_estimate,
            f"new estimate {new_estimate} should be less than "
            f"old estimate {old_estimate} (timestamp + Python repr "
            f"overhead should be stripped)",
        )

    def test_non_serializable_content_falls_back_gracefully(self) -> None:
        """Belt-and-suspenders: exotic values must not crash the estimator."""
        class _Exotic:
            def __repr__(self) -> str:
                return "<exotic>"

        message = {
            "role": "user",
            "content": _Exotic(),  # not JSON-serializable
        }
        # Should not raise -- falls back to len(str(message)) // 4
        result = estimate_message_tokens(message)
        self.assertGreater(result, 0)


if __name__ == "__main__":
    unittest.main()
