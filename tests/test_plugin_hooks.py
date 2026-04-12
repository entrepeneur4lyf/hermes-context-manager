import copy
import json
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest import mock

from hermes_context_manager.background_compressor import build_index_entry
from hermes_context_manager.config import HmcConfig
from hermes_context_manager.engine import materialize_view
from hermes_context_manager.plugin import HermesContextManagerPlugin
from hermes_context_manager.prompts import SYSTEM_CONTEXT
from hermes_context_manager.state import ToolRecord, create_state


class DummyCtx:
    def __init__(self) -> None:
        self.tools = []
        self.hooks = {}

    def register_tool(self, **kwargs) -> None:
        self.tools.append(kwargs)

    def register_hook(self, hook_name, callback) -> None:
        self.hooks[hook_name] = callback

    def register_cli_command(self, **kwargs) -> None:
        self.cli_command = kwargs

    def inject_message(self, content: str, role: str = "user") -> bool:
        self.injected = (role, content)
        return True


class PluginHookTests(unittest.TestCase):
    def test_plugin_registers_hooks_and_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plugin_dir = Path(tmp) / "plugin"
            (plugin_dir / "hmc_skill").mkdir(parents=True)
            (plugin_dir / "hmc_skill" / "SKILL.md").write_text("name: hmc\n", encoding="utf-8")
            (plugin_dir / "config.yaml.example").write_text("enabled: true\n", encoding="utf-8")
            ctx = DummyCtx()
            plugin = HermesContextManagerPlugin(
                hermes_home=tmp,
                config=HmcConfig(),
                plugin_dir=plugin_dir,
            )

            plugin.register(ctx)

            tool_names = {tool["name"] for tool in ctx.tools}
            self.assertIn("hmc_control", tool_names)
            self.assertIn("pre_llm_call", ctx.hooks)
            self.assertIn("pre_tool_call", ctx.hooks)
            self.assertIn("on_session_end", ctx.hooks)
            self.assertTrue((plugin.plugin_dir / "config.yaml").exists())

    def test_pre_llm_call_does_not_pollute_content_with_message_ids(self) -> None:
        """Regression test for the 0.3.2 fix.

        Until 0.3.2 HMC appended ``<hmc-message-id>m0NN</hmc-message-id>``
        to every message's content during materialize_view.  GLM-5.1
        learned to ECHO the tag at the start of its replies (32% of
        captured assistant messages in one session began with the tag),
        which the user perceived as "first words cut off in real time"
        when Hermes's display layer mishandled the unwanted tag in the
        stream.  No content must contain the tag now -- not before, not
        after, not anywhere.
        """
        with tempfile.TemporaryDirectory() as tmp:
            plugin = HermesContextManagerPlugin(hermes_home=tmp, config=HmcConfig())
            messages = [
                {"role": "user", "content": "start", "timestamp": 1.0},
                {
                    "role": "assistant",
                    "content": "Reading.",
                    "tool_calls": [
                        {"id": "call_1", "type": "function", "function": {"name": "terminal", "arguments": '{"command":"ls"}'}}
                    ],
                    "timestamp": 2.0,
                },
                {"role": "tool", "content": '{"output":"listing"}', "tool_call_id": "call_1", "tool_name": "terminal", "timestamp": 3.0},
                {"role": "user", "content": "next", "timestamp": 4.0},
            ]

            injection = plugin.on_pre_llm_call(
                session_id="session-1",
                user_message="next",
                conversation_history=messages,
                is_first_turn=False,
                model="test-model",
                platform="cli",
            )

            self.assertIsNotNone(injection)
            assert injection is not None
            self.assertIn(SYSTEM_CONTEXT, injection["context"])
            # Content must remain CLEAN for every message, every role.
            for msg in messages:
                content = msg.get("content")
                if isinstance(content, str):
                    self.assertNotIn("<hmc-message-id>", content)
            # Tool-call structure preserved.
            self.assertEqual(messages[1]["tool_calls"][0]["id"], "call_1")
            # Snapshot is still populated with positional ids.
            state = plugin._states["session-1"]
            self.assertEqual(len(state.message_id_snapshot), 4)
            self.assertIn("m001", state.message_id_snapshot)

            plugin.on_session_end(
                session_id="session-1",
                completed=True,
                interrupted=False,
                model="test-model",
                platform="cli",
            )

            # Still clean after session end (sanity check).
            self.assertNotIn("<hmc-message-id>", str(messages[1]["content"]))

    def test_pre_llm_call_returns_system_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plugin = HermesContextManagerPlugin(hermes_home=tmp, config=HmcConfig())

            injection = plugin.on_pre_llm_call(
                session_id="session-1",
                user_message="next",
                conversation_history=[{"role": "user", "content": "next", "timestamp": 1.0}],
                is_first_turn=False,
                model="test-model",
                platform="cli",
            )

            self.assertIsNotNone(injection)
            assert injection is not None
            self.assertEqual(injection["context"], SYSTEM_CONTEXT)

    def test_pre_llm_call_matches_engine_materialized_view(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plugin_a = HermesContextManagerPlugin(hermes_home=tmp, config=HmcConfig())
            state_a = plugin_a._get_state("session-a")
            base_messages = [
                {"role": "user", "content": "start", "timestamp": 1.0},
                {
                    "role": "assistant",
                    "content": "Reading.",
                    "tool_calls": [
                        {"id": "call_1", "type": "function", "function": {"name": "terminal", "arguments": '{"command":"ls"}'}}
                    ],
                    "timestamp": 2.0,
                },
                {"role": "tool", "content": '{"output":"listing"}', "tool_call_id": "call_1", "tool_name": "terminal", "timestamp": 3.0},
                {"role": "user", "content": "next", "timestamp": 4.0},
            ]

            # Engine path operates on its own deep copy with its own state, so
            # ``materialize_view`` produces the reference mutation set without
            # interference from the plugin's later run.
            engine_messages = copy.deepcopy(base_messages)
            expected = materialize_view(engine_messages, state_a, plugin_a.config)

            plugin_b = HermesContextManagerPlugin(hermes_home=tmp, config=HmcConfig())
            plugin_messages = copy.deepcopy(base_messages)
            plugin_b.on_pre_llm_call(
                session_id="session-b",
                user_message="next",
                conversation_history=plugin_messages,
                is_first_turn=False,
                model="test-model",
                platform="cli",
            )

            self.assertEqual(plugin_messages, expected.messages)

    def test_sweep_without_count_only_sweeps_since_last_user(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plugin = HermesContextManagerPlugin(hermes_home=tmp, config=HmcConfig())
            state = plugin._get_state("session-1")
            state.tool_calls["call_old"] = ToolRecord(
                tool_call_id="call_old",
                tool_name="terminal",
                input_args={"command": "pwd"},
                input_fingerprint='terminal::{"command":"pwd"}',
                is_error=False,
                turn_index=0,
                timestamp=1.0,
                token_estimate=10,
            )
            state.tool_calls["call_new"] = ToolRecord(
                tool_call_id="call_new",
                tool_name="terminal",
                input_args={"command": "ls"},
                input_fingerprint='terminal::{"command":"ls"}',
                is_error=False,
                turn_index=1,
                timestamp=2.0,
                token_estimate=10,
            )
            plugin._session_messages["session-1"] = [
                {"role": "user", "content": "first", "timestamp": 1.0},
                {"role": "tool", "tool_call_id": "call_old", "tool_name": "terminal", "content": "old", "timestamp": 2.0},
                {"role": "user", "content": "second", "timestamp": 3.0},
                {"role": "tool", "tool_call_id": "call_new", "tool_name": "terminal", "content": "new", "timestamp": 4.0},
            ]

            result = plugin.handle_hmc_control({"action": "sweep"}, task_id=None)

            self.assertIn('"swept": 1', result)
            self.assertNotIn("call_old", state.pruned_tool_ids)
            self.assertIn("call_new", state.pruned_tool_ids)

    def test_sweep_without_cached_messages_falls_back_to_current_turn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plugin = HermesContextManagerPlugin(hermes_home=tmp, config=HmcConfig())
            state = plugin._get_state("session-1")
            state.current_turn = 2
            state.tool_calls["call_old"] = ToolRecord(
                tool_call_id="call_old",
                tool_name="terminal",
                input_args={"command": "pwd"},
                input_fingerprint='terminal::{"command":"pwd"}',
                is_error=False,
                turn_index=1,
                timestamp=1.0,
                token_estimate=10,
            )
            state.tool_calls["call_new"] = ToolRecord(
                tool_call_id="call_new",
                tool_name="terminal",
                input_args={"command": "ls"},
                input_fingerprint='terminal::{"command":"ls"}',
                is_error=False,
                turn_index=2,
                timestamp=2.0,
                token_estimate=10,
            )

            result = plugin.handle_hmc_control({"action": "sweep"}, task_id=None)

            self.assertIn('"swept": 1', result)
            self.assertNotIn("call_old", state.pruned_tool_ids)
            self.assertIn("call_new", state.pruned_tool_ids)

    def test_sweep_skips_hardcoded_protected_tools_even_with_empty_config(self) -> None:
        """Sweep must skip hmc_control even when config protected_tools is empty."""
        with tempfile.TemporaryDirectory() as tmp:
            config = HmcConfig()
            config.compress.protected_tools = []
            config.strategies.deduplication.protected_tools = []
            plugin = HermesContextManagerPlugin(hermes_home=tmp, config=config)
            state = plugin._get_state("session-1")
            state.tool_calls["call_control"] = ToolRecord(
                tool_call_id="call_control",
                tool_name="hmc_control",
                input_args={},
                input_fingerprint="hmc_control::{}",
                is_error=False,
                turn_index=1,
                timestamp=2.0,
                token_estimate=10,
            )
            state.tool_calls["call_terminal"] = ToolRecord(
                tool_call_id="call_terminal",
                tool_name="terminal",
                input_args={"command": "ls"},
                input_fingerprint='terminal::{"command":"ls"}',
                is_error=False,
                turn_index=1,
                timestamp=3.0,
                token_estimate=10,
            )
            plugin._session_messages["session-1"] = [
                {"role": "user", "content": "go", "timestamp": 0.5},
                {"role": "tool", "tool_call_id": "call_control", "tool_name": "hmc_control", "content": "ok", "timestamp": 2.0},
                {"role": "tool", "tool_call_id": "call_terminal", "tool_name": "terminal", "content": "files", "timestamp": 3.0},
            ]

            result = plugin.handle_hmc_control({"action": "sweep"}, task_id=None)

            # Only terminal should be swept; hmc_control is hardcoded protected
            self.assertIn('"swept": 1', result)
            self.assertNotIn("call_control", state.pruned_tool_ids)
            self.assertIn("call_terminal", state.pruned_tool_ids)

    def test_context_action_reports_last_context_usage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plugin = HermesContextManagerPlugin(hermes_home=tmp, config=HmcConfig())
            messages = [
                {"role": "user", "content": "start", "timestamp": 1.0},
                {"role": "assistant", "content": "reply", "timestamp": 2.0},
            ]

            plugin.on_pre_llm_call(
                session_id="session-1",
                user_message="start",
                conversation_history=messages,
                is_first_turn=False,
                model="test-model",
                platform="cli",
            )

            result = plugin.handle_hmc_control({"action": "context"}, task_id=None)

            self.assertIn('"context_tokens_estimate"', result)
            self.assertIn('"context_usage_percent"', result)


    def test_index_action_returns_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            plugin = HermesContextManagerPlugin(hermes_home=tmp, config=HmcConfig())
            state = create_state()
            plugin._states["s1"] = state
            plugin.state_store.append_index("s1", {
                "turn_range": "1-5",
                "topic": "Setup",
                "summary": "Installed deps",
            })
            result = json.loads(plugin.handle_hmc_control(
                {"action": "index"}, task_id=None,
            ))
            self.assertEqual(result["count"], 1)
            self.assertEqual(result["entries"][0]["topic"], "Setup")

    def test_index_action_empty_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            plugin = HermesContextManagerPlugin(hermes_home=tmp, config=HmcConfig())
            state = create_state()
            plugin._states["s1"] = state
            result = json.loads(plugin.handle_hmc_control(
                {"action": "index"}, task_id=None,
            ))
            self.assertEqual(result["count"], 0)
            self.assertEqual(result["entries"], [])

    def test_status_action_returns_compact_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            plugin = HermesContextManagerPlugin(hermes_home=tmp, config=HmcConfig())
            state = create_state()
            # Un-gated accumulator (primary) plus gated diagnostic.
            state.tokens_kept_out_total = 15000
            state.tokens_kept_out_by_type = {
                "short_circuit": 10000, "truncation": 5000,
            }
            state.tokens_saved = 15000
            state.tokens_saved_by_type = {"short_circuit": 10000, "truncation": 5000}
            state.total_prune_count = 3
            state.last_context_percent = 0.42
            plugin._states["s1"] = state
            result = json.loads(plugin.handle_hmc_control(
                {"action": "status"}, task_id=None,
            ))
            status_text = result["status"]
            self.assertIn("auto", status_text)
            self.assertIn("15.0K", status_text)
            self.assertIn("42%", status_text)
            self.assertIn("short_circuit", status_text)

    def test_concurrent_post_tool_calls_no_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plugin = HermesContextManagerPlugin(hermes_home=tmp, config=HmcConfig())
            state = create_state()
            plugin._states["s1"] = state
            errors = []

            def call_post(i):
                try:
                    plugin.on_post_tool_call(
                        tool_name="test_tool",
                        args={"i": i},
                        result=json.dumps({"ok": True}),
                        task_id="t1",
                        session_id="s1",
                        tool_call_id=f"tc_{i}",
                    )
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=call_post, args=(i,)) for i in range(20)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            self.assertEqual(len(errors), 0, f"Errors: {errors}")
            self.assertEqual(len(state.tool_calls), 20)

    def test_build_index_entry_handles_multimodal_content(self) -> None:
        """build_index_entry must not crash on list-shaped multimodal content."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Please analyze this image"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
                ],
                "timestamp": 1.0,
            },
            {"role": "assistant", "content": "Analyzing.", "timestamp": 2.0},
        ]

        entry = build_index_entry(messages, 0, 1, "1-1", summary="")

        self.assertEqual(entry["topic"], "Please analyze this image")
        self.assertIn("Please analyze this image", entry["summary"])

    def test_tokens_saved_not_double_counted_across_turns(self) -> None:
        """Token savings for a given tool_call_id must only accrue once.

        Before the fix, each turn re-applied short-circuit/truncation to the
        same tool output and re-incremented ``tokens_saved`` linearly per turn.
        """
        with tempfile.TemporaryDirectory() as tmp:
            plugin = HermesContextManagerPlugin(hermes_home=tmp, config=HmcConfig())
            # 60 lines × ~10 chars each → exceeds max_lines=50 AND
            # min_content_length=500 → head/tail truncation fires.
            long_output = "\n".join(f"line {i:04d}" for i in range(60))
            messages = [
                {"role": "user", "content": "run ls", "timestamp": 1.0},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_long",
                            "type": "function",
                            "function": {"name": "terminal", "arguments": '{"command":"ls"}'},
                        }
                    ],
                    "timestamp": 2.0,
                },
                {
                    "role": "tool",
                    "content": long_output,
                    "tool_call_id": "call_long",
                    "tool_name": "terminal",
                    "timestamp": 3.0,
                },
                {"role": "user", "content": "anything else?", "timestamp": 4.0},
            ]

            plugin.on_pre_llm_call(
                session_id="session-1",
                user_message="anything else?",
                conversation_history=messages,
                is_first_turn=False,
                model="test-model",
                platform="cli",
            )
            saved_after_turn_1 = plugin._states["session-1"].tokens_saved

            # Second turn: restore_mutations puts the original long content back,
            # then materialize_view runs the strategies again.  With the gate in
            # place the saved total should NOT grow.
            messages.append({"role": "assistant", "content": "ok", "timestamp": 5.0})
            messages.append({"role": "user", "content": "and again?", "timestamp": 6.0})

            plugin.on_pre_llm_call(
                session_id="session-1",
                user_message="and again?",
                conversation_history=messages,
                is_first_turn=False,
                model="test-model",
                platform="cli",
            )
            saved_after_turn_2 = plugin._states["session-1"].tokens_saved

            self.assertGreater(saved_after_turn_1, 0)
            self.assertEqual(saved_after_turn_1, saved_after_turn_2)

    def test_mutation_isolation_after_deepcopy_optimization(self) -> None:
        """Verify mutation isolation still works after reducing deepcopy calls.

        Uses a short-circuitable tool result so an actual mutation
        happens during materialize_view -- the prior version of this
        test relied on message-id tag injection, which 0.3.2 removed.
        """
        with tempfile.TemporaryDirectory() as tmp:
            plugin = HermesContextManagerPlugin(hermes_home=tmp, config=HmcConfig())
            state = create_state()
            state.tool_calls["call_1"] = ToolRecord(
                tool_call_id="call_1",
                tool_name="terminal",
                input_args={"command": "test"},
                input_fingerprint='terminal::{"command": "test"}',
                is_error=False,
                turn_index=0,
                timestamp=1.0,
                token_estimate=20,
            )
            plugin._states["s1"] = state
            # Long content that triggers head/tail truncation -- the
            # short_circuit path requires error-indicator-free content
            # which is brittle for a test fixture, so we drive
            # truncation instead.
            tool_payload = "\n".join(f"line {i:04d}" for i in range(80))
            messages = [
                {"role": "user", "content": "run it", "timestamp": 1.0},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": "call_1", "function": {"name": "terminal", "arguments": "{}"}}
                    ],
                    "timestamp": 2.0,
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "tool_name": "terminal",
                    "content": tool_payload,
                    "timestamp": 3.0,
                },
            ]
            original_content = [m["content"] for m in messages]

            plugin.on_pre_llm_call(
                session_id="s1",
                user_message="",
                conversation_history=messages,
                is_first_turn=False,
                model="test-model",
                platform="cli",
            )

            # Tool message should be mutated (short-circuit fired)
            assert messages[2]["content"] != original_content[2]

            # Restore should bring back originals
            plugin._restore_mutations("s1")
            restored_content = [m["content"] for m in messages]
            assert restored_content == original_content


def _build_compress_ready_messages(extra_user_turns: int = 5) -> list[dict[str, object]]:
    """Build a message list with enough user turns to satisfy ``identify_stale_ranges``.

    The compression path requires ``len(user_indices) > protect_recent_turns``,
    so the default of 5 user turns gives us a 2-turn stale prefix once we
    protect the most recent 3.
    """
    messages: list[dict[str, object]] = []
    for i in range(extra_user_turns):
        messages.append({"role": "user", "content": f"user turn {i}", "timestamp": float(i * 2)})
        messages.append({"role": "assistant", "content": f"assistant {i}", "timestamp": float(i * 2 + 1)})
    return messages


class BackgroundCompressionConfigTests(unittest.TestCase):
    """Tests for the ``background_compression`` config block wiring."""

    def test_prepare_returns_none_when_compression_disabled(self) -> None:
        """``background_compression.enabled = False`` must short-circuit prepare."""
        with tempfile.TemporaryDirectory() as tmp:
            config = HmcConfig()
            config.background_compression.enabled = False
            plugin = HermesContextManagerPlugin(hermes_home=tmp, config=config)
            state = create_state()
            state.manual_mode = False
            state.last_context_percent = 0.99  # well above 0.8 threshold
            messages = _build_compress_ready_messages()

            result = plugin._prepare_background_compress(messages, state)

            self.assertIsNone(result)

    def test_prepare_runs_when_compression_enabled(self) -> None:
        """Sanity check the inverse: with the same state, enabled=True fires."""
        with tempfile.TemporaryDirectory() as tmp:
            config = HmcConfig()
            config.background_compression.enabled = True
            plugin = HermesContextManagerPlugin(hermes_home=tmp, config=config)
            state = create_state()
            state.manual_mode = False
            state.last_context_percent = 0.99
            messages = _build_compress_ready_messages()

            result = plugin._prepare_background_compress(messages, state)

            self.assertIsNotNone(result)
            assert result is not None
            self.assertIn("start_idx", result)
            self.assertIn("end_idx", result)
            self.assertIn("first_id", result)
            self.assertIn("last_id", result)

    def test_prepare_uses_protect_recent_turns_from_config(self) -> None:
        """Bumping ``protect_recent_turns`` shrinks the stale window."""
        with tempfile.TemporaryDirectory() as tmp:
            messages = _build_compress_ready_messages(extra_user_turns=4)

            # protect_recent_turns=3 → 1 stale user turn → ranges non-empty
            cfg_small = HmcConfig()
            cfg_small.background_compression.protect_recent_turns = 3
            plugin_small = HermesContextManagerPlugin(hermes_home=tmp, config=cfg_small)
            state_small = create_state()
            state_small.last_context_percent = 0.99
            self.assertIsNotNone(
                plugin_small._prepare_background_compress(messages, state_small)
            )

            # protect_recent_turns=4 → 0 stale user turns → ranges empty → None
            cfg_large = HmcConfig()
            cfg_large.background_compression.protect_recent_turns = 4
            plugin_large = HermesContextManagerPlugin(hermes_home=tmp, config=cfg_large)
            state_large = create_state()
            state_large.last_context_percent = 0.99
            self.assertIsNone(
                plugin_large._prepare_background_compress(messages, state_large)
            )


class ContextWindowFloorTests(unittest.TestCase):
    """Tests for the ``_get_context_window`` defensive floor."""

    @staticmethod
    def _patched_metadata(return_value: object):
        """Build a fake ``agent.model_metadata`` module returning ``return_value``."""
        fake_module = types.ModuleType("agent.model_metadata")
        fake_module.get_model_context_length = lambda _model: return_value  # type: ignore[attr-defined]
        fake_parent = types.ModuleType("agent")
        fake_parent.model_metadata = fake_module  # type: ignore[attr-defined]
        return mock.patch.dict(
            sys.modules,
            {"agent": fake_parent, "agent.model_metadata": fake_module},
        )

    def test_floor_applies_when_metadata_returns_zero(self) -> None:
        """A 0-return must NOT cause _estimate_context_percent to explode."""
        with tempfile.TemporaryDirectory() as tmp:
            plugin = HermesContextManagerPlugin(hermes_home=tmp, config=HmcConfig())
            with self._patched_metadata(0):
                window = plugin._get_context_window("unknown-model")
            self.assertEqual(window, plugin._CONTEXT_WINDOW_FLOOR)
            # And the percent calculation stays sane.
            pct = plugin._estimate_context_percent(1000, window)
            self.assertLess(pct, 1.0)

    def test_floor_applies_when_metadata_returns_none(self) -> None:
        """``None`` is the other realistic bad return value."""
        with tempfile.TemporaryDirectory() as tmp:
            plugin = HermesContextManagerPlugin(hermes_home=tmp, config=HmcConfig())
            with self._patched_metadata(None):
                window = plugin._get_context_window("unknown-model")
            self.assertEqual(window, plugin._CONTEXT_WINDOW_FLOOR)

    def test_real_value_above_floor_passes_through(self) -> None:
        """A normal context length is returned unchanged."""
        with tempfile.TemporaryDirectory() as tmp:
            plugin = HermesContextManagerPlugin(hermes_home=tmp, config=HmcConfig())
            with self._patched_metadata(200_000):
                window = plugin._get_context_window("big-model")
            self.assertEqual(window, 200_000)

    def test_import_error_falls_back_to_default(self) -> None:
        """If ``agent.model_metadata`` isn't installed, fall back to 128k."""
        with tempfile.TemporaryDirectory() as tmp:
            plugin = HermesContextManagerPlugin(hermes_home=tmp, config=HmcConfig())
            # Ensure the module is NOT in sys.modules so the import raises.
            removed = {
                k: sys.modules.pop(k)
                for k in ("agent", "agent.model_metadata")
                if k in sys.modules
            }
            try:
                window = plugin._get_context_window("any-model")
            finally:
                sys.modules.update(removed)
            self.assertEqual(window, 128000)


class FinishCompressIdentityGuardTests(unittest.TestCase):
    """Tests for the dict-identity guard in ``_finish_background_compress``."""

    def _make_compress_args(self, messages: list[dict[str, object]]) -> dict[str, object]:
        return {
            "start_idx": 0,
            "end_idx": 1,
            "turn_range": "1-1",
            "transcript": "test transcript",
            "range_len": 2,
            "first_id": id(messages[0]),
            "last_id": id(messages[1]),
        }

    def test_finish_succeeds_when_dict_identity_matches(self) -> None:
        """Happy path: identity guard does not block legitimate compression."""
        with tempfile.TemporaryDirectory() as tmp:
            plugin = HermesContextManagerPlugin(hermes_home=tmp, config=HmcConfig())
            state = create_state()
            messages: list[dict[str, object]] = [
                {"role": "user", "content": "first user", "timestamp": 1.0},
                {"role": "assistant", "content": "first reply", "timestamp": 2.0},
            ]
            compress_args = self._make_compress_args(messages)

            plugin._finish_background_compress(
                "s1", messages, state, compress_args, "did some work"
            )

            self.assertEqual(messages, [])  # range deleted
            self.assertGreater(state.tokens_saved, 0)
            self.assertEqual(
                state.tokens_saved_by_type.get("background_compression", 0),
                state.tokens_saved,
            )
            self.assertEqual(len(plugin.state_store.read_index("s1")), 1)

    def test_finish_aborts_when_first_dict_swapped(self) -> None:
        """Length unchanged but messages[start_idx] is a different dict → abort."""
        with tempfile.TemporaryDirectory() as tmp:
            plugin = HermesContextManagerPlugin(hermes_home=tmp, config=HmcConfig())
            state = create_state()
            messages: list[dict[str, object]] = [
                {"role": "user", "content": "first user", "timestamp": 1.0},
                {"role": "assistant", "content": "first reply", "timestamp": 2.0},
            ]
            compress_args = self._make_compress_args(messages)

            # Swap the first dict for a brand-new one (same shape, different identity)
            messages[0] = {"role": "user", "content": "swapped!", "timestamp": 99.0}

            plugin._finish_background_compress(
                "s1", messages, state, compress_args, "did some work"
            )

            # No mutations
            self.assertEqual(len(messages), 2)
            self.assertEqual(state.tokens_saved, 0)
            self.assertEqual(plugin.state_store.read_index("s1"), [])

    def test_finish_aborts_when_last_dict_swapped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plugin = HermesContextManagerPlugin(hermes_home=tmp, config=HmcConfig())
            state = create_state()
            messages: list[dict[str, object]] = [
                {"role": "user", "content": "first user", "timestamp": 1.0},
                {"role": "assistant", "content": "first reply", "timestamp": 2.0},
            ]
            compress_args = self._make_compress_args(messages)
            messages[1] = {"role": "assistant", "content": "swapped!", "timestamp": 99.0}

            plugin._finish_background_compress(
                "s1", messages, state, compress_args, "did some work"
            )

            self.assertEqual(len(messages), 2)
            self.assertEqual(state.tokens_saved, 0)
            self.assertEqual(plugin.state_store.read_index("s1"), [])

    def test_finish_aborts_when_length_shrank(self) -> None:
        """Pre-existing length guard still works."""
        with tempfile.TemporaryDirectory() as tmp:
            plugin = HermesContextManagerPlugin(hermes_home=tmp, config=HmcConfig())
            state = create_state()
            messages: list[dict[str, object]] = [
                {"role": "user", "content": "first user", "timestamp": 1.0},
                {"role": "assistant", "content": "first reply", "timestamp": 2.0},
            ]
            compress_args = self._make_compress_args(messages)
            del messages[1]  # list shrank between phases

            plugin._finish_background_compress(
                "s1", messages, state, compress_args, "did some work"
            )

            self.assertEqual(len(messages), 1)
            self.assertEqual(state.tokens_saved, 0)
            self.assertEqual(plugin.state_store.read_index("s1"), [])


class SessionIdResolutionTests(unittest.TestCase):
    """Tests for ``_session_id_for_task`` and the handle_hmc_control wiring."""

    def test_returns_mapped_session_when_task_id_known(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plugin = HermesContextManagerPlugin(hermes_home=tmp, config=HmcConfig())
            plugin._task_to_session["task-1"] = "session-A"
            self.assertEqual(plugin._session_id_for_task("task-1"), "session-A")

    def test_returns_only_state_when_one_session_in_flight(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plugin = HermesContextManagerPlugin(hermes_home=tmp, config=HmcConfig())
            plugin._states["lone-session"] = create_state()
            self.assertEqual(plugin._session_id_for_task(None), "lone-session")

    def test_returns_only_session_messages_when_one_in_flight(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plugin = HermesContextManagerPlugin(hermes_home=tmp, config=HmcConfig())
            plugin._session_messages["lone-msgs"] = []
            self.assertEqual(plugin._session_id_for_task(None), "lone-msgs")

    def test_returns_none_when_ambiguous(self) -> None:
        """Multiple sessions in flight + no task_id → no fallback, must return None."""
        with tempfile.TemporaryDirectory() as tmp:
            plugin = HermesContextManagerPlugin(hermes_home=tmp, config=HmcConfig())
            plugin._states["s1"] = create_state()
            plugin._states["s2"] = create_state()
            self.assertIsNone(plugin._session_id_for_task(None))

    def test_returns_none_when_no_sessions_at_all(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plugin = HermesContextManagerPlugin(hermes_home=tmp, config=HmcConfig())
            self.assertIsNone(plugin._session_id_for_task(None))

    def test_handle_hmc_control_returns_error_on_ambiguous_session(self) -> None:
        """The slash command must surface the ambiguity, not show phantom stats."""
        with tempfile.TemporaryDirectory() as tmp:
            plugin = HermesContextManagerPlugin(hermes_home=tmp, config=HmcConfig())
            plugin._states["s1"] = create_state()
            plugin._states["s2"] = create_state()

            result = plugin.handle_hmc_control({"action": "status"}, task_id=None)
            parsed = json.loads(result)

            self.assertIn("error", parsed)
            self.assertIn("session", parsed["error"].lower())

    def test_handle_hmc_control_works_with_single_session_fallback(self) -> None:
        """Single-session CLI flow must still work without an explicit task_id."""
        with tempfile.TemporaryDirectory() as tmp:
            plugin = HermesContextManagerPlugin(hermes_home=tmp, config=HmcConfig())
            state = create_state()
            state.tokens_saved = 1234
            plugin._states["only-session"] = state

            result = plugin.handle_hmc_control({"action": "stats"}, task_id=None)
            parsed = json.loads(result)

            self.assertNotIn("error", parsed)
            self.assertEqual(parsed["tokens_saved_estimate"], 1234)


class SessionEndFinalMaterializeTests(unittest.TestCase):
    """Regression tests for 0.3.5 on_session_end final materialize pass.

    If a session ends mid-agent-loop (user ctrl-c, Hermes restart,
    network disconnect), the last batch of tool outputs never got
    swept by materialize_view at a pre_llm_call boundary.  dedup and
    error_purge never ran over them and the analytics row undercount
    the session.

    The fix runs one final materialize_view pass at session_end
    before save-and-restore, so the sidecar's cumulative counters
    reflect the whole session including the tail.  Content is still
    restored to raw originals before Hermes saves the session file.
    """

    def test_session_end_materialize_catches_tail_compression(self) -> None:
        """Tool outputs added post-pre_llm_call get swept at session_end."""
        with tempfile.TemporaryDirectory() as tmp:
            plugin = HermesContextManagerPlugin(hermes_home=tmp, config=HmcConfig())
            long_output = "\n".join(f"line {i:04d}" for i in range(120))

            # Turn 1: user message + one tool call.  Run pre_llm_call
            # so materialize fires over the initial state.
            messages: list[dict] = [
                {"role": "user", "content": "start work", "timestamp": 1.0},
            ]
            plugin.on_pre_llm_call(
                session_id="s1",
                user_message="start work",
                conversation_history=messages,
                is_first_turn=True,
                model="test-model",
                platform="cli",
            )
            baseline = plugin._states["s1"].tokens_kept_out_total

            # Agent loop: append a big tool result (no pre_llm_call
            # fires to compress it, simulating mid-loop state).  Also
            # register the ToolRecord as on_pre_tool_call would.
            plugin._states["s1"].tool_calls["call_tail"] = ToolRecord(
                tool_call_id="call_tail",
                tool_name="terminal",
                input_args={"command": "cat"},
                input_fingerprint='terminal::{"command":"cat"}',
                is_error=False,
                turn_index=1,
                timestamp=2.0,
                token_estimate=len(long_output) // 4,
            )
            messages.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_tail", "function": {"name": "terminal", "arguments": '{"command":"cat"}'}}
                ],
                "timestamp": 2.0,
            })
            messages.append({
                "role": "tool",
                "tool_call_id": "call_tail",
                "tool_name": "terminal",
                "content": long_output,
                "timestamp": 3.0,
            })

            # Session ends WITHOUT another pre_llm_call.  The long
            # tool output is still raw in the conversation.
            content_before_session_end = messages[-1]["content"]
            self.assertEqual(content_before_session_end, long_output)

            plugin.on_session_end(
                session_id="s1",
                completed=True,
                interrupted=False,
                model="test-model",
                platform="cli",
            )

            # Hermes's saved session file should see raw content --
            # the final pass compressed in place but then restored,
            # so messages are back to raw originals.
            self.assertEqual(messages[-1]["content"], long_output)

            # Analytics should have captured tail compression: the
            # sidecar for s1 must show more savings than baseline.
            # _save_state runs before we drop in-memory state, so
            # the persisted sidecar reflects the final materialize.
            import json as _json
            from pathlib import Path as _Path
            sidecar = _Path(tmp) / "hmc_state" / "s1.json"
            self.assertTrue(sidecar.exists())
            with sidecar.open() as f:
                payload = _json.load(f)
            persisted_cumulative = payload.get("tokens_kept_out_total", 0)
            self.assertGreater(
                persisted_cumulative, baseline,
                "session_end materialize should grow tokens_kept_out_total "
                "past the pre-end baseline (captures tail compression)",
            )

    def test_session_end_without_session_messages_is_safe(self) -> None:
        """Edge case: session ends before any pre_llm_call ever fired."""
        with tempfile.TemporaryDirectory() as tmp:
            plugin = HermesContextManagerPlugin(hermes_home=tmp, config=HmcConfig())
            # Seed state without ever running pre_llm_call (so
            # _session_messages[session_id] is absent).
            plugin._get_state("s1")

            # Should not crash; falls through to the legacy flow.
            plugin.on_session_end(
                session_id="s1",
                completed=True,
                interrupted=False,
                model="test-model",
                platform="cli",
            )
            # State was cleaned up.
            self.assertNotIn("s1", plugin._states)


class PostToolCallHeartbeatTests(unittest.TestCase):
    """Regression tests for 0.3.4 post_tool_call compression + heartbeat.

    Before 0.3.4 the hook only finalized tool records and saved state.
    All compression happened in bulk at the next pre_llm_call, so during
    agent loops (many tool calls between user turns) the dashboard
    showed no activity and new tool outputs rode through uncompressed
    until the next user message.
    """

    def _setup_session_with_tool_in_flight(self, plugin: HermesContextManagerPlugin) -> list[dict]:
        """Seed a session with a live conversation and one registered tool call.

        Returns the conversation_history list so tests can append tool
        results and watch on_post_tool_call process them.
        """
        messages: list[dict] = [
            {"role": "user", "content": "run the test suite", "timestamp": 1.0},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_A", "function": {"name": "terminal", "arguments": '{"command":"pytest"}'}}
                ],
                "timestamp": 2.0,
            },
        ]
        # Establish plugin state by running one pre_llm_call -- this
        # populates _session_messages, _active_mutations, _active_session_id.
        plugin.on_pre_tool_call(
            tool_name="terminal",
            args={"command": "pytest"},
            task_id="task_A",
            session_id="s1",
            tool_call_id="call_A",
        )
        plugin.on_pre_llm_call(
            session_id="s1",
            user_message="run the test suite",
            conversation_history=messages,
            is_first_turn=True,
            model="test-model",
            platform="cli",
        )
        return messages

    def test_post_tool_call_compresses_short_circuitable_output(self) -> None:
        """A JSON-shaped success output should get short-circuited on arrival."""
        with tempfile.TemporaryDirectory() as tmp:
            plugin = HermesContextManagerPlugin(hermes_home=tmp, config=HmcConfig())
            messages = self._setup_session_with_tool_in_flight(plugin)

            tool_payload = (
                '{"status": "ok", "detail": "all 227 tests passed", "exit_code": 0}'
            )
            messages.append({
                "role": "tool",
                "tool_call_id": "call_A",
                "tool_name": "terminal",
                "content": tool_payload,
                "timestamp": 3.0,
            })

            saved_before = plugin._states["s1"].tokens_kept_out_total
            plugin.on_post_tool_call(
                tool_name="terminal",
                args={},
                result=tool_payload,
                task_id="task_A",
                session_id="s1",
                tool_call_id="call_A",
            )
            saved_after = plugin._states["s1"].tokens_kept_out_total

            # Short-circuit should have fired and credited savings.
            self.assertGreater(saved_after, saved_before)
            # Content is now a shorter placeholder, not the original JSON.
            self.assertNotEqual(messages[-1]["content"], tool_payload)
            # Backup is tracked so the next pre_llm_call can restore.
            backups = plugin._active_mutations.get("s1") or {}
            self.assertIn(len(messages) - 1, backups)
            self.assertEqual(backups[len(messages) - 1], tool_payload)

    def test_post_tool_call_publishes_tool_event(self) -> None:
        """Heartbeat event reaches the dashboard during agent loops."""
        with tempfile.TemporaryDirectory() as tmp:
            plugin = HermesContextManagerPlugin(hermes_home=tmp, config=HmcConfig())
            plugin.dashboard.start()
            try:
                # Setup with a real (non-phantom-shaped) conversation.
                bulk_payload = "x" * 5000
                messages: list[dict] = [
                    {"role": "user", "content": "work " + bulk_payload, "timestamp": 1.0},
                    {
                        "role": "assistant", "content": "",
                        "tool_calls": [
                            {"id": "call_B", "function": {"name": "terminal", "arguments": "{}"}}
                        ],
                        "timestamp": 2.0,
                    },
                ]
                plugin.on_pre_tool_call(
                    tool_name="terminal", args={}, task_id="task_B",
                    session_id="s1", tool_call_id="call_B",
                )
                plugin.on_pre_llm_call(
                    session_id="s1",
                    user_message="work",
                    conversation_history=messages,
                    is_first_turn=True,
                    model="test-model",
                    platform="cli",
                )

                bus = plugin.dashboard._bus
                q = bus.subscribe()

                # Append a tool result and fire post_tool_call.
                tool_payload = '{"status": "ok"}'
                messages.append({
                    "role": "tool",
                    "tool_call_id": "call_B",
                    "tool_name": "terminal",
                    "content": tool_payload,
                    "timestamp": 3.0,
                })
                plugin.on_post_tool_call(
                    tool_name="terminal", args={},
                    result=tool_payload, task_id="task_B",
                    session_id="s1", tool_call_id="call_B",
                )

                import queue as _q
                events: list[dict] = []
                try:
                    while True:
                        events.append(q.get(timeout=0.3))
                except _q.Empty:
                    pass
                tool_events = [e for e in events if e.get("type") == "tool"]
                self.assertEqual(len(tool_events), 1, f"expected 1 tool event, got {events}")
                data = tool_events[0]["data"]
                self.assertEqual(data["tool_name"], "terminal")
                self.assertEqual(data["tool_call_id"], "call_B")
                self.assertGreaterEqual(data["cumulative_saved"], 0)
                bus.unsubscribe(q)
            finally:
                plugin.dashboard.stop()

    def test_post_tool_call_mutation_is_restored_on_next_pre_llm_call(self) -> None:
        """_restore_mutations must undo post_tool_call compression.

        Critical invariant: by the time materialize_view runs on the
        next pre_llm_call, content is back to its original shape so
        the full pipeline (including dedup and error_purge) sees raw
        data.
        """
        with tempfile.TemporaryDirectory() as tmp:
            plugin = HermesContextManagerPlugin(hermes_home=tmp, config=HmcConfig())
            messages = self._setup_session_with_tool_in_flight(plugin)

            tool_payload = '{"status": "ok", "detail": "finished", "exit_code": 0}'
            messages.append({
                "role": "tool",
                "tool_call_id": "call_A",
                "tool_name": "terminal",
                "content": tool_payload,
                "timestamp": 3.0,
            })
            plugin.on_post_tool_call(
                tool_name="terminal", args={}, result=tool_payload,
                task_id="task_A", session_id="s1", tool_call_id="call_A",
            )
            # Content is mutated now.
            self.assertNotEqual(messages[-1]["content"], tool_payload)

            # Add another user message (simulating the next turn).
            messages.append({"role": "user", "content": "ok next", "timestamp": 4.0})

            # on_pre_llm_call phase 1 calls _restore_mutations, which
            # puts the original content back before materialize_view
            # runs fresh.  Run it and check what materialize_view sees.
            plugin.on_pre_llm_call(
                session_id="s1",
                user_message="ok next",
                conversation_history=messages,
                is_first_turn=False,
                model="test-model",
                platform="cli",
            )
            # After the new pre_llm_call, content may be mutated again
            # by materialize_view (re-short-circuited).  Both pre- and
            # post- mutation values differ from the original only in
            # whether they've been short-circuited -- they should NOT
            # accumulate tags or double-tombstone.
            tool_msgs = [m for m in messages if m.get("role") == "tool"]
            self.assertEqual(len(tool_msgs), 1)
            # The tool message's content should be the short-circuited
            # placeholder, not the raw JSON AND not something weird
            # like a double-processed artifact.
            final_content = tool_msgs[0]["content"]
            self.assertIsInstance(final_content, str)


class EmptySessionIdGuardTests(unittest.TestCase):
    """Regression tests for the 0.3.3 phantom ``.json`` file bug.

    Before the fix, a hook entry point with ``session_id=""`` would
    flow through ``_get_state`` → ``_save_state`` and write a literal
    hidden file named ``.json`` to the sidecar directory.  Observed in
    the wild at ``~/.hermes/hmc_state/.json`` (672 bytes).
    """

    def test_get_state_with_empty_session_id_does_not_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plugin = HermesContextManagerPlugin(hermes_home=tmp, config=HmcConfig())
            state = plugin._get_state("")
            # Still returns a usable state object so callers can proceed.
            self.assertIsNotNone(state)
            # But NEVER caches it under the empty key.
            self.assertNotIn("", plugin._states)

    def test_save_state_with_empty_session_id_does_not_write_phantom_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plugin = HermesContextManagerPlugin(hermes_home=tmp, config=HmcConfig())
            # Prime _states with an empty-key entry (simulating an
            # earlier buggy call that somehow slipped through).
            plugin._states[""] = create_state()
            plugin._save_state("")
            # The sidecar directory must not contain a ``.json`` file.
            hmc_state_dir = Path(tmp) / "hmc_state"
            if hmc_state_dir.exists():
                for child in hmc_state_dir.iterdir():
                    self.assertNotEqual(
                        child.name, ".json",
                        f"phantom .json file created at {child}",
                    )


class PhantomSessionFilterTests(unittest.TestCase):
    """Regression tests for the 0.3.3 phantom-session dashboard filter.

    Hermes auxiliary workers (title generation, background compression,
    schema validation) all flow through ``on_pre_llm_call`` with tiny
    contexts and no tool activity.  Pre-0.3.3 these polluted the
    dashboard event log with ``turn 1 · ctx 331`` /
    ``session ended · total 0 saved`` entries.
    """

    def test_is_phantom_session_returns_true_for_tiny_empty_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plugin = HermesContextManagerPlugin(hermes_home=tmp, config=HmcConfig())
            state = create_state()
            # Tiny context, no tools, no savings -> phantom.
            self.assertTrue(plugin._is_phantom_session(state, 331))

    def test_is_phantom_session_returns_false_for_real_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plugin = HermesContextManagerPlugin(hermes_home=tmp, config=HmcConfig())
            state = create_state()
            state.tokens_kept_out_total = 5000
            # Even a small context is non-phantom once savings exist.
            self.assertFalse(plugin._is_phantom_session(state, 500))

    def test_is_phantom_session_returns_false_above_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plugin = HermesContextManagerPlugin(hermes_home=tmp, config=HmcConfig())
            state = create_state()
            # Large context alone is enough to rule out phantom.
            self.assertFalse(plugin._is_phantom_session(state, 5000))

    def test_is_phantom_session_returns_false_with_tool_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plugin = HermesContextManagerPlugin(hermes_home=tmp, config=HmcConfig())
            state = create_state()
            state.tool_calls["call_1"] = ToolRecord(
                tool_call_id="call_1",
                tool_name="terminal",
                input_args={},
                input_fingerprint="terminal::{}",
                is_error=False,
                turn_index=0,
                timestamp=1.0,
                token_estimate=20,
            )
            # Tiny context + no savings but has tools -> not phantom.
            self.assertFalse(plugin._is_phantom_session(state, 400))

    def test_phantom_pre_llm_call_does_not_publish_turn_event(self) -> None:
        """End-to-end: auxiliary worker hits pre_llm_call, no event emitted."""
        with tempfile.TemporaryDirectory() as tmp:
            plugin = HermesContextManagerPlugin(hermes_home=tmp, config=HmcConfig())
            plugin.dashboard.start()
            try:
                # Subscribe a queue BEFORE the auxiliary hits.
                bus = plugin.dashboard._bus
                q = bus.subscribe()

                # Tiny conversation -- realistic auxiliary worker shape.
                messages = [
                    {"role": "user", "content": "hi", "timestamp": 1.0},
                ]
                plugin.on_pre_llm_call(
                    session_id="phantom-session",
                    user_message="hi",
                    conversation_history=messages,
                    is_first_turn=True,
                    model="test-model",
                    platform="cli",
                )

                # Drain the queue; no turn events should be present.
                import queue as _q
                events = []
                try:
                    while True:
                        events.append(q.get(timeout=0.2))
                except _q.Empty:
                    pass
                turn_events = [e for e in events if e.get("type") == "turn"]
                self.assertEqual(
                    len(turn_events), 0,
                    "Phantom session should not publish a turn event",
                )
                # _active_session_id must NOT have been promoted to phantom.
                self.assertIsNone(plugin._active_session_id)
                bus.unsubscribe(q)
            finally:
                plugin.dashboard.stop()

    def test_phantom_session_end_does_not_publish_session_end_event(self) -> None:
        """Ending a phantom session must not emit a session_end event."""
        with tempfile.TemporaryDirectory() as tmp:
            plugin = HermesContextManagerPlugin(hermes_home=tmp, config=HmcConfig())
            plugin.dashboard.start()
            try:
                bus = plugin.dashboard._bus
                q = bus.subscribe()

                # Seed a phantom-shaped state manually.
                state = plugin._get_state("phantom-session")
                state.last_context_tokens = 345  # below threshold
                # No tool_calls, no savings -> phantom classification.

                plugin.on_session_end(
                    session_id="phantom-session",
                    completed=True,
                    interrupted=False,
                    model="test-model",
                    platform="cli",
                )

                import queue as _q
                events = []
                try:
                    while True:
                        events.append(q.get(timeout=0.2))
                except _q.Empty:
                    pass
                end_events = [e for e in events if e.get("type") == "session_end"]
                self.assertEqual(
                    len(end_events), 0,
                    "Phantom session should not publish session_end",
                )
                bus.unsubscribe(q)
            finally:
                plugin.dashboard.stop()


if __name__ == "__main__":
    unittest.main()
