"""Hermes plugin adapter for Hermes Context Manager."""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any

from .analytics import AnalyticsStore
from .config import HmcConfig, load_config
from .dashboard import Dashboard
from .engine import (
    ALWAYS_PROTECTED_SWEEP,
    apply_strategies_to_tool_output,
    estimate_message_tokens,
    materialize_view,
)
from .persistence import JsonStateStore
from .prompts import SYSTEM_CONTEXT
from .state import ToolRecord, create_input_fingerprint, create_state

LOGGER = logging.getLogger(__name__)


class HermesContextManagerPlugin:
    """Plugin implementation for Hermes Context Manager."""

    def __init__(
        self,
        hermes_home: str | Path | None = None,
        config: HmcConfig | None = None,
        plugin_dir: str | Path | None = None,
    ) -> None:
        self.plugin_dir = Path(plugin_dir or Path(__file__).resolve().parent.parent)
        self.hermes_home = Path(hermes_home or Path.home() / ".hermes")
        self.config = config or load_config(self.plugin_dir)
        self.state_store = JsonStateStore(self.hermes_home)
        # Persistent cumulative analytics.  Lives next to the JSON sidecars
        # by default; honors HMC_DB_PATH and config.analytics.db_path.
        analytics_db = self.config.analytics.db_path or str(
            self.hermes_home / "hmc_state" / "analytics.db"
        )
        self.analytics_store = AnalyticsStore(
            db_path=analytics_db,
            retention_days=self.config.analytics.retention_days,
        )
        # Opt-in web dashboard — dormant until hmc_control dashboard
        # action=start is invoked.  The callable indirection lets the
        # HTTP handlers find the plugin without a circular import.
        self.dashboard = Dashboard(plugin_ref=lambda: self)
        self._states: dict[str, Any] = {}
        self._ctx = None
        self._lock = threading.RLock()
        self._task_to_session: dict[str, str] = {}
        self._tool_call_to_session: dict[str, str] = {}
        self._session_messages: dict[str, list[dict[str, Any]]] = {}
        # Per-session content backup: list-index -> original ``message["content"]``
        # captured before materialize_view mutated the shared dict in place.
        self._active_mutations: dict[str, dict[int, Any]] = {}
        # Most recently active session_id, used by the dashboard to
        # present a "current session" view when no task_id context is
        # available (which is always the case for HTTP requests from
        # the browser, not a Hermes tool call).
        self._active_session_id: str | None = None

    def register(self, ctx) -> None:
        """Register tools and hooks with Hermes."""
        self._ctx = ctx
        self._ensure_config_exists()
        self._ensure_skill_installed()
        ctx.register_tool(
            name="hmc_control",
            toolset="hmc_tools",
            description="Control Hermes Context Manager state and slash-command actions.",
            emoji="\U0001f9ed",
            schema={
                "description": "Hermes Context Manager control. Actions: status (compact overview), context (detailed usage), stats (token savings), index (completed work), analytics (cross-session SQLite history), dashboard (opt-in local web UI over SSE; sub_action=start|stop|status), sweep (prune tool outputs), manual_set (toggle auto mode). The analytics action accepts scope=global|project, period=all|day|month|recent|project, and limit.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string"},
                        "count": {"type": "integer"},
                        "enabled": {"type": "boolean"},
                        "scope": {"type": "string"},
                        "period": {"type": "string"},
                        "limit": {"type": "integer"},
                        "sub_action": {"type": "string"},
                    },
                    "required": ["action"],
                },
            },
            handler=self.handle_hmc_control,
        )
        ctx.register_hook("pre_tool_call", self.on_pre_tool_call)
        ctx.register_hook("post_tool_call", self.on_post_tool_call)
        ctx.register_hook("pre_llm_call", self.on_pre_llm_call)
        ctx.register_hook("on_session_end", self.on_session_end)

    def _ensure_skill_installed(self) -> None:
        source = self.plugin_dir / "hmc_skill" / "SKILL.md"
        if not source.exists():
            return
        target = self.hermes_home / "skills" / "hmc" / "SKILL.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_text(encoding="utf-8") == source.read_text(encoding="utf-8"):
            return
        shutil.copy2(source, target)

    def _ensure_config_exists(self) -> None:
        source = self.plugin_dir / "config.yaml.example"
        target = self.plugin_dir / "config.yaml"
        if target.exists() or not source.exists():
            return
        shutil.copy2(source, target)

    # Heuristic threshold: contexts smaller than this (in estimated tokens)
    # are almost certainly Hermes auxiliary workers (title generation,
    # background compression, schema validation, etc.) rather than a real
    # user conversation.  Pre-0.3.3 these polluted the dashboard event log
    # with phantom "turn 1 · ctx 331" / "session ended · total 0 saved"
    # entries.  1000 tokens is comfortably larger than any auxiliary payload
    # we've observed (which cluster in the 300-500 range) but well below
    # any real conversation's first turn, which even with an empty user
    # message ships thousands of tokens of system prompt + tool specs.
    _PHANTOM_CONTEXT_THRESHOLD = 1000

    def _is_phantom_session(self, state: Any, context_tokens: int | None) -> bool:
        """Return True if this session looks like a Hermes auxiliary call.

        Three signals must all hold:

        1. Tiny context (< _PHANTOM_CONTEXT_THRESHOLD tokens)
        2. No tool calls registered in this session
        3. No savings accumulated

        Any one of these firing on its own is ambiguous.  All three together
        is a near-lock that we're looking at an auxiliary worker the plugin
        has no business showing in the user-facing dashboard.
        """
        if state is None:
            return True
        if (context_tokens or 0) >= self._PHANTOM_CONTEXT_THRESHOLD:
            return False
        if len(state.tool_calls) > 0:
            return False
        if state.tokens_kept_out_total > 0:
            return False
        return True

    def _get_state(self, session_id: str):
        # Guard against empty/missing session_id.  Before 0.3.3 a hook
        # entry point with session_id="" would flow through here, create
        # a state, and eventually land in _save_state which wrote a
        # phantom ".json" file to the sidecar directory (literally a
        # hidden file named .json).  Return a throwaway state for the
        # rare null case so callers can still run materialize_view
        # without crashing, but never cache or persist it.
        with self._lock:
            if not session_id:
                return create_state()
            state = self._states.get(session_id)
            if state is None:
                loaded_state = self.state_store.load(session_id)
                state = loaded_state or create_state()
                if loaded_state is None:
                    state.manual_mode = self.config.manual_mode.enabled
                # Capture canonical cwd as the project key for analytics.
                # Done here (lazily, on first state creation) so the value
                # reflects the directory the session actually started in.
                if not state.project_path:
                    try:
                        state.project_path = os.path.realpath(os.getcwd())
                    except OSError:
                        state.project_path = ""
                self._states[session_id] = state
            return state

    def _save_state(self, session_id: str) -> None:
        with self._lock:
            if not session_id:
                # Same rationale as _get_state: a falsy session_id is
                # either an auxiliary caller with no identity or an
                # upstream bug.  Either way, don't persist a ".json"
                # phantom to the sidecar directory.
                return
            state = self._states.get(session_id)
            if state is not None:
                self.state_store.save(session_id, state)

    def on_pre_tool_call(
        self,
        tool_name: str,
        args: dict[str, Any],
        task_id: str,
        session_id: str,
        tool_call_id: str,
        **_: Any,
    ) -> None:
        """Track tool-call inputs and map task IDs back to sessions.

        Everything runs under a single ``with self._lock:`` block so the
        "check tool_call_id missing → insert ToolRecord" path is atomic.
        ``self._lock`` is a reentrant ``RLock``, so nested ``_get_state`` and
        ``_save_state`` calls are safe.
        """
        try:
            with self._lock:
                if task_id:
                    self._task_to_session[task_id] = session_id
                if tool_call_id:
                    self._tool_call_to_session[tool_call_id] = session_id

                state = self._get_state(session_id)
                if tool_call_id and tool_call_id not in state.tool_calls:
                    state.tool_calls[tool_call_id] = ToolRecord(
                        tool_call_id=tool_call_id,
                        tool_name=tool_name,
                        input_args=args,
                        input_fingerprint=create_input_fingerprint(tool_name, args),
                        is_error=False,
                        turn_index=state.current_turn,
                        timestamp=0.0,
                        token_estimate=0,
                    )
                    self._save_state(session_id)
        except Exception:
            LOGGER.exception("HMC pre_tool_call failed")

    def on_post_tool_call(
        self,
        tool_name: str,
        args: dict[str, Any],
        result: str,
        task_id: str,
        session_id: str,
        tool_call_id: str,
        **_: Any,
    ) -> None:
        """Finalize tool-call metadata, compress the result, publish heartbeat.

        Before 0.3.4 this hook only finalized the ``ToolRecord`` and saved
        state.  All compression happened in bulk at the next
        ``pre_llm_call`` via ``materialize_view``.  That's fine when user
        turns are frequent, but agent loops can go 10+ tool calls between
        user messages, during which the dashboard was silent (no events
        to publish) and new tool outputs rode through the conversation
        uncompressed until the next user turn.

        Now the hook runs the three single-message strategies
        (``short_circuit``, ``code_filter``, ``truncation``) on the
        just-arrived tool output and publishes a ``tool`` SSE event to
        the dashboard so browsers see continuous activity during loops.
        The mutation is tracked in ``_active_mutations`` so the next
        ``pre_llm_call``'s ``_restore_mutations`` can undo it before
        ``materialize_view`` runs fresh.

        Dedup and error_purge still run in ``materialize_view`` because
        they need the full message list and turn counter -- those fire
        once per user turn as before.
        """
        try:
            del args, task_id
            with self._lock:
                state = self._get_state(session_id)

                # --- Phase 1: finalize the ToolRecord (existing behavior) ---
                record = state.tool_calls.get(tool_call_id)
                is_error = False
                output_text = result
                try:
                    parsed = json.loads(result)
                    if isinstance(parsed, dict):
                        is_error = bool(parsed.get("error"))
                        output_text = json.dumps(parsed)
                except Exception:
                    is_error = False
                if record is None:
                    record = ToolRecord(
                        tool_call_id=tool_call_id,
                        tool_name=tool_name,
                        input_args={},
                        input_fingerprint=create_input_fingerprint(tool_name, {}),
                        is_error=is_error,
                        turn_index=state.current_turn,
                        timestamp=time.time(),
                        token_estimate=round(len(output_text) / 4),
                    )
                    state.tool_calls[tool_call_id] = record
                else:
                    record.is_error = is_error
                    record.timestamp = time.time()
                    record.token_estimate = round(len(output_text) / 4)

                # --- Phase 2: compress the new tool output in place ---
                # Requires we have a reference to the live conversation
                # list -- captured in _session_messages during the last
                # pre_llm_call.  If pre_llm_call hasn't run yet for this
                # session (e.g. tool fired before first user turn),
                # skip compression and publish; the next pre_llm_call
                # will catch everything in bulk.
                target_messages = self._session_messages.get(session_id)
                tool_msg = None
                tool_msg_index = -1
                if target_messages is not None:
                    # Scan from the end -- tool results are appended, so
                    # the one we just finalized is almost certainly the
                    # last message.
                    for i in range(len(target_messages) - 1, -1, -1):
                        m = target_messages[i]
                        if (
                            m.get("role") == "tool"
                            and str(m.get("tool_call_id", "")) == tool_call_id
                        ):
                            tool_msg = m
                            tool_msg_index = i
                            break

                delta_saved = 0
                if (
                    tool_msg is not None
                    and tool_msg_index >= 0
                    and target_messages is not None
                ):
                    # Capture original content BEFORE mutation so
                    # _restore_mutations on next pre_llm_call can put
                    # it back.  Only capture if we haven't already --
                    # this tool message may have been included in the
                    # prior pre_llm_call's content_backup.
                    backups = self._active_mutations.setdefault(session_id, {})
                    if tool_msg_index not in backups:
                        backups[tool_msg_index] = tool_msg.get("content")

                    delta_saved = apply_strategies_to_tool_output(
                        tool_msg, state, self.config,
                    )

                    # Refresh context stats using the new materialized
                    # size.  last_context_window is whatever the last
                    # pre_llm_call recorded; use it if present,
                    # otherwise leave the percent alone.
                    context_tokens = sum(
                        estimate_message_tokens(m) for m in target_messages
                    )
                    state.last_context_tokens = context_tokens
                    if state.last_context_window:
                        state.last_context_percent = (
                            context_tokens / max(state.last_context_window, 1)
                        )

                self._save_state(session_id)

                # --- Phase 3: publish heartbeat event to the dashboard ---
                # Skip for phantom auxiliary sessions, same as the
                # pre_llm_call path.  Real agent-loop tool calls have
                # plenty of context and tool activity by the time they
                # get here, so they'll publish normally.
                is_phantom = self._is_phantom_session(
                    state, state.last_context_tokens,
                )
                if not is_phantom:
                    self.dashboard.publish("tool", {
                        "turn": state.current_turn,
                        "timestamp": time.time(),
                        "tool_name": tool_name,
                        "tool_call_id": tool_call_id,
                        "is_error": is_error,
                        "context_tokens": state.last_context_tokens,
                        "context_percent": state.last_context_percent,
                        "cumulative_saved": state.tokens_kept_out_total,
                        "delta_saved": delta_saved,
                        "by_strategy": dict(state.tokens_kept_out_by_type),
                        "uniq_saved": state.tokens_saved,
                    })
        except Exception:
            LOGGER.exception("HMC post_tool_call failed")

    def _restore_mutations(self, session_id: str) -> None:
        """Write backup content values back onto the shared message dicts.

        ``materialize_view`` mutates ``message["content"]`` on the same dict
        objects that live in Hermes's conversation list (shallow copies share
        the dict references).  To restore, we look up the list we were handed
        in ``_session_messages`` and write the original content value back
        onto each index.  If the list has since shrunk (e.g. a background
        compression trimmed it), out-of-range indices are silently skipped.
        """
        with self._lock:
            backup = self._active_mutations.pop(session_id, None)
            if not backup:
                return
            target_messages = self._session_messages.get(session_id)
            if target_messages is None:
                return
            for idx, original_content in backup.items():
                if 0 <= idx < len(target_messages):
                    target_messages[idx]["content"] = original_content

    # Floor for context-window lookups.  If model_metadata returns 0 (or any
    # absurdly small value) for an unknown model, ``_estimate_context_percent``
    # would report thousands-of-percent usage and trigger compression on every
    # turn.  4096 is the smallest viable modern model context.
    _CONTEXT_WINDOW_FLOOR = 4096

    def _get_context_window(self, model: str) -> int:
        try:
            from agent.model_metadata import get_model_context_length

            window = get_model_context_length(model)
        except Exception:
            return 128000
        return max(int(window or 0), self._CONTEXT_WINDOW_FLOOR)

    def _estimate_context_percent(self, total_tokens: int, context_window: int) -> float:
        return total_tokens / max(context_window, 1)

    def _apply_materialized_view(
        self,
        session_id: str,
        content_backup: dict[int, Any],
    ) -> None:
        """Record the content backup produced by ``materialize_view``.

        The actual mutations were already written onto the shared message
        dicts by ``materialize_view``; we only need to remember the backup
        so ``_restore_mutations`` can undo them at end-of-turn.
        """
        with self._lock:
            self._active_mutations[session_id] = content_backup

    def _build_context(self, session_id):
        entries = self.state_store.read_index(session_id)
        if not entries:
            return SYSTEM_CONTEXT
        lines = [SYSTEM_CONTEXT, "", f"{len(entries)} completed phase(s) indexed:"]
        for entry in entries:
            lines.append(f"  - [{entry.get('turn_range', '?')}] {entry.get('topic', 'untitled')}: {entry.get('summary', '')}")
        return "\n".join(lines)

    def _prepare_background_compress(
        self,
        messages: list[dict[str, Any]],
        state: Any,
    ) -> dict[str, Any] | None:
        """Check whether background compression should run and gather its inputs.

        Runs under the session lock.  Returns a dict containing everything
        the unlocked LLM call needs — ``start_idx``, ``end_idx``, ``turn_range``
        and a compact ``transcript`` — or ``None`` to skip compression.

        We build the transcript here (rather than passing the raw message
        range) so the unlocked phase can't see a mutating list.
        """
        if not self.config.background_compression.enabled:
            return None
        if state.manual_mode:
            return None
        if state.last_context_percent is None:
            return None
        if state.last_context_percent < self.config.compress.max_context_percent:
            return None

        from .background_compressor import identify_stale_ranges

        ranges = identify_stale_ranges(
            messages,
            state,
            protect_recent_turns=self.config.background_compression.protect_recent_turns,
        )
        if not ranges:
            return None

        r = ranges[0]
        range_messages = messages[r["start_idx"]:r["end_idx"] + 1]

        parts = []
        for msg in range_messages[:20]:  # cap LLM input
            role = msg.get("role", "?")
            content = str(msg.get("content", ""))[:300]
            parts.append(f"[{role}] {content}")
        transcript = "\n".join(parts)

        return {
            "start_idx": r["start_idx"],
            "end_idx": r["end_idx"],
            "turn_range": r["turn_range"],
            "transcript": transcript,
            "range_len": len(range_messages),
            # Dict-identity fingerprints of the first and last messages in the
            # range.  Used in ``_finish_background_compress`` to detect the
            # (theoretical) case where a concurrent hook swapped messages at
            # those indices between the locked phases.  Length alone isn't
            # enough — same-length swaps would slip through.
            "first_id": id(messages[r["start_idx"]]),
            "last_id": id(messages[r["end_idx"]]),
        }

    def _run_compress_llm(self, transcript: str) -> str:
        """Call the auxiliary model to summarize a stale range.

        **Must not hold the session lock** — this makes an external network
        call that can take seconds.  Returns an empty string on any failure.
        """
        try:
            from agent.auxiliary_client import call_llm  # type: ignore[import-not-found]

            prompt = (
                "Summarize this completed conversation segment in 1-2 sentences. "
                "Focus on what was accomplished:\n\n" + transcript
            )
            response = call_llm(
                task="compression",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return ""

    def _finish_background_compress(
        self,
        session_id: str,
        messages: list[dict[str, Any]],
        state: Any,
        compress_args: dict[str, Any],
        summary: str,
    ) -> None:
        """Persist the compressed range and remove it from ``messages``.

        Runs under the session lock.  The index entry is built **before**
        ``del messages[...]`` so ``build_index_entry`` sees the messages it
        is describing.  If the conversation list has shrunk since the prepare
        phase (unlikely but possible under concurrent hooks), we skip the
        mutation to avoid corrupting the list.
        """
        from .background_compressor import build_index_entry

        start_idx = compress_args["start_idx"]
        end_idx = compress_args["end_idx"]
        turn_range = compress_args["turn_range"]
        expected_len = compress_args["range_len"]
        expected_first_id = compress_args["first_id"]
        expected_last_id = compress_args["last_id"]

        if end_idx >= len(messages):
            return
        current_range_len = end_idx - start_idx + 1
        if current_range_len != expected_len:
            return
        # Dict identity guard: catch the case where a concurrent mutation kept
        # the list length identical but swapped messages at our target indices
        # between phase 1 (locked) and phase 3 (locked).  In Hermes today
        # hooks within a single request are serialized so this is theoretical,
        # but the cost is two ``id()`` calls per compression and the upside
        # is bulletproof correctness if that invariant ever changes.
        if (
            id(messages[start_idx]) != expected_first_id
            or id(messages[end_idx]) != expected_last_id
        ):
            LOGGER.warning(
                "Background compression: dict identity changed between phases "
                "(turns %s); skipping",
                turn_range,
            )
            return

        entry = build_index_entry(messages, start_idx, end_idx, turn_range, summary)
        self.state_store.append_index(session_id, entry)

        range_messages = messages[start_idx:end_idx + 1]
        removed_tokens = sum(estimate_message_tokens(m) for m in range_messages)
        # Background compression deletes messages from the list entirely,
        # so there's no (tool_call_id, strategy) to gate on -- a compressed
        # range can never double-count the way an in-place strategy can.
        # Credit BOTH accumulators directly.
        state.tokens_saved += removed_tokens
        state.tokens_saved_by_type["background_compression"] = (
            state.tokens_saved_by_type.get("background_compression", 0) + removed_tokens
        )
        state.tokens_kept_out_total += removed_tokens
        state.tokens_kept_out_by_type["background_compression"] = (
            state.tokens_kept_out_by_type.get("background_compression", 0)
            + removed_tokens
        )

        del messages[start_idx:end_idx + 1]

        LOGGER.info(
            "Background compression: removed turns %s (%d msgs, ~%d tokens)",
            turn_range, len(range_messages), removed_tokens,
        )

    def on_pre_llm_call(
        self,
        session_id: str,
        user_message: str,
        conversation_history: list[dict[str, Any]],
        is_first_turn: bool,
        model: str,
        platform: str,
        **_: Any,
    ) -> dict[str, str] | None:
        """Mutate the in-memory view for this API call.

        Three-phase structure so the auxiliary LLM call for background
        compression runs **without** the session lock held:

        1. **Locked**: restore previous turn's mutations, grab session state,
           decide whether to compress, capture a transcript snapshot.
        2. **Unlocked**: auxiliary LLM call to summarize the stale range
           (can take multiple seconds — must not block other hooks).
        3. **Locked**: commit compression to the list, run ``materialize_view``
           (which mutates ``content`` in place on shared dicts), record the
           content backup, and persist state.
        """
        try:
            del user_message, is_first_turn, platform

            # ---- Phase 1: locked — prepare ----
            with self._lock:
                self._restore_mutations(session_id)
                state = self._get_state(session_id)
                self._session_messages[session_id] = conversation_history
                # Background compress uses the PREVIOUS turn's context stats.
                compress_args = self._prepare_background_compress(conversation_history, state)

            # ---- Phase 2: unlocked — external LLM call ----
            summary = ""
            if compress_args is not None:
                summary = self._run_compress_llm(compress_args["transcript"])

            # ---- Phase 3: locked — commit and materialize ----
            with self._lock:
                if compress_args is not None:
                    self._finish_background_compress(
                        session_id, conversation_history, state, compress_args, summary,
                    )

                # Reset per-turn savings gate so this turn's strategies can
                # each count once even though the set accumulates across turns.
                # (We intentionally do NOT clear it; we want the gate to
                # persist so the same tool_call isn't double-counted on a
                # later turn.)
                materialized = materialize_view(conversation_history, state, self.config)
                context_window = self._get_context_window(model)
                context_percent = self._estimate_context_percent(materialized.total_tokens, context_window)
                state.last_context_tokens = materialized.total_tokens
                state.last_context_window = context_window
                state.last_context_percent = context_percent
                self._apply_materialized_view(session_id, materialized.content_backup)

                # Observability: append per-turn snapshot to the ring
                # buffer AND publish a live event to the dashboard (if
                # it's running).  The delta uses the UN-GATED accumulator
                # so every real per-turn compression shows up, not just
                # the first firing against any given tool call.  The
                # gated counter rides along as ``uniq_saved`` so the
                # dashboard can show both numbers when it wants to.
                prev_cumulative = (
                    state.turn_history[-1]["cumulative_saved"]
                    if state.turn_history else 0
                )
                delta_saved = max(
                    0, state.tokens_kept_out_total - prev_cumulative
                )
                snapshot = {
                    "turn": state.current_turn,
                    "timestamp": time.time(),
                    "context_tokens": materialized.total_tokens,
                    "context_percent": context_percent,
                    "cumulative_saved": state.tokens_kept_out_total,
                    "delta_saved": delta_saved,
                    "by_strategy": dict(state.tokens_kept_out_by_type),
                    # Diagnostic counters: how many unique tool-call/strategy
                    # pairs fired, as opposed to raw bytes saved.
                    "uniq_saved": state.tokens_saved,
                    "uniq_by_strategy": dict(state.tokens_saved_by_type),
                }

                # Phantom auxiliary sessions (title generators, background
                # compressors, etc.) still flow through this hook with
                # tiny contexts and zero tool activity.  Don't pollute
                # the dashboard ring buffer, don't steal _active_session_id
                # from the real session, don't publish -- but still run
                # materialize_view above (harmless for empty state) and
                # persist state (the next turn may promote this session
                # to real once it grows).
                is_phantom = self._is_phantom_session(
                    state, materialized.total_tokens,
                )
                if not is_phantom:
                    state.turn_history.append(snapshot)
                    self._active_session_id = session_id

                self._save_state(session_id)

                # Non-blocking publish to dashboard SSE subscribers.
                # Dormant if no dashboard running -- returns immediately.
                # Skipped entirely for phantom sessions.
                if not is_phantom:
                    self.dashboard.publish("turn", snapshot)

                return {"context": self._build_context(session_id)}
        except Exception:
            LOGGER.exception("HMC pre_llm_call failed")
            return None

    def on_session_end(
        self,
        session_id: str,
        completed: bool,
        interrupted: bool,
        model: str,
        platform: str,
        **_: Any,
    ) -> None:
        """Persist state and restore any ephemeral mutations.

        Runs one FINAL materialize_view pass on the live conversation
        before restore-and-save.  This catches the edge case where a
        session ends mid-agent-loop (ctrl-c, timeout, Hermes restart)
        and the final batch of tool outputs never got a pre_llm_call
        sweep -- so dedup and error_purge never ran over them.  Without
        this pass, the analytics row for such sessions would undercount
        by whatever the tail batch contained.

        Order:
        1. _restore_mutations -- put content back to raw originals
           (the state as-of the last post_tool_call or pre_llm_call,
           with all our in-place mutations undone).
        2. materialize_view -- fresh full pipeline run over the raw
           conversation, mutates in place, captures content_backup.
        3. Register the new backup to _active_mutations.
        4. _restore_mutations AGAIN -- put originals back so Hermes's
           session file save sees raw content (our invariant: HMC
           mutations are ephemeral, session files on disk hold raw).
        5. _save_state -- persists the final sidecar with the
           post-pass cumulative counters.
        6. Analytics + dashboard publish use the fresh counters.

        If ``_session_messages[session_id]`` is missing (e.g. session
        ended before any pre_llm_call ever fired), the final pass is
        skipped and we fall through to the legacy flow.
        """
        try:
            del completed, interrupted, model, platform
            with self._lock:
                self._restore_mutations(session_id)

                # Final pre-save compression pass.  Operates on the
                # raw (just-restored) conversation so any tool outputs
                # that arrived after the last pre_llm_call get one
                # more chance to be dedup'd / error-purged / compressed.
                # This is where sessions that end mid-agent-loop
                # capture the tail of their savings.
                final_messages = self._session_messages.get(session_id)
                final_state = self._states.get(session_id)
                if final_messages is not None and final_state is not None:
                    try:
                        final_view = materialize_view(
                            final_messages, final_state, self.config,
                        )
                        final_state.last_context_tokens = final_view.total_tokens
                        if final_state.last_context_window:
                            final_state.last_context_percent = (
                                final_view.total_tokens
                                / max(final_state.last_context_window, 1)
                            )
                        # Stash the new backup so the follow-up restore
                        # undoes THIS pass's mutations (putting content
                        # back to raw originals for Hermes's save).
                        self._active_mutations[session_id] = final_view.content_backup
                        self._restore_mutations(session_id)
                    except Exception:
                        LOGGER.exception(
                            "HMC session_end final materialize failed",
                        )

                self._save_state(session_id)
                # Persist this session's per-strategy savings to the
                # analytics SQLite store BEFORE we drop the in-memory state.
                # Failures are swallowed inside ``record_session`` itself --
                # analytics is observability and must never crash the plugin.
                state = self._states.get(session_id)
                if self.config.analytics.enabled:
                    # Record the un-gated totals to the SQLite store --
                    # they're the true bytes-saved number we want to
                    # track across sessions.  The gated counter stays
                    # in-memory only; it's a diagnostic, not history.
                    if (
                        state is not None
                        and state.tokens_kept_out_total > 0
                    ):
                        self.analytics_store.record_session(
                            session_id=session_id,
                            project_path=state.project_path,
                            tokens_saved_by_type=dict(
                                state.tokens_kept_out_by_type
                            ),
                            last_context_tokens=state.last_context_tokens,
                        )
                # Publish session-end event to the dashboard (if running)
                # BEFORE we drop the state -- the event carries the
                # final totals so connected browsers see the session
                # close out cleanly.  Skip the publish for phantom
                # auxiliary sessions (same heuristic as the turn publish
                # path) so the dashboard event log isn't polluted with
                # "session ended · total 0 saved" entries from Hermes
                # internal worker tasks.
                if state is not None and not self._is_phantom_session(
                    state, state.last_context_tokens,
                ):
                    self.dashboard.publish("session_end", {
                        "session_id": session_id,
                        # Primary metric.
                        "total_saved": state.tokens_kept_out_total,
                        "by_strategy": dict(state.tokens_kept_out_by_type),
                        # Diagnostic (gated).
                        "uniq_saved": state.tokens_saved,
                        "uniq_by_strategy": dict(state.tokens_saved_by_type),
                        "turns": state.current_turn,
                    })
                # Clean up session-scoped data to prevent memory leaks
                self._states.pop(session_id, None)
                self._session_messages.pop(session_id, None)
                self._active_mutations.pop(session_id, None)
                if self._active_session_id == session_id:
                    self._active_session_id = None
                # Clean up task/tool_call mappings for this session
                stale_tasks = [k for k, v in self._task_to_session.items() if v == session_id]
                for k in stale_tasks:
                    del self._task_to_session[k]
                stale_tools = [k for k, v in self._tool_call_to_session.items() if v == session_id]
                for k in stale_tools:
                    del self._tool_call_to_session[k]
        except Exception:
            LOGGER.exception("HMC on_session_end failed")

    def _current_dashboard_state(self) -> dict[str, Any] | None:
        """Return a snapshot of the currently-active session for the dashboard.

        The dashboard's HTTP handlers call this from a background
        thread; we take the plugin lock for consistency with state
        mutations that may run concurrently in the hook path.  Returns
        None if no session has run any turns yet.
        """
        with self._lock:
            session_id = self._active_session_id
            if session_id is None:
                # Fall back to the single-session case
                if len(self._states) == 1:
                    session_id = next(iter(self._states))
                else:
                    return None
            state = self._states.get(session_id)
            if state is None:
                return None
            return {
                "session_id": session_id,
                # Primary (un-gated) accumulator -- the one that reflects
                # real API byte savings across the whole session.
                "tokens_kept_out_total": state.tokens_kept_out_total,
                "tokens_kept_out_by_type": dict(state.tokens_kept_out_by_type),
                # Diagnostic (gated) counters.
                "tokens_saved": state.tokens_saved,
                "tokens_saved_by_type": dict(state.tokens_saved_by_type),
                "last_context_tokens": state.last_context_tokens,
                "last_context_percent": state.last_context_percent,
                "current_turn": state.current_turn,
                "project_path": state.project_path,
            }

    def _current_dashboard_history(self) -> list[dict[str, Any]]:
        """Return the turn-history ring buffer for the active session."""
        with self._lock:
            session_id = self._active_session_id
            if session_id is None:
                if len(self._states) == 1:
                    session_id = next(iter(self._states))
                else:
                    return []
            state = self._states.get(session_id)
            if state is None:
                return []
            return list(state.turn_history)

    def _recent_sessions_for_dashboard(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return a display-ready list of recent sessions for the dashboard.

        Enriches the raw ``JsonStateStore.list_sessions`` output with
        a ``live`` flag -- True if the session is still held in
        ``self._states`` (active or pending cleanup), False for
        historical-only sidecars.  The dashboard uses this to highlight
        the current session row in the "Recent sessions" panel.

        Takes ``self._lock`` briefly to read the state membership; the
        actual disk scan happens without the lock held so a long
        sidecar directory doesn't block the hook path.
        """
        raw = self.state_store.list_sessions(limit=limit)
        with self._lock:
            live_ids = set(self._states.keys())
        for entry in raw:
            entry["live"] = entry["session_id"] in live_ids
            entry["active"] = entry["session_id"] == self._active_session_id
        return raw

    def _session_id_for_task(self, task_id: str | None) -> str | None:
        """Map a tool ``task_id`` back to its session, with single-session fallback.

        Returns ``None`` when the mapping is ambiguous.  Callers MUST handle
        ``None`` — earlier versions returned the literal string ``"default"``,
        which silently created a phantom empty session and showed bogus stats.
        """
        if task_id and task_id in self._task_to_session:
            return self._task_to_session[task_id]
        # Single-session fallback for CLI / single-user deployments where
        # task_id may not propagate to slash-command handlers.
        if len(self._states) == 1:
            return next(iter(self._states))
        if len(self._session_messages) == 1:
            return next(iter(self._session_messages))
        return None

    def _record_tool_name(self, record: ToolRecord) -> str:
        return record.tool_name

    def handle_hmc_control(self, args: dict[str, Any], task_id: str | None = None, **_: Any) -> str:
        """Handle slash-command style control operations."""
        action = str(args.get("action", "")).strip().lower()

        # Plugin-scoped actions that don't need a session context are
        # routed BEFORE taking ``self._lock``.  They operate on their
        # own internal state (``self.dashboard`` has its own lock) and
        # must NOT hold the plugin lock -- especially the ``dashboard
        # stop`` path, which calls ``server.shutdown()`` that can
        # block for up to the ``serve_forever`` poll interval (~500ms).
        # Holding ``self._lock`` across that call would stall every
        # concurrent Hermes hook for the same duration.
        if action == "dashboard":
            sub = str(args.get("sub_action", "status")).strip().lower()
            if sub == "start":
                return json.dumps({
                    "action": "dashboard",
                    "sub": "start",
                    **self.dashboard.start(),
                })
            if sub == "stop":
                return json.dumps({
                    "action": "dashboard",
                    "sub": "stop",
                    **self.dashboard.stop(),
                })
            return json.dumps({
                "action": "dashboard",
                "sub": "status",
                **self.dashboard.status(),
            })

        with self._lock:
            session_id = self._session_id_for_task(task_id)
            if session_id is None:
                # No task_id mapping AND no single-session fallback hit.  We
                # used to silently use a "default" session here, which showed
                # zeroed-out stats and confused users.  Surface the ambiguity
                # so the caller knows the request couldn't be routed.
                return json.dumps(
                    {
                        "error": (
                            "Could not determine session for hmc_control "
                            "(no task_id mapping and multiple sessions in flight)."
                        )
                    }
                )
            state = self._get_state(session_id)

            if action == "status":
                # Lead with the un-gated accumulator -- that's the real
                # "bytes kept out of the API call" number.  The gated
                # counter rides along as a diagnostic ("uniq") for users
                # who want to see how many distinct tool calls each
                # strategy touched.
                kept_out = state.tokens_kept_out_total
                kept_out_by_type = state.tokens_kept_out_by_type
                uniq = state.tokens_saved
                tool_count = len(state.tool_calls)
                prune_count = state.total_prune_count
                manual = state.manual_mode

                def _fmt(n: int) -> str:
                    return f"{n / 1000:.1f}K" if n >= 1000 else str(n)

                # Build compact breakdown off the un-gated totals since
                # they're the primary metric.
                type_parts = []
                for typ, count in sorted(
                    kept_out_by_type.items(), key=lambda x: -x[1]
                ):
                    type_parts.append(f"{typ}: {_fmt(count)}")
                breakdown = ", ".join(type_parts) if type_parts else "none yet"

                mode_str = "manual" if manual else "auto"

                # Context info
                ctx_pct = f"{state.last_context_percent * 100:.0f}%" if state.last_context_percent else "--"

                # Index count
                entries = self.state_store.read_index(session_id)

                # Only show the diagnostic counter when it differs from
                # the primary -- keeps the status line quiet on short
                # sessions where the two numbers are identical.
                saved_str = _fmt(kept_out)
                if uniq != kept_out:
                    saved_str = f"{_fmt(kept_out)} kept-out / {_fmt(uniq)} uniq"

                lines = [
                    f"HMC [{mode_str}] | ctx: {ctx_pct} | saved: {saved_str} tokens | tools: {tool_count} tracked, {prune_count} pruned",
                    f"  breakdown: {breakdown}",
                ]
                if entries:
                    lines.append(f"  index: {len(entries)} completed phase(s)")

                return json.dumps({"status": "\n".join(lines)})

            if action == "context":
                return json.dumps(
                    {
                        "context_tokens_estimate": state.last_context_tokens,
                        "context_window": state.last_context_window,
                        "context_usage_percent": state.last_context_percent,
                        "tool_calls_tracked": len(state.tool_calls),
                        "pruned_tools": len(state.pruned_tool_ids),
                        # Primary metric: un-gated real bytes kept out.
                        "tokens_kept_out_estimate": state.tokens_kept_out_total,
                        # Diagnostic: unique firings only.
                        "tokens_saved_estimate": state.tokens_saved,
                        "manual_mode": state.manual_mode,
                    }
                )

            if action == "stats":
                return json.dumps(
                    {
                        # Primary metric (un-gated).
                        "tokens_kept_out_estimate": state.tokens_kept_out_total,
                        "tokens_kept_out_by_type": dict(state.tokens_kept_out_by_type),
                        # Diagnostic (gated per-(id, strategy)).
                        "tokens_saved_estimate": state.tokens_saved,
                        "tokens_saved_by_type": dict(state.tokens_saved_by_type),
                        "total_prune_count": state.total_prune_count,
                        "manual_mode": state.manual_mode,
                    }
                )

            if action == "index":
                entries = self.state_store.read_index(session_id)
                return json.dumps({"entries": entries, "count": len(entries)})

            if action == "analytics":
                # Cumulative cross-session savings from the SQLite store.
                # ``scope`` controls project filter: "global" (default) shows
                # everything, "project" filters to the current session's
                # project_path.  ``period`` chooses the breakdown:
                # "all" (default) returns the summary, "day" returns the
                # last N days, "month" returns the last N months,
                # "recent" returns the most recent N sessions, "project"
                # returns top-N projects by savings.
                scope = str(args.get("scope", "global")).strip().lower()
                period = str(args.get("period", "all")).strip().lower()
                limit = int(args.get("limit", 10) or 10)
                project_filter = state.project_path if scope == "project" else None

                if period == "day":
                    return json.dumps({
                        "scope": scope,
                        "period": "day",
                        "project": project_filter or "",
                        "rows": self.analytics_store.get_by_day(
                            days=limit if limit > 0 else 30,
                            project_path=project_filter,
                        ),
                    })
                if period == "month":
                    return json.dumps({
                        "scope": scope,
                        "period": "month",
                        "project": project_filter or "",
                        "rows": self.analytics_store.get_by_month(
                            months=limit if limit > 0 else 12,
                            project_path=project_filter,
                        ),
                    })
                if period == "recent":
                    return json.dumps({
                        "scope": scope,
                        "period": "recent",
                        "project": project_filter or "",
                        "rows": self.analytics_store.get_recent_sessions(
                            limit=limit,
                            project_path=project_filter,
                        ),
                    })
                if period == "project":
                    return json.dumps({
                        "scope": "global",
                        "period": "project",
                        "rows": self.analytics_store.get_by_project(limit=limit),
                    })

                # Default: full summary
                summary = self.analytics_store.get_summary(project_path=project_filter)
                return json.dumps({
                    "scope": scope,
                    "period": "all",
                    "project": project_filter or "",
                    "total_saved": summary.total_saved,
                    "total_sessions": summary.total_sessions,
                    "total_input": summary.total_input,
                    "total_output": summary.total_output,
                    "savings_pct": round(summary.savings_pct, 2),
                    "by_strategy": summary.by_strategy,
                })

            if action == "sweep":
                count = int(args.get("count", 0) or 0)
                protected_tools = (
                    ALWAYS_PROTECTED_SWEEP
                    | set(self.config.compress.protected_tools)
                    | set(self.config.strategies.deduplication.protected_tools)
                )
                messages = self._session_messages.get(session_id, [])

                if messages:
                    all_tool_call_ids: list[str] = []
                    tool_call_ids_since_last_user: list[str] = []
                    last_user_index = -1

                    for index, message in enumerate(messages):
                        if message.get("role") == "user":
                            last_user_index = index

                    for index, message in enumerate(messages):
                        if message.get("role") != "tool":
                            continue
                        tool_call_id = str(message.get("tool_call_id", ""))
                        all_tool_call_ids.append(tool_call_id)
                        if last_user_index >= 0 and index > last_user_index:
                            tool_call_ids_since_last_user.append(tool_call_id)

                    candidates = all_tool_call_ids[-count:] if count > 0 else (
                        tool_call_ids_since_last_user if last_user_index >= 0 else all_tool_call_ids
                    )
                else:
                    sorted_ids = [
                        tool_call_id
                        for tool_call_id, _record in sorted(
                            state.tool_calls.items(),
                            key=lambda item: item[1].timestamp,
                        )
                    ]
                    if count > 0:
                        candidates = sorted_ids[-count:]
                    elif state.current_turn > 0:
                        candidates = [
                            tool_call_id
                            for tool_call_id, record in state.tool_calls.items()
                            if record.turn_index == state.current_turn
                        ]
                    else:
                        candidates = sorted_ids

                selected: list[str] = []
                for tool_call_id in candidates:
                    if tool_call_id in state.pruned_tool_ids:
                        continue
                    record = state.tool_calls.get(tool_call_id)
                    tool_name = self._record_tool_name(record) if record is not None else ""
                    if tool_name in protected_tools:
                        continue
                    state.pruned_tool_ids.add(tool_call_id)
                    selected.append(tool_call_id)

                self._save_state(session_id)
                return json.dumps({"status": "ok", "swept": len(selected)})

            if action == "manual_status":
                return json.dumps({"manual_mode": state.manual_mode})

            if action == "manual_set":
                state.manual_mode = bool(args.get("enabled"))
                self._save_state(session_id)
                return json.dumps({"status": "ok", "manual_mode": state.manual_mode})

            return json.dumps({"error": f"Unknown action: {action}"})
