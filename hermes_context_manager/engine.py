"""Core pruning logic for Hermes Context Manager."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .code_filter import (
    LANGUAGE_EXTENSIONS,
    detect_language,
    filter_code_block,
    filter_fenced_blocks,
)
from .config import HmcConfig
from .normalizer import normalize_for_dedup
from .short_circuits import apply_short_circuits
from .state import SessionState
from .truncation import head_tail_truncate

ALWAYS_PROTECTED_DEDUP = {"write_file", "patch"}
ALWAYS_PROTECTED_SWEEP = {"hmc_control"}
ID_ELIGIBLE_ROLES = {"user", "assistant", "tool"}


@dataclass(slots=True)
class MaterializedView:
    """Computed visible conversation view for the current turn.

    ``content_backup`` captures the original ``message["content"]`` value for
    every message that was mutated, keyed by its index in the input list.
    Callers use it to restore the conversation to its pre-mutation state
    without a full deep copy — only ``content`` is ever modified by the
    strategy functions, so backing up that one field is sufficient.
    """

    messages: list[dict[str, Any]]
    message_id_snapshot: dict[str, float]
    total_tokens: int
    content_backup: dict[int, Any] = field(default_factory=dict)


def estimate_tokens(text: str) -> int:
    """Estimate token count from text length."""
    return round(len(text) / 4)


def _capture_content_backup(
    content_backup: dict[int, Any] | None,
    idx: int,
    message: dict[str, Any],
) -> None:
    """Record a message's content before first mutation this turn.

    ``None`` means "someone else is tracking mutations for us" (the
    ``apply_strategies_to_tool_output`` path where the plugin's
    ``on_post_tool_call`` captures the backup at the outer level).

    When a dict is supplied, this function records
    ``content_backup[idx] = current_content`` but ONLY if the idx
    isn't already present -- strategies stack, and the TRUE original
    content is the one captured by the FIRST strategy that touched
    the message.  Subsequent strategies that mutate the same message
    see already-mutated content and must not overwrite the backup.

    Before 0.3.6 ``materialize_view`` pre-populated a backup dict with
    every message's content upfront, which meant a 200-message
    conversation built a 200-entry dict each turn even though only
    ~5-10 tool messages would actually be mutated.  The lazy
    copy-on-mutate approach here shrinks the backup to only the
    messages that were actually touched.
    """
    if content_backup is None:
        return
    if idx in content_backup:
        return
    content_backup[idx] = message.get("content")


def _credit_savings(
    state: SessionState,
    tool_call_id: str,
    strategy: str,
    saved: int,
) -> None:
    """Record a single strategy firing's savings on the session state.

    Two accumulators, both incremented here:

    - ``tokens_kept_out_total`` / ``tokens_kept_out_by_type``: **un-gated**.
      Every firing adds its incremental savings, even if the same
      ``(tool_call_id, strategy)`` pair fired on a previous turn.  This
      is the number the dashboard should lead with, because a tool output
      that persists across 20 turns and gets re-compressed each turn
      really does save those bytes on every API call.

    - ``tokens_saved`` / ``tokens_saved_by_type``: **gated** by
      ``counted_savings_ids`` so each ``(tool_call_id, strategy)`` pair
      credits at most once per session.  This answers a different
      question: "how many unique tool-call/strategy firings happened?"
      Useful as a diagnostic but NOT a true bytes-saved metric.

    ``saved <= 0`` or an empty ``tool_call_id`` are no-ops -- both
    represent firing paths that either didn't actually shrink the content
    or can't be de-duplicated across turns.
    """
    if saved <= 0 or not tool_call_id:
        return
    # Un-gated accumulator: always increments.  This is the number the
    # dashboard's delta_saved computes from.
    state.tokens_kept_out_total += saved
    state.tokens_kept_out_by_type[strategy] = (
        state.tokens_kept_out_by_type.get(strategy, 0) + saved
    )
    # Gated accumulator: one credit per (tool_call_id, strategy).
    gate_key = (tool_call_id, strategy)
    if gate_key not in state.counted_savings_ids:
        state.tokens_saved += saved
        state.tokens_saved_by_type[strategy] = (
            state.tokens_saved_by_type.get(strategy, 0) + saved
        )
        state.counted_savings_ids.add(gate_key)


# Fields the LLM provider actually receives on the wire.  HMC-internal
# metadata like ``timestamp`` and ``tool_name`` (duplicated from tool_calls)
# sit on the message dict for our own bookkeeping but never ship to the API,
# so they shouldn't be counted toward context usage.
_API_VISIBLE_FIELDS: tuple[str, ...] = (
    "role",
    "content",
    "tool_calls",
    "tool_call_id",
    "name",
)


def estimate_message_tokens(message: dict[str, Any]) -> int:
    """Estimate the token contribution of a message (API-accurate).

    Serializes ONLY the fields that ship to the LLM provider --
    ``role``, ``content``, ``tool_calls``, ``tool_call_id``, ``name``.
    HMC-internal fields like ``timestamp`` never go over the wire and
    must not be counted against the context budget.

    Before 0.3.6 this used ``len(str(message)) // 4`` which captured
    the whole Python dict repr including internal fields AND Python
    repr overhead (``{'role':`` vs ``{"role":``, single quotes, etc.).
    Measured impact on a real 528-message code-review session was
    modest (~0.6% reduction) because long tool outputs dominate and
    internal metadata amortizes across them -- but on short messages
    the overhead was ~15-20% of the total, and as HMC grows more
    internal bookkeeping fields the gap would widen.  This is a
    correctness fix, not a hot-path optimization: we should count
    what the API actually sees.

    The ``default=str`` fallback on ``json.dumps`` handles edge cases
    where a value isn't directly JSON-serializable (rare in practice
    for conversation messages).  On any failure we fall back to the
    old ``str()`` heuristic so the estimator can't crash the pipeline.
    """
    relevant: dict[str, Any] = {
        k: message[k] for k in _API_VISIBLE_FIELDS if k in message
    }
    try:
        return len(json.dumps(relevant, default=str)) // 4
    except (TypeError, ValueError):
        return len(str(message)) // 4


def _apply_deduplication(messages: list[dict[str, Any]], state: SessionState, config: HmcConfig) -> None:
    if not config.strategies.deduplication.enabled:
        return
    if state.manual_mode and not config.manual_mode.automatic_strategies:
        return

    protected = ALWAYS_PROTECTED_DEDUP | set(config.strategies.deduplication.protected_tools)
    fingerprints: dict[str, list[str]] = {}

    for message in messages:
        if message.get("role") != "tool":
            continue
        tool_call_id = str(message.get("tool_call_id", ""))
        record = state.tool_calls.get(tool_call_id)
        if not record:
            continue

        if record.tool_name in protected or not record.input_fingerprint:
            continue

        fingerprints.setdefault(record.input_fingerprint, []).append(tool_call_id)

    for ids in fingerprints.values():
        if len(ids) > 1:
            for dup_id in ids:
                state.dedup_group_sizes[dup_id] = len(ids)
        if len(ids) <= 1:
            continue
        for duplicate_id in ids[:-1]:
            if duplicate_id not in state.pruned_tool_ids:
                state.pruned_tool_ids.add(duplicate_id)
                state.total_prune_count += 1

    # --- Second pass: normalized content dedup ---
    content_groups: dict[str, list[str]] = {}
    for message in messages:
        if message.get("role") != "tool":
            continue
        tool_call_id = str(message.get("tool_call_id", ""))
        if tool_call_id in state.pruned_tool_ids:
            continue
        record = state.tool_calls.get(tool_call_id)
        if not record:
            continue

        if record.tool_name in protected:
            continue

        content = message.get("content")
        if not isinstance(content, str) or len(content) < 100:
            continue

        norm_fp = f"{record.tool_name}::{normalize_for_dedup(content)}"
        content_groups.setdefault(norm_fp, []).append(tool_call_id)

    for ids in content_groups.values():
        if len(ids) > 1:
            for dup_id in ids:
                state.dedup_group_sizes[dup_id] = len(ids)
        if len(ids) <= 1:
            continue
        for duplicate_id in ids[:-1]:
            if duplicate_id not in state.pruned_tool_ids:
                state.pruned_tool_ids.add(duplicate_id)
                state.total_prune_count += 1


def _apply_error_purging(messages: list[dict[str, Any]], state: SessionState, config: HmcConfig) -> None:
    if not config.strategies.purge_errors.enabled:
        return
    if state.manual_mode and not config.manual_mode.automatic_strategies:
        return

    protected = set(config.strategies.purge_errors.protected_tools)
    turns_threshold = config.strategies.purge_errors.turns

    for message in messages:
        if message.get("role") != "tool":
            continue
        tool_call_id = str(message.get("tool_call_id", ""))
        record = state.tool_calls.get(tool_call_id)
        if not record:
            continue

        if record.tool_name in protected or not record.is_error:
            continue
        if state.current_turn - record.turn_index >= turns_threshold:
            if tool_call_id not in state.pruned_tool_ids:
                state.pruned_tool_ids.add(tool_call_id)
                state.total_prune_count += 1


def _apply_tool_output_pruning(
    messages: list[dict[str, Any]],
    state: SessionState,
    content_backup: dict[int, Any] | None = None,
) -> None:
    """Replace pruned tool outputs with placeholders and credit savings.

    This is where dedup and error_purge actually mutate content -- the
    earlier identification passes only added IDs to ``state.pruned_tool_ids``.
    Both strategies were silently uncounted in ``tokens_saved`` until this
    accounting was added; ``hmc_control stats`` would show ``dedup: 0`` and
    ``error_purge: 0`` even when they were saving thousands of bytes.

    Each strategy is gated by ``(tool_call_id, strategy)`` so that the same
    tool call can credit multiple strategies that legitimately stack
    (e.g. truncation reduces 1000 -> 500, then dedup further collapses
    to 50; both contributions are real and additive).
    """
    for idx, message in enumerate(messages):
        if message.get("role") != "tool":
            continue
        tool_call_id = str(message.get("tool_call_id", ""))
        if tool_call_id not in state.pruned_tool_ids:
            continue
        record = state.tool_calls.get(tool_call_id)
        record_is_error = record.is_error if record is not None else False
        is_error = bool(message.get("is_error")) or record_is_error

        # Capture pre-pruning size so we can credit the incremental savings
        # this stage produces on top of whatever short_circuit/truncation
        # already did.  ``content`` may be a string or a list (multimodal).
        pre_content = message.get("content")
        pre_size = (
            estimate_tokens(pre_content) if isinstance(pre_content, str)
            else estimate_tokens(str(pre_content))
        )

        _capture_content_backup(content_backup, idx, message)

        if is_error:
            message["content"] = "[Error output removed - tool failed more than N turns ago]"
            strategy = "error_purge"
        else:
            count = state.dedup_group_sizes.get(tool_call_id, 0)
            if count > 1:
                tool_name = record.tool_name if record is not None else 'tool'
                message["content"] = (
                    f"[Output removed \u2014 {tool_name} called {count}\u00d7 with same args, "
                    f"showing last result only]"
                )
            else:
                message["content"] = (
                    "[Output removed to save context \u2014 information superseded "
                    "or no longer needed]"
                )
            strategy = "dedup"

        post_size = estimate_tokens(message["content"]) if isinstance(message["content"], str) else 0
        saved = pre_size - post_size
        _credit_savings(state, tool_call_id, strategy, saved)


def _inject_message_ids(messages: list[dict[str, Any]]) -> dict[str, float]:
    """Build a message-id timestamp snapshot WITHOUT polluting content.

    Until 0.3.2 this function appended ``\\n<hmc-message-id>m0NN</hmc-message-id>``
    to the content of every user/assistant/tool message.  In real-world
    sessions (verified against ~/.hermes/sessions/*.json) the assistant
    learned to ECHO the tag at the start of its replies -- 32% of GLM-5.1
    assistant messages in one captured session began with the tag, and
    several even mis-copied the format ("message_id" vs "message-id").
    The user-visible symptom was "first words cut off in real time" as
    Hermes's display layer dealt with the unwanted tag in the stream.

    The tags had no readers anywhere in HMC's codebase: nothing in
    prompts.py advertised them to the model, no tool resolved them back
    to messages, and the snapshot was only ever written, never queried.
    A pure phantom feature.

    The function is kept (rather than deleted) so legacy state sidecars
    that already have ``message_id_snapshot`` populated still hydrate
    cleanly via ``materialize_view``'s contract, and so any future
    addressing scheme that wants positional message identifiers can
    rebuild it from message metadata without being a content mutator.
    The snapshot now keys by positional ID against the message timestamp
    -- same shape as before, just without the in-content tag.
    """
    snapshot: dict[str, float] = {}
    counter = 1
    for message in messages:
        if message.get("role") not in ID_ELIGIBLE_ROLES:
            continue
        snapshot[f"m{counter:03d}"] = float(message.get("timestamp", 0))
        counter += 1
    return snapshot


def _apply_head_tail_truncation(
    messages: list[dict[str, Any]],
    state: SessionState,
    config: HmcConfig,
    content_backup: dict[int, Any] | None = None,
) -> None:
    """Apply head/tail windowing to long tool results in-place."""
    if not config.truncation.enabled:
        return

    trunc = config.truncation
    for idx, message in enumerate(messages):
        if message.get("role") != "tool":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        if len(content) < trunc.min_content_length:
            continue
        tool_call_id = str(message.get("tool_call_id", ""))
        if tool_call_id in state.pruned_tool_ids:
            continue
        # Skip error results — preserve full detail for debugging
        record = state.tool_calls.get(tool_call_id)
        if (record is not None and record.is_error) or message.get("is_error"):
            continue

        original_tokens = estimate_tokens(content)
        truncated = head_tail_truncate(
            content,
            max_lines=trunc.max_lines,
            head=trunc.head_lines,
            tail=trunc.tail_lines,
        )
        if truncated is not content and truncated != content:
            _capture_content_backup(content_backup, idx, message)
            message["content"] = truncated
            new_tokens = estimate_tokens(truncated)
            saved = original_tokens - new_tokens
            _credit_savings(state, tool_call_id, "truncation", saved)


def _detect_lang_from_record(record: Any, content: str) -> str | None:
    """Determine the language of a code-bearing tool output.

    Detection order: tool-arg file extension (most reliable) → fenced
    markdown tag → content sniffing.  Returns None when no signal is
    confident enough to apply the filter.
    """
    # 1. Tool argument file extension (e.g. read_file path="src/main.py")
    if record is not None and getattr(record, "input_fingerprint", None):
        for ext_match in re.finditer(r"\.([a-zA-Z]+)\b", record.input_fingerprint):
            ext = ext_match.group(1).lower()
            for lang, exts in LANGUAGE_EXTENSIONS.items():
                if ext in exts:
                    return lang
    # 2. Fenced markdown block
    fenced = re.search(r"```([a-zA-Z]+)\s*\n", content)
    if fenced:
        from .code_filter import _normalize_lang_hint  # local to avoid cycle
        canon = _normalize_lang_hint(fenced.group(1))
        if canon:
            return canon
    # 3. Content sniffing
    return detect_language(content)


def _apply_code_filter(
    messages: list[dict[str, Any]],
    state: SessionState,
    config: HmcConfig,
    content_backup: dict[int, Any] | None = None,
) -> None:
    """Strip function/class bodies from tool outputs containing source code.

    Slotted between short_circuit and truncation in materialize_view: a
    short-circuited output never reaches the code filter (no point), and
    truncation runs AFTER the filter so it operates on the smaller
    post-filter content.

    Trigger: content has at least ``min_lines`` lines AND a language can
    be detected via tool-arg path, fenced block, or content sniffing.
    The detected language must be in ``code_filter.languages``.

    Savings credited to the ``code_filter`` strategy bucket via the
    per-(tool_call_id, "code_filter") gate, so cross-strategy stacking
    with truncation/dedup works correctly.
    """
    cfg = config.code_filter
    if not cfg.enabled:
        return
    if state.manual_mode and not config.manual_mode.automatic_strategies:
        return

    enabled_languages = set(cfg.languages)

    for idx, message in enumerate(messages):
        if message.get("role") != "tool":
            continue
        tool_call_id = str(message.get("tool_call_id", ""))
        if tool_call_id in state.pruned_tool_ids:
            continue
        record = state.tool_calls.get(tool_call_id)
        if (record is not None and record.is_error) or message.get("is_error"):
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        if content.count("\n") + 1 < cfg.min_lines:
            continue

        lang = _detect_lang_from_record(record, content)
        if lang is None or lang not in enabled_languages:
            continue

        original_tokens = estimate_tokens(content)
        # Fenced markdown blocks: process each block independently and
        # preserve surrounding prose.  Plain code (no fences): process
        # the whole content as a single block.
        if "```" in content:
            new_content = filter_fenced_blocks(
                content, preserve_docstrings=cfg.preserve_docstrings,
            )
        else:
            new_content = filter_code_block(
                content, lang, preserve_docstrings=cfg.preserve_docstrings,
            )

        if new_content == content:
            continue

        _capture_content_backup(content_backup, idx, message)
        message["content"] = new_content
        new_tokens = estimate_tokens(new_content)
        saved = original_tokens - new_tokens
        _credit_savings(state, tool_call_id, "code_filter", saved)


def _apply_short_circuit_replacement(
    messages: list[dict[str, Any]],
    state: SessionState,
    config: HmcConfig,
    content_backup: dict[int, Any] | None = None,
) -> None:
    """Replace tool outputs that match known success patterns with one-liners."""
    if not config.short_circuits.enabled:
        return

    for idx, message in enumerate(messages):
        if message.get("role") != "tool":
            continue
        tool_call_id = str(message.get("tool_call_id", ""))
        if tool_call_id in state.pruned_tool_ids:
            continue
        # Skip error results — preserve their detail for debugging
        record = state.tool_calls.get(tool_call_id)
        if (record is not None and record.is_error) or message.get("is_error"):
            continue
        content = message.get("content")
        if not isinstance(content, str) or len(content) < 20:
            continue
        replacement = apply_short_circuits(content)
        if replacement is not None:
            original_tokens = estimate_tokens(content)
            _capture_content_backup(content_backup, idx, message)
            message["content"] = replacement
            new_tokens = estimate_tokens(replacement)
            saved = original_tokens - new_tokens
            _credit_savings(state, tool_call_id, "short_circuit", saved)


def apply_strategies_to_tool_output(
    message: dict[str, Any],
    state: SessionState,
    config: HmcConfig,
) -> int:
    """Run single-message compression strategies on one tool output.

    Used by ``on_post_tool_call`` to compress a just-arrived tool result
    during agent loops, so savings accrue continuously instead of
    waiting for the next user-initiated ``pre_llm_call``.

    Only runs ``short_circuit``, ``code_filter``, and ``truncation`` --
    those are the strategies that operate on a single message's
    ``content`` without needing the full conversation.  ``dedup`` and
    ``error_purge`` need the whole message list and continue to run
    inside ``materialize_view`` on the next ``pre_llm_call``.

    Mutates ``message["content"]`` in place on the caller's dict.
    Returns the incremental ``tokens_kept_out_total`` delta so callers
    can publish a meaningful "tool" event to the dashboard.
    """
    pre_total = state.tokens_kept_out_total
    # Each strategy handler expects a list; wrap the single message.
    single = [message]
    _apply_short_circuit_replacement(single, state, config)
    _apply_code_filter(single, state, config)
    _apply_head_tail_truncation(single, state, config)
    return state.tokens_kept_out_total - pre_total


def materialize_view(
    messages: list[dict[str, Any]],
    state: SessionState,
    config: HmcConfig,
) -> MaterializedView:
    """Compute the visible conversation view for the current turn.

    Mutates ``messages`` in place: every strategy writes through
    ``message["content"]`` on the live dicts, so mutations propagate across
    shallow copies (``list(messages)``, ``msg.copy()``) back into the
    Hermes-owned conversation list.  A ``content_backup`` dict keyed by
    list-index captures the original content of every message BEFORE its
    first mutation, enabling full restoration via
    ``plugin._restore_mutations``.

    As of 0.3.6 the backup is populated **lazily**: each mutating strategy
    calls ``_capture_content_backup(backup, idx, message)`` before writing,
    and the helper only records a message if it isn't already in the dict.
    On a 200-message conversation where 5-10 tool messages actually mutate,
    the backup ends up with 5-10 entries instead of 200.  Review point #2
    from the v0.3.3 review.

    **Important:** the caller owns the list.  HMC only edits ``content``;
    all other fields on each message dict are left untouched.
    """
    state.current_turn = sum(1 for message in messages if message.get("role") == "user")

    content_backup: dict[int, Any] = {}

    _apply_short_circuit_replacement(messages, state, config, content_backup)
    _apply_code_filter(messages, state, config, content_backup)
    _apply_head_tail_truncation(messages, state, config, content_backup)
    _apply_deduplication(messages, state, config)
    _apply_error_purging(messages, state, config)
    _apply_tool_output_pruning(messages, state, content_backup)
    message_id_snapshot = _inject_message_ids(messages)
    state.message_id_snapshot.clear()
    state.message_id_snapshot.update(message_id_snapshot)
    total_tokens = sum(estimate_message_tokens(message) for message in messages)
    return MaterializedView(
        messages=messages,
        message_id_snapshot=message_id_snapshot,
        total_tokens=total_tokens,
        content_backup=content_backup,
    )


def apply_pruning(
    messages: list[dict[str, Any]],
    state: SessionState,
    config: HmcConfig,
) -> list[dict[str, Any]]:
    """Compatibility wrapper returning the materialized message list."""
    return materialize_view(messages, state, config).messages
