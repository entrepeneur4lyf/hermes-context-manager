"""Runtime state and serialization helpers for Hermes Context Manager."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any

# How many per-turn snapshots the observability ring buffer retains.
# 50 is enough for a full day of active work without unbounded growth.
TURN_HISTORY_MAX_LEN = 50


@dataclass(slots=True)
class ToolRecord:
    tool_call_id: str
    tool_name: str
    input_args: dict[str, Any]
    input_fingerprint: str
    is_error: bool
    turn_index: int
    timestamp: float
    token_estimate: int


@dataclass(slots=True)
class SessionState:
    tool_calls: dict[str, ToolRecord] = field(default_factory=dict)
    pruned_tool_ids: set[str] = field(default_factory=set)
    message_id_snapshot: dict[str, float] = field(default_factory=dict)
    current_turn: int = 0
    tokens_saved: int = 0
    # Un-gated, monotonic accumulator of real per-turn bytes kept out of the
    # API payload.  Every strategy-firing credits this even when the
    # per-(tool_call_id, strategy) gate on ``tokens_saved`` blocks the
    # "unique events" counter from re-incrementing.  This is the metric the
    # dashboard / status surface should lead with: the gated counter is a
    # diagnostic for how many DISTINCT tool calls each strategy touched,
    # while this is a truer measure of cumulative context savings across
    # the session (a tool call that persists for 20 turns and gets
    # re-compressed each turn shows up 20x here, once in ``tokens_saved``).
    tokens_kept_out_total: int = 0
    total_prune_count: int = 0
    tokens_saved_by_type: dict[str, int] = field(default_factory=dict)
    # Un-gated per-strategy breakdown, matching ``tokens_kept_out_total``.
    tokens_kept_out_by_type: dict[str, int] = field(default_factory=dict)
    manual_mode: bool = False
    last_context_tokens: int | None = None
    last_context_window: int | None = None
    last_context_percent: float | None = None
    dedup_group_sizes: dict[str, int] = field(default_factory=dict)
    # Per-(tool_call_id, strategy) gate that prevents double-counting savings
    # both across turns AND across strategies on the same tool call.  The
    # earlier per-id gate was over-aggressive: if truncation reduced an output
    # then dedup further collapsed it, only the first strategy got credit.
    counted_savings_ids: set[tuple[str, str]] = field(default_factory=set)
    # Canonical absolute cwd at session start, used as the project key in
    # the analytics SQLite store.  Captured lazily on first state creation.
    project_path: str = ""
    # Bounded ring buffer of per-turn snapshots for the hmc_control history
    # action.  Each entry is a dict of turn-level observability data; see
    # ``plugin.on_pre_llm_call`` for the snapshot schema.  Bounded so long
    # sessions don't grow the in-memory state without limit.
    turn_history: deque = field(
        default_factory=lambda: deque(maxlen=TURN_HISTORY_MAX_LEN)
    )


def create_state() -> SessionState:
    """Return a new zeroed session state."""
    return SessionState()


def reset_state(state: SessionState) -> None:
    """Reset a state object in place."""
    state.tool_calls.clear()
    state.pruned_tool_ids.clear()
    state.message_id_snapshot.clear()
    state.current_turn = 0
    state.tokens_saved = 0
    state.tokens_kept_out_total = 0
    state.total_prune_count = 0
    state.tokens_saved_by_type.clear()
    state.tokens_kept_out_by_type.clear()
    state.manual_mode = False
    state.last_context_tokens = None
    state.last_context_window = None
    state.last_context_percent = None
    state.dedup_group_sizes.clear()
    state.counted_savings_ids.clear()
    state.project_path = ""


def sort_object_keys(value: Any) -> Any:
    """Recursively sort object keys to create stable fingerprints."""
    if isinstance(value, list):
        return [sort_object_keys(item) for item in value]
    if isinstance(value, dict):
        return {key: sort_object_keys(value[key]) for key in sorted(value)}
    return value


def create_input_fingerprint(tool_name: str, args: dict[str, Any]) -> str:
    """Return a stable tool fingerprint for deduplication."""
    import json

    return f"{tool_name}::{json.dumps(sort_object_keys(args), sort_keys=True)}"


def session_state_to_dict(state: SessionState) -> dict[str, Any]:
    """Serialize state to a JSON-safe mapping."""
    return {
        "tool_calls": {
            tool_call_id: asdict(record)
            for tool_call_id, record in state.tool_calls.items()
        },
        "pruned_tool_ids": sorted(state.pruned_tool_ids),
        "tokens_saved": state.tokens_saved,
        "tokens_kept_out_total": state.tokens_kept_out_total,
        "total_prune_count": state.total_prune_count,
        "tokens_saved_by_type": dict(state.tokens_saved_by_type),
        "tokens_kept_out_by_type": dict(state.tokens_kept_out_by_type),
        "manual_mode": state.manual_mode,
        "last_context_tokens": state.last_context_tokens,
        "last_context_window": state.last_context_window,
        "last_context_percent": state.last_context_percent,
        "dedup_group_sizes": dict(state.dedup_group_sizes),
        # Tuples can't round-trip through JSON; serialize as list of 2-lists
        # (sorted for stable on-disk diffs).  Deserializer rebuilds tuples.
        "counted_savings_ids": sorted(
            [list(key) for key in state.counted_savings_ids]
        ),
        "project_path": state.project_path,
        # Persist the per-turn ring buffer so delta_saved survives any
        # state reload.  Before 0.3.3 this was dropped from the sidecar,
        # which meant every reload reset ``prev_cumulative`` to 0 and
        # the dashboard reported the full cumulative as the per-turn
        # delta.  ``list(deque)`` is JSON-safe; the deserializer
        # rebuilds the deque with the correct maxlen.
        "turn_history": list(state.turn_history),
    }


def session_state_from_dict(payload: dict[str, Any]) -> SessionState:
    """Hydrate state from a serialized mapping."""
    state = create_state()
    _tool_record_fields = {f.name for f in ToolRecord.__dataclass_fields__.values()}
    for tool_call_id, record in (payload.get("tool_calls") or {}).items():
        filtered = {k: v for k, v in record.items() if k in _tool_record_fields}
        state.tool_calls[tool_call_id] = ToolRecord(**filtered)
    state.pruned_tool_ids = set(payload.get("pruned_tool_ids") or [])
    state.tokens_saved = payload.get("tokens_saved", 0)
    # Backward-compat: sidecars written before the un-gated accumulator
    # existed don't have these fields.  Fall back to the gated counters so
    # upgraded sessions don't suddenly zero out their displayed history.
    # The values diverge going forward.
    state.tokens_kept_out_total = payload.get(
        "tokens_kept_out_total", state.tokens_saved
    )
    state.total_prune_count = payload.get("total_prune_count", 0)
    state.tokens_saved_by_type = dict(payload.get("tokens_saved_by_type") or {})
    state.tokens_kept_out_by_type = dict(
        payload.get("tokens_kept_out_by_type")
        or state.tokens_saved_by_type
    )
    state.manual_mode = payload.get("manual_mode", False)
    state.last_context_tokens = payload.get("last_context_tokens")
    state.last_context_window = payload.get("last_context_window")
    state.last_context_percent = payload.get("last_context_percent")
    state.dedup_group_sizes = dict(payload.get("dedup_group_sizes") or {})
    # Backward-compat: legacy sidecars stored ``counted_savings_ids`` as a
    # flat list of strings (per-id gate).  Newer sidecars store list of
    # 2-element lists ([id, strategy]).  Drop legacy entries silently — the
    # gate is per-session anyway, so losing them only risks one duplicate
    # accounting per session on first load, never persisting forward.
    raw_gate = payload.get("counted_savings_ids") or []
    state.counted_savings_ids = {
        (entry[0], entry[1])
        for entry in raw_gate
        if isinstance(entry, (list, tuple)) and len(entry) == 2
    }
    state.project_path = str(payload.get("project_path") or "")
    # Rebuild the bounded ring buffer from the persisted list.  Missing
    # key (legacy sidecars written before 0.3.3) falls back to empty,
    # which is safe -- the first post-load turn will simply compute
    # delta against the cumulative just loaded, which may be slightly
    # inflated once, then converge correctly.
    raw_history = payload.get("turn_history") or []
    if isinstance(raw_history, list):
        state.turn_history.extend(
            entry for entry in raw_history if isinstance(entry, dict)
        )
    return state
