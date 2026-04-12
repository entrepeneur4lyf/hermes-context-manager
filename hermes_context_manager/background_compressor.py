"""Background compression via auxiliary model."""
from __future__ import annotations
import logging
from typing import Any
from .state import SessionState

LOGGER = logging.getLogger(__name__)


def identify_stale_ranges(
    messages: list[dict[str, Any]],
    state: SessionState,
    protect_recent_turns: int = 3,
) -> list[dict[str, Any]]:
    if not messages:
        return []
    user_indices = [i for i, m in enumerate(messages) if m.get("role") == "user"]
    if len(user_indices) <= protect_recent_turns:
        return []
    cutoff_idx = user_indices[-protect_recent_turns]
    # Find completed work before cutoff, grouped by user turns
    ranges = []
    block_start = 0
    for i, msg in enumerate(messages):
        if i >= cutoff_idx:
            break
        if msg.get("role") == "user" and i > block_start:
            ranges.append({"start_idx": block_start, "end_idx": i - 1})
            block_start = i
    if cutoff_idx > block_start:
        ranges.append({"start_idx": block_start, "end_idx": cutoff_idx - 1})
    # Merge into one range for simplicity (compress oldest chunk)
    if ranges:
        merged = {
            "start_idx": ranges[0]["start_idx"],
            "end_idx": ranges[-1]["end_idx"],
        }
        # Count user turns in range
        user_turns = sum(1 for i in range(merged["start_idx"], merged["end_idx"] + 1)
                         if messages[i].get("role") == "user")
        first_turn = sum(1 for i in range(merged["start_idx"]) if messages[i].get("role") == "user") + 1
        merged["turn_range"] = f"{first_turn}-{first_turn + user_turns - 1}"
        return [merged]
    return []


def build_index_entry(
    messages: list[dict[str, Any]],
    start_idx: int,
    end_idx: int,
    turn_range: str,
    summary: str = "",
) -> dict[str, Any]:
    import time
    topic = "Completed work"
    for msg in messages[start_idx:end_idx + 1]:
        if msg.get("role") == "user":
            raw = msg.get("content", "")
            # Multimodal messages arrive as a list of content blocks
            # (e.g. ``[{"type": "text", "text": "..."}, {"type": "image_url", ...}]``).
            # Calling ``.split`` on that list crashes — extract the first
            # text block instead.
            if isinstance(raw, list):
                text = next(
                    (
                        block.get("text", "")
                        for block in raw
                        if isinstance(block, dict) and block.get("type") == "text"
                    ),
                    "",
                )
            else:
                text = str(raw)
            first_line = text.split("\n")[0][:80]
            if first_line:
                topic = first_line
            break
    if not summary:
        parts = []
        for msg in messages[start_idx:end_idx + 1]:
            if msg.get("role") == "user":
                raw = msg.get("content", "")
                if isinstance(raw, list):
                    text = next(
                        (
                            block.get("text", "")
                            for block in raw
                            if isinstance(block, dict) and block.get("type") == "text"
                        ),
                        "",
                    )
                else:
                    text = str(raw)
                parts.append(text[:100])
        summary = "; ".join(parts[:3]) if parts else "Completed work"
    return {
        "turn_range": turn_range,
        "topic": topic,
        "summary": summary[:200],
        "timestamp": time.strftime("%H:%M"),
        "message_count": end_idx - start_idx + 1,
    }
