"""Head/tail windowing for long tool results (RTK-inspired)."""

from __future__ import annotations


def head_tail_truncate(
    content: str,
    max_lines: int = 50,
    head: int = 10,
    tail: int = 10,
) -> str:
    """Truncate content keeping first *head* and last *tail* lines.

    If the content fits within *max_lines* it is returned unchanged.
    Otherwise the first *head* lines and last *tail* lines are kept with
    a ``... (N lines omitted)`` gap marker inserted between them.
    """
    if not content:
        return content

    lines = content.splitlines(keepends=True)
    total = len(lines)

    if total <= max_lines:
        return content

    # Clamp head+tail so there is at least 1 line reserved for the gap marker
    budget = max_lines - 1  # reserve 1 for gap marker
    if budget < 1:
        budget = 1

    if tail == 0:
        # Head-only mode
        head = min(head, budget)
        omitted = total - head
        gap = f"... ({omitted} lines omitted)\n"
        return "".join(lines[:head]) + gap

    # Normal head+tail mode – distribute budget
    if head + tail > budget:
        # Scale proportionally but ensure at least 1 for each
        ratio = head / (head + tail)
        head = max(1, int(budget * ratio))
        tail = max(1, budget - head)

    omitted = total - head - tail
    if omitted <= 0:
        return content

    gap = f"... ({omitted} lines omitted)\n"
    return "".join(lines[:head]) + gap + "".join(lines[-tail:])
