# -*- coding: utf-8 -*-
"""Price metadata filter: parse budget constraint from query, filter over-budget candidates.

Dense embedding cannot reliably learn numerical price constraints — "预算600以内"
doesn't prevent a 899元 product from ranking #1 by semantic similarity. This module
adds a post-retrieval filtering step that respects the user's stated budget.

Design
──────
  1. parse_budget(query)  — extract ceiling (int yuan) from budget keywords, or None
  2. apply(chunks, budget) — remove product chunks where price > budget;
                            if ALL products exceed budget, fall back to original order
                            (graceful degradation) with a warning note

The filter only touches product chunks; policy chunks always pass through.
"""

import re

# Matches Chinese budget expressions:
#   "预算600以内", "不超过600元", "低于600", "小于600"
#   "600以内", "600元以下", "600块以内"
_BUDGET_RE = re.compile(
    r"(?:预算|不超过|低于|小于)[^\d]*(\d+)"
    r"|(\d+)\s*(?:元|块|¥)?(?:以内|以下)"
)


def parse_budget(query: str) -> int | None:
    """Return the budget ceiling in yuan, or None if no constraint detected."""
    m = _BUDGET_RE.search(query)
    if not m:
        return None
    val = m.group(1) or m.group(2)
    return int(val) if val else None


def apply(chunks: list[dict], budget: int) -> tuple[list[dict], str]:
    """Filter product chunks whose price exceeds budget.

    Returns:
        (filtered_chunks, trace_note)
        If all products are over budget, returns original chunks unchanged with a
        warning note (so the agent can still attempt an answer rather than hard-failing).
    """
    within: list[dict] = []
    over: list[dict] = []
    for c in chunks:
        if c.get("source_type") != "product":
            within.append(c)  # policies always pass through
        elif c.get("price") is None or c.get("price") <= budget:
            within.append(c)
        else:
            over.append(c)

    removed_ids = list({c["doc_id"] for c in over})

    if not removed_ids:
        return chunks, ""  # nothing filtered, no note needed

    if not within:
        # Absolutely nothing left — graceful fallback to original order with warning
        return chunks, f"价格过滤：所有候选均超预算（>{budget}元），返回原排序供参考"

    note = (
        f"价格过滤：已移除 {len(removed_ids)} 个超预算商品"
        f"（budget={budget}元）: {', '.join(removed_ids)}"
    )
    return within, note
