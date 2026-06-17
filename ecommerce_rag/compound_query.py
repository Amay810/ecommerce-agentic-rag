# -*- coding: utf-8 -*-
"""Detect and decompose compound comparison queries into entity sub-queries.

A single dense embedding for "保温杯和焖烧罐有什么区别" often surfaces one entity
strongly and misses the other. Decomposing into ["保温杯", "焖烧罐"] lets us
retrieve both separately and merge via RRF — guaranteeing dual coverage.

Design constraints
──────────────────
- Pure Python / stdlib, no new deps.
- Only called when route.name == "compare".
- Connector list ordered longest-first so regex alternation doesn't short-circuit.
- Entity A: full text before connector; 8-char cap ONLY for purely-CJK strings
  (English/mixed names like "RunBuds Clip" must not be truncated to 8 bytes).
- Entity B: stop at Chinese question-verb characters (有/是/哪/什/么/的/吗/呢).
  Whitespace is intentionally NOT a stop character so "Air Pro 2" is kept whole.
"""

import re

# Chinese connectors that signal an A-vs-B comparison. Ordered longest-first
# so regex alternation doesn't short-circuit "对比" to "比".
_CONNECTOR_RE = re.compile(
    r"对比|相比|比较|还是|和|与|vs\.?",
    re.IGNORECASE,
)

# Stop entity B at Chinese question-verb characters only.
# Deliberately excludes whitespace so "Air Pro 2" is not cut at the first space.
_STOP_RE = re.compile(r"[有是哪什么的吗呢？！、，。,.]")

_MAX_ENTITY_CJK = 8    # max chars for a purely-CJK entity
_MAX_ENTITY_TOTAL = 40  # sanity cap for any entity (prevents absurd matches)


def _n_cjk(s: str) -> int:
    return sum(1 for c in s if "一" <= c <= "鿿")


def detect(query: str) -> tuple[bool, list[str]]:
    """Return (is_compound, [entity_a, entity_b]).

    Returns (False, []) when:
    - No connector found.
    - Either extracted entity is empty or exceeds the total sanity cap.
    - Both entities are identical.
    """
    m = _CONNECTOR_RE.search(query)
    if not m:
        return False, []

    before = query[: m.start()].strip()
    after = query[m.end() :].strip()

    # Entity A: full text before connector.
    # Cap at _MAX_ENTITY_CJK ONLY when the text is purely CJK (Chinese nouns).
    # Mixed/English names ("RunBuds Clip") are kept whole.
    a = before
    if _n_cjk(a) == len(a) and len(a) > _MAX_ENTITY_CJK:
        a = a[-_MAX_ENTITY_CJK:].strip()  # right-edge: more specific noun

    # Entity B: stop at first question-verb char (not whitespace).
    stop = _STOP_RE.search(after)
    b = (after[: stop.start()].strip() if stop else after.strip())
    if _n_cjk(b) == len(b) and len(b) > _MAX_ENTITY_CJK:
        b = b[:_MAX_ENTITY_CJK].strip()  # left-edge: head noun

    if not a or not b or a == b:
        return False, []
    if len(a) > _MAX_ENTITY_TOTAL or len(b) > _MAX_ENTITY_TOTAL:
        return False, []

    return True, [a, b]
