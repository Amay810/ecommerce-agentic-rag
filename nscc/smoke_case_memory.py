# -*- coding: utf-8 -*-
"""NSCC smoke: build CaseMemory index from existing SupportCases, then run agent
with ERAG_CASE_MEMORY=1 to verify memory_hint appears in trace.

Expected flow:
  1. Build: embed all support_cases from support.db -> case_memory.db + case_memory.npy
  2. Search: ad-hoc search for Q3 / Q28 queries
  3. Agent run: 3 queries; confirm memory_hint is populated when memory matched
"""

import os
import pprint

os.environ["ERAG_CASE_MEMORY"] = "1"

from ecommerce_rag import case_memory, store
from ecommerce_rag.agent import CustomerSupportAgent
from ecommerce_rag.hybrid_retriever import HybridRetriever

SEARCH_QUERIES = [
    "预算500以内的降噪耳机",        # should hit price_constraint_ignored memory
    "推荐一款保温杯和焖烧罐",        # should hit compound_query_recall_gap
    "有没有便宜的扫地机器人",        # neutral, likely no strong match
]

AGENT_QUERIES = [
    "不超过600元的通勤降噪耳机",
    "保温杯和焖烧罐各推荐一款",
    "我的订单退款什么时候到账",
]


def main() -> None:
    # ─ Step 1: build memory ───────────────────────────────────────────────────
    retriever = HybridRetriever()
    model = retriever.model
    n = store.count()
    print(f"support.db has {n} cases")

    if n == 0:
        print("WARNING: no cases in store — run smoke_failure_memory.py first")
        return

    print("\n=== Building CaseMemory index ===")
    built = case_memory.build(model=model)
    print(f"Built {built} memory entries")

    # ─ Step 2: ad-hoc search ─────────────────────────────────────────────────
    print("\n=== Ad-hoc memory search ===")
    for q in SEARCH_QUERIES:
        hint = case_memory.search(q, model=model)
        if hint.matched:
            print(f"  Q: {q}")
            print(f"     sim={hint.top_sim:.3f}  patterns={hint.avoid_patterns}  "
                  f"suggested={hint.suggested_action}")
        else:
            print(f"  Q: {q}  -> no match above threshold")

    # ─ Step 3: agent with memory prior ────────────────────────────────────────
    print("\n=== Agent run with memory prior ===")
    agent = CustomerSupportAgent(retriever, enable_logging=True)
    for q in AGENT_QUERIES:
        r = agent.run(q)
        mh = r.get("memory_hint") or {}
        mem_s = ""
        if mh.get("matched"):
            mem_s = f"  [MEMORY sim={mh['top_sim']:.3f} action={mh.get('suggested_action')}]"
        print(f"RUN  [{r['intent']:12s}] {r['action']:8s}{mem_s}  |  {q}")
        for t in r.get("trace") or []:
            if "记忆先验" in t:
                print(f"       trace: {t}")


if __name__ == "__main__":
    main()
