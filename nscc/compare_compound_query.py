# -*- coding: utf-8 -*-
"""Before/after comparison for Q28 compound query decomposition.

Usage:  python nscc/compare_compound_query.py
"""
import importlib
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)


def run(decomp_on: bool) -> dict:
    os.environ["ERAG_COMPOUND_DECOMP"] = "1" if decomp_on else "0"
    os.environ["ERAG_SUPPORT_STORE"] = "0"
    os.environ["ERAG_PRICE_FILTER"] = "1"  # keep price filter always on

    import ecommerce_rag.config as cfg
    import ecommerce_rag.compound_query as cq
    import ecommerce_rag.agent as ag
    import ecommerce_rag.evaluate as ev
    for mod in (cfg, cq, ag, ev):
        importlib.reload(mod)

    from ecommerce_rag.evaluate import evaluate
    from ecommerce_rag.hybrid_retriever import HybridRetriever
    from ecommerce_rag.agent import CustomerSupportAgent
    from pathlib import Path

    agent = CustomerSupportAgent(HybridRetriever(), enable_logging=False)
    testset = Path(BASE) / "ecommerce_rag" / "data" / "eval_questions.jsonl"
    return evaluate(testset, agent, with_llm_metrics=False)


def main() -> None:
    print("Running baseline (ERAG_COMPOUND_DECOMP=0)...")
    r0 = run(False)

    print("Running with decomposition (ERAG_COMPOUND_DECOMP=1)...")
    r1 = run(True)

    s0, s1 = r0["summary"], r1["summary"]

    print("\n=== Q28: 保温杯和焖烧罐有什么区别 ===")
    for label, r in [("BASELINE   ", r0), ("DECOMPOSED ", r1)]:
        for q in r["details"]:
            if "焖烧罐" in q["question"]:
                print(
                    f"  {label} retrieved={q['retrieved_doc_ids']}"
                    f"  recall@1={q['recall@1']:.1f}"
                    f"  recall@3={q['recall@3']:.1f}"
                    f"  recall@5={q['recall@5']:.1f}"
                )

    print("\n=== SUMMARY (all questions) ===")
    hdr = f"{'':22s}  recall@1  recall@3  recall@5  mrr"
    print(hdr)
    print(f"{'baseline':22s}  {s0['recall@1']:.3f}     {s0['recall@3']:.3f}     {s0['recall@5']:.3f}     {s0['mrr']:.3f}")
    print(f"{'+ compound_decomp':22s}  {s1['recall@1']:.3f}     {s1['recall@3']:.3f}     {s1['recall@5']:.3f}     {s1['mrr']:.3f}")
    d1 = s1["recall@1"] - s0["recall@1"]
    d3 = s1["recall@3"] - s0["recall@3"]
    d5 = s1["recall@5"] - s0["recall@5"]
    print(f"\nΔ recall@1={d1:+.3f}  recall@3={d3:+.3f}  recall@5={d5:+.3f}  (target: Q28 P006+P014 in top-5)")


if __name__ == "__main__":
    main()
