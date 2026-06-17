# -*- coding: utf-8 -*-
"""Quick before/after comparison for Q3 price filter.
Run with:  python nscc/compare_price_filter.py
"""
import importlib
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)


def run(price_filter_on: bool) -> dict:
    os.environ["ERAG_PRICE_FILTER"] = "1" if price_filter_on else "0"
    os.environ["ERAG_SUPPORT_STORE"] = "0"

    # force reload of config + dependents so the env var takes effect
    import ecommerce_rag.config as cfg
    import ecommerce_rag.price_filter as pf
    import ecommerce_rag.agent as ag
    import ecommerce_rag.evaluate as ev
    for mod in (cfg, pf, ag, ev):
        importlib.reload(mod)
    # re-import after reload so we get the fresh function
    from ecommerce_rag.evaluate import evaluate
    from ecommerce_rag.hybrid_retriever import HybridRetriever
    from ecommerce_rag.agent import CustomerSupportAgent
    from pathlib import Path
    agent = CustomerSupportAgent(HybridRetriever(), enable_logging=False)
    testset = Path(BASE) / "ecommerce_rag" / "data" / "eval_questions.jsonl"
    return evaluate(testset, agent, with_llm_metrics=False)


def main() -> None:
    print("Running baseline (ERAG_PRICE_FILTER=0)...")
    r0 = run(False)

    print("Running with filter (ERAG_PRICE_FILTER=1)...")
    r1 = run(True)

    s0, s1 = r0["summary"], r1["summary"]

    print("\n=== Q3: 预算600以内 ===")
    for label, r in [("BASELINE ", r0), ("FILTERED ", r1)]:
        for q in r["details"]:
            if "预算600" in q["question"]:
                print(f"  {label} retrieved={q['retrieved_doc_ids']}  recall@1={q['recall@1']}")

    print("\n=== SUMMARY (28 questions) ===")
    print(f"{'':20s}  recall@1  recall@3  recall@5  mrr")
    print(f"{'baseline':20s}  {s0['recall@1']:.3f}     {s0['recall@3']:.3f}     {s0['recall@5']:.3f}     {s0['mrr']:.3f}")
    print(f"{'+ price_filter':20s}  {s1['recall@1']:.3f}     {s1['recall@3']:.3f}     {s1['recall@5']:.3f}     {s1['mrr']:.3f}")
    delta1 = s1["recall@1"] - s0["recall@1"]
    print(f"\nΔ recall@1 = {delta1:+.3f}  (Q3 fix, target: +0.036)")


if __name__ == "__main__":
    main()
