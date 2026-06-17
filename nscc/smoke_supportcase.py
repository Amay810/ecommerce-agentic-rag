# -*- coding: utf-8 -*-
"""NSCC smoke: run the agent with logging on a few queries, confirm SupportCases persist.

Runs on a CPU compute node (loads the embed model). No LLM key needed: product_qa
generation falls back gracefully, the SupportCase is still produced and stored.
"""

from ecommerce_rag import store
from ecommerce_rag.agent import CustomerSupportAgent
from ecommerce_rag.hybrid_retriever import HybridRetriever

QUERIES = [
    "保温杯可以装碳酸饮料吗？",          # product_qa
    "养猫家庭适合买哪款清洁产品？",      # recommend
    "我的订单退款什么时候到账？",        # handoff
]


def main() -> None:
    agent = CustomerSupportAgent(HybridRetriever(), enable_logging=True)
    for q in QUERIES:
        r = agent.run(q)
        fr = r.get("freshness") or {}
        fr_s = f" freshness={fr.get('status')}({'/'.join(fr.get('claims', []))})" if fr.get("triggered") else ""
        print(f"RUN  {q}  ->  intent={r['intent']} action={r['action']}{fr_s}")
    print(f"\nstore.count() = {store.count()}")
    print("recent persisted cases:")
    for row in store.recent(5):
        print(f"  {row['case_id']} | {row['intent']} | {row['action']} | "
              f"needs_review={row['needs_review']} | conf={row['confidence']:.3f}")


if __name__ == "__main__":
    main()
