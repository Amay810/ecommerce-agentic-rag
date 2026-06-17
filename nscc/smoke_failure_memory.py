# -*- coding: utf-8 -*-
"""NSCC smoke: run the agent on diverse queries, then emit a FailureMemory report.

Designed to surface real pattern instances from actual agent runs:
  - price_constraint_ignored  (Q3-style: "预算600以内通勤降噪耳机")
  - compound_query_recall_gap (Q28-style: "保温杯和焖烧罐")
  - stale_data_caution        (P005 CleanBot, updated 2026-01-01)
  - handoff_cluster           (order/refund queries)

Runs on a CPU compute node. No LLM key needed; agent falls back gracefully.
"""

from ecommerce_rag import failure_memory, store
from ecommerce_rag.agent import CustomerSupportAgent
from ecommerce_rag.hybrid_retriever import HybridRetriever

QUERIES = [
    # price constraint (Q3-style)
    "预算600以内通勤降噪耳机推荐",
    "500块以内的机械键盘有哪些",
    # compound query (Q28-style)
    "推荐保温杯和焖烧罐各一款",
    "保温杯和焖烧罐哪个保温效果更好",
    # stale product (P005 CleanBot, updated_at=2026-01-01)
    "养猫家庭适合买哪款清洁产品",
    "CleanBot 扫地机器人有货吗",
    # product_qa / recommend (mix of ok and caution)
    "保温杯可以装碳酸饮料吗",
    "有哪些适合跑步的蓝牙耳机",
    # handoff
    "我的订单退款什么时候到账",
    "我要投诉这次购物体验",
]


def main() -> None:
    agent = CustomerSupportAgent(HybridRetriever(), enable_logging=True)
    for q in QUERIES:
        r = agent.run(q)
        fr = r.get("freshness") or {}
        fr_s = (
            f" freshness={fr.get('status')}({'/'.join(fr.get('claims', []))})"
            if fr.get("triggered") else ""
        )
        print(f"RUN  [{r['intent']:12s}] {r['action']:8s}{fr_s}  |  {q}")

    print(f"\nstore.count() = {store.count()}")

    print("\n" + "=" * 60)
    print("FAILURE MEMORY REPORT")
    report = failure_memory.analyze()
    print(failure_memory.render_report(report))


if __name__ == "__main__":
    main()
