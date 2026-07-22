"""Generate a focused, unseen routing holdout for RulePolicy regression."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows: list[dict] = []

    def add(message: str, expected_tool: str, kind: str) -> None:
        rows.append({"id": f"routing_v3_{len(rows)+1:03d}", "message": message,
                     "expected_tool": expected_tool, "kind": kind})

    policy_cases = [
        ("请查平台物流条款，不是查我的订单", "get_policy"),
        ("退款有哪些正式规则？", "get_policy"),
        ("我只想了解退换货规定", "get_policy"),
        ("发票政策具体怎么写的", "get_policy"),
        ("查一下保修条款", "get_policy"),
        ("物流配送规则是什么", "get_policy"),
        ("退款到账的政策说明", "get_policy"),
        ("平台换货规则，不涉及具体订单", "get_policy"),
        ("有哪些正式的发票规定", "get_policy"),
        ("请依据政策解释保修范围", "get_policy"),
    ]
    for message, tool in policy_cases:
        add(message, tool, "policy_ambiguity")

    for i in range(1, 6):
        add(f"查订单 O{i:06d} 的物流状态，验证码 {i:06d}", "get_order", "personal_order")
        add(f"订单 O{i+10:06d} 要退货，验证码 {i+10:06d}", "check_return_eligibility", "return")
        add(f"比较 P{i:05d} 和 P{i+1:05d}", "compare_products", "compare")
        add(f"帮我搜索适合场景 {i} 的商品", "search_catalog", "catalog")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({"rows": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
