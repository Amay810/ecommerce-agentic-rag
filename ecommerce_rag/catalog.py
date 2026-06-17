# -*- coding: utf-8 -*-
"""Small deterministic helpers for recommendation and comparison views."""

from collections import defaultdict


def product_hits(chunks: list[dict]) -> list[dict]:
    grouped: dict[str, dict] = {}
    scores: dict[str, float] = defaultdict(float)
    evidence: dict[str, list[str]] = defaultdict(list)
    for c in chunks:
        if c.get("source_type") != "product":
            continue
        pid = c.get("product_id") or c["doc_id"]
        grouped.setdefault(
            pid,
            {
                "product_id": pid,
                "title": c.get("title", ""),
                "category": c.get("category", ""),
                "price": c.get("price"),
                "inventory": c.get("inventory", ""),
            },
        )
        scores[pid] += float(c.get("score", 0.0))
        evidence[pid].append(c.get("text", ""))
    rows = []
    for pid, row in grouped.items():
        rows.append({**row, "score": scores[pid], "evidence": evidence[pid][:3]})
    return sorted(rows, key=lambda x: -x["score"])


def recommendation_brief(chunks: list[dict], limit: int = 3) -> str:
    rows = product_hits(chunks)[:limit]
    if not rows:
        return "暂时没有找到足够匹配的商品，建议转人工客服。"
    lines = ["可以优先看这几款："]
    for i, row in enumerate(rows, 1):
        price = f"，价格 {row['price']} 元" if row.get("price") is not None else ""
        inv = f"，库存 {row['inventory']}" if row.get("inventory") else ""
        reason = row["evidence"][0][:80] if row["evidence"] else "与需求匹配"
        lines.append(f"{i}. {row['title']}（{row['category']}{price}{inv}）：{reason} [资料{i}]")
    lines.append("如果你有预算、使用场景或品牌偏好，我可以继续收窄推荐。")
    return "\n".join(lines)


def comparison_brief(chunks: list[dict], limit: int = 3) -> str:
    rows = product_hits(chunks)[:limit]
    if len(rows) < 2:
        return "我只找到一个明确相关的商品，暂时无法做可靠对比。可以再提供另一款商品名。"
    lines = ["按当前资料，对比如下："]
    for i, row in enumerate(rows, 1):
        price = f"{row['price']} 元" if row.get("price") is not None else "价格未注明"
        inv = row.get("inventory") or "库存未注明"
        evidence = "；".join(row["evidence"])[:120]
        lines.append(f"{i}. {row['title']}：{price}，{inv}。关键信息：{evidence} [资料{i}]")
    lines.append("如果要我给结论，请补充你更看重价格、续航、清洁力、便携性还是兼容性。")
    return "\n".join(lines)
