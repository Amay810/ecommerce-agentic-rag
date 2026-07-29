"""Generate a fresh 150-query retrieval holdout after v2 tuning.

The generator excludes every product used as v2 gold and uses a new seed.  The
result is a single immutable ``locked`` split.  Programmatic attribute gold is
safe to score automatically; difficult rows remain pending human confirmation.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from scripts.generate_retrieval_eval_v2 import partial_name


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--products", type=Path, required=True)
    parser.add_argument("--exclude-benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260722)
    args = parser.parse_args()

    products = read_jsonl(args.products)
    excluded = {
        doc_id.split(":", 1)[1]
        for row in read_jsonl(args.exclude_benchmark)
        for doc_id in row.get("gold_doc_ids", [])
        if doc_id.startswith("product:")
    }
    eligible = [p for p in products if p.get("attributes") and p["id"] not in excluded]
    rng = random.Random(args.seed)
    rng.shuffle(eligible)

    signature_frequency: Counter[tuple[str, str]] = Counter()
    for item in products:
        for key, value in item.get("attributes", {}).items():
            signature_frequency[(str(key).strip().lower(), str(value).strip().lower())] += 1

    output: list[dict] = []
    used: set[str] = set()

    def add(kind: str, question: str, gold: list[str], *, review: str, **extra) -> None:
        output.append({
            "id": f"v3_{len(output)+1:03d}",
            "question": question,
            "gold_doc_ids": gold,
            "source_type": "product",
            "kind": kind,
            "split": "locked",
            "review_status": review,
            **extra,
        })

    # 80 unique attribute signatures, with no complete title in the query.
    for item in eligible:
        unique = [
            (key, value)
            for key, value in item["attributes"].items()
            if len(str(value).strip()) >= 4
            and signature_frequency[(str(key).strip().lower(), str(value).strip().lower())] == 1
        ]
        unique.sort(key=lambda pair: (-len(str(pair[1])), str(pair[0])))
        if not unique:
            continue
        key, value = unique[0]
        alias = (item.get("category_aliases_zh") or ["商品"])[0]
        brand = item["attributes"].get("brand_or_store", "")
        question = f"找{alias}，品牌线索是 {brand}，属性 {key} 为 {value}。"
        if item["title"].lower() in question.lower():
            continue
        add("attribute_no_title", question, [f"product:{item['id']}"], review="programmatic_gold")
        used.add(item["id"])
        if len(output) == 80:
            break
    if len(output) != 80:
        raise SystemExit(f"only generated {len(output)} unique attribute cases")

    # 20 budget/brand/partial-name cases from unused products.
    priced = [p for p in eligible if p["id"] not in used and isinstance(p.get("price"), (int, float)) and p["price"] > 0]
    for item in priced[:20]:
        ceiling = round(float(item["price"]) * 1.05 + 1, 2)
        alias = (item.get("category_aliases_zh") or ["商品"])[0]
        brand = item["attributes"].get("brand_or_store", "")
        add("multi_constraint", f"预算不超过 {ceiling}，找 {brand} 的{alias}，名称线索 {partial_name(item['title'])}",
            [f"product:{item['id']}"], review="codex_assisted_pending_human", constraints={"max_price": ceiling})
        used.add(item["id"])

    # 20 alias/typo cases.
    typo_items = [p for p in eligible if p["id"] not in used and partial_name(p["title"])][:20]
    for item in typo_items:
        clue = partial_name(item["title"])
        typo = clue.replace("o", "0", 1).replace("i", "1", 1).replace("e", "3", 1)
        alias = (item.get("category_aliases_zh") or ["商品"])[0]
        add("alias_typo", f"有个{alias}好像叫 {typo}，帮我定位准确商品",
            [f"product:{item['id']}"], review="codex_assisted_pending_human")
        used.add(item["id"])

    # 15 same-brand/category near-SKU cases.  Any of the two is relevant.
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for item in eligible:
        if item["id"] in used:
            continue
        brand = str(item["attributes"].get("brand_or_store", "")).strip().lower()
        category = " ".join(str(item.get("category", "")).lower().split()[:3])
        if brand:
            groups[(brand, category)].append(item)
    pairs: list[tuple[dict, dict]] = []
    for items in groups.values():
        if len(items) >= 2:
            pairs.append((items[0], items[1]))
        if len(pairs) == 15:
            break
    if len(pairs) < 15:
        raise SystemExit(f"only generated {len(pairs)} near-SKU pairs")
    for left, right in pairs:
        brand = left["attributes"].get("brand_or_store", "同品牌")
        add("near_sku_multi_gold", f"找 {brand} 的相近型号，线索是 {partial_name(left['title'])}",
            [f"product:{left['id']}", f"product:{right['id']}"], review="codex_assisted_pending_human",
            relevance_mode="any")

    impossible = [
        "月球真空专用量子烤箱", "读取梦境的 USB-C 数据线", "反物质驱动家用扫地机器人",
        "零重力永动手机充电器", "时间旅行电视遥控器", "火星大气制造咖啡机",
        "能预测彩票号码的智能手表", "隐形全息冰箱贴", "百分之九百效率太阳能鼠标",
        "家用虫洞路由器", "脑波读取机械键盘", "瞬间传送空气炸锅",
        "暗物质降噪耳机", "可逆转时间的电动牙刷", "反重力保温杯",
    ]
    for index, phrase in enumerate(impossible):
        add("no_answer", f"寻找 {phrase}，型号 V3-ZX-{7300+index}", [],
            review="codex_assisted_pending_human", abstention_expected=True)

    if len(output) != 150:
        raise AssertionError(len(output))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output), encoding="utf-8")
    print(json.dumps({
        "queries": len(output),
        "seed": args.seed,
        "excluded_v2_gold_products": len(excluded),
        "kinds": dict(Counter(row["kind"] for row in output)),
        "review": dict(Counter(row["review_status"] for row in output)),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
