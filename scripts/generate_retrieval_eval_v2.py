"""Create a 300-query title-debiased retrieval benchmark.

The 205 deterministic rows have programmatic gold. The remaining difficult rows
are explicitly marked curated/unverified until a human signs the audit file.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path


def partial_name(title: str) -> str:
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9+._-]*", title)
    distinctive = [x for x in tokens if len(x) >= 4]
    return " ".join(distinctive[:2]) if distinctive else " ".join(tokens[:2])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--products", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260720)
    args = parser.parse_args()
    rows = [json.loads(x) for x in args.products.read_text(encoding="utf-8").splitlines() if x.strip()]
    rng = random.Random(args.seed)
    eligible = [x for x in rows if x.get("attributes")]
    rng.shuffle(eligible)
    output: list[dict] = []

    def add(kind: str, question: str, gold: list[str], *, split: str, review: str, **extra) -> None:
        output.append({"id": f"v2_{len(output)+1:03d}", "question": question, "gold_doc_ids": gold, "source_type": "product",
                       "kind": kind, "split": split, "review_status": review, **extra})

    # 180 attribute questions use a corpus-unique key/value signature, so the
    # programmatic gold remains deterministic without exposing a full title.
    signature_frequency: dict[tuple[str, str], int] = defaultdict(int)
    for item in eligible:
        for key, value in item["attributes"].items():
            signature_frequency[(str(key).strip().lower(), str(value).strip().lower())] += 1
    attribute_cases = []
    for item in eligible:
        unique = [(k, v) for k, v in item["attributes"].items()
                  if len(str(v).strip()) >= 4 and signature_frequency[(str(k).strip().lower(), str(v).strip().lower())] == 1]
        unique.sort(key=lambda pair: (0 if any(x in str(pair[0]).lower() for x in ("model", "part", "upc", "dimensions")) else 1, -len(str(pair[1]))))
        if unique: attribute_cases.append((item, *unique[0]))
        if len(attribute_cases) == 180: break
    if len(attribute_cases) < 180:
        raise SystemExit("not enough corpus-unique attribute signatures")
    for i, (item, key, value) in enumerate(attribute_cases):
        alias = (item.get("category_aliases_zh") or [item.get("category", "商品")])[0]
        brand = item["attributes"].get("brand_or_store", "")
        question = f"找{alias}，品牌 {brand}；{key} 为 {value}。"
        assert item["title"].lower() not in question.lower()
        add("attribute_no_title", question, [f"product:{item['id']}"], split="dev" if i < 90 else "locked", review="programmatic_gold")

    priced = [x for x in eligible if isinstance(x.get("price"), (int, float)) and x["price"] > 0]
    if len(priced) < 40:
        raise SystemExit("not enough priced products")
    for i, item in enumerate(priced[:40]):
        ceiling = round(float(item["price"]) + max(1, float(item["price"]) * .05), 2)
        alias = (item.get("category_aliases_zh") or [item.get("category", "商品")])[0]
        brand = item.get("attributes", {}).get("brand_or_store", "")
        question = f"预算不超过 {ceiling}，想买 {brand} 的{alias}，关键词 {partial_name(item['title'])}"
        add("multi_constraint", question, [f"product:{item['id']}"], split="dev" if i < 20 else "locked",
            review="curated_unverified", constraints={"max_price": ceiling})

    for i, item in enumerate(eligible[200:230]):
        clue = partial_name(item["title"])
        typo = clue.replace("o", "0", 1).replace("i", "1", 1)
        alias = (item.get("category_aliases_zh") or ["商品"])[0]
        add("alias_typo", f"有个{alias}好像叫 {typo}，帮我找准确商品", [f"product:{item['id']}"],
            split="dev" if i < 15 else "locked", review="curated_unverified")

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for item in eligible:
        brand = str(item.get("attributes", {}).get("brand_or_store", "")).strip().lower()
        category = " ".join(str(item.get("category", "")).split()[:3]).lower()
        if brand: groups[(brand, category)].append(item)
    pairs = []
    for (_, _), items in sorted(groups.items()):
        if len(items) >= 2:
            pairs.append((items[0], items[1]))
        if len(pairs) == 25: break
    if len(pairs) < 25:
        pairs.extend((eligible[300+2*i], eligible[301+2*i]) for i in range(25-len(pairs)))
    for i, (left, right) in enumerate(pairs[:25]):
        brand = left.get("attributes", {}).get("brand_or_store", "同品牌")
        category = " ".join(str(left.get("category", "商品")).split()[-2:])
        add("near_sku_multi_gold", f"找 {brand} 的 {category} 相近型号，线索是 {partial_name(left['title'])}",
            [f"product:{left['id']}", f"product:{right['id']}"], split="dev" if i < 13 else "locked",
            review="curated_unverified", relevance_mode="any")

    impossible = [
        "量子悬浮全息烤箱 第九代 火星专供", "零重力永动手机充电器 900%效率", "能读取梦境的USB线 紫色版本",
        "时间旅行电视遥控器 2099款", "反物质驱动扫地机器人 家用版",
    ]
    for i in range(25):
        add("no_answer", f"{impossible[i % len(impossible)]} 编号ZX-{9000+i}", [], split="dev" if i < 13 else "locked",
            review="curated_unverified", abstention_expected=True)

    if len(output) != 300:
        raise AssertionError(len(output))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in output), encoding="utf-8")
    print(json.dumps({"queries": len(output), "full_title_leaks": 0,
                      "splits": {s: sum(x["split"] == s for x in output) for s in ("dev", "locked")},
                      "review": {s: sum(x["review_status"] == s for x in output) for s in {x["review_status"] for x in output}}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
