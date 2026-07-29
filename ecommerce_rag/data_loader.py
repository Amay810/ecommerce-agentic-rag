# -*- coding: utf-8 -*-
"""Build retrievable child chunks and parent evidence cards.

Products and policies are indexed together, but retain metadata so the agent can
route product QA, recommendation, comparison, and policy questions differently.
"""

import json
import time
from pathlib import Path
from typing import Iterable

from . import config


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_products(path: Path = config.PRODUCT_DATA_PATH) -> list[dict]:
    return _read_jsonl(path)


def load_policies(path: Path = config.POLICY_DATA_PATH) -> list[dict]:
    return _read_jsonl(path)


def product_card(p: dict) -> str:
    attrs = "；".join(f"{k}: {v}" for k, v in p.get("attributes", {}).items())
    parts = [
        f"商品：{p['title']}（{p.get('category', '')}）",
        f"价格：{p.get('price')} 元" if p.get("price") is not None else "",
        f"库存：{p.get('inventory')}" if p.get("inventory") else "",
        f"规格：{attrs}" if attrs else "",
        f"描述：{p.get('description', '')}" if p.get("description") else "",
    ]
    if p.get("reviews"):
        parts.append("用户评价：" + " ".join(p["reviews"]))
    if p.get("qa"):
        parts.append("常见问答：" + " ".join(f"问：{x['q']} 答：{x['a']}" for x in p["qa"]))
    return "\n".join(x for x in parts if x)


def policy_card(p: dict) -> str:
    return "\n".join(
        x
        for x in [
            f"政策：{p['title']}（{p.get('policy_type', '')}）",
            f"适用范围：{p.get('scope', '')}" if p.get("scope") else "",
            f"内容：{p.get('content', '')}",
        ]
        if x
    )


def build_product_chunks(products: Iterable[dict]) -> tuple[list[dict], dict[str, str]]:
    chunks: list[dict] = []
    parents: dict[str, str] = {}
    for p in products:
        pid = p["id"]
        doc_id = f"product:{pid}"
        parents[doc_id] = product_card(p)
        base = {
            "doc_id": doc_id,
            "source_type": "product",
            "product_id": pid,
            "title": p["title"],
            "category": p.get("category", ""),
            "price": p.get("price"),
            "inventory": p.get("inventory", ""),
            "updated_at": p.get("updated_at"),  # optional; feeds the freshness guardrail
        }

        def add(chunk_type: str, text: str, i: int) -> None:
            chunks.append({**base, "chunk_id": f"{doc_id}:{chunk_type}:{i}", "chunk_type": chunk_type, "text": text.strip()})

        if p.get("description"):
            add("desc", f"{p['title']}：{p['description']}", 0)
        for i, (key, value) in enumerate(list(p.get("attributes", {}).items())[:8]):
            add("attr", f"{p['title']} specification: {key} {value}", i)
        for i, qa in enumerate(p.get("qa", [])):
            add("qa", f"{p['title']} 问：{qa['q']} 答：{qa['a']}", i)
        # Separate review chunks improve evidence density and scale to roughly
        # 25k-50k retrievable units for a 5k-product corpus.
        for i, review in enumerate(p.get("reviews", [])[:5]):
            review_text = review.get("text", "") if isinstance(review, dict) else str(review)
            if review_text.strip():
                add("review", f"{p['title']} user review: {review_text}", i)
    return chunks, parents


def build_policy_chunks(policies: Iterable[dict]) -> tuple[list[dict], dict[str, str]]:
    chunks: list[dict] = []
    parents: dict[str, str] = {}
    for p in policies:
        pid = p["id"]
        doc_id = f"policy:{pid}"
        parents[doc_id] = policy_card(p)
        chunks.append(
            {
                "chunk_id": f"{doc_id}:body:0",
                "doc_id": doc_id,
                "source_type": "policy",
                "policy_id": pid,
                "title": p["title"],
                "category": p.get("policy_type", ""),
                "price": None,
                "inventory": "",
                "chunk_type": "policy",
                "text": policy_card(p),
                "updated_at": p.get("updated_at"),  # optional; feeds the freshness guardrail
            }
        )
    return chunks, parents


def build_chunks(products: list[dict], policies: list[dict]) -> tuple[list[dict], dict[str, str]]:
    product_chunks, product_parents = build_product_chunks(products)
    policy_chunks, policy_parents = build_policy_chunks(policies)
    return product_chunks + policy_chunks, {**product_parents, **policy_parents}


def build_index(index_dir: Path = config.INDEX_DIR, product_path: Path = config.PRODUCT_DATA_PATH,
                policy_path: Path = config.POLICY_DATA_PATH) -> dict:
    started = time.perf_counter()
    import numpy as np
    from sentence_transformers import SentenceTransformer

    products = load_products(product_path)
    policies = load_policies(policy_path)
    chunks, parents = build_chunks(products, policies)
    index_dir.mkdir(parents=True, exist_ok=True)

    model_load_started = time.perf_counter()
    model = SentenceTransformer(config.EMBED_MODEL)
    model_load_ms = (time.perf_counter() - model_load_started) * 1000
    embedding_started = time.perf_counter()
    embeddings = model.encode(
        [c["text"] for c in chunks],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype("float32")
    embedding_ms = (time.perf_counter() - embedding_started) * 1000
    persist_started = time.perf_counter()
    np.save(index_dir / "embeddings.npy", embeddings)
    with open(index_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    with open(index_dir / "parents.json", "w", encoding="utf-8") as f:
        json.dump(parents, f, ensure_ascii=False, indent=2)
    persist_ms = (time.perf_counter() - persist_started) * 1000
    stats = {
        "products": len(products),
        "policies": len(policies),
        "chunks": len(chunks),
        "parents": len(parents),
        "embed_model": config.EMBED_MODEL,
        "model_load_ms": round(model_load_ms, 2),
        "embedding_ms": round(embedding_ms, 2),
        "persist_ms": round(persist_ms, 2),
        "build_time_ms": round((time.perf_counter() - started) * 1000, 2),
        "index_size_bytes": sum(p.stat().st_size for p in index_dir.rglob("*") if p.is_file()),
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return stats


if __name__ == "__main__":
    import argparse
    from collections import Counter
    parser = argparse.ArgumentParser()
    parser.add_argument("--products", type=Path, default=config.PRODUCT_DATA_PATH)
    parser.add_argument("--policies", type=Path, default=config.POLICY_DATA_PATH)
    parser.add_argument("--index-dir", type=Path, default=config.INDEX_DIR)
    parser.add_argument("--stats-output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    products = load_products(args.products)
    policies = load_policies(args.policies)
    chunks, parents = build_chunks(products, policies)
    if args.dry_run:
        print(f"{len(products)} products + {len(policies)} policies -> {len(chunks)} chunks, {len(parents)} parents")
        print("source types:", dict(Counter(c["source_type"] for c in chunks)))
        print("chunk types:", dict(Counter(c["chunk_type"] for c in chunks)))
    else:
        stats = build_index(args.index_dir, args.products, args.policies)
        if args.stats_output:
            args.stats_output.parent.mkdir(parents=True, exist_ok=True)
            args.stats_output.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
