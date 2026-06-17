# -*- coding: utf-8 -*-
"""Stdlib-only tests for case_memory.py.

Uses a fake embedding model (returns deterministic vectors) and a temp SQLite DB
so no sentence-transformers or real support.db is needed.
"""

import json
import sqlite3
import tempfile
from pathlib import Path

import numpy as np

from ecommerce_rag import case_memory


# ── fake model ────────────────────────────────────────────────────────────────

class _FakeModel:
    """Returns vectors derived from the first char's codepoint for determinism."""
    def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False):
        vecs = []
        for t in texts:
            seed = ord(t[0]) if t else 42
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(16).astype("float32")
            if normalize_embeddings:
                v /= np.linalg.norm(v) + 1e-9
            vecs.append(v)
        return np.array(vecs)


# ── helpers ───────────────────────────────────────────────────────────────────

def _fake_case(case_id: str, query: str, intent: str, action: str,
               patterns: list[str], sim_seed: str | None = None) -> tuple[dict, str]:
    """Return (metadata dict, memory_text) for injection into the memory DB."""
    text = f"意图:{intent} | 查询:{query} | 结果:{action}"
    if patterns:
        text += f" | 失败模式:{' '.join(patterns)}"
    return ({
        "mem_id": case_id,
        "case_id": case_id,
        "intent": intent,
        "action": action,
        "query": query,
        "memory_text": text,
        "patterns_json": json.dumps(patterns, ensure_ascii=False),
        "embedding_idx": 0,  # will be overwritten
        "created_at": "2026-06-12T00:00:00+00:00",
    }, text)


def _build_fake_memory(entries: list[tuple[dict, str]], model=None):
    """Build a fake memory DB + npy in temp paths. Returns (mem_db_path, embed_path)."""
    if model is None:
        model = _FakeModel()

    mem_db = Path(tempfile.mktemp(suffix="_mem.db"))
    embed_p = Path(tempfile.mktemp(suffix="_mem.npy"))

    texts = [text for _, text in entries]
    embs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    np.save(embed_p, embs.astype("float32"))

    conn = sqlite3.connect(mem_db)
    conn.row_factory = sqlite3.Row
    cols = list(case_memory._MEM_SCHEMA.keys())
    col_defs = ", ".join(f"{k} {v}" for k, v in case_memory._MEM_SCHEMA.items())
    conn.execute(f"CREATE TABLE case_memory ({col_defs})")
    ph = ", ".join("?" for _ in cols)
    for idx, (meta, _) in enumerate(entries):
        meta["embedding_idx"] = idx
        meta["mem_id"] = str(idx)
        conn.execute(f"INSERT INTO case_memory ({', '.join(cols)}) VALUES ({ph})",
                     [meta.get(c) for c in cols])
    conn.commit()
    conn.close()
    return mem_db, embed_p


# ── tests ─────────────────────────────────────────────────────────────────────

def test_search_finds_price_constraint():
    """Searching '不超过500元耳机' should surface a price_constraint_ignored memory."""
    model = _FakeModel()
    entries = [
        _fake_case("sc_p1", "预算600以内通勤降噪耳机推荐", "recommend", "caution",
                   ["price_constraint_ignored"]),
        _fake_case("sc_ok1", "这款耳机音质怎么样", "product_qa", "ok", []),
    ]
    db, ep = _build_fake_memory(entries, model)

    # A query semantically close to "预算600以内" should hit with our fake model
    # since both start with '预', seed will be similar — but we lower threshold to 0.0
    # for testing to always trigger.
    hint = case_memory.search("预算500以内耳机", model=model,
                               mem_db_path=db, embed_path=ep, threshold=0.0, n=3)
    assert hint.matched is True
    assert len(hint.memories) > 0
    print("PASS test_search_finds_price_constraint")


def test_search_returns_no_match_above_threshold():
    """With very high threshold, nothing should match."""
    model = _FakeModel()
    entries = [_fake_case("sc_1", "保温杯怎么清洗", "product_qa", "ok", [])]
    db, ep = _build_fake_memory(entries, model)
    hint = case_memory.search("保温杯怎么清洗", model=model,
                               mem_db_path=db, embed_path=ep, threshold=2.0)
    assert hint.matched is False
    print("PASS test_search_returns_no_match_above_threshold")


def test_suggested_action_price_filter():
    """price_constraint_ignored pattern → suggested_action = apply_price_filter."""
    model = _FakeModel()
    entries = [
        _fake_case("sc_p2", "不超过800元的机械键盘", "recommend", "caution",
                   ["price_constraint_ignored"]),
    ]
    db, ep = _build_fake_memory(entries, model)
    hint = case_memory.search("不超过800元的机械键盘", model=model,
                               mem_db_path=db, embed_path=ep, threshold=0.0)
    assert hint.matched
    assert hint.suggested_action == "apply_price_filter"
    assert "price_constraint_ignored" in hint.avoid_patterns
    print("PASS test_suggested_action_price_filter")


def test_suggested_action_query_decomposition():
    """compound_query_recall_gap → suggested_action = use_query_decomposition."""
    model = _FakeModel()
    entries = [
        _fake_case("sc_c1", "保温杯和焖烧罐哪个好", "compare", "caution",
                   ["compound_query_recall_gap"]),
    ]
    db, ep = _build_fake_memory(entries, model)
    hint = case_memory.search("保温杯和焖烧罐哪个保温效果更好", model=model,
                               mem_db_path=db, embed_path=ep, threshold=0.0)
    assert hint.matched
    assert hint.suggested_action == "use_query_decomposition"
    print("PASS test_suggested_action_query_decomposition")


def test_hint_to_trace():
    """hint_to_trace returns a non-empty string for matched hints."""
    hint = case_memory.MemoryHint(
        matched=True, top_sim=0.82,
        avoid_patterns=["price_constraint_ignored"],
        suggested_action="apply_price_filter",
        memories=[],
        note="test note",
    )
    t = case_memory.hint_to_trace(hint)
    assert t is not None
    assert "0.82" in t
    assert "price_constraint" in t

    no_match = case_memory.MemoryHint(matched=False)
    assert case_memory.hint_to_trace(no_match) is None
    print("PASS test_hint_to_trace")


def test_no_memory_index_returns_no_match():
    """When memory files don't exist, search returns matched=False gracefully."""
    model = _FakeModel()
    hint = case_memory.search("随便问个问题", model=model,
                               mem_db_path=Path("/nonexistent/mem.db"),
                               embed_path=Path("/nonexistent/mem.npy"))
    assert hint.matched is False
    print("PASS test_no_memory_index_returns_no_match")


def test_make_memory_text_includes_patterns():
    """_make_memory_text embeds pattern info into the text."""
    case = {
        "intent": "recommend", "query": "预算600以内耳机",
        "action": "caution", "evidence": [
            {"doc_id": "product:P009", "source_type": "product"}
        ],
        "freshness": {"status": "fresh"},
    }
    text = case_memory._make_memory_text(case, ["price_constraint_ignored"])
    assert "price_constraint_ignored" in text
    assert "recommend" in text
    assert "预算600以内耳机" in text
    print("PASS test_make_memory_text_includes_patterns")


def test_case_patterns_detects_price():
    """_case_patterns correctly detects price_constraint_ignored on a synthetic case."""
    case = {
        "query": "预算600以内通勤降噪耳机",
        "intent": "recommend",
        "action": "caution",
        "needs_review": True,
        "evidence": [
            {"chunk_id": "c9", "doc_id": "product:P009", "source_type": "product",
             "title": "ProNoise", "score": 0.88, "dense_sim": 0.70, "citation_index": 1}
        ],
        "snapshot": {
            "products": [{"doc_id": "product:P009", "title": "ProNoise",
                          "price": 899, "inventory": "现货",
                          "version": None, "default_updated_at": "2026-06-12"}],
            "policies": [],
        },
        "case_id": "sc_test_price", "ts": "2026-06-12T00:00:00+00:00",
        "freshness": None, "trace": [], "grounding_ratio": None,
        "citation_ok": None, "consistency_verdict": None, "confidence": 0.7,
    }
    patterns = case_memory._case_patterns(case)
    assert "price_constraint_ignored" in patterns, f"got {patterns}"
    print("PASS test_case_patterns_detects_price")


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
    print(f"\nAll {len(fns)} case_memory tests passed.")


if __name__ == "__main__":
    _run_all()
