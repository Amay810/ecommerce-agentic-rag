# -*- coding: utf-8 -*-
"""CaseMemory (B-lite): lightweight semantic memory over SupportCase history.

Architecture
────────────
  Offline build  : read cases from SQLite → generate memory_text per case
                   → embed (same SentenceTransformer as the retriever) → persist:
                     CASE_MEMORY_EMBED_PATH  (float32 .npy, one row per memory)
                     CASE_MEMORY_DB_PATH     (SQLite case_memory table, metadata)

  Online search  : embed incoming query → cosine sim vs stored memories
                   → threshold filter → top-k → return a memory_hint dict

The hint is advisory in v1: it is logged in the agent trace and included in the
SupportCase result, but does not override retrieval or answer generation. It seeds
the "action prior" story for Q3 (price filter) and Q28 (query decomposition).

CLI
───
  python -m ecommerce_rag.case_memory build          # build/rebuild memory
  python -m ecommerce_rag.case_memory search "query" # ad-hoc search
  python -m ecommerce_rag.case_memory info           # show memory stats
"""

import argparse
import json
import os
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import config, failure_memory, store

# ── config knobs ──────────────────────────────────────────────────────────────

CASE_MEMORY_DB_PATH = Path(
    os.environ.get("ERAG_MEMORY_DB", config.LOG_DIR / "case_memory.db")
)
CASE_MEMORY_EMBED_PATH = Path(
    os.environ.get("ERAG_MEMORY_EMBED", config.LOG_DIR / "case_memory.npy")
)
CASE_MEMORY_SIM_THRESHOLD = float(
    os.environ.get("ERAG_MEMORY_SIM_THRESHOLD", "0.70")
)
CASE_MEMORY_TOP_K = int(os.environ.get("ERAG_MEMORY_TOP_K", "3"))
CASE_MEMORY_ENABLED = os.environ.get("ERAG_CASE_MEMORY", "0") == "1"

# ── data class ────────────────────────────────────────────────────────────────

@dataclass
class MemoryHint:
    matched: bool
    top_sim: float = 0.0
    avoid_patterns: list[str] = field(default_factory=list)
    suggested_action: str | None = None  # "apply_price_filter" | "use_query_decomposition" | None
    memories: list[dict] = field(default_factory=list)
    note: str = ""


# ── memory text generation ────────────────────────────────────────────────────

def _case_patterns(case: dict) -> list[str]:
    """Run per-case failure pattern detectors and return matched pattern type strings."""
    patterns: list[str] = []
    single = [case]
    for detector in (
        failure_memory._detect_price_constraint,
        failure_memory._detect_compound_recall_gap,
        failure_memory._detect_stale_data,
        failure_memory._detect_zero_retrieval,
    ):
        results = detector(single)
        for r in results:
            if r.count > 0:
                patterns.append(r.pattern_type)
    return patterns


def _make_memory_text(case: dict, patterns: list[str]) -> str:
    """Generate a compact, embedding-friendly representation of a SupportCase experience."""
    parts = [
        f"意图:{case.get('intent','')}",
        f"查询:{case.get('query','')}",
        f"结果:{case.get('action','')}",
    ]
    if patterns:
        parts.append(f"失败模式:{' '.join(patterns)}")
    doc_ids = [
        e.get("doc_id", "")
        for e in (case.get("evidence") or [])[:3]
        if e.get("source_type") == "product"
    ]
    if doc_ids:
        parts.append(f"文档:{' '.join(doc_ids)}")
    fr_status = (case.get("freshness") or {}).get("status")
    if fr_status and fr_status not in ("n/a", "fresh"):
        parts.append(f"新鲜度:{fr_status}")
    return " | ".join(parts)


# ── SQLite memory store ───────────────────────────────────────────────────────

_MEM_SCHEMA = {
    "mem_id": "TEXT PRIMARY KEY",      # embedding row index as text (stable after build)
    "case_id": "TEXT",
    "intent": "TEXT",
    "action": "TEXT",
    "query": "TEXT",
    "memory_text": "TEXT",
    "patterns_json": "TEXT",           # JSON list of pattern_type strings
    "embedding_idx": "INTEGER",        # row in case_memory.npy
    "created_at": "TEXT",
}


def _connect_mem(path: Path | None = None) -> sqlite3.Connection:
    p = Path(path or CASE_MEMORY_DB_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p)
    conn.row_factory = sqlite3.Row
    return conn


def _init_mem_db(conn: sqlite3.Connection) -> None:
    cols = ", ".join(f"{k} {v}" for k, v in _MEM_SCHEMA.items())
    conn.execute(f"CREATE TABLE IF NOT EXISTS case_memory ({cols})")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_intent ON case_memory(intent)")
    conn.commit()


def _clear_mem_db(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM case_memory")
    conn.commit()


# ── offline build ─────────────────────────────────────────────────────────────

def build(
    model=None,
    max_cases: int = 500,
    db_path: Path | None = None,
    mem_db_path: Path | None = None,
    embed_path: Path | None = None,
) -> int:
    """Embed SupportCase history and save memory index.

    model: a SentenceTransformer instance (or None to load from config.EMBED_MODEL)
    Returns the number of memories built.
    """
    import numpy as np

    if model is None:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(config.EMBED_MODEL)

    # Load all cases from the support store
    all_rows = store.recent(max_cases, db_path)
    if not all_rows:
        print("No cases in store — run the agent first.")
        return 0

    cases = [failure_memory._parse_row(r) for r in all_rows]

    texts: list[str] = []
    metadata: list[dict] = []
    for idx, c in enumerate(cases):
        patterns = _case_patterns(c) if c.get("needs_review") else []
        text = _make_memory_text(c, patterns)
        texts.append(text)
        metadata.append({
            "mem_id": str(idx),
            "case_id": c.get("case_id", ""),
            "intent": c.get("intent", ""),
            "action": c.get("action", ""),
            "query": c.get("query", ""),
            "memory_text": text,
            "patterns_json": json.dumps(patterns, ensure_ascii=False),
            "embedding_idx": idx,
            "created_at": c.get("ts", ""),
        })

    print(f"Embedding {len(texts)} cases…")
    embs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True)

    ep = Path(embed_path or CASE_MEMORY_EMBED_PATH)
    ep.parent.mkdir(parents=True, exist_ok=True)
    np.save(ep, embs.astype("float32"))
    print(f"Saved embeddings → {ep}")

    conn = _connect_mem(mem_db_path or CASE_MEMORY_DB_PATH)
    try:
        _init_mem_db(conn)
        _clear_mem_db(conn)
        cols = list(_MEM_SCHEMA.keys())
        placeholders = ", ".join("?" for _ in cols)
        for m in metadata:
            conn.execute(
                f"INSERT INTO case_memory ({', '.join(cols)}) VALUES ({placeholders})",
                [m.get(c) for c in cols],
            )
        conn.commit()
    finally:
        conn.close()

    print(f"Saved memory index ({len(metadata)} entries) → {mem_db_path or CASE_MEMORY_DB_PATH}")
    return len(metadata)


# ── online search ─────────────────────────────────────────────────────────────

_cached_embs = None  # module-level cache: loaded once per process


def _load_embs(embed_path: Path | None = None):
    global _cached_embs
    if _cached_embs is None:
        import numpy as np
        ep = Path(embed_path or CASE_MEMORY_EMBED_PATH)
        if ep.exists():
            _cached_embs = np.load(ep)
    return _cached_embs


def search(
    query: str,
    intent: str | None = None,
    model=None,
    n: int = CASE_MEMORY_TOP_K,
    threshold: float = CASE_MEMORY_SIM_THRESHOLD,
    mem_db_path: Path | None = None,
    embed_path: Path | None = None,
) -> MemoryHint:
    """Return a MemoryHint for the given query (advisory only in v1).

    model: SentenceTransformer with encode(); pass in from agent to avoid reloading.
    """
    import numpy as np

    embs = _load_embs(embed_path)
    if embs is None or not Path(mem_db_path or CASE_MEMORY_DB_PATH).exists():
        return MemoryHint(matched=False)

    if model is None:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(config.EMBED_MODEL)

    q_emb = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
    sims = embs @ q_emb  # cosine similarity (embeddings are L2-normalized)

    above = sims >= threshold
    if not above.any():
        return MemoryHint(matched=False)

    top_idxs = [int(i) for i in np.argsort(-sims) if above[i]][:n]

    # Load metadata from SQLite
    conn = _connect_mem(mem_db_path)
    try:
        _init_mem_db(conn)
        rows = []
        for idx in top_idxs:
            r = conn.execute(
                "SELECT * FROM case_memory WHERE embedding_idx = ?", (idx,)
            ).fetchone()
            if r:
                rows.append((dict(r), float(sims[idx])))
    finally:
        conn.close()

    if not rows:
        return MemoryHint(matched=False)

    memories = []
    all_patterns: list[str] = []
    for row, sim in rows:
        pats = json.loads(row.get("patterns_json") or "[]")
        all_patterns.extend(pats)
        memories.append({
            "query": row["query"],
            "intent": row["intent"],
            "action": row["action"],
            "patterns": pats,
            "sim": round(sim, 4),
        })

    unique_patterns = list(dict.fromkeys(all_patterns))  # dedup, preserve order

    suggested: str | None = None
    if "price_constraint_ignored" in unique_patterns:
        suggested = "apply_price_filter"
    elif "compound_query_recall_gap" in unique_patterns:
        suggested = "use_query_decomposition"

    top_sim = rows[0][1]
    note_parts = []
    if suggested == "apply_price_filter":
        note_parts.append("历史相似查询存在价格约束忽略问题，建议在检索后施加预算过滤。")
    elif suggested == "use_query_decomposition":
        note_parts.append("历史相似查询存在复合实体漏召问题，建议拆分子查询分别检索。")
    if unique_patterns and not suggested:
        note_parts.append(f"历史相似查询已知失败模式：{', '.join(unique_patterns)}。")

    return MemoryHint(
        matched=True,
        top_sim=top_sim,
        avoid_patterns=unique_patterns,
        suggested_action=suggested,
        memories=memories,
        note=" ".join(note_parts),
    )


# ── agent helper ──────────────────────────────────────────────────────────────

def hint_to_trace(hint: MemoryHint) -> str | None:
    """Format a MemoryHint as a trace line for agent logging."""
    if not hint.matched:
        return None
    pats = f"pattern={','.join(hint.avoid_patterns)}" if hint.avoid_patterns else ""
    action = f" suggested={hint.suggested_action}" if hint.suggested_action else ""
    return f"记忆先验：sim={hint.top_sim:.3f} {pats}{action}"


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="CaseMemory: build or search semantic memory.")
    sub = parser.add_subparsers(dest="cmd")

    b = sub.add_parser("build", help="embed all SupportCases and save memory index")
    b.add_argument("--max-cases", type=int, default=500)
    b.add_argument("--db", metavar="PATH", help="support.db path")
    b.add_argument("--mem-db", metavar="PATH")
    b.add_argument("--embed-path", metavar="PATH")

    s = sub.add_parser("search", help="search memory for a query")
    s.add_argument("query")
    s.add_argument("--intent", default=None)
    s.add_argument("--n", type=int, default=3)
    s.add_argument("--threshold", type=float, default=CASE_MEMORY_SIM_THRESHOLD)

    sub.add_parser("info", help="show memory stats")

    args = parser.parse_args()

    if args.cmd == "build":
        n = build(max_cases=args.max_cases,
                  db_path=Path(args.db) if args.db else None,
                  mem_db_path=Path(args.mem_db) if args.mem_db else None,
                  embed_path=Path(args.embed_path) if args.embed_path else None)
        print(f"Built {n} memory entries.")

    elif args.cmd == "search":
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(config.EMBED_MODEL)
        hint = search(args.query, intent=args.intent, model=model,
                      n=args.n, threshold=args.threshold)
        if not hint.matched:
            print("No similar cases found above threshold.")
        else:
            print(f"matched=True  top_sim={hint.top_sim:.3f}")
            print(f"avoid_patterns: {hint.avoid_patterns}")
            print(f"suggested_action: {hint.suggested_action}")
            print(f"note: {hint.note}")
            for m in hint.memories:
                print(f"  sim={m['sim']}  [{m['intent']}] {m['action']}  Q: {m['query']}")
                if m["patterns"]:
                    print(f"    patterns: {m['patterns']}")

    elif args.cmd == "info":
        if not CASE_MEMORY_DB_PATH.exists():
            print("Memory not built yet. Run: python -m ecommerce_rag.case_memory build")
        else:
            conn = _connect_mem()
            try:
                _init_mem_db(conn)
                n = conn.execute("SELECT COUNT(*) FROM case_memory").fetchone()[0]
                by_intent = conn.execute(
                    "SELECT intent, COUNT(*) AS n FROM case_memory GROUP BY intent ORDER BY n DESC"
                ).fetchall()
                print(f"Memory entries: {n}")
                print(f"Embed file: {CASE_MEMORY_EMBED_PATH} (exists={CASE_MEMORY_EMBED_PATH.exists()})")
                for row in by_intent:
                    print(f"  intent={row[0]:15s}  n={row[1]}")
            finally:
                conn.close()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
