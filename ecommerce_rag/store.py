# -*- coding: utf-8 -*-
"""SQLite persistence for SupportCases (stdlib sqlite3 only, no extra deps).

SQLite is the source of truth for the memory flywheel; telemetry.py keeps the JSONL
mirror. v1 deliberately avoids any vector DB — semantic case reuse comes later (Step 5).
"""

import argparse
import json
import sqlite3
from pathlib import Path

from . import config
from .support_case import SupportCase

# column name -> SQLite type; order defines table layout and insert order
_SCHEMA = {
    "case_id": "TEXT PRIMARY KEY",
    "ts": "TEXT",
    "query": "TEXT",
    "intent": "TEXT",
    "action": "TEXT",
    "grounding_ratio": "REAL",
    "citation_ok": "INTEGER",
    "consistency_verdict": "TEXT",
    "confidence": "REAL",
    "answer": "TEXT",
    "needs_review": "INTEGER",
    "evidence_json": "TEXT",
    "snapshot_json": "TEXT",
    "trace_json": "TEXT",
    "freshness_json": "TEXT",
}


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = Path(db_path or config.SUPPORT_DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    cols = ", ".join(f"{name} {decl}" for name, decl in _SCHEMA.items())
    conn.execute(f"CREATE TABLE IF NOT EXISTS support_cases ({cols})")
    # lightweight migration: add any columns missing from an older DB
    existing = {r["name"] for r in conn.execute("PRAGMA table_info(support_cases)")}
    for name, decl in _SCHEMA.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE support_cases ADD COLUMN {name} {decl.replace(' PRIMARY KEY', '')}")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_needs_review ON support_cases(needs_review)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_intent ON support_cases(intent)")
    conn.commit()


def insert_case(case: SupportCase, db_path: Path | None = None) -> None:
    row = case.to_row()
    columns = list(_SCHEMA.keys())
    placeholders = ", ".join("?" for _ in columns)
    values = [row.get(c) for c in columns]
    conn = connect(db_path)
    try:
        init_db(conn)
        conn.execute(
            f"INSERT OR REPLACE INTO support_cases ({', '.join(columns)}) VALUES ({placeholders})",
            values,
        )
        conn.commit()
    finally:
        conn.close()


def _query(sql: str, params: tuple = (), db_path: Path | None = None) -> list[sqlite3.Row]:
    conn = connect(db_path)
    try:
        init_db(conn)
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


def recent(n: int = 20, db_path: Path | None = None) -> list[sqlite3.Row]:
    return _query("SELECT * FROM support_cases ORDER BY ts DESC LIMIT ?", (n,), db_path)


def needs_review(n: int = 50, db_path: Path | None = None) -> list[sqlite3.Row]:
    return _query(
        "SELECT * FROM support_cases WHERE needs_review = 1 ORDER BY ts DESC LIMIT ?", (n,), db_path
    )


def count(db_path: Path | None = None) -> int:
    return _query("SELECT COUNT(*) AS n FROM support_cases", (), db_path)[0]["n"]


def export_jsonl(path: Path, db_path: Path | None = None) -> int:
    rows = _query("SELECT * FROM support_cases ORDER BY ts", (), db_path)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(_row_to_case_dict(r), ensure_ascii=False) + "\n")
    return len(rows)


def _row_to_case_dict(row: sqlite3.Row) -> dict:
    """Rehydrate a row into the SupportCase-shaped dict (JSON columns decoded)."""
    d = dict(row)
    for f in ("evidence", "snapshot", "trace", "freshness"):
        col = f"{f}_json"
        if col in d:
            d[f] = json.loads(d.pop(col) or "null")
    d["needs_review"] = bool(d.get("needs_review"))
    if d.get("citation_ok") is not None:
        d["citation_ok"] = bool(d["citation_ok"])
    return d


def _fmt(row: sqlite3.Row) -> str:
    flag = "⚠ review" if row["needs_review"] else "ok"
    gr = row["grounding_ratio"]
    gr_s = f"{gr:.2f}" if gr is not None else "-"
    return (
        f"[{row['ts']}] {row['case_id']}\n"
        f"  intent={row['intent']} action={row['action']} ({flag}) "
        f"conf={row['confidence']:.3f} grounding={gr_s}\n"
        f"  Q: {row['query']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect/export the SupportCase store.")
    parser.add_argument("--recent", type=int, metavar="N", help="show N most recent cases")
    parser.add_argument("--needs-review", action="store_true", help="show cases flagged needs_review")
    parser.add_argument("--export", metavar="PATH", help="export all cases to JSONL")
    parser.add_argument("--count", action="store_true", help="print total case count")
    args = parser.parse_args()

    did_something = False
    if args.count:
        print(f"total cases: {count()}")
        did_something = True
    if args.recent:
        for r in recent(args.recent):
            print(_fmt(r))
        did_something = True
    if args.needs_review:
        rows = needs_review()
        print(f"needs_review cases: {len(rows)}")
        for r in rows:
            print(_fmt(r))
        did_something = True
    if args.export:
        n = export_jsonl(Path(args.export))
        print(f"exported {n} cases -> {args.export}")
        did_something = True
    if not did_something:
        parser.print_help()


if __name__ == "__main__":
    main()
