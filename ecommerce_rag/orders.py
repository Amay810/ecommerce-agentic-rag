"""Deterministic SQLite retail environment used by the agent harness."""

from __future__ import annotations

import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path


ORDER_STATES = ("pending", "processed", "delivered", "cancelled")


def connect(path: Path | str) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(path: Path | str) -> None:
    conn = connect(path)
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
              user_id TEXT PRIMARY KEY, name TEXT NOT NULL, verification_code TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS orders (
              order_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, product_id TEXT NOT NULL,
              status TEXT NOT NULL, ordered_at TEXT NOT NULL, delivered_at TEXT,
              opened INTEGER NOT NULL DEFAULT 0, quality_issue INTEGER NOT NULL DEFAULT 0,
              inventory_status TEXT NOT NULL DEFAULT 'available',
              return_status TEXT, version INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS handoffs (
              handoff_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, order_id TEXT,
              reason TEXT NOT NULL, created_at TEXT NOT NULL
            );
            """
        )
        columns = {row[1] for row in conn.execute("PRAGMA table_info(orders)")}
        if "inventory_status" not in columns:
            conn.execute("ALTER TABLE orders ADD COLUMN inventory_status TEXT NOT NULL DEFAULT 'available'")
        conn.commit()
    finally:
        conn.close()


def seed_database(path: Path | str, users: int = 1000, orders: int = 10000, seed: int = 20260720) -> dict:
    """Create a reproducible environment. Existing rows are replaced deliberately."""
    init_db(path)
    rng = random.Random(seed)
    today = date(2026, 7, 20)
    conn = connect(path)
    try:
        conn.execute("DELETE FROM handoffs")
        conn.execute("DELETE FROM orders")
        conn.execute("DELETE FROM users")
        conn.executemany(
            "INSERT INTO users(user_id,name,verification_code) VALUES(?,?,?)",
            [(f"U{i:04d}", f"User {i}", f"{(i * 7919) % 1000000:06d}") for i in range(1, users + 1)],
        )
        rows = []
        for i in range(1, orders + 1):
            uid = f"U{rng.randint(1, users):04d}"
            state = ORDER_STATES[(i - 1) % len(ORDER_STATES)]
            ordered = today - timedelta(days=rng.randint(1, 90))
            delivered = ordered + timedelta(days=rng.randint(1, 5)) if state == "delivered" else None
            rows.append(
                (f"O{i:06d}", uid, f"P{rng.randint(1, 5000):05d}", state, ordered.isoformat(),
                 delivered.isoformat() if delivered else None, int(i % 5 == 0), int(i % 11 == 0),
                 "out_of_stock" if i % 17 == 0 else "available", None, 0)
            )
        conn.executemany(
            """INSERT INTO orders(order_id,user_id,product_id,status,ordered_at,delivered_at,
               opened,quality_issue,inventory_status,return_status,version) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
            rows,
        )
        conn.commit()
        return {"users": users, "orders": orders, "seed": seed}
    finally:
        conn.close()


def get_order(path: Path | str, order_id: str) -> dict | None:
    conn = connect(path)
    try:
        row = conn.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def snapshot(path: Path | str, order_ids: list[str] | None = None) -> dict:
    conn = connect(path)
    try:
        if order_ids:
            marks = ",".join("?" for _ in order_ids)
            rows = conn.execute(f"SELECT * FROM orders WHERE order_id IN ({marks}) ORDER BY order_id", order_ids)
        else:
            rows = conn.execute("SELECT * FROM orders ORDER BY order_id")
        return {r["order_id"]: dict(r) for r in rows}
    finally:
        conn.close()
