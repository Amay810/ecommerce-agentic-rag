"""SQLite store for flywheel AgentCases (stdlib only).

Formal ``dev``/``locked`` audit cases may be inserted for schema checks, but
``memory_status`` must never be ``approved`` for those splits.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from . import config
from .agent_case import AgentCase, MEMORY_STATUSES


_SCHEMA = {
    "case_id": "TEXT PRIMARY KEY",
    "source_split": "TEXT NOT NULL",
    "workflow": "TEXT",
    "progress_signature": "TEXT",
    "pending": "TEXT",
    "blocked_by": "TEXT",
    "guard_state": "TEXT",
    "eligible": "TEXT",
    "cancelled": "INTEGER",
    "allowed_actions_json": "TEXT",
    "chosen_action_json": "TEXT",
    "tool_result_type": "TEXT",
    "terminal_state_json": "TEXT",
    "success": "INTEGER",
    "failure_owner": "TEXT",
    "reusable_pattern": "TEXT",
    "avoid_pattern": "TEXT",
    "paired_replay_result_json": "TEXT",
    "training_approved": "INTEGER",
    "memory_status": "TEXT",
    "memory_approved": "INTEGER",
    "created_at": "TEXT",
    "user_goal": "TEXT",
    "task_id": "TEXT",
    "step": "INTEGER",
    "causal_credit": "TEXT",
    "constraint_remapped": "INTEGER",
    "source_hash": "TEXT",
    "payload_json": "TEXT",
}


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = Path(db_path or config.AGENT_CASE_DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    cols = ", ".join(f"{name} {decl}" for name, decl in _SCHEMA.items())
    conn.execute(f"CREATE TABLE IF NOT EXISTS agent_cases ({cols})")
    existing = {row["name"] for row in conn.execute("PRAGMA table_info(agent_cases)")}
    for name, decl in _SCHEMA.items():
        if name not in existing:
            conn.execute(
                f"ALTER TABLE agent_cases ADD COLUMN {name} {decl.replace(' PRIMARY KEY', '').replace(' NOT NULL', '')}"
            )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_agent_cases_memory "
        "ON agent_cases(memory_status, workflow, guard_state, pending)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_agent_cases_signature "
        "ON agent_cases(progress_signature, memory_status)"
    )
    conn.commit()


def _row_from_case(case: AgentCase) -> dict[str, Any]:
    progress = case.progress_before or {}
    return {
        "case_id": case.case_id,
        "source_split": case.split,
        "workflow": case.workflow or progress.get("workflow") or "",
        "progress_signature": case.progress_signature or "",
        "pending": ",".join(
            str(item) for item in (progress.get("pending") or ())
            if not isinstance(progress.get("pending"), str)
        ) if not isinstance(progress.get("pending"), str) else str(progress.get("pending") or ""),
        "blocked_by": (
            progress.get("blocked_by")
            if not isinstance(progress.get("blocked_by"), (list, tuple))
            else ",".join(str(item) for item in progress.get("blocked_by") or ())
        ),
        "guard_state": progress.get("guard_state"),
        "eligible": "" if progress.get("eligible") is None else str(progress.get("eligible")),
        "cancelled": int(bool(progress.get("cancelled"))),
        "allowed_actions_json": json.dumps(case.allowed_actions, ensure_ascii=False),
        "chosen_action_json": json.dumps(
            case.executed_action or case.chosen_action, ensure_ascii=False
        ),
        "tool_result_type": case.tool_result_type or "",
        "terminal_state_json": json.dumps(case.terminal_state, ensure_ascii=False),
        "success": int(bool(case.success)),
        "failure_owner": case.failure_owner,
        "reusable_pattern": case.reusable_pattern or "",
        "avoid_pattern": case.avoid_pattern or "",
        "paired_replay_result_json": json.dumps(case.paired_replay_result, ensure_ascii=False),
        "training_approved": int(bool(case.training_approved)),
        "memory_status": case.memory_status,
        "memory_approved": int(bool(case.memory_approved)),
        "created_at": case.created_at or "",
        "user_goal": case.user_goal,
        "task_id": case.task_id,
        "step": case.step,
        "causal_credit": case.causal_credit,
        "constraint_remapped": int(bool(case.constraint_remapped)),
        "source_hash": case.source_hash or "",
        "payload_json": json.dumps(case.to_dict(), ensure_ascii=False),
    }


def insert_case(case: AgentCase, db_path: Path | None = None) -> None:
    if case.memory_status not in MEMORY_STATUSES:
        raise ValueError(f"invalid memory_status: {case.memory_status}")
    if case.split in {"dev", "locked"} and case.memory_status == "approved":
        raise ValueError("dev/locked cases cannot be memory-approved")
    if case.split in {"dev", "locked"} and case.memory_approved:
        raise ValueError("dev/locked cases cannot set memory_approved=true")
    row = _row_from_case(case)
    columns = list(_SCHEMA.keys())
    placeholders = ", ".join("?" for _ in columns)
    conn = connect(db_path)
    try:
        init_db(conn)
        conn.execute(
            f"INSERT OR REPLACE INTO agent_cases ({', '.join(columns)}) VALUES ({placeholders})",
            [row[name] for name in columns],
        )
        conn.commit()
    finally:
        conn.close()


def get_case(case_id: str, db_path: Path | None = None) -> AgentCase | None:
    conn = connect(db_path)
    try:
        init_db(conn)
        row = conn.execute(
            "SELECT payload_json FROM agent_cases WHERE case_id=?", (case_id,)
        ).fetchone()
    finally:
        conn.close()
    if not row:
        return None
    return AgentCase.from_dict(json.loads(row["payload_json"]))


def list_cases(
    *,
    memory_status: str | None = None,
    source_split: str | None = None,
    db_path: Path | None = None,
) -> list[AgentCase]:
    clauses: list[str] = []
    params: list[Any] = []
    if memory_status is not None:
        clauses.append("memory_status=?")
        params.append(memory_status)
    if source_split is not None:
        clauses.append("source_split=?")
        params.append(source_split)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    conn = connect(db_path)
    try:
        init_db(conn)
        rows = conn.execute(
            f"SELECT payload_json FROM agent_cases {where} ORDER BY created_at",
            params,
        ).fetchall()
    finally:
        conn.close()
    return [AgentCase.from_dict(json.loads(row["payload_json"])) for row in rows]


def query_memory_candidates(
    *,
    workflow: str,
    pending: Iterable[str] | None = None,
    guard_state: str | None = None,
    blocked_by: str | None = None,
    allowed_actions: Iterable[str] | None = None,
    db_path: Path | None = None,
    limit: int = 20,
) -> list[AgentCase]:
    """Return only memory-approved, non-dev/locked cases with matching workflow."""
    pending_key = ",".join(pending or ())
    allowed = set(allowed_actions or ())
    conn = connect(db_path)
    try:
        init_db(conn)
        rows = conn.execute(
            """
            SELECT payload_json, success, allowed_actions_json, pending, guard_state, blocked_by
            FROM agent_cases
            WHERE memory_status='approved'
              AND memory_approved=1
              AND source_split NOT IN ('dev', 'locked')
              AND workflow=?
            ORDER BY success DESC, created_at DESC
            LIMIT ?
            """,
            (workflow, max(limit * 5, limit)),
        ).fetchall()
    finally:
        conn.close()

    scored: list[tuple[int, AgentCase]] = []
    for row in rows:
        case = AgentCase.from_dict(json.loads(row["payload_json"]))
        score = 0
        if guard_state and row["guard_state"] == guard_state:
            score += 4
        if pending_key and row["pending"] == pending_key:
            score += 4
        if blocked_by is not None and row["blocked_by"] == blocked_by:
            score += 2
        case_allowed = set(json.loads(row["allowed_actions_json"] or "[]"))
        if allowed and case_allowed & allowed:
            score += 3
        elif allowed and not case_allowed:
            continue
        if score == 0 and (guard_state or pending_key):
            # Require at least one state match when query is specific.
            continue
        if case.success:
            score += 2
        if case.reusable_pattern:
            score += 1
        scored.append((score, case))
    scored.sort(key=lambda item: (-item[0], item[1].case_id))
    return [case for _, case in scored[:limit]]
