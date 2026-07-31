"""Structured Case Memory: SQL retrieval + advisory action priors.

Memory never expands TaskProgress allowlists, never supplies credentials or
write authority, and never blocks the main task when retrieval fails.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .agent_case import (
    AgentCase,
    admit_for_memory,
    case_id_for,
    progress_signature_from_progress,
    sanitize_case,
)
from .agent_case_store import get_case, insert_case, query_memory_candidates
from .domain import GradeResult, Trajectory


@dataclass(frozen=True)
class MemoryAdvice:
    preferred_actions: tuple[str, ...] = ()
    avoid_actions: tuple[str, ...] = ()
    successful_cases: int = 0
    failed_cases: int = 0
    applicable_pattern: str | None = None
    retrieved_case_ids: tuple[str, ...] = ()
    matched: bool = False
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryTrace:
    memory_query: dict[str, Any] = field(default_factory=dict)
    retrieved_case_ids: list[str] = field(default_factory=list)
    memory_advice: dict[str, Any] = field(default_factory=dict)
    advice_used: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _action_key(action: dict[str, Any]) -> str | None:
    action_type = action.get("action_type")
    if action_type == "tool_call":
        return action.get("tool_name")
    if action_type == "handoff":
        return "handoff"
    if action_type == "final_answer":
        if action.get("requires_user_response"):
            requested = action.get("requested_input_type")
            return f"ask_user:{requested or 'unknown'}"
        return "final_answer"
    return None


def build_memory_advice(
    progress: dict[str, Any],
    *,
    db_path: Path | str | None = None,
    limit: int = 8,
) -> MemoryAdvice:
    allowed = list(progress.get("allowed_next_actions") or progress.get("allowed_actions") or [])
    workflow = str(progress.get("workflow") or "")
    if workflow != "return_resolution" or not allowed:
        return MemoryAdvice(note="memory_not_applicable")
    try:
        cases = query_memory_candidates(
            workflow=workflow,
            pending=progress.get("pending") or (),
            guard_state=progress.get("guard_state"),
            blocked_by=progress.get("blocked_by"),
            allowed_actions=allowed,
            db_path=Path(db_path) if db_path else None,
            limit=limit,
        )
    except Exception as exc:  # never block the agent
        return MemoryAdvice(note=f"memory_retrieval_failed:{type(exc).__name__}")

    preferred: list[str] = []
    avoid: list[str] = []
    patterns: list[str] = []
    success_n = fail_n = 0
    for case in cases:
        if case.success:
            success_n += 1
            key = _action_key(case.chosen_action) if case.chosen_action else None
            # Successful cases may store the *corrected* preferred pattern.
            if case.reusable_pattern and "verification" in case.reusable_pattern:
                for item in allowed:
                    if item.startswith("ask_user:verification"):
                        preferred.append(item)
            if key and key in allowed:
                preferred.append(key)
            elif case.allowed_actions:
                preferred.extend(action for action in case.allowed_actions if action in allowed)
        else:
            fail_n += 1
            key = _action_key(case.chosen_action) if case.chosen_action else None
            if key:
                avoid.append(key)
        # Successful cases may also carry avoid_patterns (e.g. prior bad handoffs).
        if case.avoid_pattern:
            avoid.extend(part.strip() for part in case.avoid_pattern.replace(";", "|").split("|") if part.strip())
        if case.reusable_pattern:
            patterns.append(case.reusable_pattern)

    # Memory may only recommend actions already legal for this progress.
    preferred_f = tuple(dict.fromkeys(action for action in preferred if action in allowed))
    avoid_f = tuple(dict.fromkeys(
        action for action in avoid
        if action not in preferred_f and action != "create_return_request"
    ))
    return MemoryAdvice(
        preferred_actions=preferred_f,
        avoid_actions=avoid_f,
        successful_cases=success_n,
        failed_cases=fail_n,
        applicable_pattern=patterns[0] if patterns else None,
        retrieved_case_ids=tuple(case.case_id for case in cases),
        matched=bool(cases),
        note="ok" if cases else "no_match",
    )


def advice_used(advice: MemoryAdvice | dict[str, Any], chosen: dict[str, Any]) -> bool:
    payload = advice.to_dict() if isinstance(advice, MemoryAdvice) else advice
    key = _action_key(chosen)
    preferred = payload.get("preferred_actions") or []
    return bool(key and key in preferred)


def candidate_from_trajectory(
    *,
    task_id: str,
    split: str,
    user_goal: str,
    trajectory: Trajectory,
    grade: GradeResult | dict[str, Any],
    failure_owner: str = "none",
    first_causal_failure: str = "",
    reusable_pattern: str = "",
    avoid_pattern: str = "",
) -> AgentCase:
    grade_payload = grade.to_dict() if hasattr(grade, "to_dict") else dict(grade)
    progress = {}
    if trajectory.progress_spans:
        progress = {
            key: trajectory.progress_spans[0].get(key)
            for key in (
                "workflow", "completed", "pending", "blocked_by",
                "allowed_next_actions", "requested_input_type", "guard_state",
                "eligible", "cancelled",
            )
            if key in trajectory.progress_spans[0]
        }
    chosen = dict(trajectory.actions[0]) if trajectory.actions else {}
    if chosen.get("requires_user_response") and progress.get("requested_input_type"):
        chosen.setdefault("requested_input_type", progress.get("requested_input_type"))
    tool_type = ""
    if trajectory.tool_calls:
        tool_type = trajectory.tool_calls[0].name
    success = bool(grade_payload.get("success") or grade_payload.get("operational_success"))
    terminal = {
        "final_answer": trajectory.final_answer,
        "final_state": trajectory.final_state,
        "illegal_state_change": bool(grade_payload.get("illegal_state_change")),
        "failed_closed": trajectory.failed_closed,
    }
    case = AgentCase(
        case_id=case_id_for(
            task_id,
            first_causal_failure or ("success" if success else "failure"),
            commit=hashlib.sha1(trajectory.trajectory_id.encode()).hexdigest()[:8],
        ),
        split=split,
        training_approved=False,
        memory_status="candidate",
        memory_approved=False,
        user_goal=user_goal,
        task_id=task_id,
        progress_before=progress,
        allowed_actions=list(progress.get("allowed_next_actions") or []),
        chosen_action=dict(chosen),
        tool_result_type=tool_type,
        terminal_state=terminal,
        success=success,
        first_causal_failure=first_causal_failure,
        failure_owner=failure_owner if not success else "none",
        reusable_pattern=reusable_pattern,
        avoid_pattern=avoid_pattern,
        workflow=str(progress.get("workflow") or ""),
        progress_signature=progress_signature_from_progress(progress) if progress else "",
        source={"trajectory_id": trajectory.trajectory_id, "split": split},
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    return sanitize_case(case)


def write_candidate(
    case: AgentCase,
    *,
    db_path: Path | str | None = None,
    approve: bool = False,
) -> tuple[AgentCase, dict[str, Any]]:
    cleaned = sanitize_case(case)
    admission = admit_for_memory(cleaned, approve=approve)
    cleaned.memory_status = admission.status
    cleaned.memory_approved = admission.accepted and admission.status == "approved"
    # Always persist candidates/rejects for audit; approved only when allowed.
    insert_case(cleaned, Path(db_path) if db_path else None)
    return cleaned, admission.to_dict()


def approve_case(case_id: str, *, db_path: Path | str | None = None) -> AgentCase:
    case = get_case(case_id, Path(db_path) if db_path else None)
    if case is None:
        raise KeyError(case_id)
    admission = admit_for_memory(case, approve=True)
    if not admission.accepted:
        raise ValueError(f"cannot approve: {admission.reasons}")
    case.memory_status = "approved"
    case.memory_approved = True
    insert_case(case, Path(db_path) if db_path else None)
    return case


def memory_enabled() -> bool:
    return os.environ.get("ERAG_AGENT_CASE_MEMORY", "0") == "1"
