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
    provenance_hash,
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
    policy_followed_advice: bool = False
    advice_used: bool = False  # alias of policy_followed_advice (pre-constraint)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _action_key(action: dict[str, Any] | None) -> str | None:
    if not action:
        return None
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


def _step_outcome(executed: dict[str, Any], *, remapped: bool, allowed: list[str]) -> str:
    key = _action_key(executed)
    if remapped:
        return "constraint_remapped"
    if key and key in allowed:
        return f"allowed:{key}"
    if key:
        return f"out_of_allowlist:{key}"
    return "unknown"


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
        # Remapped constraint recoveries are not policy-success experience.
        if case.constraint_remapped and case.causal_credit == "constraint":
            fail_n += 1
            raw_key = _action_key(case.raw_policy_action)
            if raw_key:
                avoid.append(raw_key)
            if case.avoid_pattern:
                avoid.extend(
                    part.strip()
                    for part in case.avoid_pattern.replace(";", "|").split("|")
                    if part.strip()
                )
            continue
        if case.success:
            success_n += 1
            action = case.executed_action or case.chosen_action
            key = _action_key(action)
            if case.reusable_pattern and "verification" in case.reusable_pattern:
                for item in allowed:
                    if item.startswith("ask_user:verification"):
                        preferred.append(item)
            if key and key in allowed:
                preferred.append(key)
            elif case.allowed_actions:
                preferred.extend(a for a in case.allowed_actions if a in allowed)
        else:
            fail_n += 1
            key = _action_key(case.raw_policy_action or case.chosen_action)
            if key:
                avoid.append(key)
        if case.avoid_pattern:
            avoid.extend(
                part.strip()
                for part in case.avoid_pattern.replace(";", "|").split("|")
                if part.strip()
            )
        if case.reusable_pattern:
            patterns.append(case.reusable_pattern)

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
    """Whether *raw policy* action matched memory preferred (pre-constraint)."""
    payload = advice.to_dict() if isinstance(advice, MemoryAdvice) else advice
    key = _action_key(chosen)
    preferred = payload.get("preferred_actions") or []
    return bool(key and key in preferred)


def _progress_from_span(span: dict[str, Any]) -> dict[str, Any]:
    return {
        key: span.get(key)
        for key in (
            "workflow", "completed", "pending", "blocked_by",
            "allowed_next_actions", "requested_input_type", "guard_state",
            "eligible", "cancelled",
        )
        if key in span
    }


def _memory_span_for_step(trajectory: Trajectory, step: int) -> dict[str, Any]:
    for span in trajectory.memory_spans or []:
        if span.get("step") == step:
            return span
    return {}


def _tool_result_type_for_step(
    trajectory: Trajectory,
    index: int,
    executed: dict[str, Any],
) -> str:
    """Align tool_result_type to this decision step; leave empty if unreliable."""
    action_type = (executed or {}).get("action_type")
    if action_type not in {"tool_call", "handoff"}:
        return ""
    ordinal = -1
    for i, action in enumerate(trajectory.actions or []):
        if action.get("action_type") in {"tool_call", "handoff"}:
            ordinal += 1
            if i == index:
                break
    else:
        return ""
    calls = list(trajectory.tool_calls or [])
    if ordinal < 0 or ordinal >= len(calls):
        return ""
    call = calls[ordinal]
    name = getattr(call, "name", None) or (call.get("name") if isinstance(call, dict) else None)
    if not name:
        return ""
    if action_type == "tool_call":
        expected = (executed or {}).get("tool_name")
        if expected and name != expected:
            return ""
    elif action_type == "handoff" and name != "escalate_to_human":
        return ""
    return str(name)


def candidates_from_trajectory(
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
) -> list[AgentCase]:
    """One AgentCase per decision step (progress span / action)."""
    grade_payload = grade.to_dict() if hasattr(grade, "to_dict") else dict(grade)
    terminal_success = bool(
        grade_payload.get("success") or grade_payload.get("operational_success")
    )
    terminal_outcome = {
        "success": terminal_success,
        "illegal_state_change": bool(grade_payload.get("illegal_state_change")),
        "failed_closed": trajectory.failed_closed,
        "final_answer": trajectory.final_answer,
    }
    terminal_state = {
        "final_answer": trajectory.final_answer,
        "final_state": trajectory.final_state,
        "illegal_state_change": bool(grade_payload.get("illegal_state_change")),
        "failed_closed": trajectory.failed_closed,
    }
    progress_spans = list(trajectory.progress_spans or [])
    actions = list(trajectory.actions or [])
    last_step_index = max(len(actions) - 1, 0)
    cases: list[AgentCase] = []
    traj_digest = hashlib.sha1(trajectory.trajectory_id.encode()).hexdigest()[:8]

    for index, span in enumerate(progress_spans):
        if index >= len(actions):
            break
        step = int(span.get("step", index))
        progress = _progress_from_span(span)
        allowed = list(progress.get("allowed_next_actions") or [])
        executed = dict(actions[index])
        if executed.get("requires_user_response") and progress.get("requested_input_type"):
            executed.setdefault("requested_input_type", progress.get("requested_input_type"))
        mem = _memory_span_for_step(trajectory, step)
        raw = dict(mem.get("raw_policy_action") or executed)
        constrained = dict(mem.get("constrained_action") or executed)
        if mem.get("executed_action"):
            executed = dict(mem["executed_action"])
        remapped = bool(
            mem.get("constraint_remapped")
            or (mem.get("constraint_result") or {}).get("remapped")
        )
        followed = mem.get("policy_followed_advice")
        if followed is None:
            followed = mem.get("advice_used")
        credit = "constraint" if remapped else "policy"
        step_outcome = _step_outcome(executed, remapped=remapped, allowed=allowed)
        executed_key = _action_key(executed)
        step_ok = bool(executed_key and executed_key in allowed) and not remapped
        # Precision over recall: remapped steps never inherit terminal success.
        # Non-remapped last steps, or steps that followed advice, may.
        decision_success = bool(
            terminal_success
            and step_ok
            and (index == last_step_index or followed is True)
        )
        raw_key = _action_key(raw)
        step_avoid = avoid_pattern
        if remapped and raw_key:
            step_avoid = "|".join(filter(None, [avoid_pattern, raw_key]))
        owner = failure_owner
        if decision_success:
            owner = "none"
        elif remapped:
            owner = "policy" if failure_owner == "none" else failure_owner
        elif not terminal_success and failure_owner == "none":
            owner = "policy"
        source = {
            "trajectory_id": trajectory.trajectory_id,
            "split": split,
            "step": step,
            "attribution_source": "harness_decision_writeback",
            "attribution": "decision_level_auto_candidate",
        }
        case = AgentCase(
            case_id=case_id_for(
                f"{task_id}_s{step}",
                first_causal_failure or ("success" if decision_success else "failure"),
                commit=traj_digest,
            ),
            split=split,
            training_approved=False,
            memory_status="candidate",
            memory_approved=False,
            user_goal=user_goal,
            task_id=task_id,
            progress_before=progress,
            allowed_actions=allowed,
            chosen_action=dict(executed),
            raw_policy_action=raw,
            constrained_action=constrained,
            executed_action=dict(executed),
            constraint_remapped=remapped,
            policy_followed_advice=bool(followed) if followed is not None else None,
            step=step,
            step_outcome=step_outcome,
            terminal_outcome=dict(terminal_outcome),
            causal_credit=credit,
            tool_result_type=_tool_result_type_for_step(trajectory, index, executed),
            terminal_state=dict(terminal_state),
            success=decision_success,
            first_causal_failure=first_causal_failure,
            failure_owner=owner,
            reusable_pattern=reusable_pattern if decision_success else "",
            avoid_pattern=step_avoid,
            workflow=str(progress.get("workflow") or ""),
            progress_signature=(
                progress_signature_from_progress(progress) if progress else ""
            ),
            source=source,
            source_hash=provenance_hash(source, task_id, step),
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        cases.append(sanitize_case(case))
    return cases


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
    """Backward-compatible: last decision case from the trajectory."""
    cases = candidates_from_trajectory(
        task_id=task_id,
        split=split,
        user_goal=user_goal,
        trajectory=trajectory,
        grade=grade,
        failure_owner=failure_owner,
        first_causal_failure=first_causal_failure,
        reusable_pattern=reusable_pattern,
        avoid_pattern=avoid_pattern,
    )
    if not cases:
        raise ValueError("trajectory produced no decision cases")
    return cases[-1]


def write_candidate(
    case: AgentCase,
    *,
    db_path: Path | str | None = None,
    approve: bool = False,
    approved_by: str = "",
    approval_reason: str = "",
    require_paired_replay: bool = True,
) -> tuple[AgentCase, dict[str, Any]]:
    cleaned = sanitize_case(case)
    if approve:
        cleaned.approved_by = approved_by or cleaned.approved_by
        cleaned.approval_reason = approval_reason or cleaned.approval_reason
        cleaned.approved_at = datetime.now(timezone.utc).isoformat()
    admission = admit_for_memory(
        cleaned,
        approve=approve,
        approved_by=cleaned.approved_by,
        approval_reason=cleaned.approval_reason,
        require_paired_replay=require_paired_replay,
    )
    cleaned.memory_status = admission.status
    cleaned.memory_approved = admission.accepted and admission.status == "approved"
    insert_case(cleaned, Path(db_path) if db_path else None)
    return cleaned, admission.to_dict()


def write_candidates(
    cases: list[AgentCase],
    *,
    db_path: Path | str | None = None,
) -> list[tuple[AgentCase, dict[str, Any]]]:
    return [write_candidate(case, db_path=db_path, approve=False) for case in cases]


def approve_case(
    case_id: str,
    *,
    db_path: Path | str | None = None,
    approved_by: str,
    approval_reason: str,
    require_paired_replay: bool = True,
) -> AgentCase:
    case = get_case(case_id, Path(db_path) if db_path else None)
    if case is None:
        raise KeyError(case_id)
    case.approved_by = approved_by
    case.approval_reason = approval_reason
    case.approved_at = datetime.now(timezone.utc).isoformat()
    admission = admit_for_memory(
        case,
        approve=True,
        approved_by=approved_by,
        approval_reason=approval_reason,
        require_paired_replay=require_paired_replay,
    )
    if not admission.accepted:
        raise ValueError(f"cannot approve: {admission.reasons}")
    case.memory_status = "approved"
    case.memory_approved = True
    insert_case(case, Path(db_path) if db_path else None)
    return case


def memory_enabled() -> bool:
    return os.environ.get("ERAG_AGENT_CASE_MEMORY", "0") == "1"
