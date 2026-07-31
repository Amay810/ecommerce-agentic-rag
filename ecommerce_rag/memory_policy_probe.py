"""Frozen Memory off/on causal probe: memory_policy_probe_v1.

Does not touch formal 40/40 dev/locked. Answers only whether Memory advice
makes the *raw* Policy choose preferred legal actions more often under the
same TaskProgress + Action Constraint.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .agent_case import AgentCase, progress_signature_from_progress, provenance_hash
from .agent_case_memory import _action_key, build_memory_advice, write_candidate
from .domain import AgentAction, AgentObservation, GradeResult, TaskSpec, Trajectory
from .legacy_closure_benchmark import prepare_database

PROTOCOL = "memory_policy_probe_v1"
FROZEN_PROBE_TASK_SHA256 = "3dc886ba36d36b0ff7bb73268e8114dd6da47063ec6e168f108096c22a7d16fd"
FROZEN_TRAIN_CASE_SHA256 = "3f679e26eeff27342387dd8013dc483018a81d08d9e1f7d0c65bee79ee63fcd9"

PROBE_STATES: tuple[str, ...] = (
    "missing_order_id",
    "missing_verification",
    "missing_return_reason",
    "awaiting_confirmation",
)

PREFERRED_BY_STATE: dict[str, str] = {
    "missing_order_id": "ask_user:order_id",
    "missing_verification": "ask_user:verification_code",
    "missing_return_reason": "ask_user:return_reason",
    "awaiting_confirmation": "ask_user:confirmation",
}

GUARD_BY_STATE: dict[str, str] = {
    "missing_order_id": "read_only",
    "missing_verification": "identity_required",
    "missing_return_reason": "reason_required",
    "awaiting_confirmation": "confirmation_required",
}


@dataclass(frozen=True)
class MemoryProbeTask:
    task_id: str
    state_id: str
    preferred_action: str
    seed: int
    user_id: str
    order_id: str
    initial_message: str
    user_responses: tuple[str, ...]
    database_fixture: str = "eligible"
    split: str = "memory_probe"


@dataclass(frozen=True)
class TrainCaseSpec:
    case_id: str
    state_id: str
    preferred_action: str
    user_goal: str
    order_tag: str
    reusable_pattern: str
    avoid_pattern: str = "handoff|final_answer|create_return_request"


class ScriptedThenLLMPolicy:
    """Run a fixed legal prefix, then delegate to ``llm``."""

    privileged = False

    def __init__(self, prefix: list[AgentAction], llm: Any):
        self.prefix = list(prefix)
        self.llm = llm
        self._index = 0
        self.probe_step: int | None = None
        self.last_trace = None
        self.retry_count = 0
        self.max_parse_retries = getattr(llm, "max_parse_retries", 0)

    def act(self, observation: AgentObservation) -> AgentAction:
        if self._index < len(self.prefix):
            action = self.prefix[self._index]
            self._index += 1
            self.last_trace = {
                "resolution": "scripted_prefix",
                "prefix_index": self._index - 1,
            }
            return action
        if self.probe_step is None:
            self.probe_step = observation.step
        decided = self.llm.act(observation)
        self.last_trace = getattr(self.llm, "last_trace", None)
        self.retry_count = int(getattr(self.llm, "retry_count", 0))
        return decided


def _progress_for_state(state_id: str) -> dict[str, Any]:
    preferred = PREFERRED_BY_STATE[state_id]
    pending = {
        "missing_order_id": ["order_id"],
        "missing_verification": ["identity_verification"],
        "missing_return_reason": ["return_reason"],
        "awaiting_confirmation": ["explicit_confirmation"],
    }[state_id]
    requested = preferred.split(":", 1)[1]
    return {
        "workflow": "return_resolution",
        "pending": pending,
        "blocked_by": "user_input",
        "guard_state": GUARD_BY_STATE[state_id],
        "eligible": True if state_id in {
            "missing_return_reason", "awaiting_confirmation",
        } else None,
        "cancelled": False,
        "allowed_next_actions": [preferred],
        "requested_input_type": requested,
    }


def build_train_case_specs(seed: int = 20260731) -> list[TrainCaseSpec]:
    rng = random.Random(seed)
    patterns = {
        "missing_order_id": "collect order id before any write or handoff",
        "missing_verification": "identity verification must complete first",
        "missing_return_reason": "return reason must be collected before confirmation",
        "awaiting_confirmation": "explicit confirmation required before create_return_request",
    }
    goals = {
        "missing_order_id": (
            "I want to return something but do not have the order number yet",
            "help me start a return, order id still missing",
            "prepare a return but the order number is not ready",
            "return request without an order id so far",
        ),
        "missing_verification": (
            "order {oid} needs a return; reason: wrong size",
            "please return {oid}, return reason: damaged box",
            "order {oid} apply return, reason: dislike color",
            "process return for {oid}; reason: bought by mistake",
        ),
        "missing_return_reason": (
            "order {oid} return after identity passed, reason still missing",
            "identity ok for {oid}; still need a return reason",
            "order {oid} verified, still missing return reason",
            "verified {oid}, waiting on return reason only",
        ),
        "awaiting_confirmation": (
            "order {oid} is eligible; waiting for explicit confirm",
            "ready to submit return for {oid} after user confirms",
            "order {oid} eligible, waiting for user confirmation",
            "eligible return {oid}; confirmation is the next ask",
        ),
    }
    specs: list[TrainCaseSpec] = []
    n = 0
    for state_id in PROBE_STATES:
        for phrase in goals[state_id]:
            n += 1
            oid = f"O{200000 + n:06d}"
            specs.append(TrainCaseSpec(
                case_id=f"ac_mprobe_train_{state_id}_{n:02d}",
                state_id=state_id,
                preferred_action=PREFERRED_BY_STATE[state_id],
                user_goal=phrase.format(oid=oid),
                order_tag=oid,
                reusable_pattern=patterns[state_id],
            ))
            rng.random()
    return specs


def train_case_from_spec(spec: TrainCaseSpec) -> AgentCase:
    progress = _progress_for_state(spec.state_id)
    requested = progress["requested_input_type"]
    action = {
        "action_type": "final_answer",
        "requires_user_response": True,
        "requested_input_type": requested,
        "content": f"please provide {requested}",
    }
    source = {
        "attribution": "human_curated_memory_probe_train",
        "attribution_source": "memory_policy_probe_v1_train_specs",
        "protocol": PROTOCOL,
        "state_id": spec.state_id,
        "order_tag": spec.order_tag,
    }
    return AgentCase(
        case_id=spec.case_id,
        split="train",
        training_approved=False,
        user_goal=spec.user_goal,
        task_id=spec.case_id,
        progress_before=progress,
        allowed_actions=[spec.preferred_action],
        forbidden_actions=["handoff", "final_answer", "create_return_request"],
        chosen_action=dict(action),
        executed_action=dict(action),
        raw_policy_action=dict(action),
        constrained_action=dict(action),
        constraint_remapped=False,
        policy_followed_advice=True,
        step=0,
        step_outcome=f"allowed:{spec.preferred_action}",
        terminal_state={"illegal_state_change": False, "return_status": None},
        terminal_outcome={"success": True, "illegal_state_change": False},
        success=True,
        failure_owner="none",
        causal_credit="seed",
        reusable_pattern=spec.reusable_pattern,
        avoid_pattern=spec.avoid_pattern,
        workflow="return_resolution",
        progress_signature=progress_signature_from_progress(progress),
        source=source,
        source_hash=provenance_hash(source, spec.case_id, 0),
        paired_replay_result={
            "ok": True,
            "protocol": PROTOCOL,
            "note": "curated_legal_ask_user_seed",
            "constraint_remapped": False,
        },
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def build_probe_tasks(seed: int = 20260731) -> list[MemoryProbeTask]:
    rng = random.Random(seed)
    phrasings: dict[str, tuple[str, ...]] = {
        "missing_order_id": (
            "I need to return an item but cannot find my order number.",
            "help me return something; I still do not know the order id",
            "want a return but the order number is temporarily missing",
            "start a return for me without an order id for now",
            "I want to process a return and do not have the order id ready",
            "return please — order number unavailable at the moment",
        ),
        "missing_verification": (
            "order {oid} should be returned; 原因：size mismatch",
            "please handle return for {oid}. 原因：color issue",
            "order {oid} needs return, 原因：average quality",
            "return {oid}; 原因：arrived late and unused",
            "help return {oid}, 原因：bought too many",
            "process a return on {oid}; 原因：wrong model ordered",
        ),
        "missing_return_reason": (
            "order {oid} needs a return",
            "please return order {oid}",
            "apply return for order {oid}",
            "I want to return {oid}",
            "help process the return for {oid}",
            "start return workflow for {oid}",
        ),
        "awaiting_confirmation": (
            "order {oid} return; 原因：fabric defect",
            "please return {oid}. 原因：stitching problem",
            "order {oid} return, 原因：missing accessory",
            "return {oid}; 原因：does not fit as expected",
            "help return {oid}, 原因：differs from page",
            "process return for {oid}; 原因：changed my mind",
        ),
    }
    responses: dict[str, tuple[str, ...]] = {
        "missing_order_id": ("{oid}", "123456", "yes"),
        "missing_verification": ("123456", "yes"),
        "missing_return_reason": ("123456", "size not suitable", "yes"),
        "awaiting_confirmation": ("123456", "yes"),
    }
    tasks: list[MemoryProbeTask] = []
    n = 0
    for state_id in PROBE_STATES:
        for phrase in phrasings[state_id]:
            n += 1
            oid = f"O{300000 + n:06d}"
            uid = f"P{n:04d}"
            tasks.append(MemoryProbeTask(
                task_id=f"mprobe_{state_id}_{n:02d}",
                state_id=state_id,
                preferred_action=PREFERRED_BY_STATE[state_id],
                seed=rng.randrange(1, 2**31),
                user_id=uid,
                order_id=oid,
                initial_message=phrase.format(oid=oid),
                user_responses=tuple(item.format(oid=oid) for item in responses[state_id]),
            ))
    return tasks


def manifest_sha256(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256(json.dumps(
        rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()


def freeze_manifest_digests() -> dict[str, str]:
    return {
        "train_case_sha256": manifest_sha256(
            [asdict(spec) for spec in build_train_case_specs()]),
        "probe_task_sha256": manifest_sha256(
            [asdict(task) for task in build_probe_tasks()]),
    }


def assert_frozen_manifests() -> None:
    digests = freeze_manifest_digests()
    if digests["probe_task_sha256"] != FROZEN_PROBE_TASK_SHA256:
        raise ValueError(
            f"probe task hash mismatch: {digests['probe_task_sha256']} "
            f"!= {FROZEN_PROBE_TASK_SHA256}"
        )
    if digests["train_case_sha256"] != FROZEN_TRAIN_CASE_SHA256:
        raise ValueError(
            f"train case hash mismatch: {digests['train_case_sha256']} "
            f"!= {FROZEN_TRAIN_CASE_SHA256}"
        )


def to_probe_task_spec(task: MemoryProbeTask) -> TaskSpec:
    return TaskSpec(
        task.task_id, "return", task.user_id, task.initial_message, task.seed,
        allowed_tools=[
            "get_order", "check_return_eligibility", "create_return_request",
        ],
        forbidden_tools=[],
        expected_state={task.order_id: {"return_status": None}},
        metadata={
            "order_id": task.order_id,
            "verification_code": "123456",
            "probe_state_id": task.state_id,
            "preferred_action": task.preferred_action,
        },
        split=task.split,
    )


def scripted_prefix_for(task: MemoryProbeTask) -> list[AgentAction]:
    if task.state_id in {"missing_order_id", "missing_verification"}:
        return []
    code = "123456"
    oid = task.order_id
    return [
        AgentAction.answer(
            "please provide the six-digit verification code.",
            requires_user_response=True,
        ),
        AgentAction.tool_call("get_order", order_id=oid, verification_code=code),
        AgentAction.tool_call(
            "check_return_eligibility", order_id=oid, verification_code=code,
        ),
    ]


def prepare_probe_database(tasks: Iterable[MemoryProbeTask], path: Path | str) -> Path:
    from .legacy_closure_benchmark import M1Task

    mirrored = [
        M1Task(
            task_id=task.task_id,
            split="dev",
            scenario="memory_probe",
            seed=task.seed,
            user_id=task.user_id,
            order_id=task.order_id,
            initial_message=task.initial_message,
            user_responses=task.user_responses,
            database_fixture=task.database_fixture,
            expected={"status": "WAITING_USER", "write": False},
        )
        for task in tasks
    ]
    return prepare_database(mirrored, path)


def seed_train_cases(
    db_path: Path | str,
    *,
    approved_by: str,
    approval_reason: str,
    approve: bool = True,
) -> list[dict[str, Any]]:
    rows = []
    for spec in build_train_case_specs():
        case = train_case_from_spec(spec)
        stored, admission = write_candidate(
            case,
            db_path=db_path,
            approve=approve,
            approved_by=approved_by,
            approval_reason=approval_reason,
            require_paired_replay=True,
        )
        rows.append({
            "case_id": stored.case_id,
            "state_id": spec.state_id,
            "admission": admission,
            "memory_approved": stored.memory_approved,
        })
    return rows


def decision_at_step(trajectory: Trajectory, step: int) -> dict[str, Any] | None:
    for span in trajectory.decision_spans or []:
        if span.get("step") == step:
            return span
    return None


def resolve_probe_step(task: MemoryProbeTask, policy: Any) -> int:
    if getattr(policy, "probe_step", None) is not None:
        return int(policy.probe_step)
    return len(scripted_prefix_for(task))


def offline_preferred(progress: dict[str, Any], *, db_path: Path | str) -> list[str]:
    advice = build_memory_advice(progress, db_path=db_path)
    return list(advice.preferred_actions)


def score_pair(
    *,
    task: MemoryProbeTask,
    off_trajectory: Trajectory,
    on_trajectory: Trajectory,
    off_grade: GradeResult | dict[str, Any],
    on_grade: GradeResult | dict[str, Any],
    probe_step_off: int,
    probe_step_on: int,
    offline_preferred_actions: list[str],
) -> dict[str, Any]:
    def _grade(g: GradeResult | dict[str, Any]) -> dict[str, Any]:
        return g.to_dict() if hasattr(g, "to_dict") else dict(g)

    off_g, on_g = _grade(off_grade), _grade(on_grade)
    off_d = decision_at_step(off_trajectory, probe_step_off) or {}
    on_d = decision_at_step(on_trajectory, probe_step_on) or {}
    preferred = list(offline_preferred_actions) or [task.preferred_action]
    off_raw = off_d.get("raw_policy_action") or {}
    on_raw = on_d.get("raw_policy_action") or {}
    off_key = _action_key(off_raw)
    on_key = _action_key(on_raw)
    allowed = list(
        (off_d.get("progress") or on_d.get("progress") or {})
        .get("allowed_next_actions") or []
    )
    if not allowed:
        allowed = [task.preferred_action]
    return {
        "task_id": task.task_id,
        "state_id": task.state_id,
        "preferred_action": task.preferred_action,
        "offline_preferred_actions": preferred,
        "retrieval_coverage": bool(preferred),
        "off": {
            "raw_action_key": off_key,
            "raw_in_allowlist": bool(off_key and off_key in allowed),
            "raw_matches_preferred": bool(off_key and off_key in preferred),
            "constraint_remapped": bool(off_d.get("constraint_remapped")),
            "terminal_success": bool(
                off_g.get("success") or off_g.get("operational_success")),
            "illegal_state_change": bool(off_g.get("illegal_state_change")),
            "probe_step": probe_step_off,
        },
        "on": {
            "raw_action_key": on_key,
            "raw_in_allowlist": bool(on_key and on_key in allowed),
            "raw_matches_preferred": bool(on_key and on_key in preferred),
            "constraint_remapped": bool(on_d.get("constraint_remapped")),
            "policy_followed_advice": on_d.get("policy_followed_advice"),
            "terminal_success": bool(
                on_g.get("success") or on_g.get("operational_success")),
            "illegal_state_change": bool(on_g.get("illegal_state_change")),
            "probe_step": probe_step_on,
        },
        "repaired_by_memory": bool(
            off_key and off_key not in preferred
            and on_key and on_key in preferred
        ),
        "regressed_by_memory": bool(
            off_key and off_key in preferred
            and on_key and on_key not in preferred
        ),
    }


def aggregate_probe(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    off_errors = [row for row in pairs if not row["off"]["raw_matches_preferred"]]
    repaired = [row for row in pairs if row["repaired_by_memory"]]
    regressed = [row for row in pairs if row["regressed_by_memory"]]
    return {
        "n_pairs": len(pairs),
        "retrieval_coverage": sum(1 for row in pairs if row["retrieval_coverage"]),
        "off_raw_errors": len(off_errors),
        "repaired_by_memory": len(repaired),
        "regressed_by_memory": len(regressed),
        "off_constraint_remap_count": sum(
            1 for row in pairs if row["off"]["constraint_remapped"]),
        "on_constraint_remap_count": sum(
            1 for row in pairs if row["on"]["constraint_remapped"]),
        "off_terminal_success": sum(
            1 for row in pairs if row["off"]["terminal_success"]),
        "on_terminal_success": sum(
            1 for row in pairs if row["on"]["terminal_success"]),
        "illegal_state_change_total": sum(
            1 for row in pairs
            if row["off"]["illegal_state_change"] or row["on"]["illegal_state_change"]
        ),
        "off_error_task_ids": [row["task_id"] for row in off_errors],
        "repaired_task_ids": [row["task_id"] for row in repaired],
        "regressed_task_ids": [row["task_id"] for row in regressed],
    }


def conclude(summary: dict[str, Any]) -> dict[str, Any]:
    """Preregistered decision rule for memory_policy_probe_v1."""
    off_errors = int(summary["off_raw_errors"])
    repaired = int(summary["repaired_by_memory"])
    regressed = int(summary["regressed_by_memory"])
    remap_down = (
        int(summary["on_constraint_remap_count"])
        < int(summary["off_constraint_remap_count"])
    )
    success_ok = (
        int(summary["on_terminal_success"]) >= int(summary["off_terminal_success"])
    )
    illegal_ok = int(summary["illegal_state_change_total"]) == 0
    if off_errors < 4:
        return {
            "verdict": "neutral_underpowered",
            "policy_memory_gain": "not_identifiable",
            "reason": "fewer than 4 off-arm raw policy errors",
        }
    if (
        repaired * 2 >= off_errors
        and regressed == 0
        and remap_down
        and success_ok
        and illegal_ok
    ):
        return {
            "verdict": "positive",
            "policy_memory_gain": "positive",
            "reason": (
                "raw policy repairs >= half of off errors; "
                "no regress; remap down"
            ),
        }
    return {
        "verdict": "negative_or_inconclusive",
        "policy_memory_gain": "negative_or_inconclusive",
        "reason": "preregistered positive criteria not met",
    }
