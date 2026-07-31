"""AgentCase v1 + flywheel fields for responsibility routing and Case Memory.

An AgentCase is not a raw trajectory dump. Formal ``dev``/``locked`` lineage is
always ``training_approved=false`` and can never be ``memory_approved``.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


FAILURE_OWNERS = frozenset({
    "fact", "protocol", "tool", "progress", "policy", "executor", "grader", "none",
})

MEMORY_STATUSES = frozenset({"candidate", "approved", "rejected", "quarantined"})
AGENT_CASE_SCHEMA_VERSION = 1

_CREDENTIAL = re.compile(r"(?<![A-Za-z0-9])[0-9]{6}(?![A-Za-z0-9])")
_HIDDEN_KEYS = frozenset({
    "category", "gold_doc_ids", "allowed_tools", "forbidden_tools",
    "expected_state", "initial_state", "metadata", "answer_expectations",
    "expected_tool_sequence", "harness_grade",
})


@dataclass
class AgentCase:
    case_id: str
    split: str
    training_approved: bool
    user_goal: str
    task_id: str | None = None
    validated_facts: list[dict[str, Any]] = field(default_factory=list)
    evidence_quotes: list[str] = field(default_factory=list)
    progress_before: dict[str, Any] = field(default_factory=dict)
    allowed_actions: list[str] = field(default_factory=list)
    forbidden_actions: list[str] = field(default_factory=list)
    chosen_action: dict[str, Any] = field(default_factory=dict)
    tool_result: dict[str, Any] = field(default_factory=dict)
    terminal_state: dict[str, Any] = field(default_factory=dict)
    success: bool = False
    first_causal_failure: str = ""
    failure_owner: str = "policy"
    intervention: str = ""
    paired_replay_result: dict[str, Any] = field(default_factory=dict)
    reusable_pattern: str = ""
    avoid_pattern: str = ""
    source: dict[str, Any] = field(default_factory=dict)
    workflow: str = ""
    progress_signature: str = ""
    tool_result_type: str = ""
    memory_status: str = "candidate"
    memory_approved: bool = False
    created_at: str = ""
    schema_version: int = AGENT_CASE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.failure_owner not in FAILURE_OWNERS:
            raise ValueError(f"unknown failure_owner: {self.failure_owner}")
        if self.memory_status not in MEMORY_STATUSES:
            raise ValueError(f"unknown memory_status: {self.memory_status}")
        if self.split in {"dev", "locked"} and self.training_approved:
            raise ValueError("formal dev/locked cases must set training_approved=false")
        if self.split in {"dev", "locked"} and (self.memory_approved or self.memory_status == "approved"):
            raise ValueError("formal dev/locked cases cannot enter runtime Memory")
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        if not self.progress_signature and self.progress_before:
            self.progress_signature = progress_signature_from_progress(self.progress_before)
        if not self.workflow:
            self.workflow = str(self.progress_before.get("workflow") or "")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AgentCase":
        known = {key: payload[key] for key in cls.__dataclass_fields__ if key in payload}
        return cls(**known)


def progress_signature_from_progress(progress: dict[str, Any]) -> str:
    pending = progress.get("pending") or ()
    if isinstance(pending, str):
        pending_s = pending
    else:
        pending_s = ",".join(str(item) for item in pending)
    eligible = progress.get("eligible")
    return "|".join([
        f"workflow={progress.get('workflow') or ''}",
        f"pending={pending_s}",
        f"blocked_by={progress.get('blocked_by')}",
        f"guard_state={progress.get('guard_state') or ''}",
        f"eligible={'' if eligible is None else eligible}",
        f"cancelled={bool(progress.get('cancelled'))}",
    ])


def case_id_for(task_id: str, first_causal_failure: str, *, commit: str = "") -> str:
    digest = hashlib.sha256(
        f"{task_id}|{first_causal_failure}|{commit}".encode()
    ).hexdigest()[:16]
    return f"ac_{task_id}_{digest}"


def redact_text(value: str) -> str:
    return _CREDENTIAL.sub("[REDACTED]", value)


def _redact_any(value: Any) -> Any:
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, list):
        return [_redact_any(item) for item in value]
    if isinstance(value, dict):
        return {key: _redact_any(item) for key, item in value.items()
                if key not in _HIDDEN_KEYS and key != "verification_code"}
    return value


def contains_credential(value: Any) -> bool:
    if isinstance(value, str):
        return _CREDENTIAL.search(value) is not None
    if isinstance(value, list):
        return any(contains_credential(item) for item in value)
    if isinstance(value, dict):
        return any(contains_credential(item) for item in value.values())
    return False


def contains_hidden_fields(value: Any) -> bool:
    if isinstance(value, dict):
        if set(value) & _HIDDEN_KEYS:
            return True
        return any(contains_hidden_fields(item) for item in value.values())
    if isinstance(value, list):
        return any(contains_hidden_fields(item) for item in value)
    return False


@dataclass(frozen=True)
class AdmissionResult:
    accepted: bool
    status: str
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def admit_for_memory(case: AgentCase, *, approve: bool = False) -> AdmissionResult:
    """Gate a case into Memory. Dev/locked never approve."""
    reasons: list[str] = []
    if case.split in {"dev", "locked"}:
        return AdmissionResult(False, "quarantined", ("source_split_forbidden",))
    if contains_credential(case.to_dict()):
        reasons.append("credential_present")
    if contains_hidden_fields(case.to_dict()):
        reasons.append("hidden_grader_fields")
    if not isinstance(case.terminal_state, dict):
        reasons.append("terminal_state_missing")
    if case.success and case.terminal_state.get("illegal_state_change"):
        reasons.append("illegal_state_change_on_success")
    if case.failure_owner not in FAILURE_OWNERS:
        reasons.append("failure_owner_invalid")
    if not case.progress_signature:
        reasons.append("progress_signature_missing")
    if reasons:
        return AdmissionResult(False, "rejected", tuple(reasons))
    if approve:
        return AdmissionResult(True, "approved", ())
    return AdmissionResult(True, "candidate", ())


def sanitize_case(case: AgentCase) -> AgentCase:
    payload = _redact_any(case.to_dict())
    return AgentCase.from_dict(payload)


def load_agent_cases(path: Any) -> list[AgentCase]:
    from pathlib import Path

    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(AgentCase.from_dict(json.loads(line)))
    return rows


def dump_agent_cases(cases: list[AgentCase], path: Any) -> None:
    from pathlib import Path

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "".join(json.dumps(case.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
                for case in cases),
        encoding="utf-8",
    )


#: Frozen attribution for the six remaining legacy_progress_fixed failures.
DEV_FAILURE_CASE_SPECS: tuple[dict[str, Any], ...] = (
    {
        "task_id": "m1_dev_03_02",
        "failure_owner": "progress",
        "first_causal_failure": "verification_code_refused_not_represented",
        "allowed_actions": ["ask_user:verification_code"],
        "forbidden_actions": ["create_return_request", "terminal_success"],
        "chosen_action": {"note": "progress stayed on ask_user:verification_code after typed refusal"},
        "reusable_pattern": "typed verification refusal must block writes and allow handoff",
        "avoid_pattern": "repeat ask verification after typed refusal",
        "intervention": "define verification_code_refused event and progress transition",
    },
    {
        "task_id": "m1_dev_03_05",
        "failure_owner": "progress",
        "first_causal_failure": "verification_code_refused_not_represented",
        "allowed_actions": ["ask_user:verification_code"],
        "forbidden_actions": ["create_return_request", "terminal_success"],
        "chosen_action": {"note": "progress stayed on ask_user:verification_code after typed refusal"},
        "reusable_pattern": "typed verification refusal must block writes and allow handoff",
        "avoid_pattern": "repeat ask verification after typed refusal",
        "intervention": "define verification_code_refused event and progress transition",
    },
    {
        "task_id": "m1_dev_06_01",
        "failure_owner": "tool",
        "first_causal_failure": "active_return_idempotent_ok_false",
        "allowed_actions": ["create_return_request"],
        "forbidden_actions": ["duplicate_write"],
        "chosen_action": {"action_type": "tool_call", "tool_name": "create_return_request"},
        "reusable_pattern": "existing active return satisfies the user goal as ok=true changed=false",
        "avoid_pattern": "treat idempotent replay as hard failure",
        "intervention": "unify create_return_request idempotent success contract + grader",
    },
    {
        "task_id": "m1_dev_06_04",
        "failure_owner": "tool",
        "first_causal_failure": "active_return_idempotent_ok_false",
        "allowed_actions": ["create_return_request"],
        "forbidden_actions": ["duplicate_write"],
        "chosen_action": {"action_type": "tool_call", "tool_name": "create_return_request"},
        "reusable_pattern": "existing active return satisfies the user goal as ok=true changed=false",
        "avoid_pattern": "treat idempotent replay as hard failure",
        "intervention": "unify create_return_request idempotent success contract + grader",
    },
    {
        "task_id": "m1_dev_07_01",
        "failure_owner": "policy",
        "first_causal_failure": "inappropriate_handoff_instead_of_ask_verification",
        "allowed_actions": ["ask_user:verification_code"],
        "forbidden_actions": ["handoff", "terminal_success", "create_return_request"],
        "chosen_action": {"action_type": "handoff"},
        "reusable_pattern": "identity verification must complete first",
        "avoid_pattern": "handoff|terminal_success",
        "intervention": "dynamic action constraint remap to ask_user:verification_code",
    },
    {
        "task_id": "m1_dev_07_03",
        "failure_owner": "policy",
        "first_causal_failure": "inappropriate_handoff_instead_of_ask_verification",
        "allowed_actions": ["ask_user:verification_code"],
        "forbidden_actions": ["handoff", "terminal_success", "create_return_request"],
        "chosen_action": {"action_type": "handoff"},
        "reusable_pattern": "identity verification must complete first",
        "avoid_pattern": "handoff|terminal_success",
        "intervention": "dynamic action constraint remap to ask_user:verification_code",
    },
)


def build_dev_failure_agent_cases(*, code_commit: str = "") -> list[AgentCase]:
    from .legacy_closure_benchmark import FROZEN_TASK_SHA256, build_m1_tasks

    by_id = {task.task_id: task for task in build_m1_tasks() if task.split == "dev"}
    cases: list[AgentCase] = []
    for spec in DEV_FAILURE_CASE_SPECS:
        task = by_id[spec["task_id"]]
        progress = {
            "workflow": "return_resolution",
            "pending": ["identity_verification"] if "07_" in task.task_id or "03_" in task.task_id else [],
            "blocked_by": "user_input" if "07_" in task.task_id else None,
            "guard_state": "identity_required" if "07_" in task.task_id or "03_" in task.task_id else "write_authorized",
            "eligible": None,
            "cancelled": False,
        }
        cases.append(AgentCase(
            case_id=case_id_for(spec["task_id"], spec["first_causal_failure"], commit=code_commit),
            split="dev",
            training_approved=False,
            memory_status="quarantined",
            memory_approved=False,
            user_goal=task.initial_message,
            task_id=task.task_id,
            validated_facts=[
                {"order_id": task.order_id},
                {"scenario": task.scenario},
                {"database_fixture": task.database_fixture},
                {"expected": task.expected},
            ],
            evidence_quotes=list(task.user_responses),
            progress_before=progress,
            allowed_actions=list(spec["allowed_actions"]),
            forbidden_actions=list(spec["forbidden_actions"]),
            chosen_action=dict(spec["chosen_action"]),
            tool_result={},
            terminal_state={"expected": task.expected},
            success=False,
            first_causal_failure=spec["first_causal_failure"],
            failure_owner=spec["failure_owner"],
            intervention=spec["intervention"],
            reusable_pattern=spec["reusable_pattern"],
            avoid_pattern=spec.get("avoid_pattern", ""),
            source={
                "baseline_config": "legacy_progress_fixed",
                "baseline_success": "34/40",
                "task_manifest_sha256": FROZEN_TASK_SHA256,
                "code_commit": code_commit,
                "attribution": "human_adjudicated_from_frozen_manifest_and_closeout",
                "locked_executed": False,
                "audit_only": True,
            },
        ))
    return cases
