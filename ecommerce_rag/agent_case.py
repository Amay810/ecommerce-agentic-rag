"""AgentCase v1: auditable failure/improvement assets for responsibility routing.

An AgentCase is not a raw trajectory dump.  It records the first causal failure,
its owner layer, and whether the case may ever enter training.  Formal ``dev``
and ``locked`` lineage is always ``training_approved=false``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


FAILURE_OWNERS = frozenset({
    "fact", "protocol", "tool", "progress", "policy", "executor", "grader",
})

AGENT_CASE_SCHEMA_VERSION = 1


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
    source: dict[str, Any] = field(default_factory=dict)
    schema_version: int = AGENT_CASE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.failure_owner not in FAILURE_OWNERS:
            raise ValueError(f"unknown failure_owner: {self.failure_owner}")
        if self.split in {"dev", "locked"} and self.training_approved:
            raise ValueError("formal dev/locked cases must set training_approved=false")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AgentCase":
        known = {key: payload[key] for key in cls.__dataclass_fields__ if key in payload}
        return cls(**known)


def case_id_for(task_id: str, first_causal_failure: str, *, commit: str = "") -> str:
    digest = hashlib.sha256(
        f"{task_id}|{first_causal_failure}|{commit}".encode()
    ).hexdigest()[:16]
    return f"ac_{task_id}_{digest}"


def load_agent_cases(path: Path | str) -> list[AgentCase]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(AgentCase.from_dict(json.loads(line)))
    return rows


def dump_agent_cases(cases: list[AgentCase], path: Path | str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "".join(json.dumps(case.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
                for case in cases),
        encoding="utf-8",
    )


#: Frozen attribution for the six remaining legacy_progress_fixed failures.
#: Owners were human-adjudicated from the frozen task manifest + closeout;
#: formal trajectory blobs remain external and are referenced by hash when present.
DEV_FAILURE_CASE_SPECS: tuple[dict[str, Any], ...] = (
    {
        "task_id": "m1_dev_03_02",
        "failure_owner": "progress",
        "first_causal_failure": "verification_code_refused_not_represented",
        "allowed_actions": ["ask_user:verification_code"],
        "forbidden_actions": ["create_return_request", "terminal_success"],
        "chosen_action": {"note": "progress stayed on ask_user:verification_code after typed refusal"},
        "reusable_pattern": "typed verification refusal must block writes and allow handoff",
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
        "intervention": "unify create_return_request idempotent success contract + grader",
    },
    {
        "task_id": "m1_dev_07_01",
        "failure_owner": "policy",
        "first_causal_failure": "inappropriate_handoff_instead_of_ask_verification",
        "allowed_actions": ["ask_user:verification_code"],
        "forbidden_actions": ["handoff", "terminal_success", "create_return_request"],
        "chosen_action": {"action_type": "handoff"},
        "reusable_pattern": "identity_required forbids handoff and premature terminal success",
        "intervention": "deferred: dynamic action mask or independent policy data (not LoRA from these two)",
    },
    {
        "task_id": "m1_dev_07_03",
        "failure_owner": "policy",
        "first_causal_failure": "inappropriate_handoff_instead_of_ask_verification",
        "allowed_actions": ["ask_user:verification_code"],
        "forbidden_actions": ["handoff", "terminal_success", "create_return_request"],
        "chosen_action": {"action_type": "handoff"},
        "reusable_pattern": "identity_required forbids handoff and premature terminal success",
        "intervention": "deferred: dynamic action mask or independent policy data (not LoRA from these two)",
    },
)


def build_dev_failure_agent_cases(*, code_commit: str = "") -> list[AgentCase]:
    from .legacy_closure_benchmark import FROZEN_TASK_SHA256, build_m1_tasks

    by_id = {task.task_id: task for task in build_m1_tasks() if task.split == "dev"}
    cases: list[AgentCase] = []
    for spec in DEV_FAILURE_CASE_SPECS:
        task = by_id[spec["task_id"]]
        cases.append(AgentCase(
            case_id=case_id_for(spec["task_id"], spec["first_causal_failure"], commit=code_commit),
            split="dev",
            training_approved=False,
            user_goal=task.initial_message,
            task_id=task.task_id,
            validated_facts=[
                {"order_id": task.order_id},
                {"scenario": task.scenario},
                {"database_fixture": task.database_fixture},
                {"expected": task.expected},
            ],
            evidence_quotes=list(task.user_responses),
            progress_before={},
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
            source={
                "baseline_config": "legacy_progress_fixed",
                "baseline_success": "34/40",
                "task_manifest_sha256": FROZEN_TASK_SHA256,
                "code_commit": code_commit,
                "attribution": "human_adjudicated_from_frozen_manifest_and_closeout",
                "locked_executed": False,
            },
        ))
    return cases
