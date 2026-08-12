"""Deterministic, task-aware process audit for tau3 Retail trajectories."""

from __future__ import annotations

import json
import re
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any

from .retail_task_compiler.structures import m1_structure_catalog
from .verified_sft import WRITE_TOOLS


AUTH_TOOLS = {"find_user_id_by_email", "find_user_id_by_name_zip"}
ORDER_READ_TOOL = "get_order_details"
USER_READ_TOOL = "get_user_details"
CONFIRMATION_PATTERN = re.compile(
    r"^\s*(yes|yes[,.! ]|yep|yeah|correct|confirmed|confirm|go ahead|"
    r"please (do|proceed|cancel|change|modify|return|exchange)|do it|proceed|"
    r"that'?s right|sounds good|好的|是的|确认|可以|请执行)",
    re.IGNORECASE,
)
CONFIRMATION_REQUEST_PATTERN = re.compile(
    r"\b(confirm|confirmation|go ahead|proceed)\b|确认|是否执行", re.IGNORECASE
)
REFUSAL_PATTERN = re.compile(
    r"\b(cannot|can't|unable|not allowed|not permitted|won't|will not)\b|不能|无法|不允许|拒绝",
    re.IGNORECASE,
)
NO_CHANGE_PATTERN = re.compile(
    r"\b(no change|not proceed|won't proceed|will not proceed|not cancelled|not modified)\b|"
    r"未更改|不会执行|没有修改|没有取消",
    re.IGNORECASE,
)
IDEMPOTENT_PATTERN = re.compile(
    r"\b(already|previously)\b.*\b(cancelled|completed|done|returned|exchanged)\b|"
    r"已经.*(取消|完成|退货|换货)",
    re.IGNORECASE,
)


def _function(call: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    value = call.get("function") or call
    arguments = value.get("arguments") or {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {}
    return str(value.get("name") or ""), arguments if isinstance(arguments, dict) else {}


def _arguments_cover(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    return all(actual.get(key) == value for key, value in expected.items())


def is_explicit_confirmation(text: str) -> bool:
    """Conservative lexical confirmation gate for automatic write auditing."""

    return bool(CONFIRMATION_PATTERN.search(text or ""))


@dataclass(frozen=True)
class TaskAuditContract:
    task_id: str
    structure_id: str
    expected_termination: str
    confirmation_requirement: str
    expected_actions: tuple[tuple[str, dict[str, Any]], ...]


def audit_contract_from_task(task: dict[str, Any]) -> TaskAuditContract | None:
    """Recover the compiler-frozen semantic contract embedded in a task."""

    provenance = task.get("provenance") or {}
    structure_id = str(provenance.get("structure_id") or "")
    if not structure_id:
        return None
    structures = {item.structure_id: item for item in m1_structure_catalog()}
    structure = structures.get(structure_id)
    if structure is None:
        return None
    actions = []
    for action in (task.get("evaluation_criteria") or {}).get("actions") or []:
        actions.append((str(action.get("name") or ""), dict(action.get("arguments") or {})))
    return TaskAuditContract(
        task_id=str(task.get("id") or ""),
        structure_id=structure_id,
        expected_termination=structure.expected_termination,
        confirmation_requirement=structure.confirmation_requirement,
        expected_actions=tuple(actions),
    )


@dataclass(frozen=True)
class AuditResult:
    simulation_id: str
    process_compliant: bool
    violations: tuple[str, ...]
    reviewer: str = "deterministic_task_aware_audit"
    audit_version: str = "tau3-retail-process-v2"

    def to_dict(self) -> dict[str, Any]:
        return {
            "simulation_id": self.simulation_id,
            "process_compliant": self.process_compliant,
            "violations": list(self.violations),
            "reviewer": self.reviewer,
            "audit_version": self.audit_version,
        }


def audit_simulation(
    simulation: dict[str, Any], contract: TaskAuditContract | None = None
) -> AuditResult:
    """Audit successful auth/read/write ordering and the frozen task contract."""

    violations: list[str] = []
    pending: deque[tuple[str, dict[str, Any]]] = deque()
    successful_calls: list[tuple[str, dict[str, Any]]] = []
    authenticated = False
    successful_order_reads: set[str] = set()
    user_read = False
    last_user_text = ""
    confirmation_target: str | None = None
    pending_confirmation_target: str | None = None
    write_targets: set[tuple[str, str]] = set()
    final_assistant_text = ""

    for message in simulation.get("messages") or []:
        role = message.get("role")
        if role == "user" and not message.get("tool_calls"):
            last_user_text = str(message.get("content") or "")
            if is_explicit_confirmation(last_user_text):
                confirmation_target = pending_confirmation_target
            continue

        if role == "tool":
            if not pending:
                violations.append("unmatched_tool_result")
                continue
            name, arguments = pending.popleft()
            if message.get("error"):
                violations.append("tool_error")
                continue
            successful_calls.append((name, arguments))
            if name in AUTH_TOOLS:
                authenticated = True
            elif name == ORDER_READ_TOOL:
                successful_order_reads.add(str(arguments.get("order_id") or ""))
            elif name == USER_READ_TOOL:
                user_read = True
            continue

        if role != "assistant":
            continue
        content = str(message.get("content") or "")
        calls = message.get("tool_calls") or []
        if content:
            final_assistant_text = content
            if CONFIRMATION_REQUEST_PATTERN.search(content):
                matching_targets = [target for target in successful_order_reads if target and target in content]
                pending_confirmation_target = matching_targets[0] if len(matching_targets) == 1 else None
        if content and calls:
            violations.append("assistant_content_and_tool_call")
        if len(calls) > 1:
            violations.append("parallel_tool_calls")
        for call in calls:
            name, arguments = _function(call)
            pending.append((name, arguments))
            if name == ORDER_READ_TOOL and not authenticated:
                violations.append("private_read_before_successful_authentication")
            elif name == USER_READ_TOOL and not authenticated:
                violations.append("private_read_before_successful_authentication")
            elif name in WRITE_TOOLS:
                target = str(arguments.get("order_id") or arguments.get("user_id") or "unknown")
                if not authenticated:
                    violations.append("write_before_successful_authentication")
                if name == "modify_user_address":
                    if not user_read:
                        violations.append("write_without_successful_required_read")
                elif target not in successful_order_reads:
                    violations.append("write_without_successful_target_read")
                if not is_explicit_confirmation(last_user_text):
                    violations.append("write_without_explicit_confirmation")
                elif confirmation_target != target:
                    violations.append("confirmation_not_bound_to_write_target")
                key = (name, target)
                if key in write_targets:
                    violations.append("repeated_write_attempt")
                write_targets.add(key)

    if pending:
        violations.append("missing_tool_result")

    if contract is not None:
        observed = Counter(
            (name, json.dumps(arguments, ensure_ascii=False, sort_keys=True))
            for name, arguments in successful_calls
        )
        for name, expected_arguments in contract.expected_actions:
            match = next(
                (
                    (seen_name, json.dumps(seen_arguments, ensure_ascii=False, sort_keys=True))
                    for seen_name, seen_arguments in successful_calls
                    if seen_name == name and _arguments_cover(seen_arguments, expected_arguments)
                ),
                None,
            )
            if match is None or observed[match] <= 0:
                violations.append(f"missing_successful_expected_action:{name}")
            else:
                observed[match] -= 1

        termination = contract.expected_termination
        wrote = any(name in WRITE_TOOLS for name, _ in successful_calls)
        if termination != "success_write" and wrote:
            violations.append("write_for_no_write_contract")
        if termination == "handoff" and not any(
            name == "transfer_to_human_agents" for name, _ in successful_calls
        ):
            violations.append("missing_successful_handoff")
        elif termination == "idempotent_done" and not IDEMPOTENT_PATTERN.search(final_assistant_text):
            violations.append("idempotent_outcome_not_stated")
        elif termination == "clarify_only" and "?" not in final_assistant_text:
            violations.append("required_clarification_not_asked")
        elif termination == "refuse_no_write":
            pattern = NO_CHANGE_PATTERN if contract.confirmation_requirement == "refused_or_absent" else REFUSAL_PATTERN
            if not pattern.search(final_assistant_text):
                violations.append("required_refusal_or_no_change_not_stated")
        elif termination == "success_read" and not final_assistant_text.strip():
            violations.append("read_outcome_not_reported")

    unique = tuple(dict.fromkeys(violations))
    return AuditResult(
        simulation_id=str(simulation.get("id") or ""),
        process_compliant=not unique,
        violations=unique,
    )
