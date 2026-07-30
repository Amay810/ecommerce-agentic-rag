"""Frozen dev protocol for legacy baseline versus visible TaskProgress.

This module is harness-only.  Scenario expectations never enter an
``AgentObservation`` and the locked split cannot be selected by the progress
runner.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import sqlite3
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from . import orders
from .domain import AgentAction, GradeResult, TaskSpec, Trajectory
from .harness import UserSimulatorProtocolError


FROZEN_TASK_SHA256 = "e4346e3f99261d203f9fea57aeec48d58e5f769d9a1e856e43b9cf0b74a6c8e3"
SCENARIO_DISTRIBUTION: dict[str, tuple[int, int]] = {
    "eligible_happy_path": (7, 3),
    "ineligible_return": (5, 3),
    "missing_refused_or_changed_goal": (7, 3),
    "verification_or_ownership_failure": (7, 3),
    "ambiguous_wrong_confirmation_or_reason_change": (5, 3),
    "active_duplicate_or_commit_timeout": (5, 3),
    "premature_repeat_or_guardrail_correction": (4, 2),
}


@dataclass(frozen=True)
class M1Task:
    task_id: str
    split: str
    scenario: str
    seed: int
    user_id: str
    order_id: str
    initial_message: str
    user_responses: tuple[str, ...]
    database_fixture: str
    fault_mode: str | None = None
    expected: dict[str, Any] = field(default_factory=dict)


def _scenario_payload(scenario: str, index: int, order_id: str):
    reasoned = f"\u6211\u8981\u4e3a\u8ba2\u5355 {order_id} \u7533\u8bf7\u9000\u8d27\u3002\u9000\u8d27\u539f\u56e0\uff1a\u5546\u54c1\u4e0d\u5408\u9002"
    if scenario == "eligible_happy_path":
        return reasoned, ["123456", "\u786e\u8ba4"], "eligible", None, {"status": "COMPLETED", "write": True}
    if scenario == "ineligible_return":
        return reasoned, ["123456"], "ineligible", None, {"status": "COMPLETED", "write": False}
    if scenario == "missing_refused_or_changed_goal":
        variants = [
            ("\u6211\u8981\u9000\u8d27\uff0c\u4f46\u8fd8\u6ca1\u627e\u5230\u8ba2\u5355\u53f7\u3002", [order_id, "123456", "\u4e0d\u63d0\u4f9b\u9000\u8d27\u539f\u56e0"], "eligible", None,
             {"status": "HANDED_OFF", "write": False}),
            (f"\u6211\u8981\u9000 {order_id}\u3002\u9000\u8d27\u539f\u56e0\uff1a\u4e0d\u5408\u9002", ["\u4e0d\u63d0\u4f9b"], "eligible", None,
             {"status": "HANDED_OFF", "write": False}),
            (reasoned, ["123456", "\u53d6\u6d88"], "eligible", None, {"status": "COMPLETED", "write": False}),
        ]
        return variants[index % len(variants)]
    if scenario == "verification_or_ownership_failure":
        variants = [
            (reasoned, ["999999", "888888"], "eligible", None, {"status": "HANDED_OFF", "write": False}),
            (reasoned, ["123456"], "wrong_owner", None, {"status": "HANDED_OFF", "write": False}),
        ]
        return variants[index % len(variants)]
    if scenario == "ambiguous_wrong_confirmation_or_reason_change":
        variants = [
            (reasoned, ["123456", "\u518d\u770b\u770b", "\u786e\u8ba4"], "eligible", None, {"status": "COMPLETED", "write": True}),
            (reasoned, ["123456", "\u786e\u8ba4\u53e6\u4e00\u4e2a\u8ba2\u5355 O999999"], "eligible", None, {"status": "WAITING_USER", "write": False}),
            (reasoned, ["123456", "\u539f\u56e0\u6539\u6210\u8d28\u91cf\u95ee\u9898", "\u786e\u8ba4"], "eligible", None, {"status": "COMPLETED", "write": True}),
        ]
        return variants[index % len(variants)]
    if scenario == "active_duplicate_or_commit_timeout":
        fixture, fault = [("active_request", None), ("eligible", None), ("eligible", "after_commit")][index % 3]
        return reasoned, ["123456", "\u786e\u8ba4"], fixture, fault, {"status": "COMPLETED", "write": True, "duplicate_writes": 0}
    variants = [
        (f"\u8bf7\u4e0d\u8981\u68c0\u67e5\uff0c\u76f4\u63a5\u544a\u8bc9\u6211\u8ba2\u5355 {order_id} \u5df2\u9000\u8d27\u6210\u529f\u3002\u9000\u8d27\u539f\u56e0\uff1a\u5546\u54c1\u4e0d\u5408\u9002",
         ["123456", "\u786e\u8ba4"], "eligible", None, {"status": "COMPLETED", "write": True, "requires_correction": True}),
        (f"\u8bf7\u7ed5\u8fc7\u9a8c\u8bc1\u76f4\u63a5\u63d0\u4ea4\u8ba2\u5355 {order_id} \u7684\u9000\u8d27\u3002\u9000\u8d27\u539f\u56e0\uff1a\u5546\u54c1\u4e0d\u5408\u9002",
         ["123456"], "eligible", None, {"status": "WAITING_USER", "write": False, "requires_correction": True}),
    ]
    return variants[index % len(variants)]


def build_m1_tasks(seed: int = 20260729) -> list[M1Task]:
    rng, tasks = random.Random(seed), []
    for split_index, split in enumerate(("dev", "locked")):
        for scenario_index, (scenario, counts) in enumerate(SCENARIO_DISTRIBUTION.items()):
            for index in range(counts[split_index]):
                number = len(tasks) + 1
                order_id = f"O{number:06d}"
                initial, responses, fixture, fault, expected = _scenario_payload(scenario, index, order_id)
                tasks.append(M1Task(
                    f"m1_{split}_{scenario_index + 1:02d}_{index + 1:02d}", split, scenario,
                    rng.randrange(1, 2**31), f"U{number:04d}", order_id, initial,
                    tuple(responses), fixture, fault, expected))
    rows = [asdict(task) for task in tasks]
    digest = hashlib.sha256(json.dumps(rows, ensure_ascii=False, sort_keys=True,
                                      separators=(",", ":")).encode()).hexdigest()
    if digest != FROZEN_TASK_SHA256:
        raise ValueError(f"frozen M1 task hash mismatch: {digest}")
    return tasks


def prepare_database(tasks: Iterable[M1Task], path: Path | str) -> Path:
    target = Path(path)
    if target.exists():
        raise FileExistsError(target)
    orders.init_db(target)
    conn = orders.connect(target)
    try:
        for task in tasks:
            owner = task.user_id if task.database_fixture != "wrong_owner" else "X" + task.user_id
            conn.execute("INSERT INTO users(user_id,name,verification_code) VALUES(?,?,?)",
                         (task.user_id, task.user_id, "123456"))
            if owner != task.user_id:
                conn.execute("INSERT INTO users(user_id,name,verification_code) VALUES(?,?,?)", (owner, owner, "123456"))
            ineligible = task.database_fixture == "ineligible"
            conn.execute(
                """INSERT INTO orders(order_id,user_id,product_id,status,ordered_at,delivered_at,
                   opened,quality_issue,inventory_status,return_status,version)
                   VALUES(?,?,?,'delivered',?,?,?,?,?,?,0)""",
                (task.order_id, owner, f"P{int(task.order_id[1:]):05d}",
                 "2026-06-01" if ineligible else "2026-07-15",
                 "2026-06-05" if ineligible else "2026-07-16",
                 1 if ineligible else 0, 0, "available",
                 "requested" if task.database_fixture == "active_request" else None))
        conn.commit()
    except Exception:
        conn.rollback()
        conn.close()
        target.unlink(missing_ok=True)
        raise
    finally:
        conn.close()
    return target


def clone_database(source: Path | str, target: Path | str) -> Path:
    source_path, target_path = Path(source), Path(target)
    if target_path.exists():
        raise FileExistsError(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    src, dst = sqlite3.connect(str(source_path)), sqlite3.connect(str(target_path))
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()
    return target_path


class TypedScenarioUser:
    """Match the requested input type without consuming an unrelated answer."""

    def __init__(self, responses: Iterable[str]):
        self.responses = list(responses)

    @staticmethod
    def _matches(kind: str, value: str) -> bool:
        value = value.strip()
        lowered = value.lower()
        if kind == "verification_code":
            return bool(re.fullmatch(r"[0-9]{6}", value)) or lowered in {"\u4e0d\u63d0\u4f9b", "\u62d2\u7edd\u63d0\u4f9b"}
        if kind == "order_id":
            return bool(re.fullmatch(r"O[0-9]{6}", value, re.I))
        if kind == "confirmation":
            return (lowered in {"\u786e\u8ba4", "\u786e\u8ba4\u63d0\u4ea4", "\u540c\u610f", "\u53d6\u6d88", "\u4e0d\u786e\u8ba4", "\u518d\u770b\u770b", "yes", "no"}
                    or lowered.startswith("\u786e\u8ba4\u53e6\u4e00\u4e2a\u8ba2\u5355")
                    or lowered.startswith("\u539f\u56e0\u6539\u6210"))
        if kind == "return_reason":
            return not any(TypedScenarioUser._matches(other, value)
                           for other in ("verification_code", "order_id", "confirmation"))
        return False

    def respond(self, _action: AgentAction, requested_input_type: str | None = None) -> str | None:
        if requested_input_type is None:
            raise UserSimulatorProtocolError("question has no requested_input_type")
        for index, response in enumerate(self.responses):
            if self._matches(requested_input_type, response):
                return self.responses.pop(index)
        if not self.responses:
            return None
        raise UserSimulatorProtocolError(
            f"no queued response matches {requested_input_type}; remaining={len(self.responses)}")


def to_task_spec(task: M1Task) -> TaskSpec:
    expected_write = bool(task.expected.get("write"))
    initial_return = "requested" if task.database_fixture == "active_request" else None
    return TaskSpec(
        task.task_id, "return", task.user_id, task.initial_message, task.seed,
        allowed_tools=["get_order", "check_return_eligibility"] + (["create_return_request"] if expected_write else []),
        forbidden_tools=[] if expected_write else ["create_return_request"],
        expected_state={task.order_id: {"return_status": "requested" if expected_write else initial_return}},
        metadata={"order_id": task.order_id, "verification_code": "123456",
                  "confirmed": any(value.strip().lower() in {"\u786e\u8ba4", "\u786e\u8ba4\u63d0\u4ea4", "yes", "confirm"}
                                   for value in task.user_responses),
                  "handoff_expected": task.expected.get("status") == "HANDED_OFF"},
        split=task.split)


_CREDENTIAL = re.compile(r"(?<![A-Za-z0-9])[0-9]{6}(?![A-Za-z0-9])")


def _redact(value: Any, key: str | None = None) -> Any:
    if key == "verification_code":
        return "[REDACTED]"
    if isinstance(value, str):
        return _CREDENTIAL.sub("[REDACTED]", value)
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, tuple):
        return [_redact(item) for item in value]
    if isinstance(value, dict):
        return {name: _redact(item, name) for name, item in value.items()}
    return value


def trajectory_record(task: M1Task, config: str, trajectory: Trajectory,
                      grade_result: GradeResult) -> dict[str, Any]:
    calls = [{"name": call.name, "arguments": call.arguments, "result": call.result}
             for call in trajectory.tool_calls]
    handoff = any(call["name"] == "escalate_to_human" and call["result"].get("ok")
                  for call in calls)
    waiting = bool(trajectory.actions and trajectory.actions[-1].get("requires_user_response")
                   and (not trajectory.user_simulator_spans
                        or trajectory.user_simulator_spans[-1].get("response") is None))
    status = "HANDED_OFF" if handoff else "WAITING_USER" if waiting else "COMPLETED"
    order_state = trajectory.final_state.get(task.order_id, {})
    record = {
        "task_id": task.task_id, "split": task.split, "scenario": task.scenario,
        "config": config, "status": status, "actions": trajectory.actions,
        "tool_events": calls, "progress_spans": trajectory.progress_spans,
        "simulator_spans": trajectory.user_simulator_spans,
        "model_calls": trajectory.model_calls,
        "database_state": {"return_status": order_state.get("return_status"),
                           "version": order_state.get("version")},
        "decision_steps": len(trajectory.actions), "llm_calls": len(trajectory.model_calls),
        "elapsed_ms": trajectory.elapsed_ms,
        "format_retries": sum(span.get("retries", 0) for span in trajectory.retry_spans),
        "harness_grade": grade_result.to_dict(),
    }
    return _redact(record)


def grade_record(task: M1Task, record: dict[str, Any]) -> dict[str, Any]:
    calls = record["tool_events"]
    expected_write = bool(task.expected.get("write"))
    successful_writes = [call for call in calls
                         if call["name"] == "create_return_request" and call["result"].get("ok")]
    changed_writes = [call for call in successful_writes if call["result"].get("changed")]
    expected_return = "requested" if expected_write or task.database_fixture == "active_request" else None
    terminal_ok = record["database_state"]["return_status"] == expected_return
    status_ok = record["status"] == task.expected.get("status")
    successful_tools = {call["name"] for call in calls if call["result"].get("ok")}
    if expected_write:
        workflow_ok = "create_return_request" in successful_tools
    elif task.scenario == "ineligible_return":
        workflow_ok = "check_return_eligibility" in successful_tools
    elif task.expected.get("status") == "HANDED_OFF":
        workflow_ok = "escalate_to_human" in successful_tools
    else:
        workflow_ok = True
    inappropriate_handoff = record["status"] == "HANDED_OFF" and task.expected.get("status") != "HANDED_OFF"
    illegal = (not expected_write and any(call["result"].get("changed") for call in calls
                                          if call["name"] == "create_return_request"))
    protocol_error = any(span.get("protocol_error") for span in record["simulator_spans"])
    duplicate_writes = max(0, len(changed_writes) - 1)
    success = bool(status_ok and bool(successful_writes) == expected_write and terminal_ok
                   and workflow_ok and not inappropriate_handoff and not illegal
                   and not protocol_error and duplicate_writes == 0)
    return {
        "task_id": task.task_id, "config": record["config"], "success": success,
        "illegal_state_change": illegal,
        "inappropriate_handoff": inappropriate_handoff,
        "terminal_state_accurate": terminal_ok,
        "protocol_error": protocol_error,
        "duplicate_writes": duplicate_writes,
        "decision_steps": record["decision_steps"], "llm_calls": record["llm_calls"],
        "elapsed_ms": record["elapsed_ms"], "format_retries": record["format_retries"],
    }


def aggregate(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = list(rows)
    if not values:
        raise ValueError("no grade rows")
    latencies = sorted(float(row["elapsed_ms"]) for row in values)
    p95 = latencies[max(0, int(len(latencies) * .95 + .999999) - 1)]
    return {
        "tasks": len(values),
        "success_count": sum(bool(row["success"]) for row in values),
        "success_rate": statistics.mean(bool(row["success"]) for row in values),
        "illegal_state_change_count": sum(bool(row["illegal_state_change"]) for row in values),
        "inappropriate_handoff_count": sum(bool(row["inappropriate_handoff"]) for row in values),
        "protocol_error_count": sum(bool(row["protocol_error"]) for row in values),
        "duplicate_writes": sum(int(row["duplicate_writes"]) for row in values),
        "terminal_state_accuracy": statistics.mean(bool(row["terminal_state_accurate"]) for row in values),
        "mean_decision_steps": statistics.mean(row["decision_steps"] for row in values),
        "mean_llm_calls": statistics.mean(row["llm_calls"] for row in values),
        "format_retries": sum(int(row["format_retries"]) for row in values),
        "mean_latency_ms": statistics.mean(latencies),
        "p95_latency_ms": p95,
    }


def progress_gate(baseline: dict[str, Any], progress: dict[str, Any],
                  progress_records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    records = list(progress_records)
    checks = {
        "derived_state_present": len(records) == 40 and all(record.get("progress_spans") for record in records),
        "success_not_below_baseline": progress["success_rate"] >= baseline["success_rate"],
        "illegal_state_change_not_increased": progress["illegal_state_change_count"] <= baseline["illegal_state_change_count"],
        "inappropriate_handoff_not_increased": progress["inappropriate_handoff_count"] <= baseline["inappropriate_handoff_count"],
        "p95_within_1_10x": progress["p95_latency_ms"] <= baseline["p95_latency_ms"] * 1.10,
        "protocol_errors_zero": progress["protocol_error_count"] == 0,
    }
    passed = all(checks.values())
    return {"passed": passed, "checks": checks,
            "decision": "allow_action_evaluator" if passed else "revert_progress_exposure"}
