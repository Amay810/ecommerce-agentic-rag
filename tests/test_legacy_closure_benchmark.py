from __future__ import annotations

import sqlite3

import pytest

from ecommerce_rag.domain import AgentAction
from ecommerce_rag.harness import UserSimulatorProtocolError, _requested_input_type
from ecommerce_rag.legacy_closure_benchmark import (
    FROZEN_TASK_SHA256,
    FROZEN_ACTION_CORRECTION_CHALLENGE_SHA256,
    TypedScenarioUser,
    _redact,
    action_evaluator_gate,
    build_action_correction_challenges,
    build_m1_tasks,
    clone_database,
    prepare_database,
    progress_gate,
)


def test_frozen_m1_manifest_shape_and_hash():
    tasks = build_m1_tasks()
    assert FROZEN_TASK_SHA256 == "e4346e3f99261d203f9fea57aeec48d58e5f769d9a1e856e43b9cf0b74a6c8e3"
    assert len(tasks) == len({task.task_id for task in tasks}) == 60
    assert sum(task.split == "dev" for task in tasks) == 40
    assert sum(task.split == "locked" for task in tasks) == 20


def test_frozen_action_correction_challenge_is_separate_from_main_tasks():
    tasks = build_action_correction_challenges()
    assert FROZEN_ACTION_CORRECTION_CHALLENGE_SHA256 == (
        "f3443e8d2336aa9c66bb4da37972597688c9edcf8e93b397e4a01f77cf0de729")
    assert len(tasks) == len({task.task_id for task in tasks}) == 10
    assert {task.split for task in tasks} == {"correction_challenge"}
    assert not ({task.task_id for task in tasks}
                & {task.task_id for task in build_m1_tasks()})


def test_database_is_pristine_and_clones_are_independent(tmp_path):
    tasks = build_m1_tasks()
    pristine = prepare_database(tasks, tmp_path / "pristine.sqlite")
    first = clone_database(pristine, tmp_path / "first.sqlite")
    second = clone_database(pristine, tmp_path / "second.sqlite")
    with sqlite3.connect(first) as conn:
        conn.execute("UPDATE orders SET return_status='requested' WHERE order_id=?", (tasks[0].order_id,))
        conn.commit()
    with sqlite3.connect(second) as conn:
        assert conn.execute("SELECT return_status FROM orders WHERE order_id=?",
                            (tasks[0].order_id,)).fetchone()[0] is None
    with pytest.raises(FileExistsError):
        prepare_database(tasks, pristine)
    with pytest.raises(FileExistsError):
        clone_database(pristine, first)


def test_typed_user_does_not_consume_confirmation_as_verification():
    user = TypedScenarioUser(["确认"])
    action = AgentAction.answer("请输入验证码", requires_user_response=True)
    with pytest.raises(UserSimulatorProtocolError):
        user.respond(action, "verification_code")
    assert user.responses == ["确认"]
    assert user.respond(action, "confirmation") == "确认"


@pytest.mark.parametrize(
    ("task_id", "question"),
    [
        ("m1_dev_03_01", "订单已确认可退货，请告诉我退货的具体原因。"),
        ("m1_dev_03_04", "已确认该订单符合条件；还需要您提供退货原因。"),
        ("m1_dev_03_07", "I confirmed eligibility. What is the return reason?"),
    ],
)
def test_return_reason_question_wins_over_confirmation_recap(task_id, question):
    action = AgentAction.answer(question, requires_user_response=True)
    assert _requested_input_type(action, None) == "return_reason", task_id


def test_confirmation_question_wins_when_reason_is_only_a_recap():
    action = AgentAction.answer(
        "退货原因是商品不合适，请确认是否提交。", requires_user_response=True)
    assert _requested_input_type(action, None) == "confirmation"


def test_typed_user_consumes_return_reason_for_mixed_wording():
    user = TypedScenarioUser(["不提供退货原因"])
    action = AgentAction.answer(
        "订单已确认可退货，请告诉我退货的具体原因。", requires_user_response=True)
    kind = _requested_input_type(action, None)
    assert kind == "return_reason"
    assert user.respond(action, kind) == "不提供退货原因"
    assert user.responses == []


def _grade(task_id, *, success, handoff=False):
    return {"task_id": task_id, "success": success,
            "inappropriate_handoff": handoff}


def test_progress_gate_enforces_layer_responsibility():
    baseline = {"success_rate": .5, "illegal_state_change_count": 1,
                "inappropriate_handoff_count": 0, "p95_latency_ms": 100,
                "protocol_error_count": 0}
    progress = {**baseline, "p95_latency_ms": 150}
    records = [{"progress_spans": [{"step": 0}]} for _ in range(40)]
    baseline_grades = [_grade(str(index), success=index < 20) for index in range(40)]
    progress_grades = [_grade(str(index), success=index < 20) for index in range(40)]
    gate = progress_gate(baseline, progress, baseline_grades, progress_grades, records)
    assert gate["passed"]
    assert gate["latency_diagnostic"]["p95_ratio"] == 1.5
    progress["success_rate"] = .475
    progress_grades[0] = _grade("0", success=False)
    failed = progress_gate(baseline, progress, baseline_grades, progress_grades, records)
    assert not failed["passed"]
    assert failed["decision"] == "hold_for_progress_diagnosis"
    assert failed["paired_diagnostics"]["success_regression_task_ids"] == ["0"]


def test_progress_gate_ignores_safer_handoff_on_existing_failure_only():
    baseline = {"success_rate": .5, "illegal_state_change_count": 1,
                "inappropriate_handoff_count": 0, "p95_latency_ms": 100,
                "protocol_error_count": 0}
    progress = {**baseline, "inappropriate_handoff_count": 1}
    records = [{"progress_spans": [{"step": 0}]} for _ in range(40)]
    baseline_grades = [_grade(str(index), success=index < 20) for index in range(40)]
    progress_grades = [_grade(str(index), success=index < 20, handoff=index == 30)
                       for index in range(40)]
    assert progress_gate(baseline, progress, baseline_grades,
                         progress_grades, records)["passed"]


def test_progress_gate_rejects_new_handoff_on_baseline_success():
    baseline = {"success_rate": .5, "illegal_state_change_count": 1,
                "inappropriate_handoff_count": 0, "p95_latency_ms": 100,
                "protocol_error_count": 0}
    progress = {**baseline}
    records = [{"progress_spans": [{"step": 0}]} for _ in range(40)]
    baseline_grades = [_grade(str(index), success=index < 20) for index in range(40)]
    progress_grades = [_grade(str(index), success=index < 20, handoff=index == 3)
                       for index in range(40)]
    gate = progress_gate(baseline, progress, baseline_grades, progress_grades, records)
    assert not gate["passed"]
    assert gate["paired_diagnostics"][
        "new_inappropriate_handoff_on_baseline_success_task_ids"] == ["3"]


def test_progress_gate_rejects_duplicate_or_missing_task_pairs():
    baseline = {"success_rate": .5, "illegal_state_change_count": 1,
                "inappropriate_handoff_count": 0, "p95_latency_ms": 100,
                "protocol_error_count": 0}
    progress = {**baseline}
    records = [{"progress_spans": [{"step": 0}]} for _ in range(40)]
    baseline_grades = [_grade(str(index), success=index < 20) for index in range(40)]
    progress_grades = [_grade(str(index), success=index < 20) for index in range(39)]
    progress_grades.append(_grade("38", success=False))
    gate = progress_gate(baseline, progress, baseline_grades, progress_grades, records)
    assert not gate["passed"]
    assert not gate["checks"]["paired_task_sets_match"]


def _action_summary(success_count, *, eligible=10, recovered=5):
    return {
        "success_count": success_count,
        "illegal_state_change_count": 0,
        "terminal_state_accuracy": .9,
        "inappropriate_handoff_count": 0,
        "protocol_error_count": 0,
        "rejected_tool_execution_count": 0,
        "correction_eligible_tasks": eligible,
        "correction_recovery_count": recovered,
        "correction_task_recovery_rate": recovered / eligible if eligible else None,
    }


def test_action_evaluator_gate_uses_paired_gain_and_fixed_refusal_proof():
    fixed_grades = [_grade(str(index), success=index < 20) for index in range(40)]
    evaluated_grades = [_grade(str(index), success=index < 21) for index in range(40)]
    fixed_records = [{"task_id": str(index), "progress_spans": [], "tool_events": [],
                      "database_state": {"return_status": None}}
                     for index in range(40)]
    for task_id in ("m1_dev_03_01", "m1_dev_03_04", "m1_dev_03_07"):
        fixed_grades.append(_grade(task_id, success=False))
        evaluated_grades.append(_grade(task_id, success=False))
        fixed_records.append({
            "task_id": task_id,
            "progress_spans": [{
                "blocked_by": "return_reason_refused",
                "completed": [],
                "allowed_next_actions": ["handoff"],
            }],
            "tool_events": [],
            "database_state": {"return_status": None},
        })
    # Keep the formal gate shape at exactly 40 paired tasks.
    fixed_grades = fixed_grades[:37] + fixed_grades[-3:]
    evaluated_grades = evaluated_grades[:37] + evaluated_grades[-3:]
    fixed_records = fixed_records[:37] + fixed_records[-3:]
    gate = action_evaluator_gate(
        _action_summary(20), _action_summary(21),
        fixed_grades, evaluated_grades, fixed_records,
        _action_summary(0, eligible=0, recovered=0))
    assert gate["passed"]
    assert gate["decision"] == "allow_completion_evaluator"


def test_action_evaluator_gate_holds_when_correction_sample_is_too_small():
    fixed_grades = [_grade(str(index), success=index < 20) for index in range(37)]
    evaluated_grades = [_grade(str(index), success=index < 21) for index in range(37)]
    fixed_records = [{"task_id": str(index), "progress_spans": [], "tool_events": [],
                      "database_state": {"return_status": None}}
                     for index in range(37)]
    for task_id in ("m1_dev_03_01", "m1_dev_03_04", "m1_dev_03_07"):
        fixed_grades.append(_grade(task_id, success=False))
        evaluated_grades.append(_grade(task_id, success=False))
        fixed_records.append({
            "task_id": task_id,
            "progress_spans": [{
                "blocked_by": "return_reason_refused", "completed": [],
                "allowed_next_actions": ["handoff"],
            }],
            "tool_events": [],
            "database_state": {"return_status": None},
        })
    gate = action_evaluator_gate(
        _action_summary(20), _action_summary(21, eligible=2, recovered=2),
        fixed_grades, evaluated_grades, fixed_records,
        _action_summary(0, eligible=7, recovered=4))
    assert not gate["passed"]
    assert not gate["checks"]["correction_sample_size_at_least_ten"]


def test_record_redaction_removes_credentials_without_destroying_order_ids():
    payload = _redact({
        "verification_code": "123456",
        "prompt": "user supplied 123456 for O000001",
        "arguments": {"order_id": "O000001"},
    })
    assert "123456" not in str(payload)
    assert payload["prompt"] == "user supplied [REDACTED] for O000001"
    assert payload["arguments"]["order_id"] == "O000001"
