from __future__ import annotations

import pytest

from ecommerce_rag.agent_case import (
    AgentCase,
    build_dev_failure_agent_cases,
    load_agent_cases,
)
from ecommerce_rag.domain import AgentAction
from ecommerce_rag.harness import HarnessRunner, RulePolicy
from ecommerce_rag.legacy_closure import LegacyTaskProgressReducer
from ecommerce_rag.legacy_closure_benchmark import (
    TypedScenarioUser,
    build_m1_tasks,
    clone_database,
    grade_record,
    prepare_database,
    to_task_spec,
    trajectory_record,
)
from ecommerce_rag.orders import seed_database
from ecommerce_rag.tools import RetailTools


@pytest.mark.parametrize(
    "refusal",
    ["不提供", "拒绝提供", "拒绝提供验证码", "不愿提供验证码"],
)
def test_verification_code_refusal_blocks_writes_and_allows_handoff(refusal):
    history = [
        {"role": "user", "content": "订单 O000001 想退货。退货原因：商品不合适"},
        {"role": "assistant", "content": "请提供六位验证码",
         "requested_input_type": "verification_code"},
        {"role": "user", "content": refusal},
    ]
    progress = LegacyTaskProgressReducer().derive(history)
    assert progress.blocked_by == "verification_code_refused"
    assert progress.allowed_next_actions == ("handoff",)
    assert "verification_supplied" not in progress.completed
    assert progress.requested_input_type is None


def test_verification_refusal_is_not_a_code_and_does_not_repeat_ask():
    history = [
        {"role": "user", "content": "订单 O000001 想退货"},
        {"role": "assistant", "content": "请提供验证码",
         "requested_input_type": "verification_code"},
        {"role": "user", "content": "不提供"},
        {"role": "assistant", "content": "请再提供一次验证码",
         "requested_input_type": "verification_code"},
    ]
    progress = LegacyTaskProgressReducer().derive(history)
    assert progress.blocked_by == "verification_code_refused"
    assert "ask_user:verification_code" not in progress.allowed_next_actions


def test_real_verification_code_is_not_mistaken_for_refusal():
    progress = LegacyTaskProgressReducer().derive([
        {"role": "user", "content": "订单 O000001 想退货"},
        {"role": "assistant", "content": "请提供验证码",
         "requested_input_type": "verification_code"},
        {"role": "user", "content": "123456"},
    ])
    assert progress.blocked_by is None
    assert "verification_supplied" in progress.completed


def test_active_return_idempotent_success_contract(tmp_path):
    db = tmp_path / "env.sqlite"
    seed_database(db, users=20, orders=100)
    tools = RetailTools(db)
    order_id, user_id, code = "O000001", "U0001", None
    from ecommerce_rag.orders import connect
    conn = connect(db)
    try:
        row = conn.execute(
            "SELECT order_id,user_id FROM orders WHERE status='delivered' LIMIT 1"
        ).fetchone()
        order_id, user_id = row["order_id"], row["user_id"]
        code = conn.execute(
            "SELECT verification_code FROM users WHERE user_id=?", (user_id,)
        ).fetchone()[0]
        conn.execute(
            "UPDATE orders SET return_status='requested', delivered_at='2026-07-16', opened=0 WHERE order_id=?",
            (order_id,),
        )
        conn.commit()
        version_before = conn.execute(
            "SELECT version FROM orders WHERE order_id=?", (order_id,)
        ).fetchone()[0]
    finally:
        conn.close()

    result = tools.call(
        "create_return_request", order_id=order_id, user_id=user_id,
        verification_code=code, confirmed=True)
    assert result["ok"] is True
    assert result["changed"] is False
    assert result["idempotent_replay"] is True
    assert result["request_id"] == f"RR-{order_id}"
    assert result["status"] == "active"

    conn = connect(db)
    try:
        row = conn.execute(
            "SELECT return_status, version FROM orders WHERE order_id=?", (order_id,)
        ).fetchone()
        assert row["return_status"] == "requested"
        assert row["version"] == version_before
    finally:
        conn.close()

    progress = LegacyTaskProgressReducer().derive([
        {"role": "user", "content": f"订单 {order_id} 想退货，验证码 {code}。退货原因：不合适"},
        {"role": "tool", "name": "check_return_eligibility",
         "result": {"ok": True, "eligible": True, "order": {"order_id": order_id}}},
        {"role": "assistant", "content": "请确认", "requested_input_type": "confirmation"},
        {"role": "user", "content": "确认"},
        {"role": "tool", "name": "create_return_request", "result": result},
    ])
    assert "return_request_created" in progress.completed
    assert progress.guard_state == "complete"


def test_fresh_return_still_changes_state(tmp_path):
    db = tmp_path / "env.sqlite"
    seed_database(db, users=20, orders=100)
    from ecommerce_rag.orders import connect
    conn = connect(db)
    try:
        row = conn.execute(
            """SELECT o.order_id,o.user_id,u.verification_code FROM orders o
               JOIN users u ON u.user_id=o.user_id
               WHERE o.status='delivered' AND o.return_status IS NULL
               AND o.opened=0 AND o.delivered_at >= '2026-07-13' LIMIT 1"""
        ).fetchone()
        order_id, user_id, code = row["order_id"], row["user_id"], row["verification_code"]
    finally:
        conn.close()
    first = RetailTools(db).call(
        "create_return_request", order_id=order_id, user_id=user_id,
        verification_code=code, confirmed=True)
    second = RetailTools(db).call(
        "create_return_request", order_id=order_id, user_id=user_id,
        verification_code=code, confirmed=True)
    assert first["ok"] and first["changed"] and not first["idempotent_replay"]
    assert second["ok"] and not second["changed"] and second["idempotent_replay"]
    assert first["request_id"] == second["request_id"] == f"RR-{order_id}"


def test_protocol_tasks_grade_success_with_rule_policy(tmp_path):
    tasks = [task for task in build_m1_tasks() if task.task_id in {
        "m1_dev_03_02", "m1_dev_03_05", "m1_dev_06_01", "m1_dev_06_04"}]
    pristine = prepare_database(tasks, tmp_path / "pristine.sqlite")
    for task in tasks:
        db = clone_database(pristine, tmp_path / f"{task.task_id}.sqlite")
        runner = HarnessRunner(
            db, policy=RulePolicy(), progress_reducer=LegacyTaskProgressReducer(),
            expose_task_progress=True,
            user_simulator_factory=lambda _task, responses=task.user_responses: TypedScenarioUser(responses),
        )
        trajectory, grade = runner.run(to_task_spec(task))
        record = trajectory_record(task, "protocol_fix", trajectory, grade)
        graded = grade_record(task, record)
        assert graded["success"], (task.task_id, graded, record["status"], record["tool_events"])
        assert not graded["illegal_state_change"]
        assert not graded["protocol_error"]


def test_agent_cases_v1_cover_six_failures_and_forbid_training():
    path = "ecommerce_rag/data/agent_cases_dev_failures_v1.jsonl"
    cases = load_agent_cases(path)
    assert len(cases) == 6
    assert {case.task_id for case in cases} == {
        "m1_dev_03_02", "m1_dev_03_05", "m1_dev_06_01",
        "m1_dev_06_04", "m1_dev_07_01", "m1_dev_07_03",
    }
    assert all(case.training_approved is False for case in cases)
    assert all(case.split == "dev" for case in cases)
    owners = {case.task_id: case.failure_owner for case in cases}
    assert owners["m1_dev_03_02"] == owners["m1_dev_03_05"] == "progress"
    assert owners["m1_dev_06_01"] == owners["m1_dev_06_04"] == "tool"
    assert owners["m1_dev_07_01"] == owners["m1_dev_07_03"] == "policy"
    rebuilt = build_dev_failure_agent_cases(code_commit="b466db7379c117d0f3bcc9617d4c82f0c3f23f99")
    assert [case.case_id for case in rebuilt] == [case.case_id for case in cases]


def test_agent_case_rejects_training_approved_on_dev():
    with pytest.raises(ValueError):
        AgentCase(
            case_id="bad", split="dev", training_approved=True,
            user_goal="x", failure_owner="policy",
        )


def test_typed_user_accepts_verification_refusal_phrases():
    user = TypedScenarioUser(["拒绝提供验证码"])
    action = AgentAction.answer("请提供验证码", requires_user_response=True)
    assert user.respond(action, "verification_code") == "拒绝提供验证码"
