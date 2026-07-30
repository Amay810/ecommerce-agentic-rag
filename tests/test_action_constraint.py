from __future__ import annotations

from ecommerce_rag.action_constraint import (
    apply_action_constraint,
    contract_from_progress,
)
from ecommerce_rag.domain import AgentAction, TaskSpec
from ecommerce_rag.harness import HarnessRunner
from ecommerce_rag.legacy_closure import LegacyTaskProgressReducer, TaskProgress
from ecommerce_rag.legacy_closure_benchmark import (
    TypedScenarioUser,
    action_constraint_gate,
    build_m1_tasks,
    clone_database,
    grade_record,
    prepare_database,
    to_task_spec,
    trajectory_record,
)
from ecommerce_rag.orders import seed_database


def _identity_progress():
    return TaskProgress(
        "return_resolution",
        ("order_id_collected", "return_reason_collected"),
        ("identity_verification",),
        "user_input",
        ("ask_user:verification_code",),
        "verification_code",
        "identity_required",
    )


def test_contract_forbids_handoff_and_terminal_when_identity_pending():
    contract = contract_from_progress(_identity_progress())
    assert contract is not None
    assert contract.allowed_actions == ("ask_user:verification_code",)
    assert contract.preferred_action == "ask_user:verification_code"
    assert contract.terminal_allowed is False
    assert "handoff" in contract.forbidden_actions
    assert "final_answer" in contract.forbidden_actions


def test_constraint_remaps_handoff_without_extra_llm_call():
    result = apply_action_constraint(
        AgentAction.handoff("user asked to skip checks"),
        _identity_progress(),
        requested_input_type=None,
    )
    assert result.remapped and result.accepted and not result.fail_closed
    assert result.llm_calls_added == 0
    assert result.action.action_type == "final_answer"
    assert result.action.requires_user_response is True
    assert "验证码" in result.action.content


def test_constraint_remaps_premature_terminal_success():
    result = apply_action_constraint(
        AgentAction.answer("退货已成功。"),
        _identity_progress(),
        requested_input_type=None,
    )
    assert result.remapped
    assert result.original_action_key == "final_answer"
    assert result.action.requires_user_response is True


def test_constraint_passes_legal_ask():
    result = apply_action_constraint(
        AgentAction.answer("请提供六位验证码。", requires_user_response=True),
        _identity_progress(),
        requested_input_type="verification_code",
    )
    assert result.accepted and not result.remapped and not result.fail_closed


def test_harness_constraint_recovers_premature_handoff_tasks(tmp_path):
    tasks = [task for task in build_m1_tasks()
             if task.task_id in {"m1_dev_07_01", "m1_dev_07_03"}]
    pristine = prepare_database(tasks, tmp_path / "pristine.sqlite")

    class BadThenUseful:
        privileged = False

        def __init__(self):
            self.calls = 0

        def act(self, observation):
            self.calls += 1
            progress = (observation.session or {}).get("task_progress") or {}
            if progress.get("guard_state") == "identity_required" and self.calls == 1:
                return AgentAction.handoff("skip checks")
            text = "\n".join(
                str(item.get("content", "")) for item in observation.history
                if item.get("role") == "user")
            if "123456" in text and "确认" not in text.lower():
                # after code supplied
                order_id = next(
                    (item for item in text.split() if item.startswith("O")),
                    progress.get("order_id") or "O000037",
                )
                # find O###### 
                import re
                match = re.search(r"O\d{6}", text)
                order_id = match.group(0) if match else "O000037"
                return AgentAction.tool_call(
                    "check_return_eligibility", order_id=order_id,
                    user_id=observation.session["user_id"], verification_code="123456")
            if any(item.get("role") == "tool" and item.get("name") == "check_return_eligibility"
                   for item in observation.history):
                tools = [item for item in observation.history if item.get("role") == "tool"]
                last = tools[-1] if tools else None
                if last and last.get("name") == "create_return_request":
                    return AgentAction.answer("退货申请已处理。")
                if last and last.get("name") == "check_return_eligibility":
                    if "确认" in text:
                        import re
                        match = re.search(r"O\d{6}", text)
                        order_id = match.group(0) if match else "O000037"
                        return AgentAction.tool_call(
                            "create_return_request", order_id=order_id,
                            user_id=observation.session["user_id"],
                            verification_code="123456", confirmed=True)
                    return AgentAction.answer("是否确认提交退货？", requires_user_response=True)
            return AgentAction.answer("请提供六位验证码。", requires_user_response=True)

    for task in tasks:
        db = clone_database(pristine, tmp_path / f"{task.task_id}.sqlite")
        policy = BadThenUseful()
        trajectory, _ = HarnessRunner(
            db, policy=policy, max_steps=8,
            progress_reducer=LegacyTaskProgressReducer(),
            expose_task_progress=True,
            enforce_action_constraint=True,
            user_simulator_factory=lambda _spec, current=task: TypedScenarioUser(
                current.user_responses),
        ).run(to_task_spec(task))
        record = trajectory_record(task, "constrained", trajectory, _)
        graded = grade_record(task, record)
        assert any(span.get("remapped") for span in trajectory.constraint_spans)
        assert all(span.get("llm_calls_added", 0) == 0 for span in trajectory.constraint_spans)
        assert graded["success"], (task.task_id, graded, record["status"], record["actions"])


def test_action_constraint_gate_requires_observe_recovery_without_regression():
    fixed = [{"task_id": f"m1_dev_{i:02d}_01", "success": i != 7}
             for i in range(1, 41)]
    # simplify: use real task ids from builder
    tasks = [task.task_id for task in build_m1_tasks() if task.split == "dev"]
    fixed = [{"task_id": tid, "success": tid not in {"m1_dev_07_01", "m1_dev_07_03"}}
             for tid in tasks]
    constrained = []
    for row in fixed:
        success = row["success"] or row["task_id"] == "m1_dev_07_01"
        constrained.append({**row, "success": success})
    records = []
    for row in constrained:
        records.append({
            "task_id": row["task_id"],
            "status": "COMPLETED" if row["success"] else "HANDED_OFF",
            "tool_events": (
                [{"name": "create_return_request", "result": {"ok": True, "changed": True}}]
                if row["task_id"] == "m1_dev_07_01" and row["success"] else []),
            "constraint_spans": [{"remapped": row["task_id"] == "m1_dev_07_01",
                                  "llm_calls_added": 0}],
            "correction_spans": [],
        })
    summary_fixed = {"success_count": 38, "illegal_state_change_count": 0,
                     "protocol_error_count": 0, "duplicate_writes": 0}
    summary_constrained = {"success_count": 39, "illegal_state_change_count": 0,
                           "protocol_error_count": 0, "duplicate_writes": 0}
    gate = action_constraint_gate(
        summary_fixed, summary_constrained, fixed, constrained, records)
    assert gate["passed"]
    assert gate["diagnostics"]["observe_policy_recovered_task_ids"] == ["m1_dev_07_01"]
