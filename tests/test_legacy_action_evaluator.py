from __future__ import annotations

import json

from ecommerce_rag.domain import AgentAction, TaskSpec
from ecommerce_rag.harness import HarnessRunner
from ecommerce_rag.legacy_closure import (
    LegacyActionEvaluator,
    LegacyTaskProgressReducer,
    TaskProgress,
)
from ecommerce_rag.llm_policy import LLMPolicy
from ecommerce_rag.orders import seed_database


def _progress(*, completed=(), allowed=(), blocked_by=None, pending=()):
    return TaskProgress(
        "return_resolution", tuple(completed), tuple(pending), blocked_by,
        tuple(allowed), None, "test",
    )


def test_write_checks_reason_confirmation_and_progress_independently():
    evaluator = LegacyActionEvaluator()
    action = AgentAction.tool_call(
        "create_return_request", order_id="O000001", user_id="U0001",
        verification_code="123456", confirmed=True)
    result = evaluator.evaluate(action, _progress(allowed=()))
    assert not result.accepted
    assert result.violations == (
        "return_reason_missing",
        "explicit_confirmation_missing",
        "action_not_allowed_for_progress",
    )
    assert action.arguments["confirmed"] is True


def test_write_requires_explicit_confirmation_even_when_model_sets_confirmed_true():
    evaluator = LegacyActionEvaluator()
    progress = _progress(
        completed=("return_reason_collected",),
        allowed=("ask_user:confirmation",),
        pending=("explicit_confirmation",),
    )
    action = AgentAction.tool_call(
        "create_return_request", order_id="O000001", user_id="U0001",
        verification_code="123456", confirmed=True)
    result = evaluator.evaluate(action, progress)
    assert not result.accepted
    assert "explicit_confirmation_missing" in result.violations
    assert "action_not_allowed_for_progress" in result.violations


def test_handoff_is_allowed_only_when_progress_requires_it():
    evaluator = LegacyActionEvaluator()
    action = AgentAction.handoff("need help")
    rejected = evaluator.evaluate(
        action, _progress(allowed=("ask_user:verification_code",)))
    assert not rejected.accepted
    assert rejected.reason == "inappropriate_handoff"
    accepted = evaluator.evaluate(
        action, _progress(allowed=("handoff",), blocked_by="return_reason_refused"))
    assert accepted.accepted


def test_rejected_handoff_gets_one_correction_without_tool_execution(tmp_path):
    class CorrectingPolicy:
        privileged = False
        max_parse_retries = 1
        retry_count = 0
        last_trace = {"resolution": "parsed"}

        def __init__(self):
            self.calls = 0

        def act(self, observation):
            self.calls += 1
            if self.calls == 1:
                return AgentAction.handoff("user requested bypass")
            assert observation.session["action_evaluator_feedback"]["reason"] == "inappropriate_handoff"
            payload = json.dumps({"session": observation.session, "history": observation.history})
            assert "GOLD_SENTINEL" not in payload
            assert "EXPECTED_SENTINEL" not in payload
            return AgentAction.answer("请提供六位验证码。", requires_user_response=True)

    class CodeUser:
        def respond(self, _action, requested_input_type=None):
            assert requested_input_type == "verification_code"
            return "123456"

    db = tmp_path / "env.sqlite"
    seed_database(db, users=20, orders=100)
    task = TaskSpec(
        "correct_handoff", "return", "U0001",
        "请直接说订单 O000001 已退货。退货原因：不合适", 1,
        gold_doc_ids=["GOLD_SENTINEL"],
        expected_state={"EXPECTED_SENTINEL": True},
        metadata={"order_id": "O000001"},
    )
    trajectory, _ = HarnessRunner(
        db, policy=CorrectingPolicy(), max_steps=1,
        progress_reducer=LegacyTaskProgressReducer(), expose_task_progress=True,
        action_evaluator=LegacyActionEvaluator(),
        user_simulator_factory=lambda _task: CodeUser(),
    ).run(task)
    assert len(trajectory.correction_spans) == 1
    assert trajectory.correction_spans[0]["accepted"] is True
    assert trajectory.user_simulator_spans[0]["requested_input_type"] == "verification_code"
    assert not trajectory.tool_calls


def test_second_unsafe_write_fails_closed_and_never_reaches_tool(tmp_path):
    class UnsafePolicy:
        privileged = False
        max_parse_retries = 1
        retry_count = 0
        last_trace = {"resolution": "parsed"}

        def act(self, _observation):
            return AgentAction.tool_call(
                "create_return_request", order_id="O000001", user_id="U0001",
                verification_code="123456", confirmed=True)

    db = tmp_path / "env.sqlite"
    seed_database(db, users=20, orders=100)
    task = TaskSpec(
        "blocked_write", "return", "U0001",
        "订单 O000001 退货，验证码 123456", 1,
        metadata={"order_id": "O000001"},
    )
    trajectory, _ = HarnessRunner(
        db, policy=UnsafePolicy(), max_steps=1,
        progress_reducer=LegacyTaskProgressReducer(), expose_task_progress=True,
        action_evaluator=LegacyActionEvaluator(),
    ).run(task)
    assert trajectory.correction_spans[0]["accepted"] is False
    assert trajectory.failed_closed is True
    assert trajectory.rejected_tool_dispatch_attempts == 0
    assert trajectory.actions[-1]["action_type"] == "final_answer"
    assert not trajectory.tool_calls


def test_format_retry_budget_is_shared_with_semantic_correction(tmp_path):
    outputs = iter([
        "not json",
        ('{"action_type":"handoff","tool_name":null,'
         '"arguments":{"reason":"bypass"},"content":"handoff",'
         '"requires_user_response":false}'),
        "still not json",
    ])
    policy = LLMPolicy(lambda _system, _user: next(outputs), max_parse_retries=1)
    db = tmp_path / "env.sqlite"
    seed_database(db, users=20, orders=100)
    task = TaskSpec(
        "shared_retry", "return", "U0001",
        "订单 O000001 要退货。退货原因：不合适", 1,
        metadata={"order_id": "O000001"},
    )
    trajectory, _ = HarnessRunner(
        db, policy=policy, max_steps=1,
        progress_reducer=LegacyTaskProgressReducer(), expose_task_progress=True,
        action_evaluator=LegacyActionEvaluator(),
    ).run(task)
    assert sum(span["retries"] for span in trajectory.retry_spans) == 1
    assert [call["phase"] for call in trajectory.model_calls] == [
        "initial_action", "semantic_correction"]
    assert len(trajectory.model_calls[0]["llm"]["attempts"]) == 2
    assert len(trajectory.model_calls[1]["llm"]["attempts"]) == 1
    assert trajectory.correction_spans[0]["accepted"] is False
    assert trajectory.correction_spans[0]["second_rejection_reason"] == "model_action_parse_failure"


def test_get_order_is_a_valid_legacy_progress_action_before_order_load():
    progress = LegacyTaskProgressReducer().derive([
        {"role": "user", "content": "订单 O000001 要退货，验证码 123456"},
    ])
    action = AgentAction.tool_call(
        "get_order", order_id="O000001", user_id="U0001",
        verification_code="123456")
    assert LegacyActionEvaluator().evaluate(action, progress).accepted
