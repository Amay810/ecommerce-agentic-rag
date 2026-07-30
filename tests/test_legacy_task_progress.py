from __future__ import annotations

import json

import pytest

from ecommerce_rag.domain import AgentAction, TaskSpec
from ecommerce_rag.harness import HarnessRunner
from ecommerce_rag.legacy_closure import LegacyTaskProgressReducer
from ecommerce_rag.orders import seed_database


def test_progress_is_derived_from_visible_events_and_never_contains_the_code():
    history = [
        {"role": "user", "content": "订单 O000001 想退货"},
        {"role": "assistant", "content": "请提供验证码", "requested_input_type": "verification_code"},
        {"role": "user", "content": "123456"},
        {"role": "assistant", "content": "checking", "requested_input_type": None},
        {"role": "tool", "name": "check_return_eligibility",
         "result": {"ok": True, "eligible": True, "order": {"order_id": "O000001"}}},
    ]
    progress = LegacyTaskProgressReducer().derive(history)
    payload = json.dumps(progress.to_dict(), ensure_ascii=False)
    assert progress.pending == ("return_reason",)
    assert progress.requested_input_type == "return_reason"
    assert progress.allowed_next_actions == ("ask_user:return_reason",)
    assert "verification_supplied" in progress.completed
    assert "123456" not in payload


def test_order_change_invalidates_identity_and_downstream_progress():
    history = [
        {"role": "user", "content": "订单 O000001 想退货，验证码 123456"},
        {"role": "tool", "name": "check_return_eligibility",
         "result": {"ok": True, "eligible": True, "order": {"order_id": "O000001"}}},
        {"role": "user", "content": "改成订单 O000002"},
    ]
    progress = LegacyTaskProgressReducer().derive(history)
    assert progress.pending == ("identity_verification",)
    assert progress.requested_input_type == "verification_code"
    assert "identity_verified" not in progress.completed


def test_non_return_conversation_has_no_task_closure_directive():
    progress = LegacyTaskProgressReducer().derive([
        {"role": "user", "content": "比较一下这两款耳机"},
    ])
    assert progress.workflow == "unclassified"
    assert progress.allowed_next_actions == ()


def test_eligible_order_without_user_reason_requests_reason_before_confirmation():
    progress = LegacyTaskProgressReducer().derive([
        {"role": "user", "content": "我要退货，订单 O000001，验证码 123456"},
        {"role": "tool", "name": "check_return_eligibility",
         "result": {"ok": True, "eligible": True, "order": {"order_id": "O000001"}}},
    ])
    assert progress.pending == ("return_reason",)
    assert progress.requested_input_type == "return_reason"
    assert progress.allowed_next_actions == ("ask_user:return_reason",)


@pytest.mark.parametrize(
    "refusal",
    ["不提供退货原因", "拒绝说明原因", "不提供", "退货原因：不提供"],
)
def test_return_reason_refusal_blocks_write_and_is_not_stored(refusal):
    history = [
        {"role": "user", "content": "我要退货，订单 O000001，验证码 123456"},
        {"role": "tool", "name": "check_return_eligibility",
         "result": {"ok": True, "eligible": True,
                    "order": {"order_id": "O000001"}}},
        {"role": "assistant", "content": "请提供退货原因",
         "requested_input_type": "return_reason"},
        {"role": "user", "content": refusal},
    ]
    reducer = LegacyTaskProgressReducer()
    facts = reducer.facts(history)
    progress = reducer.derive(history)
    assert facts["return_reason"] is None
    assert progress.blocked_by == "return_reason_refused"
    assert "return_reason_collected" not in progress.completed
    assert progress.allowed_next_actions == ("handoff",)
    assert progress.requested_input_type is None


def test_real_return_reason_is_preserved_and_advances_to_confirmation():
    history = [
        {"role": "user", "content": "我要退货，订单 O000001，验证码 123456"},
        {"role": "tool", "name": "check_return_eligibility",
         "result": {"ok": True, "eligible": True,
                    "order": {"order_id": "O000001"}}},
        {"role": "assistant", "content": "请提供退货原因",
         "requested_input_type": "return_reason"},
        {"role": "user", "content": "不合适"},
    ]
    reducer = LegacyTaskProgressReducer()
    facts = reducer.facts(history)
    progress = reducer.derive(history)
    assert facts["return_reason"] == "不合适"
    assert facts["blocked_by"] is None
    assert progress.blocked_by != "return_reason_refused"
    assert "return_reason_collected" in progress.completed
    assert progress.allowed_next_actions == ("ask_user:confirmation",)


def test_identity_available_allows_legacy_get_order_or_direct_eligibility_check():
    progress = LegacyTaskProgressReducer().derive([
        {"role": "user", "content": "订单 O000001 要退货，验证码 123456"},
    ])
    assert progress.allowed_next_actions == (
        "get_order", "check_return_eligibility")

    loaded = LegacyTaskProgressReducer().derive([
        {"role": "user", "content": "订单 O000001 要退货，验证码 123456"},
        {"role": "tool", "name": "get_order",
         "result": {"ok": True, "order": {"order_id": "O000001"}}},
    ])
    assert loaded.allowed_next_actions == ("check_return_eligibility",)


def test_harness_exposes_progress_only_when_enabled_and_never_exposes_gold(tmp_path):
    class SpyPolicy:
        privileged = False

        def __init__(self):
            self.observation = None

        def act(self, observation):
            self.observation = observation
            return AgentAction.answer("done")

    db = tmp_path / "env.sqlite"
    seed_database(db, users=20, orders=100)
    task = TaskSpec(
        "progress_leak", "hidden_category", "U0001", "订单 O000001 想退货", 1,
        gold_doc_ids=["GOLD_SENTINEL"], allowed_tools=["SECRET_TOOL"],
        expected_state={"SECRET_STATE": True}, split="SPLIT_SENTINEL",
        answer_expectations={"SECRET_EXPECTATION": True},
        expected_tool_sequence=["SECRET_SEQUENCE"],
    )
    reducer = LegacyTaskProgressReducer()

    baseline = SpyPolicy()
    HarnessRunner(db, policy=baseline, progress_reducer=reducer).run(task)
    assert "task_progress" not in baseline.observation.session

    exposed = SpyPolicy()
    trajectory, _ = HarnessRunner(
        db, policy=exposed, progress_reducer=reducer, expose_task_progress=True).run(task)
    payload = json.dumps(exposed.observation.session, ensure_ascii=False)
    assert exposed.observation.session["task_progress"]["workflow"] == "return_resolution"
    assert trajectory.progress_spans
    for secret in ("GOLD_SENTINEL", "SECRET_TOOL", "SECRET_STATE",
                   "SECRET_EXPECTATION", "SECRET_SEQUENCE", "hidden_category", "SPLIT_SENTINEL"):
        assert secret not in payload


def test_exposing_progress_without_reducer_is_rejected(tmp_path):
    try:
        HarnessRunner(tmp_path / "x.sqlite", expose_task_progress=True)
    except ValueError as exc:
        assert "progress_reducer" in str(exc)
    else:
        raise AssertionError("missing reducer must fail closed")
