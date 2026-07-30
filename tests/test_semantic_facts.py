from __future__ import annotations

import json

import pytest

from ecommerce_rag.semantic_facts import (
    DeterministicFactValidator,
    LLMSemanticFactExtractor,
    SessionSemanticFactPipeline,
    UserTurnContext,
)
from ecommerce_rag.domain import AgentAction, TaskSpec
from ecommerce_rag.harness import HarnessRunner
from ecommerce_rag.orders import seed_database


def _candidate(*, reason_status="not_mentioned", reason_text=None,
               reason_quote=None, intent_status="present", intent_quote="退货",
               goal_kind="none", goal_text=None, goal_quote=None):
    return {
        "return_intent": {
            "status": intent_status,
            "evidence_quote": intent_quote,
        },
        "return_reason": {
            "status": reason_status,
            "text": reason_text,
            "evidence_quote": reason_quote,
        },
        "goal_change": {
            "kind": goal_kind,
            "text": goal_text,
            "evidence_quote": goal_quote,
        },
    }


@pytest.mark.parametrize(
    ("message", "quote"),
    [
        ("我要退货，我不想提供退货原因", "我不想提供退货原因"),
        ("这个订单退货，但我拒绝说明原因", "我拒绝说明原因"),
    ],
)
def test_refusal_is_semantic_and_must_carry_exact_evidence(message, quote):
    events, codes = DeterministicFactValidator().validate(
        _candidate(reason_status="refused", reason_quote=quote),
        UserTurnContext(1, message),
    )
    assert not codes
    assert any(event.kind == "return_reason_refused" for event in events)


def test_provided_reason_keeps_original_substring_without_rewrite():
    message = "订单要退货，原因是尺码偏小"
    events, codes = DeterministicFactValidator().validate(
        _candidate(reason_status="provided", reason_text="尺码偏小",
                   reason_quote="原因是尺码偏小"),
        UserTurnContext(2, message),
    )
    assert not codes
    reason = next(event for event in events if event.kind == "return_reason_provided")
    assert reason.value == "尺码偏小"
    assert reason.evidence_quote == "原因是尺码偏小"


def test_hallucinated_or_rewritten_reason_fails_closed():
    events, codes = DeterministicFactValidator().validate(
        _candidate(reason_status="provided", reason_text="质量问题",
                   reason_quote="原因是尺寸不合适"),
        UserTurnContext(3, "我要退货，原因是尺寸不合适"),
    )
    assert "return_reason_not_grounded" in codes
    assert not any(event.kind == "return_reason_provided" for event in events)
    assert any(event.kind == "semantic_ambiguous" for event in events)


def test_goal_change_requires_a_verbatim_quote():
    events, codes = DeterministicFactValidator().validate(
        _candidate(goal_kind="order_query_only", goal_text="只查订单",
                   goal_quote="先不退了，只查订单",
                   intent_status="not_present", intent_quote=None),
        UserTurnContext(4, "先不退了，只查订单"),
    )
    assert not codes
    event = next(event for event in events if event.kind == "goal_change_observed")
    assert event.value == "order_query_only"


class RecordingExtractor:
    def __init__(self, candidate):
        self.candidate = candidate
        self.calls = []

    def extract(self, context):
        self.calls.append(context)
        return self.candidate


def test_session_cache_calls_extractor_at_most_once_per_user_turn():
    extractor = RecordingExtractor(_candidate(reason_status="provided",
                                               reason_text="不合适",
                                               reason_quote="原因是不合适"))
    pipeline = SessionSemanticFactPipeline(extractor)
    context = UserTurnContext(5, "我要退货，原因是不合适")
    first = pipeline.process(context)
    second = pipeline.process(context)
    third = pipeline.process(UserTurnContext(6, context.text))
    assert len(extractor.calls) == 2
    assert first.extractor_called and not first.cache_hit
    assert second.cache_hit and not second.extractor_called
    assert third.extractor_called


def test_credentials_are_redacted_before_extraction_and_never_serialized():
    extractor = RecordingExtractor(_candidate(reason_status="provided",
                                               reason_text="不合适",
                                               reason_quote="原因是不合适"))
    pipeline = SessionSemanticFactPipeline(extractor)
    result = pipeline.process(UserTurnContext(
        7, "我要退货，原因是不合适，验证码 123456"))
    assert "123456" not in extractor.calls[0].text
    assert "[VERIFICATION_CODE]" in extractor.calls[0].text
    assert "123456" not in json.dumps(result.to_dict(), ensure_ascii=False)


def test_malformed_llm_output_becomes_ambiguous_without_raw_output():
    extractor = LLMSemanticFactExtractor(lambda _system, _user: "not json")
    result = SessionSemanticFactPipeline(extractor).process(
        UserTurnContext(8, "我要退货"))
    assert result.validation_codes == ("extractor_output_invalid",)
    assert result.events[0].kind == "semantic_ambiguous"
    assert "not json" not in json.dumps(result.to_dict(), ensure_ascii=False)


def test_llm_extractor_rejects_extra_schema_fields():
    payload = _candidate()
    payload["unexpected"] = True
    extractor = LLMSemanticFactExtractor(
        lambda _system, _user: json.dumps(payload, ensure_ascii=False))
    result = SessionSemanticFactPipeline(extractor).process(
        UserTurnContext(9, "我要退货"))
    assert result.validation_codes == ("extractor_output_invalid",)


def test_nested_extra_fields_fail_closed():
    candidate = _candidate()
    candidate["return_reason"]["confidence"] = 1.0
    events, codes = DeterministicFactValidator().validate(
        candidate, UserTurnContext(10, "我要退货"))
    assert "invalid_return_reason_schema" in codes
    assert not any(event.kind == "return_reason_provided" for event in events)


def test_shadow_generation_error_does_not_escape_pipeline():
    class FailingExtractor:
        def extract(self, _context):
            raise RuntimeError("backend unavailable")

    result = SessionSemanticFactPipeline(FailingExtractor()).process(
        UserTurnContext(11, "我要退货"))
    assert result.validation_codes == ("extractor_output_invalid",)
    assert "backend unavailable" not in json.dumps(result.to_dict(), ensure_ascii=False)


def test_harness_shadow_extracts_each_user_turn_without_changing_actions(tmp_path):
    class TwoTurnPolicy:
        privileged = False

        def __init__(self):
            self.calls = 0

        def act(self, _observation):
            self.calls += 1
            if self.calls == 1:
                return AgentAction.answer("请提供六位验证码。", requires_user_response=True)
            return AgentAction.answer("已收到。")

    class CodeUser:
        def respond(self, _action, requested_input_type=None):
            assert requested_input_type == "verification_code"
            return "123456"

    extractor = RecordingExtractor(_candidate(
        reason_status="provided", reason_text="不合适", reason_quote="原因是不合适"))
    baseline_db = tmp_path / "baseline.sqlite"
    shadow_db = tmp_path / "shadow.sqlite"
    seed_database(baseline_db, users=20, orders=100)
    seed_database(shadow_db, users=20, orders=100)
    task = TaskSpec(
        "semantic_shadow", "return", "U0001",
        "订单 O000001 要退货，原因是不合适", 1,
        metadata={"order_id": "O000001"},
    )
    baseline, baseline_grade = HarnessRunner(
        baseline_db,
        policy=TwoTurnPolicy(),
        max_steps=2,
        user_simulator_factory=lambda _task: CodeUser(),
    ).run(task)
    trajectory, shadow_grade = HarnessRunner(
        shadow_db,
        policy=TwoTurnPolicy(),
        max_steps=2,
        user_simulator_factory=lambda _task: CodeUser(),
        semantic_fact_pipeline_factory=lambda: SessionSemanticFactPipeline(extractor),
    ).run(task)
    assert len(extractor.calls) == 2
    assert "123456" not in extractor.calls[1].text
    assert [action["action_type"] for action in trajectory.actions] == [
        "final_answer", "final_answer"]
    assert len(trajectory.semantic_fact_spans) == 2
    assert "123456" not in json.dumps(trajectory.semantic_fact_spans, ensure_ascii=False)
    assert trajectory.actions == baseline.actions
    assert trajectory.tool_calls == baseline.tool_calls
    assert trajectory.final_state == baseline.final_state
    assert shadow_grade.success == baseline_grade.success
    assert shadow_grade.terminal_state_match == baseline_grade.terminal_state_match
