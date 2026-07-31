"""End-to-end Data Flywheel MVP v1.1 checks (minimal regression, not a new eval suite)."""

from __future__ import annotations

import pytest

from ecommerce_rag.agent_case import (
    AgentCase,
    admit_for_memory,
    build_dev_failure_agent_cases,
    case_has_credential,
    progress_signature_from_progress,
    provenance_hash,
    redact_text,
    sanitize_case,
)
from ecommerce_rag.agent_case_memory import (
    advice_used,
    approve_case,
    build_memory_advice,
    candidates_from_trajectory,
    write_candidate,
)
from ecommerce_rag.agent_case_store import list_cases
from ecommerce_rag.domain import AgentAction, TaskSpec, Trajectory
from ecommerce_rag.harness import HarnessRunner
from ecommerce_rag.legacy_closure import LegacyTaskProgressReducer
from ecommerce_rag.orders import seed_database


def _identity_progress(**overrides):
    payload = {
        "workflow": "return_resolution",
        "pending": ["identity_verification"],
        "blocked_by": "user_input",
        "guard_state": "identity_required",
        "eligible": None,
        "cancelled": False,
        "allowed_next_actions": ["ask_user:verification_code"],
        "requested_input_type": "verification_code",
    }
    payload.update(overrides)
    return payload


def _ask_verification_action():
    return {
        "action_type": "final_answer",
        "requires_user_response": True,
        "requested_input_type": "verification_code",
        "content": "please provide verification code",
    }


def _seed_train_case(*, case_id: str, **overrides) -> AgentCase:
    progress = _identity_progress()
    action = _ask_verification_action()
    source = {
        "attribution": "human_seed",
        "attribution_source": "unit_test_seed",
        "split": "train",
    }
    payload = {
        "case_id": case_id,
        "split": "train",
        "training_approved": False,
        "memory_status": "candidate",
        "user_goal": "order O000001 wants a return",
        "progress_before": progress,
        "allowed_actions": ["ask_user:verification_code"],
        "forbidden_actions": ["handoff", "final_answer", "create_return_request"],
        "chosen_action": dict(action),
        "executed_action": dict(action),
        "raw_policy_action": dict(action),
        "constrained_action": dict(action),
        "constraint_remapped": False,
        "policy_followed_advice": True,
        "step": 0,
        "step_outcome": "allowed:ask_user:verification_code",
        "terminal_state": {"return_status": "requested", "illegal_state_change": False},
        "terminal_outcome": {"success": True, "illegal_state_change": False},
        "success": True,
        "failure_owner": "none",
        "causal_credit": "seed",
        "reusable_pattern": "identity verification must complete first",
        "avoid_pattern": "handoff|final_answer",
        "workflow": "return_resolution",
        "progress_signature": progress_signature_from_progress(progress),
        "source": source,
        "source_hash": provenance_hash(source, case_id, 0),
        "paired_replay_result": {"ok": True, "note": "seed_paired_replay"},
    }
    payload.update(overrides)
    return AgentCase(**payload)


def test_dev_cases_cannot_be_memory_approved(tmp_path):
    case = build_dev_failure_agent_cases(code_commit="x")[0]
    assert case.split == "dev"
    assert case.memory_status == "quarantined"
    with pytest.raises(ValueError, match="cannot enter runtime Memory"):
        AgentCase(**{**case.to_dict(), "memory_status": "approved", "memory_approved": True})
    admission = admit_for_memory(
        case, approve=True, approved_by="test", approval_reason="nope")
    assert not admission.accepted
    assert admission.status == "quarantined"


def test_train_case_can_be_approved_and_retrieved(tmp_path):
    db = tmp_path / "cases.db"
    progress = _identity_progress()
    case = _seed_train_case(case_id="ac_train_identity_01")
    stored, admission = write_candidate(case, db_path=db, approve=False)
    assert admission["status"] == "candidate"
    assert stored.memory_status == "candidate"
    approved = approve_case(
        stored.case_id,
        db_path=db,
        approved_by="test_reviewer",
        approval_reason="identity seed with paired replay",
    )
    assert approved.memory_approved
    assert approved.approved_by == "test_reviewer"
    assert approved.created_at and "[REDACTED]" not in approved.created_at
    advice = build_memory_advice(progress, db_path=db)
    assert advice.matched
    assert "ask_user:verification_code" in advice.preferred_actions
    assert "handoff" in advice.avoid_actions
    assert advice.successful_cases == 1


def test_approve_rejects_weak_or_remapped_success(tmp_path):
    db = tmp_path / "cases.db"
    weak = _seed_train_case(case_id="ac_weak", paired_replay_result={})
    weak.source = {}
    weak.source_hash = ""
    stored, _ = write_candidate(weak, db_path=db, approve=False)
    with pytest.raises(ValueError, match="cannot approve"):
        approve_case(
            stored.case_id,
            db_path=db,
            approved_by="r",
            approval_reason="try",
        )

    remapped = _seed_train_case(
        case_id="ac_remapped",
        constraint_remapped=True,
        causal_credit="policy",
        raw_policy_action={"action_type": "handoff"},
        chosen_action={"action_type": "handoff"},
        executed_action=_ask_verification_action(),
    )
    stored2, adm = write_candidate(
        remapped,
        db_path=db,
        approve=True,
        approved_by="r",
        approval_reason="should fail",
    )
    assert not adm["accepted"]
    assert stored2.memory_status == "rejected"


def test_memory_advice_never_expands_allowlist(tmp_path):
    db = tmp_path / "cases.db"
    case = _seed_train_case(
        case_id="ac_train_bad_pref",
        allowed_actions=["ask_user:verification_code", "create_return_request"],
        chosen_action={"action_type": "tool_call", "tool_name": "create_return_request"},
        executed_action={"action_type": "tool_call", "tool_name": "create_return_request"},
        raw_policy_action={"action_type": "tool_call", "tool_name": "create_return_request"},
        reusable_pattern="should not authorize write",
        avoid_pattern="",
    )
    write_candidate(
        case,
        db_path=db,
        approve=True,
        approved_by="test",
        approval_reason="negative control seed",
    )
    advice = build_memory_advice(
        _identity_progress(allowed_next_actions=["ask_user:verification_code"]),
        db_path=db,
    )
    assert "create_return_request" not in advice.preferred_actions


def test_policy_followed_advice_is_pre_constraint(tmp_path):
    db = tmp_path / "cases.db"
    env = tmp_path / "env.sqlite"
    seed_database(env, users=20, orders=50)
    write_candidate(
        _seed_train_case(case_id="ac_train_identity_seed"),
        db_path=db,
        approve=True,
        approved_by="test",
        approval_reason="seed for attribution check",
    )

    class MemoryIgnoringPolicy:
        privileged = False

        def __init__(self):
            self.seen_advice = None

        def act(self, observation):
            self.seen_advice = (observation.session or {}).get("memory_advice")
            return AgentAction.handoff("skip checks")

    task = TaskSpec(
        "flywheel_train_01", "return", "U0001",
        "order O000001 wants a return. reason: size issue", 11,
        split="train",
        metadata={"order_id": "O000001", "verification_code": "123456"},
    )
    policy = MemoryIgnoringPolicy()
    trajectory, _grade = HarnessRunner(
        env,
        policy=policy,
        max_steps=1,
        progress_reducer=LegacyTaskProgressReducer(),
        expose_task_progress=True,
        enforce_action_constraint=True,
        enable_case_memory=True,
        case_memory_db=db,
        enable_case_writeback=True,
    ).run(task)

    assert policy.seen_advice and policy.seen_advice.get("matched")
    span = trajectory.memory_spans[0]
    assert span["raw_policy_action"]["action_type"] == "handoff"
    assert span["policy_followed_advice"] is False
    assert span["advice_used"] is False
    assert span["constraint_remapped"] is True
    assert span["executed_action"]["requires_user_response"] is True
    assert advice_used(span["memory_advice"], span["raw_policy_action"]) is False
    assert advice_used(span["memory_advice"], span["executed_action"]) is True

    candidates = list_cases(memory_status="candidate", source_split="train", db_path=db)
    written = [case for case in candidates if case.task_id == "flywheel_train_01"]
    assert written
    assert all(case.step is not None for case in written)
    assert all(case.constraint_remapped for case in written)
    assert all(not case.success for case in written)
    assert all(case.causal_credit == "constraint" for case in written)

    policy2 = MemoryIgnoringPolicy()
    HarnessRunner(
        env,
        policy=policy2,
        max_steps=1,
        progress_reducer=LegacyTaskProgressReducer(),
        expose_task_progress=True,
        enable_case_memory=False,
        case_memory_db=db,
        enable_case_writeback=False,
    ).run(task)
    assert policy2.seen_advice is None


def test_decision_level_cases_do_not_credit_first_step_with_terminal_success():
    progress0 = _identity_progress()
    progress1 = {
        "workflow": "return_resolution",
        "pending": [],
        "blocked_by": None,
        "guard_state": "complete",
        "allowed_next_actions": ["final_answer"],
        "eligible": True,
        "cancelled": False,
    }
    trajectory = Trajectory(
        "tr_x", "task_x", 1,
        progress_spans=[
            {"step": 0, **progress0},
            {"step": 1, **progress1},
        ],
        actions=[
            {"action_type": "handoff", "content": "bad"},
            {"action_type": "final_answer", "content": "done", "requires_user_response": False},
        ],
        memory_spans=[
            {
                "step": 0,
                "raw_policy_action": {"action_type": "handoff"},
                "constrained_action": _ask_verification_action(),
                "executed_action": _ask_verification_action(),
                "constraint_remapped": True,
                "policy_followed_advice": False,
                "constraint_result": {"remapped": True},
            },
            {
                "step": 1,
                "raw_policy_action": {
                    "action_type": "final_answer",
                    "requires_user_response": False,
                    "content": "done",
                },
                "constrained_action": {
                    "action_type": "final_answer",
                    "requires_user_response": False,
                    "content": "done",
                },
                "executed_action": {
                    "action_type": "final_answer",
                    "requires_user_response": False,
                    "content": "done",
                },
                "constraint_remapped": False,
                "policy_followed_advice": False,
            },
        ],
    )
    cases = candidates_from_trajectory(
        task_id="task_x",
        split="train",
        user_goal="order O000001 wants a return",
        trajectory=trajectory,
        grade={"success": True, "illegal_state_change": False},
    )
    assert len(cases) == 2
    assert cases[0].constraint_remapped is True
    assert cases[0].success is False
    assert cases[0].causal_credit == "constraint"
    assert "handoff" in cases[0].avoid_pattern
    assert cases[1].success is True
    assert cases[1].causal_credit == "policy"


def test_credential_scan_skips_timestamps_and_rejects_user_codes():
    ts = "2026-07-31T06:24:48.123456+00:00"
    assert "[REDACTED]" not in redact_text(ts)
    case = _seed_train_case(case_id="ac_ts", created_at=ts)
    cleaned = sanitize_case(case)
    assert cleaned.created_at == ts
    assert not case_has_credential(cleaned)

    bad = _seed_train_case(case_id="ac_bad", user_goal="code 123456")
    admission = admit_for_memory(
        bad, approve=True, approved_by="r", approval_reason="x")
    assert not admission.accepted
    assert "credential_present" in admission.reasons
