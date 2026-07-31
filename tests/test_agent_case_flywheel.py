"""End-to-end Data Flywheel MVP checks (minimal regression, not a new eval suite)."""

from __future__ import annotations

import pytest

from ecommerce_rag.agent_case import (
    AgentCase,
    admit_for_memory,
    build_dev_failure_agent_cases,
    progress_signature_from_progress,
)
from ecommerce_rag.agent_case_memory import (
    advice_used,
    approve_case,
    build_memory_advice,
    candidate_from_trajectory,
    write_candidate,
)
from ecommerce_rag.agent_case_store import get_case, insert_case, list_cases
from ecommerce_rag.domain import AgentAction, TaskSpec
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


def test_dev_cases_cannot_be_memory_approved(tmp_path):
    case = build_dev_failure_agent_cases(code_commit="x")[0]
    assert case.split == "dev"
    assert case.memory_status == "quarantined"
    with pytest.raises(ValueError, match="cannot enter runtime Memory"):
        AgentCase(**{**case.to_dict(), "memory_status": "approved", "memory_approved": True})
    admission = admit_for_memory(case, approve=True)
    assert not admission.accepted
    assert admission.status == "quarantined"


def test_train_case_can_be_approved_and_retrieved(tmp_path):
    db = tmp_path / "cases.db"
    progress = _identity_progress()
    case = AgentCase(
        case_id="ac_train_identity_01",
        split="train",
        training_approved=False,
        memory_status="candidate",
        user_goal="order O000001 wants a return",
        progress_before=progress,
        allowed_actions=["ask_user:verification_code"],
        forbidden_actions=["handoff", "final_answer", "create_return_request"],
        chosen_action={
            "action_type": "final_answer",
            "requires_user_response": True,
            "requested_input_type": "verification_code",
            "content": "please provide verification code",
        },
        terminal_state={"return_status": "requested", "illegal_state_change": False},
        success=True,
        failure_owner="none",
        reusable_pattern="identity verification must complete first",
        avoid_pattern="handoff|final_answer",
        workflow="return_resolution",
        progress_signature=progress_signature_from_progress(progress),
    )
    stored, admission = write_candidate(case, db_path=db, approve=False)
    assert admission["status"] == "candidate"
    assert stored.memory_status == "candidate"
    approved = approve_case(stored.case_id, db_path=db)
    assert approved.memory_approved
    advice = build_memory_advice(progress, db_path=db)
    assert advice.matched
    assert "ask_user:verification_code" in advice.preferred_actions
    assert "handoff" in advice.avoid_actions
    assert advice.successful_cases == 1


def test_memory_advice_never_expands_allowlist(tmp_path):
    db = tmp_path / "cases.db"
    progress = _identity_progress()
    case = AgentCase(
        case_id="ac_train_bad_pref",
        split="train",
        training_approved=False,
        user_goal="x",
        progress_before=progress,
        allowed_actions=["ask_user:verification_code", "create_return_request"],
        chosen_action={"action_type": "tool_call", "tool_name": "create_return_request"},
        terminal_state={"illegal_state_change": False},
        success=True,
        failure_owner="none",
        reusable_pattern="should not authorize write",
        workflow="return_resolution",
        progress_signature=progress_signature_from_progress(progress),
    )
    write_candidate(case, db_path=db, approve=True)
    advice = build_memory_advice(
        _identity_progress(allowed_next_actions=["ask_user:verification_code"]),
        db_path=db,
    )
    assert "create_return_request" not in advice.preferred_actions


def test_flywheel_end_to_end_with_constraint_and_writeback(tmp_path):
    db = tmp_path / "cases.db"
    env = tmp_path / "env.sqlite"
    seed_database(env, users=20, orders=50)
    progress = _identity_progress()
    seed = AgentCase(
        case_id="ac_train_identity_seed",
        split="train",
        training_approved=False,
        user_goal="seed",
        progress_before=progress,
        allowed_actions=["ask_user:verification_code"],
        chosen_action={
            "action_type": "final_answer",
            "requires_user_response": True,
            "requested_input_type": "verification_code",
            "content": "please provide verification code",
        },
        terminal_state={"illegal_state_change": False},
        success=True,
        failure_owner="none",
        reusable_pattern="identity verification must complete first",
        avoid_pattern="handoff",
        workflow="return_resolution",
        progress_signature=progress_signature_from_progress(progress),
    )
    write_candidate(seed, db_path=db, approve=True)

    class MemoryAwarePolicy:
        privileged = False

        def __init__(self):
            self.seen_advice = None

        def act(self, observation):
            self.seen_advice = (observation.session or {}).get("memory_advice")
            # Illegal without constraint/memory; memory should prefer ask.
            return AgentAction.handoff("skip checks")

    # Use ASCII "return" intent so file encoding cannot break progress matching.
    task = TaskSpec(
        "flywheel_train_01", "return", "U0001",
        "order O000001 wants a return. reason: size issue", 11,
        split="train",
        metadata={"order_id": "O000001", "verification_code": "123456"},
    )
    policy = MemoryAwarePolicy()
    runner = HarnessRunner(
        env,
        policy=policy,
        max_steps=1,
        progress_reducer=LegacyTaskProgressReducer(),
        expose_task_progress=True,
        enforce_action_constraint=True,
        enable_case_memory=True,
        case_memory_db=db,
        enable_case_writeback=True,
    )
    trajectory, grade = runner.run(task)
    assert policy.seen_advice and policy.seen_advice.get("matched")
    assert "ask_user:verification_code" in policy.seen_advice["preferred_actions"]
    assert trajectory.memory_spans and trajectory.memory_spans[0].get("retrieved_case_ids")
    assert trajectory.constraint_spans and trajectory.constraint_spans[0]["remapped"]
    assert trajectory.actions[0]["requires_user_response"] is True
    assert advice_used(policy.seen_advice, {
        "action_type": "final_answer",
        "requires_user_response": True,
        "requested_input_type": "verification_code",
    })
    # Writeback created a candidate from the train split run.
    candidates = list_cases(memory_status="candidate", source_split="train", db_path=db)
    assert any(case.task_id == "flywheel_train_01" for case in candidates)

    # Memory off restores advice-free observation.
    policy2 = MemoryAwarePolicy()
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


def test_credential_and_hidden_fields_are_rejected(tmp_path):
    case = AgentCase(
        case_id="ac_bad",
        split="train",
        training_approved=False,
        user_goal="code 123456",
        progress_before=_identity_progress(),
        allowed_actions=["ask_user:verification_code"],
        terminal_state={"illegal_state_change": False},
        success=True,
        failure_owner="none",
        workflow="return_resolution",
        progress_signature=progress_signature_from_progress(_identity_progress()),
    )
    admission = admit_for_memory(case, approve=True)
    assert not admission.accepted
    assert "credential_present" in admission.reasons
