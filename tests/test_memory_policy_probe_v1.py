"""Unit tests for memory_policy_probe_v1 protocol (no LLM required)."""

from __future__ import annotations

from ecommerce_rag.agent_case_memory import _tool_result_type_for_step, candidates_from_trajectory
from ecommerce_rag.domain import ToolCall, Trajectory
from ecommerce_rag.memory_policy_probe import (
    FROZEN_PROBE_TASK_SHA256,
    FROZEN_TRAIN_CASE_SHA256,
    aggregate_probe,
    assert_frozen_manifests,
    build_probe_tasks,
    build_train_case_specs,
    conclude,
    freeze_manifest_digests,
    offline_preferred,
    score_pair,
    seed_train_cases,
    train_case_from_spec,
)


def test_frozen_manifests_match():
    digests = freeze_manifest_digests()
    assert digests["probe_task_sha256"] == FROZEN_PROBE_TASK_SHA256
    assert digests["train_case_sha256"] == FROZEN_TRAIN_CASE_SHA256
    assert_frozen_manifests()
    assert len(build_probe_tasks()) == 24
    assert len(build_train_case_specs()) == 16


def test_seed_train_cases_are_approvable_contract_seeds(tmp_path):
    db = tmp_path / "cases.db"
    rows = seed_train_cases(
        db,
        approved_by="unit",
        approval_reason="probe train freeze",
        approve=True,
    )
    assert len(rows) == 16
    assert all(row["memory_approved"] for row in rows)
    assert all(row["admission"]["status"] == "approved" for row in rows)
    case = train_case_from_spec(build_train_case_specs()[0])
    assert case.source["source_kind"] == "curated_contract_seed"
    assert case.source["validation_type"] == "deterministic_contract_check"
    assert case.source["experience_case"] is False
    assert case.paired_replay_result == {}


def test_conclude_rules_preregistered():
    under = conclude({
        "n_pairs": 24,
        "retrieval_matched_count": 24,
        "off_raw_errors": 2,
        "repaired_by_memory": 2,
        "regressed_by_memory": 0,
        "off_constraint_remap_count": 2,
        "on_constraint_remap_count": 0,
        "off_terminal_success": 24,
        "on_terminal_success": 24,
        "illegal_state_change_total": 0,
    })
    assert under["verdict"] == "neutral_underpowered"

    positive = conclude({
        "n_pairs": 24,
        "retrieval_matched_count": 24,
        "off_raw_errors": 6,
        "repaired_by_memory": 4,
        "regressed_by_memory": 0,
        "off_constraint_remap_count": 6,
        "on_constraint_remap_count": 2,
        "off_terminal_success": 20,
        "on_terminal_success": 22,
        "illegal_state_change_total": 0,
    })
    assert positive["verdict"] == "positive"

    no_coverage = conclude({
        "n_pairs": 24,
        "retrieval_matched_count": 20,
        "off_raw_errors": 6,
        "repaired_by_memory": 4,
        "regressed_by_memory": 0,
        "off_constraint_remap_count": 6,
        "on_constraint_remap_count": 2,
        "off_terminal_success": 20,
        "on_terminal_success": 22,
        "illegal_state_change_total": 0,
    })
    assert no_coverage["verdict"] == "negative_or_inconclusive"

    negative = conclude({
        "n_pairs": 24,
        "retrieval_matched_count": 24,
        "off_raw_errors": 6,
        "repaired_by_memory": 1,
        "regressed_by_memory": 0,
        "off_constraint_remap_count": 6,
        "on_constraint_remap_count": 6,
        "off_terminal_success": 20,
        "on_terminal_success": 20,
        "illegal_state_change_total": 0,
    })
    assert negative["verdict"] == "negative_or_inconclusive"


def test_repair_requires_retrieval_and_policy_followed(tmp_path):
    from ecommerce_rag.memory_policy_probe import MemoryProbeTask

    task = MemoryProbeTask(
        task_id="t1",
        state_id="missing_verification",
        preferred_action="ask_user:verification_code",
        seed=1,
        user_id="P0001",
        order_id="O300001",
        initial_message="x",
        user_responses=("123456",),
    )
    off_tr = Trajectory(
        "off", "t1", 1,
        decision_spans=[{
            "step": 0,
            "raw_policy_action": {"action_type": "handoff"},
            "constraint_remapped": True,
            "policy_followed_advice": None,
            "progress": {
                "workflow": "return_resolution",
                "allowed_next_actions": ["ask_user:verification_code"],
            },
        }],
    )
    on_tr = Trajectory(
        "on", "t1", 1,
        decision_spans=[{
            "step": 0,
            "raw_policy_action": {
                "action_type": "final_answer",
                "requires_user_response": True,
                "requested_input_type": "verification_code",
            },
            "constraint_remapped": False,
            "policy_followed_advice": True,
            "progress": {
                "workflow": "return_resolution",
                "allowed_next_actions": ["ask_user:verification_code"],
            },
        }],
    )
    grade = {"success": True, "illegal_state_change": False}

    # No retrieval: lucky on-arm match must not count as Memory repair.
    lucky = score_pair(
        task=task,
        off_trajectory=off_tr,
        on_trajectory=on_tr,
        off_grade=grade,
        on_grade=grade,
        probe_step_off=0,
        probe_step_on=0,
        offline_preferred_payload={
            "retrieval_matched": False,
            "scoring_preferred_actions": [],
            "fallback_expected_action": "ask_user:verification_code",
            "retrieved_case_ids": [],
        },
    )
    assert lucky["retrieval_coverage"] is False
    assert lucky["repaired_by_memory"] is False

    # Retrieval + followed advice: counts.
    repaired = score_pair(
        task=task,
        off_trajectory=off_tr,
        on_trajectory=on_tr,
        off_grade=grade,
        on_grade=grade,
        probe_step_off=0,
        probe_step_on=0,
        offline_preferred_payload={
            "retrieval_matched": True,
            "scoring_preferred_actions": ["ask_user:verification_code"],
            "fallback_expected_action": "ask_user:verification_code",
            "retrieved_case_ids": ["ac_x"],
        },
    )
    assert repaired["retrieval_coverage"] is True
    assert repaired["repaired_by_memory"] is True

    # Retrieval but policy did not follow advice: no repair credit.
    on_unfollowed = Trajectory(
        "on2", "t1", 1,
        decision_spans=[{
            "step": 0,
            "raw_policy_action": {
                "action_type": "final_answer",
                "requires_user_response": True,
                "requested_input_type": "verification_code",
            },
            "constraint_remapped": False,
            "policy_followed_advice": False,
            "progress": {
                "workflow": "return_resolution",
                "allowed_next_actions": ["ask_user:verification_code"],
            },
        }],
    )
    no_follow = score_pair(
        task=task,
        off_trajectory=off_tr,
        on_trajectory=on_unfollowed,
        off_grade=grade,
        on_grade=grade,
        probe_step_off=0,
        probe_step_on=0,
        offline_preferred_payload={
            "retrieval_matched": True,
            "scoring_preferred_actions": ["ask_user:verification_code"],
            "fallback_expected_action": "ask_user:verification_code",
            "retrieved_case_ids": ["ac_x"],
        },
    )
    assert no_follow["repaired_by_memory"] is False


def test_offline_preferred_does_not_fallback_into_coverage(tmp_path):
    payload = offline_preferred(
        {
            "workflow": "return_resolution",
            "pending": ["identity_verification"],
            "guard_state": "identity_required",
            "blocked_by": "user_input",
            "allowed_next_actions": ["ask_user:verification_code"],
        },
        db_path=tmp_path / "empty.db",
        fallback_expected_action="ask_user:verification_code",
    )
    assert payload["retrieval_matched"] is False
    assert payload["scoring_preferred_actions"] == []
    assert payload["fallback_expected_action"] == "ask_user:verification_code"


def test_aggregate_counts_repairs():
    pairs = [
        {
            "task_id": "a",
            "retrieval_matched": True,
            "retrieval_coverage": True,
            "off": {
                "raw_matches_preferred": False,
                "raw_matches_fallback_expected": False,
                "constraint_remapped": True,
                "terminal_success": True,
                "illegal_state_change": False,
            },
            "on": {
                "raw_matches_preferred": True,
                "constraint_remapped": False,
                "terminal_success": True,
                "illegal_state_change": False,
            },
            "repaired_by_memory": True,
            "regressed_by_memory": False,
        },
        {
            "task_id": "b",
            "retrieval_matched": True,
            "retrieval_coverage": True,
            "off": {
                "raw_matches_preferred": True,
                "raw_matches_fallback_expected": True,
                "constraint_remapped": False,
                "terminal_success": True,
                "illegal_state_change": False,
            },
            "on": {
                "raw_matches_preferred": True,
                "constraint_remapped": False,
                "terminal_success": True,
                "illegal_state_change": False,
            },
            "repaired_by_memory": False,
            "regressed_by_memory": False,
        },
    ]
    summary = aggregate_probe(pairs)
    assert summary["off_raw_errors"] == 1
    assert summary["repaired_by_memory"] == 1
    assert summary["retrieval_matched_count"] == 2


def test_tool_result_type_is_step_aligned_or_empty():
    trajectory = Trajectory(
        "tr", "t", 1,
        actions=[
            {"action_type": "tool_call", "tool_name": "get_order"},
            {"action_type": "tool_call", "tool_name": "check_return_eligibility"},
            {"action_type": "final_answer", "content": "done"},
        ],
        tool_calls=[
            ToolCall("get_order", {}, "c1", {"ok": True}, "t0"),
            ToolCall("check_return_eligibility", {}, "c2", {"ok": True}, "t1"),
        ],
        progress_spans=[
            {"step": 0, "workflow": "return_resolution",
             "pending": ["eligibility"], "allowed_next_actions": ["get_order"]},
            {"step": 1, "workflow": "return_resolution",
             "pending": ["eligibility"],
             "allowed_next_actions": ["check_return_eligibility"]},
            {"step": 2, "workflow": "return_resolution",
             "pending": [], "allowed_next_actions": ["final_answer"]},
        ],
        memory_spans=[
            {"step": 0, "raw_policy_action": {"action_type": "tool_call", "tool_name": "get_order"},
             "executed_action": {"action_type": "tool_call", "tool_name": "get_order"},
             "constraint_remapped": False},
            {"step": 1,
             "raw_policy_action": {
                 "action_type": "tool_call", "tool_name": "check_return_eligibility"},
             "executed_action": {
                 "action_type": "tool_call", "tool_name": "check_return_eligibility"},
             "constraint_remapped": False},
            {"step": 2,
             "raw_policy_action": {"action_type": "final_answer", "content": "done"},
             "executed_action": {"action_type": "final_answer", "content": "done"},
             "constraint_remapped": False},
        ],
    )
    assert _tool_result_type_for_step(
        trajectory, 0, {"action_type": "tool_call", "tool_name": "get_order"},
    ) == "get_order"
    cases = candidates_from_trajectory(
        task_id="t",
        split="train",
        user_goal="order O000001 wants a return",
        trajectory=trajectory,
        grade={"success": True, "illegal_state_change": False},
    )
    assert cases[0].tool_result_type == "get_order"
    assert cases[1].tool_result_type == "check_return_eligibility"
    assert cases[2].tool_result_type == ""


def test_train_case_specs_have_no_constraint_remap():
    for spec in build_train_case_specs():
        case = train_case_from_spec(spec)
        assert case.constraint_remapped is False
        assert case.success is True
        assert case.split == "train"
