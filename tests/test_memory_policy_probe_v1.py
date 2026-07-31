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


def test_seed_train_cases_are_approvable(tmp_path):
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


def test_conclude_rules_preregistered():
    under = conclude({
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

    negative = conclude({
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


def test_aggregate_counts_repairs():
    pairs = [
        {
            "task_id": "a",
            "retrieval_coverage": True,
            "off": {
                "raw_matches_preferred": False,
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
            "retrieval_coverage": True,
            "off": {
                "raw_matches_preferred": True,
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
    assert _tool_result_type_for_step(
        trajectory, 1,
        {"action_type": "tool_call", "tool_name": "check_return_eligibility"},
    ) == "check_return_eligibility"
    assert _tool_result_type_for_step(
        trajectory, 2, {"action_type": "final_answer", "content": "done"},
    ) == ""
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
