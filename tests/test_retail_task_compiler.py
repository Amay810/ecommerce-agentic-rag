# -*- coding: utf-8 -*-
"""Unit tests for Retail Task Compiler P1 v0 (cancel_pending slice)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ecommerce_rag.retail_task_compiler import (
    RetailTaskCompiler,
    coverage_from_blueprints,
    load_retail_tool_graph,
    replay_reference_path_twice,
    structure_signature,
    validate_blueprint,
)
from ecommerce_rag.retail_task_compiler.contamination import (
    check_contamination,
    signature_from_blueprint,
)
from ecommerce_rag.retail_task_compiler.replay import CancelPendingMockExecutor
from ecommerce_rag.retail_task_compiler.tool_graph import validate_edge_batch


ROOT = Path(__file__).resolve().parents[1]
SIGNATURES = ROOT / "docs" / "tau3_retail_test40_structure_signatures.json"


def _user() -> dict:
    return {
        "user_id": "sara_demo_001",
        "email": "sara_demo_001@example.com",
        "first_name": "Sara",
        "last_name": "Demo",
        "zip": "19122",
    }


def _order() -> dict:
    return {"order_id": "#W9000999", "status": "pending"}


def test_tool_graph_edges_all_have_provenance():
    graph = load_retail_tool_graph()
    assert graph.version.startswith("retail_tool_graph.")
    assert graph.graph_hash()
    for edge in graph.edges:
        assert edge.provenance
        for item in edge.provenance:
            assert item.kind in {"policy", "tools_impl", "schema", "human_reviewed"}


def test_candidate_edge_without_provenance_is_rejected():
    with pytest.raises(ValueError, match="missing provenance"):
        validate_edge_batch(
            [{"source": "a", "target": "b", "relation": "enables", "provenance": []}]
        )


def test_blueprint_validation_rejects_unknown_tool():
    graph = load_retail_tool_graph()
    with pytest.raises(ValueError, match="unknown"):
        validate_blueprint(
            {
                "task_id": "bad",
                "environment": "tau3_retail",
                "source_policy_version": "x",
                "tool_graph_hash": graph.graph_hash(),
                "db_snapshot_hash": "db",
                "initial_state": {},
                "user_goal": {},
                "private_user_facts": {},
                "disclosure_schedule": [],
                "required_effects": [],
                "forbidden_effects": [],
                "acceptable_terminal_conditions": [{"ok": True}],
                "reference_tool_paths": [
                    [{"name": "not_a_real_tool", "arguments": {}}]
                ],
                "behavior_profile": "cooperative",
                "generator_version": "retail_task_compiler.v0.cancel_pending",
                "generator_prompt_hash": "abc",
            }
        )


def test_write_path_without_auth_is_rejected_by_graph():
    graph = load_retail_tool_graph()
    with pytest.raises(ValueError, match="authentication"):
        graph.assert_path_allowed(["get_order_details", "cancel_pending_order"])


def test_cancel_pending_compiles_and_replays_twice():
    compiler = RetailTaskCompiler(test_signatures={"signatures": []})
    result = compiler.compile_cancel_pending(
        task_id="rtc_test_cancel_001",
        user=_user(),
        order=_order(),
        db_snapshot_hash="demo_db",
    )
    assert result.accepted
    assert result.replay is not None and result.replay.accepted
    assert result.blueprint.task_family == "cancel_pending"
    assert result.blueprint.environment == "tau3_retail"


def test_contamination_detects_exact_tool_path_collision():
    compiler = RetailTaskCompiler(test_signatures={"signatures": []})
    result = compiler.compile_cancel_pending(
        task_id="rtc_test_cancel_002",
        user=_user(),
        order=_order(),
        db_snapshot_hash="demo_db",
        run_replay=False,
    )
    sig = signature_from_blueprint(result.blueprint)
    fake_test = {
        "signatures": [
            {
                "task_id": "5",
                "tool_path": sig["tool_path"],
                "signature_hash": "not-the-same",
            }
        ]
    }
    report = check_contamination(result.blueprint, fake_test)
    assert report.contaminated
    assert "5" in report.matched_task_ids


def test_structure_signature_is_entity_agnostic():
    left = structure_signature(
        task_family="cancel_pending",
        tool_path=["find_user_id_by_email", "cancel_pending_order"],
        state_predicates=["order_status=pending"],
        required_effect_kinds=["order_status_change"],
    )
    right = structure_signature(
        task_family="cancel_pending",
        tool_path=["find_user_id_by_email", "cancel_pending_order"],
        state_predicates=["order_status=pending"],
        required_effect_kinds=["order_status_change"],
    )
    assert left["signature_hash"] == right["signature_hash"]


@pytest.mark.skipif(not SIGNATURES.exists(), reason="frozen test signatures missing")
def test_demo_cancel_blueprints_do_not_contaminate_frozen_test40():
    payload = json.loads(SIGNATURES.read_text(encoding="utf-8"))
    assert payload["count"] == 40
    compiler = RetailTaskCompiler(test_signatures=payload)
    result = compiler.compile_cancel_pending(
        task_id="rtc_test_cancel_003",
        user=_user(),
        order=_order(),
        auth_mode="name_zip",
        reason="ordered by mistake",
        db_snapshot_hash="demo_db",
    )
    assert not result.contamination.contaminated
    assert result.accepted


def test_coverage_axes_are_emitted():
    compiler = RetailTaskCompiler(test_signatures={"signatures": []})
    result = compiler.compile_cancel_pending(
        task_id="rtc_test_cancel_004",
        user=_user(),
        order=_order(),
        db_snapshot_hash="demo_db",
    )
    report = coverage_from_blueprints([result.blueprint])
    assert report.totals["blueprints"] == 1
    assert "cancel_pending" in report.axes["task_family"]
    assert "cooperative" in report.axes["user_behavior"]


def test_replay_rejects_missing_required_effect():
    compiler = RetailTaskCompiler(test_signatures={"signatures": []})
    result = compiler.compile_cancel_pending(
        task_id="rtc_test_cancel_005",
        user=_user(),
        order=_order(),
        db_snapshot_hash="demo_db",
        run_replay=False,
    )
    broken = result.blueprint.to_dict()
    broken["required_effects"] = [
        {
            "kind": "order_status_change",
            "entity": "#W9000999",
            "field": "status",
            "after": "exchange requested",
        }
    ]
    report = replay_reference_path_twice(broken, CancelPendingMockExecutor())
    assert not report.accepted
    assert any("missing required effect" in reason for reason in report.reasons)
