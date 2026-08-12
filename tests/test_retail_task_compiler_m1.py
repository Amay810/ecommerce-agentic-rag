# -*- coding: utf-8 -*-
"""Tests for M1 structure catalog, splits, and contamination gating."""

from __future__ import annotations

import json
from pathlib import Path

from ecommerce_rag.retail_task_compiler import (
    assign_structure_splits,
    check_contamination,
    compile_m1_dataset,
    m1_structure_catalog,
)
from ecommerce_rag.retail_task_compiler.contamination import load_test_signatures
from ecommerce_rag.retail_task_compiler.structures import BEHAVIOR_FAMILIES


ROOT = Path(__file__).resolve().parents[1]
SIGNATURES = ROOT / "docs" / "tau3_retail_test40_structure_signatures.json"
TAU_DB = Path(r"E:\cv_codex\external\tau2-bench\data\tau2\domains\retail\db.json")


def test_m1_catalog_covers_required_families_and_structure_count():
    catalog = m1_structure_catalog()
    assert 30 <= len(catalog) <= 80
    families = {item.task_family for item in catalog}
    assert set(BEHAVIOR_FAMILIES) <= families
    assert 8 <= len(families) <= 12
    hashes = {item.signature_hash() for item in catalog}
    assert len(hashes) == len(catalog)
    by_id = {item.structure_id: item for item in catalog}
    assert "S47_report_failure_no_false_success" not in by_id
    assert by_id["S16_cancel_after_success_idempotent"].confirmation_requirement == "not_applicable"


def test_structure_splits_are_structure_first_and_nonempty():
    catalog = m1_structure_catalog()
    splits = assign_structure_splits([item.structure_id for item in catalog])
    assert set(splits) == {item.structure_id for item in catalog}
    assert {"train", "dev", "held_out"} <= set(splits.values())
    # Same structure never maps to multiple splits.
    assert len(splits) == len(catalog)


def test_compiled_m1_has_zero_test40_contamination():
    if not SIGNATURES.exists() or not TAU_DB.exists():
        return
    signatures = load_test_signatures(SIGNATURES)
    payload = compile_m1_dataset(
        db_path=TAU_DB,
        test_signatures=signatures,
        instances_per_structure=2,
    )
    assert payload["contamination_count"] == 0
    assert payload["accepted_instances"] >= 60
    assert len(payload["split_tasks"]["pilot"]) == payload["structure_count"] == 47
    tasks = {task["id"]: task for task in payload["tasks"]}
    for task_id in payload["split_tasks"]["pilot"]:
        task = tasks[task_id]
        structure_id = task["provenance"]["structure_id"]
        if structure_id == "S16_cancel_after_success_idempotent":
            assert "already cancelled" in task["user_scenario"]["instructions"]["reason_for_call"]
        if "DB" in task["evaluation_criteria"]["reward_basis"] and not any(
            action["name"] in {
                "cancel_pending_order",
                "modify_pending_order_address",
                "modify_pending_order_items",
                "modify_pending_order_payment",
                "modify_user_address",
                "return_delivered_order_items",
                "exchange_delivered_order_items",
            }
            for action in task["evaluation_criteria"]["actions"]
        ):
            assert "ACTION" in task["evaluation_criteria"]["reward_basis"]
            assert task["evaluation_criteria"]["nl_assertions"]
    for instance in payload["instances"]:
        report = check_contamination(instance["blueprint"], signatures)
        assert not report.contaminated
    # Structure isolation across splits.
    by_split = {}
    for instance in payload["instances"]:
        by_split.setdefault(instance["split"], set()).add(
            instance["structure"]["structure_id"]
        )
    assert by_split["train"].isdisjoint(by_split["dev"])
    assert by_split["train"].isdisjoint(by_split["held_out"])
    assert by_split["dev"].isdisjoint(by_split["held_out"])
