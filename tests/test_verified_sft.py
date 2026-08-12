from __future__ import annotations

import json

import pytest

from ecommerce_rag.verified_sft import (
    assign_structure_splits,
    assign_isolated_splits,
    DatasetBuildConfig,
    ProcessAudit,
    build_verified_dataset,
    convert_messages,
    split_for_signature,
    structure_signature,
)


def _simulation(simulation_id="S1", task_id="1", reward=1.0):
    return {
        "id": simulation_id,
        "task_id": task_id,
        "trial": 0,
        "seed": 7,
        "termination_reason": "user_stop",
        "reward_info": {"reward": reward, "db_check": {"db_match": True}},
        "messages": [
            {"role": "user", "content": "cancel my order"},
            {"role": "assistant", "content": "What is your email?"},
            {"role": "user", "content": "a@example.com"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "function": {
                            "name": "find_user_id_by_email",
                            "arguments": {"email": "a@example.com"},
                        },
                    }
                ],
            },
            {"role": "tool", "content": '"U1"', "error": False},
            {"role": "assistant", "content": "Please confirm cancellation."},
            {"role": "user", "content": "yes"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c2",
                        "function": {
                            "name": "cancel_pending_order",
                            "arguments": {
                                "order_id": "O1",
                                "reason": "no longer needed",
                            },
                        },
                    }
                ],
            },
            {"role": "tool", "content": '{"status":"cancelled"}', "error": False},
            {"role": "assistant", "content": "The order was cancelled."},
        ],
    }


def _config(require_process_audit=True):
    return DatasetBuildConfig(
        source_split="train",
        teacher_model="Qwen3-Teacher",
        teacher_usage_rights="approved_open_weights",
        require_process_audit=require_process_audit,
    )


def test_terms_gate_and_train_split_are_fail_closed():
    with pytest.raises(ValueError, match="train split"):
        DatasetBuildConfig(
            source_split="test",
            teacher_model="teacher",
            teacher_usage_rights="approved_open_weights",
        )
    with pytest.raises(ValueError, match="terms gate"):
        DatasetBuildConfig(
            source_split="train",
            teacher_model="teacher",
            teacher_usage_rights="unknown",
        )


def test_structure_split_is_deterministic_and_entity_agnostic():
    left = structure_signature(_simulation()["messages"])
    right_messages = json.loads(json.dumps(_simulation()["messages"]))
    right_messages[2]["content"] = "other@example.com"
    right = structure_signature(right_messages)
    assert left["signature_hash"] == right["signature_hash"]
    assert split_for_signature(left["signature_hash"]) == split_for_signature(
        right["signature_hash"]
    )


def test_structure_allocator_keeps_nonempty_dev_and_held_out():
    allocation = assign_structure_splits(f"signature-{index}" for index in range(12))
    assert set(allocation.values()) == {"train", "dev", "held_out"}


def test_isolated_allocator_keeps_connected_task_and_structure_together():
    records = []
    for simulation_id, task_id, signature in (
        ("S1", "T1", "A"),
        ("S2", "T1", "B"),
        ("S3", "T2", "B"),
        ("S4", "T3", "C"),
        ("S5", "T4", "D"),
    ):
        records.append(
            {
                "provenance": {
                    "simulation_id": simulation_id,
                    "task_id": task_id,
                    "structure": {"signature_hash": signature},
                }
            }
        )
    allocation = assign_isolated_splits(records)
    assert allocation["S1"] == allocation["S2"] == allocation["S3"]


def test_strict_builder_requires_process_audit(tmp_path):
    results = tmp_path / "results.json"
    results.write_text(json.dumps({"simulations": [_simulation()]}), encoding="utf-8")
    manifest = build_verified_dataset(
        payload=json.loads(results.read_text()),
        source_results=results,
        output_dir=tmp_path / "dataset",
        system_prompt="system",
        tools=[],
        allowed_task_ids={"1"},
        process_audits={},
        config=_config(),
    )
    assert manifest["accepted"] == 0
    assert manifest["rejections"]["missing_process_audit"] == 1


def test_builder_accepts_audited_trajectory_and_keeps_structure_in_one_split(tmp_path):
    simulations = [_simulation("S1"), _simulation("S2")]
    results = tmp_path / "results.json"
    results.write_text(json.dumps({"simulations": simulations}), encoding="utf-8")
    audits = {
        sim["id"]: ProcessAudit(
            simulation_id=sim["id"], process_compliant=True, reviewer="test"
        )
        for sim in simulations
    }
    manifest = build_verified_dataset(
        payload={"simulations": simulations},
        source_results=results,
        output_dir=tmp_path / "dataset",
        system_prompt="system",
        tools=[{"type": "function", "function": {"name": "cancel_pending_order"}}],
        allowed_task_ids={"1"},
        process_audits=audits,
        config=_config(),
    )
    assert manifest["accepted"] == 2
    assert manifest["unique_structures"] == 1
    assert sum(manifest["split_structure_counts"].values()) == 1


def test_converter_rejects_tool_errors_and_mixed_assistant_output():
    _, reason = convert_messages(
        [{"role": "tool", "content": "failed", "error": True}]
    )
    assert reason == "tool_error_in_trajectory"
    _, reason = convert_messages(
        [
            {
                "role": "assistant",
                "content": "also text",
                "tool_calls": [{"function": {"name": "x", "arguments": {}}}],
            }
        ]
    )
    assert reason == "assistant_content_and_tool_call"


def test_compiled_task_keeps_frozen_structure_identity_and_split(tmp_path):
    simulation = _simulation("S-frozen", "compiled-T1")
    results = tmp_path / "results.json"
    results.write_text(json.dumps({"simulations": [simulation]}), encoding="utf-8")
    config = DatasetBuildConfig(
        source_split="train",
        teacher_model="Qwen3-Teacher",
        teacher_usage_rights="approved_open_weights",
        task_structures={
            "compiled-T1": {
                "structure_id": "S13_cancel_email_confirmed",
                "signature_hash": "compiler-frozen-hash",
                "behavior_family": "cancel_pending",
            }
        },
        preassigned_task_splits={"compiled-T1": "train"},
    )
    manifest = build_verified_dataset(
        payload={"simulations": [simulation]},
        source_results=results,
        output_dir=tmp_path / "dataset",
        system_prompt="system",
        tools=[],
        allowed_task_ids={"compiled-T1"},
        process_audits={
            "S-frozen": ProcessAudit(simulation_id="S-frozen", process_compliant=True)
        },
        config=config,
    )
    record = json.loads((tmp_path / "dataset" / "train.jsonl").read_text())
    assert manifest["accepted"] == 1
    assert record["provenance"]["structure"]["structure_id"] == "S13_cancel_email_confirmed"
    assert record["provenance"]["structure"]["signature_hash"] == "compiler-frozen-hash"
    assert record["provenance"]["split"] == "train"
