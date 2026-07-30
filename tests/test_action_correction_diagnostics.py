from __future__ import annotations

import json

import pytest

from scripts.export_action_correction_diagnostics import (
    TASK_IDS,
    build_diagnostic_samples,
)
from ecommerce_rag.legacy_closure_benchmark import FROZEN_TASK_SHA256


def _record(task_id):
    progress = {
        "step": 0,
        "workflow": "return_resolution",
        "completed": ["order_id_collected", "return_reason_collected"],
        "pending": ["identity_verification"],
        "blocked_by": "user_input",
        "allowed_next_actions": ["ask_user:verification_code"],
        "requested_input_type": "verification_code",
        "guard_state": "identity_required",
        "eligible": None,
        "cancelled": False,
    }
    rejected = {
        "action_type": "handoff",
        "tool_name": None,
        "arguments": {"reason": "cannot bypass verification"},
        "content": "转人工处理。",
        "requires_user_response": False,
    }
    return {
        "task_id": task_id,
        "config": "legacy_progress_action_eval",
        "progress_spans": [progress],
        "correction_spans": [{
            "step": 0,
            "reason": "inappropriate_handoff",
            "rejected_action": rejected,
            "feedback": {
                "reason": "inappropriate_handoff",
                "pending": ["identity_verification"],
                "allowed_next_actions": ["ask_user:verification_code"],
            },
        }],
    }


def _write_records(path, task_ids=TASK_IDS):
    path.write_text("".join(
        json.dumps(_record(task_id), ensure_ascii=False) + "\n"
        for task_id in task_ids), encoding="utf-8")


def _write_report(path):
    path.write_text(json.dumps({
        "protocol": "legacy_task_closure_action_eval_dev_v1",
        "task_manifest_sha256": FROZEN_TASK_SHA256,
        "code_commit": "ff6af987ff034ec3140679070038ae928ec65ca0",
        "locked_executed": False,
    }), encoding="utf-8")


def test_exported_dev_failures_are_diagnostic_only_and_credential_free(tmp_path):
    records = tmp_path / "records.jsonl"
    report = tmp_path / "report.json"
    _write_records(records)
    _write_report(report)
    samples = build_diagnostic_samples(records, report)
    assert len(samples) == 2
    assert len({sample["sample_id"] for sample in samples}) == 2
    for sample in samples:
        assert sample["usage"] == "diagnostic_fixture_only"
        assert sample["training_approved"] is False
        assert sample["source"]["split"] == "dev"
        assert sample["target_action"]["requires_user_response"] is True
        assert sample["supervision"]["target_requested_input_type"] == "verification_code"
        payload = json.dumps(sample, ensure_ascii=False)
        assert "123456" not in payload
        for forbidden in ("expected_state", "grader", "gold_doc_ids"):
            assert forbidden not in sample["policy_input"]


def test_export_requires_both_exact_formal_failure_records(tmp_path):
    records = tmp_path / "records.jsonl"
    report = tmp_path / "report.json"
    _write_records(records, TASK_IDS[:1])
    _write_report(report)
    with pytest.raises(ValueError, match="both frozen correction tasks"):
        build_diagnostic_samples(records, report)


def test_export_rejects_unverified_report_provenance(tmp_path):
    records = tmp_path / "records.jsonl"
    report = tmp_path / "report.json"
    _write_records(records)
    _write_report(report)
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["code_commit"] = "wrong"
    report.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen v1 commit"):
        build_diagnostic_samples(records, report)
