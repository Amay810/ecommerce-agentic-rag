"""Export two dev failures as diagnostic next-action fixtures, never training data."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from ecommerce_rag.domain import AgentAction
from ecommerce_rag.harness import _requested_input_type
from ecommerce_rag.legacy_closure import LegacyActionEvaluator, TaskProgress
from ecommerce_rag.legacy_closure_benchmark import build_m1_tasks
from ecommerce_rag.legacy_closure_benchmark import FROZEN_TASK_SHA256
from ecommerce_rag.llm_policy import LLMPolicy
from ecommerce_rag.tool_schema import TOOL_SCHEMAS


TASK_IDS = ("m1_dev_07_01", "m1_dev_07_03")
TARGET = {
    "action_type": "final_answer",
    "tool_name": None,
    "arguments": {},
    "content": "请提供用于身份验证的六位验证码。",
    "requires_user_response": True,
}
_CREDENTIAL = re.compile(r"(?<![A-Za-z0-9])[0-9]{6}(?![A-Za-z0-9])")
_FORBIDDEN_POLICY_KEYS = {
    "gold_doc_ids", "allowed_tools", "forbidden_tools", "expected_state",
    "initial_state", "answer_expectations", "expected_tool_sequence",
    "grader", "grade", "metadata", "split",
}


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()


def _contains_forbidden_key(value: Any) -> bool:
    if isinstance(value, dict):
        return bool(_FORBIDDEN_POLICY_KEYS.intersection(value)) or any(
            _contains_forbidden_key(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_forbidden_key(item) for item in value)
    return False


def build_diagnostic_samples(records_path: Path, report_path: Path) -> list[dict[str, Any]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("protocol") != "legacy_task_closure_action_eval_dev_v1":
        raise ValueError("wrong formal report protocol")
    if report.get("task_manifest_sha256") != FROZEN_TASK_SHA256:
        raise ValueError("formal report task manifest does not match frozen tasks")
    if report.get("code_commit") != "ff6af987ff034ec3140679070038ae928ec65ca0":
        raise ValueError("formal report code commit is not the frozen v1 commit")
    if report.get("locked_executed") is not False:
        raise ValueError("diagnostic export requires locked_executed=false")

    rows: dict[str, tuple[dict[str, Any], str]] = {}
    for raw_line in records_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        record = json.loads(raw_line)
        if (record.get("task_id") in TASK_IDS
                and record.get("config") == "legacy_progress_action_eval"):
            if record["task_id"] in rows:
                raise ValueError(f"duplicate formal record for {record['task_id']}")
            rows[record["task_id"]] = (record, hashlib.sha256(raw_line.encode()).hexdigest())
    if set(rows) != set(TASK_IDS):
        raise ValueError("formal records must contain both frozen correction tasks")

    task_by_id = {task.task_id: task for task in build_m1_tasks()}
    target_action, target_violations = LLMPolicy._parse(
        json.dumps(TARGET, ensure_ascii=False),
        {schema["name"] for schema in TOOL_SCHEMAS},
    )
    if target_violations:
        raise ValueError(f"target action is not strict: {target_violations}")

    samples = []
    for task_id in TASK_IDS:
        record, source_hash = rows[task_id]
        if record.get("config") != "legacy_progress_action_eval":
            raise ValueError(f"{task_id}: wrong source config")
        corrections = record.get("correction_spans") or []
        if len(corrections) != 1 or corrections[0].get("reason") != "inappropriate_handoff":
            raise ValueError(f"{task_id}: expected one inappropriate-handoff correction")
        correction = corrections[0]
        step = correction.get("step")
        if step != 0:
            raise ValueError(f"{task_id}: only a step-zero policy input can be reconstructed exactly")
        progress_rows = [span for span in record.get("progress_spans", [])
                         if span.get("step") == step]
        if len(progress_rows) != 1:
            raise ValueError(f"{task_id}: missing unique progress span")
        progress_payload = {key: value for key, value in progress_rows[0].items() if key != "step"}
        progress = TaskProgress(
            progress_payload["workflow"], tuple(progress_payload["completed"]),
            tuple(progress_payload["pending"]), progress_payload.get("blocked_by"),
            tuple(progress_payload["allowed_next_actions"]),
            progress_payload.get("requested_input_type"), progress_payload["guard_state"],
            progress_payload.get("eligible"), bool(progress_payload.get("cancelled", False)),
        )
        requested = _requested_input_type(target_action, progress)
        if requested != "verification_code":
            raise ValueError("target must request a verification code")
        if not LegacyActionEvaluator().evaluate(
                target_action, progress, requested_input_type=requested).accepted:
            raise ValueError("target is not accepted by the deterministic action contract")

        rejected = correction["rejected_action"]
        _rejected_action, rejected_violations = LLMPolicy._parse(
            json.dumps(rejected, ensure_ascii=False),
            {schema["name"] for schema in TOOL_SCHEMAS})
        if rejected_violations:
            raise ValueError(f"{task_id}: rejected action is not strict")
        policy_input = {
            "history": [{"role": "user", "content": task_by_id[task_id].initial_message}],
            "session": {"task_progress": progress_payload},
            "action_evaluator_feedback": correction["feedback"],
        }
        if _contains_forbidden_key(policy_input):
            raise ValueError("hidden evaluation field leaked into policy input")
        payload = {
            "schema_version": 1,
            "schema": "legacy_next_action_correction_v1",
            "usage": "diagnostic_fixture_only",
            "training_approved": False,
            "source": {
                "split": "dev",
                "task_id": task_id,
                "config": record["config"],
                "decision_step": step,
                "source_record_sha256": source_hash,
                "code_commit": report["code_commit"],
                "task_manifest_sha256": report["task_manifest_sha256"],
                "initial_message_source": "reconstructed_from_hash_verified_frozen_task_manifest",
            },
            "policy_input": policy_input,
            "rejected_action": rejected,
            "target_action": TARGET,
            "supervision": {
                "failure_label": "inappropriate_handoff",
                "label_source": "deterministic_allowed_action_plus_human_adjudication",
                "target_requested_input_type": "verification_code",
                "no_chain_of_thought": True,
            },
            "audit": {
                "contains_hidden_fields": False,
                "credential_scan_passed": True,
                "replay_status": "diagnostic_only_not_training_approved",
            },
        }
        if _CREDENTIAL.search(json.dumps(payload, ensure_ascii=False)):
            raise ValueError("credential-like six-digit value in diagnostic fixture")
        payload["sample_id"] = _canonical_hash(payload)
        samples.append(payload)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    samples = build_diagnostic_samples(args.records, args.report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(json.dumps({"samples": len(samples), "training_approved": False,
                      "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
