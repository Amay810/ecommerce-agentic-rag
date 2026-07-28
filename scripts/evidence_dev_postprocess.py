"""Read-only post-processing for the frozen Phase-A dev_v3 experiment.

This module intentionally does not import or execute the agent.  It snapshots the
SQLite file set, opens only the snapshot in SQLite read-only/query-only mode, and
produces versioned integrity, aggregate, attribution, and human-audit artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import sqlite3
import statistics
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from ecommerce_rag.evidence import EVIDENCE_BEARING_TOOLS
from ecommerce_rag.tools import READ_TOOLS, WRITE_TOOLS


VARIANTS = ("base", "evidence_verify", "evidence_verify_repair")
RUN_NAME = "evidence_phase_a_dev_v3"
CONTRACT = "evidence_phase_a_tasks_v2"
RESULT_COMMIT = "d3f5063"
REQUIRED_TRAJECTORY_FIELDS = {
    "trajectory_id", "task_id", "messages", "model_calls", "tool_calls",
    "final_answer", "evidence_ledger", "evidence_conversion_spans",
    "verification_spans", "repair_spans",
}
REQUIRED_GRADE_FIELDS = {
    "task_id", "operational_success", "policy_compliant",
    "terminal_state_match", "joint_success", "failure_type",
    "handoff_expected", "handoff_observed", "answer_fact_applicable",
    "hard_verification_pass", "answer_fact_pass", "citation_binding_pass",
    "required_evidence_coverage", "repair_attempted", "repair_succeeded",
    "repair_hard_recovery", "repair_diagnostic_improvement",
    "unsupported_high_risk_claims", "contradicted_claims",
    "omitted_required_facts", "citation_diagnostics", "latency_ms",
}
FACT_STATUS = {"supported", "contradicted", "unsupported", "not_factual", "unclear"}
CITATION_STATUS = {"correct", "incorrect", "missing", "not_required", "unclear"}
BOOL_OR_BLANK = {"", "true", "false"}
ANSWER_HUMAN_FIELDS = {
    "human_answer_complete", "human_handoff_appropriate", "human_overall_pass",
    "human_claim_segmentation_complete", "missing_claim_notes", "review_notes",
}
CLAIM_HUMAN_FIELDS = {"human_fact_status", "human_citation_status", "review_notes"}
_CITATION_RE = re.compile(r"\[(E\d+)\]")
_PRODUCT_ID_RE = re.compile(r"\bP[0-9]{5}\b")
_SENTENCE_BOUNDARY = re.compile(r"(?:\r?\n)+|(?<=[。！？；;])")


class PostprocessError(RuntimeError):
    """A fail-closed post-processing error."""


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "name": path.name,
        "sha256": _sha256(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    }


def sqlite_file_set(path: Path) -> list[Path]:
    candidates = [path, Path(f"{path}-wal"), Path(f"{path}-shm")]
    return [candidate for candidate in candidates if candidate.exists()]


def inventory_sqlite(path: Path) -> list[dict]:
    if not path.exists():
        raise PostprocessError(f"missing SQLite: {path}")
    return [file_record(item) for item in sqlite_file_set(path)]


def _inventory_signature(records: list[dict]) -> list[tuple[str, int, int, str]]:
    return sorted((row["name"], row["size"], row["mtime_ns"], row["sha256"]) for row in records)


def ensure_inventory_unchanged(before: list[dict], after: list[dict], label: str) -> None:
    if _inventory_signature(before) != _inventory_signature(after):
        raise PostprocessError(f"{label} SQLite file set or hash changed during read-only analysis")


def snapshot_sqlite(path: Path, snapshot_dir: Path) -> tuple[Path, list[dict]]:
    """Byte-copy main/WAL/SHM after hashing; callers verify the source again."""
    before = inventory_sqlite(path)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for record in before:
        shutil.copy2(record["path"], snapshot_dir / record["name"])
    after = inventory_sqlite(path)
    ensure_inventory_unchanged(before, after, str(path))
    return snapshot_dir / path.name, before


def open_readonly(path: Path) -> sqlite3.Connection:
    uri = f"{path.resolve().as_uri()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    if conn.execute("PRAGMA query_only").fetchone()[0] != 1:
        conn.close()
        raise PostprocessError(f"query_only could not be enabled: {path}")
    return conn


@dataclass(frozen=True)
class StoredTrajectory:
    variant: str
    trajectory_id: str
    task_id: str
    seed: int
    trajectory: dict
    grade: dict


def load_store(path: Path, variant: str) -> list[StoredTrajectory]:
    conn = open_readonly(path)
    try:
        rows = conn.execute(
            "SELECT trajectory_id,task_id,seed,trajectory_json,grade_json "
            "FROM trajectories ORDER BY task_id"
        ).fetchall()
    finally:
        conn.close()
    loaded = []
    for row in rows:
        loaded.append(StoredTrajectory(
            variant=variant,
            trajectory_id=str(row["trajectory_id"]),
            task_id=str(row["task_id"]),
            seed=int(row["seed"]),
            trajectory=json.loads(row["trajectory_json"]),
            grade=json.loads(row["grade_json"]),
        ))
    return loaded


def _raw_source_count(tool_name: str, result: dict) -> int:
    if tool_name == "search_catalog":
        return len(result.get("items") or [])
    if tool_name == "get_policy":
        return len(result.get("policies") or [])
    if tool_name == "compare_products":
        return len(result.get("products") or [])
    if tool_name == "get_product":
        return int(bool(result.get("product")))
    if tool_name == "get_order":
        return int(bool(result.get("order")))
    if tool_name == "check_return_eligibility":
        return int(bool(result.get("eligibility") or any(key != "ok" for key in result)))
    return int(any(key != "ok" for key in result))


def conversion_errors(trajectory: dict) -> list[str]:
    """Validate call -> one span -> zero-or-many evidence ownership."""
    errors: list[str] = []
    tool_calls = trajectory.get("tool_calls")
    spans = trajectory.get("evidence_conversion_spans")
    ledger = trajectory.get("evidence_ledger")
    if not isinstance(tool_calls, list):
        return ["tool_calls is missing or not a list"]
    if not isinstance(spans, list):
        return ["evidence_conversion_spans is missing or not a list"]
    if not isinstance(ledger, list):
        return ["evidence_ledger is missing or not a list"]

    calls = {
        str(call.get("call_id")): call for call in tool_calls
        if call.get("name") in EVIDENCE_BEARING_TOOLS
    }
    if "None" in calls:
        errors.append("evidence-producing call lacks call_id")
    spans_by_call: dict[str, list[dict]] = defaultdict(list)
    for span in spans:
        spans_by_call[str(span.get("tool_call_id"))].append(span)
    for call_id in calls:
        count = len(spans_by_call.get(call_id, []))
        if count != 1:
            errors.append(f"{call_id}: expected exactly one conversion span, observed {count}")
    for call_id in spans_by_call:
        if call_id not in calls:
            errors.append(f"{call_id}: conversion span has no evidence-producing tool call")

    evidence_ids = [str(item.get("evidence_id")) for item in ledger]
    if "None" in evidence_ids:
        errors.append("evidence item lacks evidence_id")
    duplicates = sorted(item for item, count in Counter(evidence_ids).items() if count > 1)
    if duplicates:
        errors.append(f"duplicate evidence IDs: {duplicates}")
    ledger_by_call: dict[str, list[dict]] = defaultdict(list)
    for item in ledger:
        call_id = str(item.get("tool_call_id"))
        ledger_by_call[call_id].append(item)
        if call_id not in calls:
            errors.append(f"{item.get('evidence_id')}: orphan evidence owner {call_id}")
        elif len(spans_by_call.get(call_id, [])) != 1:
            errors.append(f"{item.get('evidence_id')}: owner has no unique conversion span")

    for call_id, span_rows in spans_by_call.items():
        if len(span_rows) != 1:
            continue
        span = span_rows[0]
        call = calls.get(call_id)
        if call is None:
            continue
        status = span.get("status")
        if status not in {"converted", "valid_empty", "tool_failed", "converter_missing"}:
            errors.append(f"{call_id}: invalid conversion status {status!r}")
        if status == "converter_missing":
            errors.append(f"{call_id}: converter_missing")
        actual_items = ledger_by_call.get(call_id, [])
        actual_ids = [str(item.get("evidence_id")) for item in actual_items]
        declared_ids = [str(item) for item in (span.get("evidence_ids") or [])]
        if declared_ids != actual_ids:
            errors.append(f"{call_id}: span evidence_ids differ from ledger ownership/order")
        if span.get("evidence_item_count") != len(actual_items):
            errors.append(f"{call_id}: declared evidence count differs from ledger")
        result = call.get("result") if isinstance(call.get("result"), dict) else {}
        source_count = _raw_source_count(str(call.get("name")), result)
        if span.get("source_item_count") != source_count:
            errors.append(f"{call_id}: declared source count differs from raw result")
        if status == "converted" and (not result.get("ok") or source_count <= 0 or not actual_items):
            errors.append(f"{call_id}: invalid converted span")
        if status == "valid_empty" and (not result.get("ok") or source_count != 0 or actual_items):
            errors.append(f"{call_id}: invalid valid_empty span")
        if status == "tool_failed" and (result.get("ok") or actual_items):
            errors.append(f"{call_id}: invalid tool_failed span")
        name = str(call.get("name"))
        if name in READ_TOOLS and result.get("ok") and source_count > 0 and not actual_items:
            errors.append(f"{call_id}: successful nonempty read produced no evidence")
        # Writes may legitimately produce zero evidence; no count equality is imposed.
        if name in WRITE_TOOLS and actual_items and status != "converted":
            errors.append(f"{call_id}: write evidence exists without converted status")
    return errors


def _all_llm_traces(trajectory: dict) -> Iterable[dict]:
    for model_call in trajectory.get("model_calls") or []:
        trace = model_call.get("llm")
        if isinstance(trace, dict):
            yield trace
            repair = trace.get("repair_llm")
            if isinstance(repair, dict):
                yield repair
    for repair in trajectory.get("repair_spans") or []:
        trace = repair.get("llm")
        if isinstance(trace, dict):
            yield trace


def _attempts(trajectory: dict) -> Iterable[dict]:
    seen: set[int] = set()
    for trace in _all_llm_traces(trajectory):
        for attempt in trace.get("attempts") or []:
            if id(attempt) not in seen:
                seen.add(id(attempt))
                yield attempt


def validate_rows(
    by_variant: dict[str, list[StoredTrajectory]], expected_ids: list[str]
) -> tuple[list[dict], dict]:
    checks: list[dict] = []
    details: dict[str, Any] = {}

    def add(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    expected = set(expected_ids)
    add("manifest_has_80_unique_tasks", len(expected_ids) == 80 and len(expected) == 80,
        {"rows": len(expected_ids), "unique": len(expected)})
    all_trajectory_ids: list[str] = []
    for variant in VARIANTS:
        rows = by_variant.get(variant, [])
        ids = [row.task_id for row in rows]
        trajectory_ids = [row.trajectory_id for row in rows]
        all_trajectory_ids.extend(trajectory_ids)
        add(f"{variant}:80_rows_80_unique_tasks", len(rows) == 80 and len(set(ids)) == 80,
            {"rows": len(rows), "unique_tasks": len(set(ids))})
        add(f"{variant}:task_ids_match_manifest", set(ids) == expected,
            {"missing": sorted(expected - set(ids)), "extra": sorted(set(ids) - expected)})
        add(f"{variant}:trajectory_ids_unique", len(trajectory_ids) == len(set(trajectory_ids)),
            {"rows": len(trajectory_ids), "unique": len(set(trajectory_ids))})
        generation_errors = fallback = repeated_repairs = 0
        field_errors: list[str] = []
        evidence_errors: list[str] = []
        for row in rows:
            missing_t = sorted(REQUIRED_TRAJECTORY_FIELDS - set(row.trajectory))
            missing_g = sorted(REQUIRED_GRADE_FIELDS - set(row.grade))
            if missing_t:
                field_errors.append(f"{row.task_id}: trajectory missing {missing_t}")
            if missing_g:
                field_errors.append(f"{row.task_id}: grade missing {missing_g}")
            if row.trajectory.get("task_id") != row.task_id or row.grade.get("task_id") != row.task_id:
                field_errors.append(f"{row.task_id}: embedded task_id mismatch")
            repairs = row.trajectory.get("repair_spans")
            if not isinstance(repairs, list):
                field_errors.append(f"{row.task_id}: repair_spans is not a list")
            elif len(repairs) > 1:
                repeated_repairs += 1
            traces = list(_all_llm_traces(row.trajectory))
            if not isinstance(row.trajectory.get("model_calls"), list):
                field_errors.append(f"{row.task_id}: model_calls is not a list")
            fallback += int(any(trace.get("resolution") == "fallback_handoff" for trace in traces))
            generation_errors += sum(
                attempt.get("parse_stage") == "generation_error" for attempt in _attempts(row.trajectory)
            )
            evidence_errors.extend(f"{row.task_id}: {error}" for error in conversion_errors(row.trajectory))
        add(f"{variant}:generation_errors_zero", generation_errors == 0, generation_errors)
        add(f"{variant}:fallback_only_zero", fallback == 0, fallback)
        add(f"{variant}:repair_at_most_once", repeated_repairs == 0, repeated_repairs)
        add(f"{variant}:required_fields_complete", not field_errors, field_errors[:20])
        add(f"{variant}:evidence_references_complete", not evidence_errors, evidence_errors[:30])
        details[variant] = {
            "generation_errors": generation_errors,
            "fallback_trajectories": fallback,
            "repeated_repairs": repeated_repairs,
            "field_error_count": len(field_errors),
            "evidence_error_count": len(evidence_errors),
        }
    duplicates = sorted(item for item, count in Counter(all_trajectory_ids).items() if count > 1)
    add("trajectory_ids_unique_across_variants", not duplicates, duplicates)
    return checks, details


def rate(numerator: int, denominator: int) -> dict:
    return {
        "numerator": int(numerator),
        "denominator": int(denominator),
        "rate": (numerator / denominator if denominator else None),
    }


def distribution(values: Iterable[float]) -> dict:
    rows = sorted(float(value) for value in values)
    if not rows:
        return {"count": 0, "mean": None, "p50": None, "p95": None}
    p95_index = max(0, math.ceil(0.95 * len(rows)) - 1)
    return {
        "count": len(rows),
        "mean": statistics.fmean(rows),
        "p50": statistics.median(rows),
        "p95": rows[p95_index],
    }


def _terminal_is_handoff(row: StoredTrajectory) -> bool:
    if row.grade.get("handoff_observed"):
        return True
    actions = row.trajectory.get("actions") or []
    return bool(actions and actions[-1].get("action_type") == "handoff")


def _initial_verification(row: StoredTrajectory) -> dict | None:
    spans = row.trajectory.get("verification_spans") or []
    return next((span for span in spans if span.get("phase") == "initial"), None)


def _token_counts(row: StoredTrajectory) -> tuple[int, int, int]:
    attempts = list(_attempts(row.trajectory))
    return (
        sum(int(item.get("prompt_tokens") or 0) for item in attempts),
        sum(int(item.get("completion_tokens") or 0) for item in attempts),
        len(attempts),
    )


def _repair_token_counts(row: StoredTrajectory) -> tuple[int, int, int]:
    prompt = completion = generations = 0
    for repair in row.trajectory.get("repair_spans") or []:
        trace = repair.get("llm")
        if not isinstance(trace, dict):
            continue
        for attempt in trace.get("attempts") or []:
            prompt += int(attempt.get("prompt_tokens") or 0)
            completion += int(attempt.get("completion_tokens") or 0)
            generations += 1
    return prompt, completion, generations


def variant_metrics(rows: list[StoredTrajectory]) -> dict:
    applicable = [row for row in rows if row.grade.get("answer_fact_applicable")]
    expected_answers = [row for row in rows if not row.grade.get("handoff_expected")]
    accepted = [row for row in applicable if not _terminal_is_handoff(row)]
    initial_rows: list[tuple[StoredTrajectory, bool]] = []
    direct_handoff = 0
    for row in applicable:
        initial = _initial_verification(row)
        if initial is not None:
            initial_rows.append((row, bool(initial.get("hard_verification_pass"))))
        elif not _terminal_is_handoff(row):
            # Base has no runtime verification span: its accepted final answer is its initial answer.
            initial_rows.append((row, bool(row.grade.get("hard_verification_pass"))))
        else:
            direct_handoff += 1
    initial_pass = sum(passed for _, passed in initial_rows)
    accepted_pass = sum(bool(row.grade.get("hard_verification_pass")) for row in accepted)
    coverage_count = sum(not _terminal_is_handoff(row) for row in expected_answers)
    required = [float(row.grade["required_evidence_coverage"]) for row in applicable
                if row.grade.get("required_evidence_coverage") is not None]
    repairs = [row for row in rows if row.grade.get("repair_attempted")]
    repair_tokens = [_repair_token_counts(row) for row in repairs]
    adopted_repairs = [row for row in repairs if row.grade.get("repair_succeeded")]
    rejected_repairs = [row for row in repairs if not row.grade.get("repair_succeeded")]
    handoff_tp = sum(row.grade.get("handoff_expected") and row.grade.get("handoff_observed") for row in rows)
    handoff_fp = sum(not row.grade.get("handoff_expected") and row.grade.get("handoff_observed") for row in rows)
    handoff_fn = sum(row.grade.get("handoff_expected") and not row.grade.get("handoff_observed") for row in rows)
    totals = [_token_counts(row) for row in rows]
    return {
        "task_count": len(rows),
        "operational_success": rate(sum(bool(row.grade.get("operational_success")) for row in rows), len(rows)),
        "joint_success": rate(sum(bool(row.grade.get("joint_success")) for row in rows), len(rows)),
        "policy_compliance": rate(sum(bool(row.grade.get("policy_compliant")) for row in rows), len(rows)),
        "terminal_state_match": rate(sum(bool(row.grade.get("terminal_state_match")) for row in rows), len(rows)),
        "initial_answer_hard_pass": rate(initial_pass, len(initial_rows)),
        "initial_direct_handoff_count": direct_handoff,
        "accepted_answer_hard_pass": rate(accepted_pass, len(accepted)),
        "accepted_answer_error_rate": rate(len(accepted) - accepted_pass, len(accepted)),
        "answer_coverage": rate(coverage_count, len(expected_answers)),
        "final_hard_pass": rate(sum(bool(row.grade.get("hard_verification_pass")) for row in rows), len(rows)),
        "handoff_rate": rate(sum(bool(row.grade.get("handoff_observed")) for row in rows), len(rows)),
        "handoff_precision": rate(handoff_tp, handoff_tp + handoff_fp),
        "handoff_recall": rate(handoff_tp, handoff_tp + handoff_fn),
        "citation_binding_accepted_answers": rate(
            sum(bool(row.grade.get("citation_binding_pass")) for row in accepted), len(accepted)),
        "unsupported_answer_rate": rate(
            sum(bool(row.grade.get("unsupported_high_risk_claims")) for row in applicable), len(applicable)),
        "contradicted_answer_rate": rate(
            sum(bool(row.grade.get("contradicted_claims")) for row in applicable), len(applicable)),
        "required_coverage": {
            "sum": sum(required),
            "denominator": len(required),
            "mean": statistics.fmean(required) if required else None,
            "fully_covered": rate(sum(value >= 1.0 for value in required), len(required)),
        },
        "repair_attempted": rate(len(repairs), len(rows)),
        "repair_adopted": rate(sum(bool(row.grade.get("repair_succeeded")) for row in repairs), len(repairs)),
        "repair_hard_recovery": rate(sum(bool(row.grade.get("repair_hard_recovery")) for row in repairs), len(repairs)),
        "repair_diagnostic_improvement": rate(
            sum(bool(row.grade.get("repair_diagnostic_improvement")) for row in repairs), len(repairs)),
        "repair_extra_cost": {
            "attempted_trajectories": len(repairs),
            "generation_attempts": sum(item[2] for item in repair_tokens),
            "prompt_tokens": {"total": sum(item[0] for item in repair_tokens),
                              "per_attempted_trajectory": distribution(item[0] for item in repair_tokens)},
            "completion_tokens": {"total": sum(item[1] for item in repair_tokens),
                                  "per_attempted_trajectory": distribution(item[1] for item in repair_tokens)},
            "adopted": {
                "count": len(adopted_repairs),
                "initial_hard_pass": rate(
                    sum(bool((_initial_verification(row) or {}).get("hard_verification_pass"))
                        for row in adopted_repairs), len(adopted_repairs)),
                "repair_hard_pass": rate(
                    sum(bool((next((span for span in row.trajectory.get("verification_spans") or []
                                    if span.get("phase") == "repair"), {})
                              ).get("hard_verification_pass")) for row in adopted_repairs),
                    len(adopted_repairs)),
            },
            "rejected": {
                "count": len(rejected_repairs),
                "initial_hard_pass": rate(
                    sum(bool((_initial_verification(row) or {}).get("hard_verification_pass"))
                        for row in rejected_repairs), len(rejected_repairs)),
                "repair_hard_pass": rate(
                    sum(bool((next((span for span in row.trajectory.get("verification_spans") or []
                                    if span.get("phase") == "repair"), {})
                              ).get("hard_verification_pass")) for row in rejected_repairs),
                    len(rejected_repairs)),
            },
        },
        "latency_ms": distribution(row.grade.get("latency_ms") or row.trajectory.get("elapsed_ms") or 0 for row in rows),
        "prompt_tokens": {"total": sum(item[0] for item in totals), "per_task": distribution(item[0] for item in totals)},
        "completion_tokens": {"total": sum(item[1] for item in totals), "per_task": distribution(item[1] for item in totals)},
        "model_generations": {"total": sum(item[2] for item in totals), "per_task": distribution(item[2] for item in totals)},
        "tool_calls": {"total": sum(len(row.trajectory.get("tool_calls") or []) for row in rows),
                       "per_task": distribution(len(row.trajectory.get("tool_calls") or []) for row in rows)},
        "verifier_wall_time_ms": {"available": False, "reason": "not independently recorded in frozen trace"},
    }


def paired_cost_deltas(by_variant: dict[str, list[StoredTrajectory]]) -> dict:
    indexed = {variant: {row.task_id: row for row in rows} for variant, rows in by_variant.items()}
    base = indexed["base"]
    result = {}
    for variant in VARIANTS[1:]:
        latency = []
        prompt = []
        completion = []
        generations = []
        tools = []
        for task_id, base_row in base.items():
            other = indexed[variant][task_id]
            bp, bc, bg = _token_counts(base_row)
            op, oc, og = _token_counts(other)
            latency.append(float(other.grade.get("latency_ms") or 0) - float(base_row.grade.get("latency_ms") or 0))
            prompt.append(op - bp)
            completion.append(oc - bc)
            generations.append(og - bg)
            tools.append(len(other.trajectory.get("tool_calls") or []) - len(base_row.trajectory.get("tool_calls") or []))
        result[variant] = {
            "paired_latency_ms_delta": distribution(latency),
            "paired_prompt_token_delta": distribution(prompt),
            "paired_completion_token_delta": distribution(completion),
            "paired_generation_delta": distribution(generations),
            "paired_tool_call_delta": distribution(tools),
            "repair_extra_generation_and_tokens": {
                "note": "included in the paired deltas; per-repair totals are also reported from repair spans",
                "repair_attempts": sum(len(row.trajectory.get("repair_spans") or []) for row in indexed[variant].values()),
            },
            "handoff_reduced_followup_calls": {
                "available": False,
                "reason": "counterfactual calls after handoff are not observable in frozen trajectories",
            },
        }
    return result


def _tool_sequence(row: StoredTrajectory, successful_only: bool = False) -> list[str]:
    calls = row.trajectory.get("tool_calls") or []
    if successful_only:
        calls = [call for call in calls if isinstance(call.get("result"), dict) and call["result"].get("ok")]
    return [str(call.get("name")) for call in calls]


def _verification_summary(row: StoredTrajectory) -> dict:
    spans = row.trajectory.get("verification_spans") or []
    return {
        "initial": next((span for span in spans if span.get("phase") == "initial"), None),
        "repair": next((span for span in spans if span.get("phase") == "repair"), None),
        "final_grade": {
            key: row.grade.get(key) for key in (
                "hard_verification_pass", "unsupported_high_risk_claims", "contradicted_claims",
                "omitted_required_facts", "citation_diagnostics", "citation_binding_pass",
            )
        },
    }


def _product_contract_context(row: StoredTrajectory) -> dict:
    searches = [call for call in row.trajectory.get("tool_calls") or [] if call.get("name") == "search_catalog"]
    products = [call for call in row.trajectory.get("tool_calls") or [] if call.get("name") == "get_product"]
    search_fields = sorted({key for call in searches for item in ((call.get("result") or {}).get("items") or []) for key in item})
    product_fields = sorted({key for call in products for key in (((call.get("result") or {}).get("product") or {}).keys())})
    return {
        "search_catalog_result_fields": search_fields,
        "get_product_result_fields": product_fields,
        "final_answer": row.trajectory.get("final_answer", ""),
        "contract_necessity_assessment": "unclear",
        "allowed_assessments": [
            "business_required", "redundant_for_observed_answer",
            "search_evidence_insufficient", "unclear",
        ],
        "note": "Requires human comparison of observed answer facts with fields available from each tool.",
    }


def _recommend_handoff_candidate(row: StoredTrajectory) -> str:
    verification = _verification_summary(row)
    initial = verification["initial"] or {}
    if initial.get("contradicted_claims") or initial.get("citation_oppositions"):
        return "true_contradiction_candidate"
    if initial.get("unsupported_high_risk_claims"):
        return "true_unsupported_candidate"
    if initial.get("omitted_required_facts"):
        return "required_coverage_candidate"
    if initial.get("citation_diagnostics"):
        return "citation_only_candidate"
    conversion = conversion_errors(row.trajectory)
    if conversion:
        return "evidence_conversion_error"
    return "unclear"


def failure_attribution(by_variant: dict[str, list[StoredTrajectory]], tasks: dict[str, dict]) -> dict:
    indexed = {variant: {row.task_id: row for row in rows} for variant, rows in by_variant.items()}
    entries = []
    for task_id in sorted(tasks):
        variant_rows = {variant: indexed[variant][task_id] for variant in VARIANTS}
        failures = {variant: (row.grade.get("failure_type") or "ok") for variant, row in variant_rows.items()}
        operational = {variant: bool(row.grade.get("operational_success")) for variant, row in variant_rows.items()}
        has_interest = len(set(operational.values())) > 1 or any(value != "ok" for value in failures.values()) \
            or any(row.grade.get("repair_attempted") for row in variant_rows.values())
        if not has_interest:
            continue
        category = tasks[task_id].get("category")
        item = {
            "task_id": task_id,
            "category": category,
            "operational_disagreement": len(set(operational.values())) > 1,
            "variants": {},
            "attribution_status": "requires_human_confirmation",
        }
        for variant, row in variant_rows.items():
            item["variants"][variant] = {
                "operational_success": operational[variant],
                "joint_success": bool(row.grade.get("joint_success")),
                "failure_type": failures[variant],
                "raw_observed_sequence": row.grade.get("raw_observed_tool_sequence") or _tool_sequence(row),
                "successful_sequence": row.grade.get("successful_tool_sequence") or _tool_sequence(row, True),
                "failed_or_empty_calls": row.grade.get("failed_or_empty_tool_calls") or [],
                "handoff_observed": bool(row.grade.get("handoff_observed")),
                "verification": _verification_summary(row),
                "repair_spans": row.trajectory.get("repair_spans") or [],
                "final_answer": row.trajectory.get("final_answer", ""),
            }
            if category == "product_qa":
                item["variants"][variant]["product_contract"] = _product_contract_context(row)
            if category == "recommend" and row.grade.get("handoff_observed"):
                item["variants"][variant]["recommend_handoff_candidate"] = _recommend_handoff_candidate(row)
                item["variants"][variant]["allowed_human_classes"] = [
                    "true_contradiction", "true_unsupported", "required_coverage",
                    "citation_only", "claim_extraction_error", "evidence_conversion_error",
                    "verifier_false_positive", "unclear",
                ]
        entries.append(item)
    failure_counts = {
        variant: dict(sorted(Counter(row.grade.get("failure_type") or "ok" for row in rows).items()))
        for variant, rows in by_variant.items()
    }
    return {
        "schema_version": 1,
        "run_name": RUN_NAME,
        "failure_counts": failure_counts,
        "task_attributions": entries,
        "warning": "Automated classes are evidence for review, not verified causal labels.",
    }


def mandatory_task_ids(by_variant: dict[str, list[StoredTrajectory]]) -> tuple[set[str], dict[str, list[str]]]:
    indexed = {variant: {row.task_id: row for row in rows} for variant, rows in by_variant.items()}
    disagreements = {
        task_id for task_id in indexed["base"]
        if len({bool(indexed[variant][task_id].grade.get("operational_success")) for variant in VARIANTS}) > 1
    }
    adopted = {
        row.task_id for row in by_variant["evidence_verify_repair"] if row.grade.get("repair_succeeded")
    }
    return disagreements | adopted, {
        "operational_disagreement": sorted(disagreements),
        "adopted_repair": sorted(adopted),
    }


def _sampling_labels(task_id: str, indexed: dict[str, dict[str, StoredTrajectory]], task: dict) -> set[str]:
    rows = {variant: indexed[variant][task_id] for variant in VARIANTS}
    labels = {f"category:{task.get('category', 'unknown')}"}
    if any((_initial_verification(row) or {}).get("hard_verification_pass") is False for row in rows.values()):
        labels.add("initial_hard_failure")
    repair_row = rows["evidence_verify_repair"]
    if repair_row.grade.get("repair_attempted") and not repair_row.grade.get("repair_succeeded"):
        labels.add("rejected_repair")
    if any(rows[variant].grade.get("citation_binding_pass") and not rows["base"].grade.get("citation_binding_pass")
           for variant in VARIANTS[1:]):
        labels.add("citation_improvement")
    if any(rows[variant].grade.get("handoff_observed") and not rows["base"].grade.get("handoff_observed")
           for variant in VARIANTS[1:]):
        labels.add("extra_handoff")
    if any(row.grade.get("failure_type") == "missing-required-tool" for row in rows.values()):
        labels.add("missing_required_tool")
    failures = [row.grade.get("failure_type") or "ok" for row in rows.values()]
    if len(set(failures)) == 1 and failures[0] != "ok":
        labels.add(f"shared_failure:{failures[0]}")
    if all(row.grade.get("joint_success") for row in rows.values()):
        labels.add("clean_control")
    return labels


def select_audit_tasks(
    by_variant: dict[str, list[StoredTrajectory]], tasks: dict[str, dict], target: int = 32
) -> tuple[list[str], dict]:
    indexed = {variant: {row.task_id: row for row in rows} for variant, rows in by_variant.items()}
    mandatory, components = mandatory_task_ids(by_variant)
    if len(mandatory) > target:
        raise PostprocessError(
            f"mandatory audit union has {len(mandatory)} tasks, exceeding fixed target {target}; refusing to truncate"
        )
    selected = set(mandatory)
    labels = {task_id: _sampling_labels(task_id, indexed, tasks[task_id]) for task_id in tasks}
    covered = set().union(*(labels[task_id] for task_id in selected)) if selected else set()
    universe = set().union(*labels.values()) if labels else set()
    while len(selected) < target:
        candidates = sorted(set(tasks) - selected)
        if not candidates:
            raise PostprocessError(f"only {len(selected)} eligible task groups exist; expected {target}")
        choice = min(candidates, key=lambda task_id: (-len(labels[task_id] - covered), task_id))
        selected.add(choice)
        covered.update(labels[choice])
    ordered = sorted(selected)
    return ordered, {
        "target_groups": target,
        "mandatory_count": len(mandatory),
        "mandatory_components": components,
        "mandatory_union": sorted(mandatory),
        "greedy_added": sorted(selected - mandatory),
        "covered_strata": sorted(covered),
        "uncovered_strata": sorted(universe - covered),
        "selection_reasons": {task_id: sorted(labels[task_id]) for task_id in ordered},
    }


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).casefold()


def answer_id(task_id: str, variant: str) -> str:
    return "A" + hashlib.sha256(f"{RUN_NAME}|{task_id}|{variant}".encode()).hexdigest()[:20]


def _claim_id(answer: str, source_span: str, claim_text: str, fact_key: str) -> str:
    material = "|".join((answer, source_span, _normalized(claim_text), fact_key))
    return "C" + hashlib.sha256(material.encode()).hexdigest()[:24]


def _atomic_claims(text: str) -> list[dict]:
    """Stable, conservative splitter with offsets and separate structured facts."""
    claims: list[dict] = []
    cursor = 0
    for match in _SENTENCE_BOUNDARY.finditer(text):
        end = match.end()
        segment = text[cursor:end].strip()
        raw_start = cursor
        cursor = end
        if segment:
            claims.extend(_claims_from_segment(segment, raw_start, end))
    tail = text[cursor:].strip()
    if tail:
        claims.extend(_claims_from_segment(tail, cursor, len(text)))
    if not claims:
        claims.append({"claim_text": text.strip(), "source_span": f"0:{len(text)}", "fact_key": "prose"})
    return claims


def _claims_from_segment(segment: str, start: int, end: int) -> list[dict]:
    structured: list[tuple[str, str]] = []
    date_spans: list[tuple[int, int]] = []
    date_patterns = (
        r"\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日",
        r"\d{4}-\d{1,2}-\d{1,2}",
        r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
        r"Dec(?:ember)?)\s+\d{1,2},?\s+\d{4}",
    )
    for pattern in date_patterns:
        for match in re.finditer(pattern, segment, re.I):
            structured.append(("date", match.group(0)))
            date_spans.append(match.span())
    for product_id in _PRODUCT_ID_RE.findall(segment):
        structured.append(("product_id", product_id))
    masked = list(segment)
    for span_start, span_end in date_spans:
        masked[span_start:span_end] = " " * (span_end - span_start)
    for number in re.findall(r"(?<![A-Za-z0-9])\d+(?:\.\d+)?\s*(?:元|天|日|件|个|kg|g|cm|mm|%)", "".join(masked), re.I):
        structured.append(("numeric_fact", number.strip()))
    rows = []
    seen = set()
    for fact_key, value in structured:
        key = (fact_key, _normalized(value))
        if key in seen:
            continue
        seen.add(key)
        rows.append({"claim_text": value, "source_span": f"{start}:{end}", "fact_key": fact_key})
    # Keep the prose claim even when structured facts were extracted so reviewers can catch
    # semantic claims the deterministic splitter cannot enumerate.
    rows.append({"claim_text": segment, "source_span": f"{start}:{end}", "fact_key": "prose"})
    return rows


def _auto_claim_status(claim: str, row: StoredTrajectory) -> tuple[str, str]:
    normalized = _normalized(claim)
    contradicted = json.dumps(row.grade.get("contradicted_claims") or [], ensure_ascii=False)
    unsupported = json.dumps(row.grade.get("unsupported_high_risk_claims") or [], ensure_ascii=False)
    omitted = json.dumps(row.grade.get("omitted_required_facts") or [], ensure_ascii=False)
    if normalized and normalized in _normalized(contradicted):
        fact = "contradicted"
    elif normalized and normalized in _normalized(unsupported):
        fact = "unsupported"
    elif normalized and normalized in _normalized(omitted):
        fact = "required_coverage_missing"
    else:
        fact = "unclassified"
    citations = _CITATION_RE.findall(claim)
    diagnostic = json.dumps(row.grade.get("citation_diagnostics") or [], ensure_ascii=False)
    if not citations:
        citation = "missing_or_not_required"
    elif any(value in diagnostic for value in citations):
        citation = "diagnostic_failure"
    else:
        citation = "no_automatic_failure"
    return fact, citation


def _csv_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _immutable_row_hash(row: dict, mutable_fields: set[str]) -> str:
    fixed = {key: str(value) for key, value in row.items() if key not in mutable_fields}
    return hashlib.sha256(_csv_json(fixed).encode("utf-8")).hexdigest()


def build_audit_rows(
    selected: list[str], by_variant: dict[str, list[StoredTrajectory]], tasks: dict[str, dict], selection: dict
) -> tuple[list[dict], list[dict]]:
    indexed = {variant: {row.task_id: row for row in rows} for variant, rows in by_variant.items()}
    answer_rows: list[dict] = []
    claim_rows: list[dict] = []
    for task_id in selected:
        task = tasks[task_id]
        for variant in VARIANTS:
            row = indexed[variant][task_id]
            aid = answer_id(task_id, variant)
            answer = str(row.trajectory.get("final_answer") or "")
            answer_rows.append({
                "answer_id": aid,
                "task_id": task_id,
                "variant": variant,
                "category": task.get("category", ""),
                "sample_reasons": "|".join(selection["selection_reasons"][task_id]),
                "user_prompt": task.get("user_goal") or task.get("user_prompt") or task.get("prompt") or task.get("question") or "",
                "final_answer": answer,
                "handoff_expected": str(bool(row.grade.get("handoff_expected"))).lower(),
                "handoff_observed": str(bool(row.grade.get("handoff_observed"))).lower(),
                "operational_success": str(bool(row.grade.get("operational_success"))).lower(),
                "auto_initial_hard_pass": (
                    "" if _initial_verification(row) is None
                    else str(bool(_initial_verification(row).get("hard_verification_pass"))).lower()
                ),
                "auto_final_hard_pass": str(bool(row.grade.get("hard_verification_pass"))).lower(),
                "auto_citation_binding_pass": str(bool(row.grade.get("citation_binding_pass"))).lower(),
                "auto_failure_type": row.grade.get("failure_type", ""),
                "evidence_ledger_json": _csv_json(row.trajectory.get("evidence_ledger") or []),
                "human_answer_complete": "",
                "human_handoff_appropriate": "",
                "human_overall_pass": "",
                "human_claim_segmentation_complete": "",
                "missing_claim_notes": "",
                "review_notes": "",
                "derived_all_factual_claims_supported": "",
                "derived_citation_binding_pass": "",
                "derived_answer_fact_pass": "",
                "derived_overall_pass": "",
            })
            for claim in _atomic_claims(answer):
                fact_auto, citation_auto = _auto_claim_status(claim["claim_text"], row)
                cited = _CITATION_RE.findall(claim["claim_text"])
                ledger = [item for item in row.trajectory.get("evidence_ledger") or []
                          if item.get("evidence_id") in cited]
                claim_rows.append({
                    "claim_id": _claim_id(aid, claim["source_span"], claim["claim_text"], claim["fact_key"]),
                    "answer_id": aid,
                    "task_id": task_id,
                    "variant": variant,
                    "source_span": claim["source_span"],
                    "fact_key": claim["fact_key"],
                    "claim_text": claim["claim_text"],
                    "cited_evidence_ids": "|".join(cited),
                    "cited_evidence_json": _csv_json(ledger),
                    "auto_fact_status": fact_auto,
                    "auto_citation_status": citation_auto,
                    "human_fact_status": "",
                    "human_citation_status": "",
                    "review_notes": "",
                })
    return answer_rows, claim_rows


def _write_json(path: Path, value: Any) -> None:
    if path.exists():
        raise PostprocessError(f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if path.exists():
        raise PostprocessError(f"refusing to overwrite: {path}")
    if not rows:
        raise PostprocessError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _input_record(path: Path) -> dict:
    return file_record(path)


def run_analysis(args: argparse.Namespace) -> dict:
    outputs = {
        "integrity": args.output_dir / f"{RUN_NAME}_integrity.json",
        "aggregate": args.output_dir / f"{RUN_NAME}_aggregate.json",
        "attribution": args.output_dir / f"{RUN_NAME}_failure_attribution.json",
        "answers": args.output_dir / "audit_32_answers.csv",
        "claims": args.output_dir / "audit_32_claims.csv",
        "audit_manifest": args.output_dir / "audit_32_manifest.json",
    }
    for key, path in outputs.items():
        if path.exists():
            raise PostprocessError(f"refusing to overwrite {key}: {path}")

    manifest = _json(args.manifest)
    manifest_contract = manifest.get("task_contract", manifest.get("contract"))
    if manifest.get("run_name") != RUN_NAME or manifest_contract != CONTRACT:
        raise PostprocessError(
            f"manifest must declare run_name={RUN_NAME!r} and contract={CONTRACT!r}"
        )
    task_rows = _jsonl(args.tasks)
    tasks = {row["task_id"]: row for row in task_rows}
    if len(tasks) != len(task_rows):
        raise PostprocessError("task contract contains duplicate task IDs")
    expected_ids = list(manifest.get("task_ids") or [])
    unknown_manifest_ids = set(expected_ids) - set(tasks)
    if unknown_manifest_ids:
        raise PostprocessError(f"manifest contains IDs absent from task contract: {sorted(unknown_manifest_ids)}")
    # The versioned contract also contains smoke/final candidates.  This run is the
    # exact 80-ID dev subset declared by the frozen run manifest.
    tasks = {task_id: tasks[task_id] for task_id in expected_ids}

    reports = {}
    report_records = []
    for variant in VARIANTS:
        path = args.report_dir / f"{RUN_NAME}_{variant}_report.json"
        reports[variant] = _json(path)
        report_records.append(_input_record(path))

    provenance = {
        "run_code_commit": args.run_code_commit,
        "result_commit": args.result_commit,
        "postprocess_code_commit": args.postprocess_code_commit,
    }
    if len(set(provenance.values())) < 2:
        raise PostprocessError("run/result/postprocess provenance must be explicit; do not conflate all commits")

    source_inventory: dict[str, list[dict]] = {}
    by_variant: dict[str, list[StoredTrajectory]] = {}
    with tempfile.TemporaryDirectory(prefix="evidence-dev-v3-snapshot-") as temp:
        snapshot_root = Path(temp)
        for variant in VARIANTS:
            source = args.stores[variant]
            snapshot, inventory = snapshot_sqlite(source, snapshot_root / variant)
            source_inventory[variant] = inventory
            by_variant[variant] = load_store(snapshot, variant)

        checks, integrity_details = validate_rows(by_variant, expected_ids)
        for variant in VARIANTS:
            report_ids = reports[variant].get("task_ids") or [row.get("task_id") for row in reports[variant].get("details", [])]
            passed = len(report_ids) == 80 and len(set(report_ids)) == 80 and set(report_ids) == set(expected_ids)
            checks.append({
                "name": f"{variant}:report_ids_match_manifest",
                "passed": passed,
                "detail": {"rows": len(report_ids), "unique": len(set(report_ids))},
            })
        if args.pbs_log:
            checks.append({"name": "pbs_log_exists", "passed": args.pbs_log.exists(), "detail": str(args.pbs_log)})
        for variant in VARIANTS:
            after = inventory_sqlite(args.stores[variant])
            try:
                ensure_inventory_unchanged(source_inventory[variant], after, variant)
            except PostprocessError as exc:
                checks.append({"name": f"{variant}:source_unchanged", "passed": False, "detail": str(exc)})
            else:
                checks.append({"name": f"{variant}:source_unchanged", "passed": True, "detail": "hash/file set unchanged"})

        integrity = {
            "schema_version": 1,
            "run_name": RUN_NAME,
            "contract": CONTRACT,
            "passed": all(check["passed"] for check in checks),
            "checks": checks,
            "details": integrity_details,
            "provenance": provenance,
            "source_sqlite_files": source_inventory,
            "source_reports": report_records,
            "manifest": _input_record(args.manifest),
            "tasks": _input_record(args.tasks),
            "pbs_log": _input_record(args.pbs_log) if args.pbs_log and args.pbs_log.exists() else None,
            "read_mode": {"snapshot": True, "sqlite_uri_mode": "ro", "pragma_query_only": True,
                          "checkpoint_performed": False},
        }
        _write_json(outputs["integrity"], integrity)
        if not integrity["passed"]:
            raise PostprocessError(f"integrity gate failed; see {outputs['integrity']}")

        metrics = {variant: variant_metrics(by_variant[variant]) for variant in VARIANTS}
        aggregate = {
            "schema_version": 3,
            "run_name": RUN_NAME,
            "contract": CONTRACT,
            "provenance": provenance,
            "metrics": metrics,
            "paired_cost_deltas_vs_base": paired_cost_deltas(by_variant),
            "selective_risk_note": (
                "Final accepted-answer hard pass can rise because hard failures are handed off. "
                "Interpret it together with initial pass, answer coverage, handoff rate, and joint success."
            ),
            "final_ready": False,
            "human_review_status": "pending",
            "dpo_pairs_generated": 0,
        }
        attribution = failure_attribution(by_variant, tasks)
        selected, selection = select_audit_tasks(by_variant, tasks, target=32)
        answer_rows, claim_rows = build_audit_rows(selected, by_variant, tasks, selection)
        if len(selected) != 32 or len(answer_rows) != 96 or len({row["answer_id"] for row in answer_rows}) != 96:
            raise PostprocessError("audit cardinality must be exactly 32 groups / 96 answer IDs")
        if any(tasks[task_id].get("split") == "final" for task_id in selected):
            raise PostprocessError("final split task selected for calibration")

        _write_json(outputs["aggregate"], aggregate)
        _write_json(outputs["attribution"], attribution)
        _write_csv(outputs["answers"], answer_rows)
        _write_csv(outputs["claims"], claim_rows)
        audit_manifest = {
            "schema_version": 1,
            "run_name": RUN_NAME,
            "contract": CONTRACT,
            "provenance": provenance,
            "selection": selection,
            "task_groups": selected,
            "task_group_count": len(selected),
            "answer_ids": [row["answer_id"] for row in answer_rows],
            "answer_count": len(answer_rows),
            "claim_ids": [row["claim_id"] for row in claim_rows],
            "claim_count": len(claim_rows),
            "answer_fields": list(answer_rows[0]),
            "claim_fields": list(claim_rows[0]),
            "answer_immutable_hashes": {
                row["answer_id"]: _immutable_row_hash(row, ANSWER_HUMAN_FIELDS) for row in answer_rows
            },
            "claim_immutable_hashes": {
                row["claim_id"]: _immutable_row_hash(row, CLAIM_HUMAN_FIELDS) for row in claim_rows
            },
            "source_sqlite_files": source_inventory,
            "source_reports": report_records,
            "answer_template_sha256": _sha256(outputs["answers"]),
            "claim_template_sha256": _sha256(outputs["claims"]),
            "warning": "This paired sample calibrates the verifier. Its rates must not be extrapolated to all 80 tasks.",
            "dpo_use_prohibited": True,
        }
        _write_json(outputs["audit_manifest"], audit_manifest)
        for variant in VARIANTS:
            ensure_inventory_unchanged(source_inventory[variant], inventory_sqlite(args.stores[variant]), variant)
    return {"outputs": {key: str(path) for key, path in outputs.items()}, "integrity_passed": True}


def _read_csv(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _tri_and(values: Iterable[str]) -> str:
    rows = list(values)
    if any(value == "false" for value in rows):
        return "false"
    if rows and all(value == "true" for value in rows):
        return "true"
    return "unclear"


def summarize_human_audit(answers_path: Path, claims_path: Path, manifest_path: Path) -> tuple[dict, list[dict]]:
    answer_fields, answers = _read_csv(answers_path)
    claim_fields, claims = _read_csv(claims_path)
    manifest = _json(manifest_path)
    expected_answers = set(manifest.get("answer_ids") or [])
    observed_answers = [row.get("answer_id", "") for row in answers]
    if len(answers) != 96 or set(observed_answers) != expected_answers or len(set(observed_answers)) != 96:
        raise PostprocessError("answer audit rows/IDs differ from the audit manifest")
    if len({row.get("claim_id") for row in claims}) != len(claims):
        raise PostprocessError("claim audit has duplicate claim IDs")
    expected_claims = set(manifest.get("claim_ids") or [])
    if expected_claims and ({row.get("claim_id") for row in claims} != expected_claims
                            or len(claims) != len(expected_claims)):
        raise PostprocessError("claim audit rows/IDs differ from the audit manifest")
    if set(row.get("answer_id") for row in claims) - expected_answers:
        raise PostprocessError("claim audit contains unknown answer IDs")
    if answer_fields != manifest.get("answer_fields"):
        raise PostprocessError("answer audit header differs from the audit manifest")
    if claim_fields != manifest.get("claim_fields"):
        raise PostprocessError("claim audit header differs from the audit manifest")
    answer_hashes = manifest.get("answer_immutable_hashes") or {}
    claim_hashes = manifest.get("claim_immutable_hashes") or {}
    if set(answer_hashes) != expected_answers or any(
        _immutable_row_hash(row, ANSWER_HUMAN_FIELDS) != answer_hashes.get(row["answer_id"])
        for row in answers
    ):
        raise PostprocessError("an immutable answer-audit field was changed")
    if set(claim_hashes) != expected_claims or any(
        _immutable_row_hash(row, CLAIM_HUMAN_FIELDS) != claim_hashes.get(row["claim_id"])
        for row in claims
    ):
        raise PostprocessError("an immutable claim-audit field was changed")
    for row in answers:
        for field in ("human_answer_complete", "human_handoff_appropriate", "human_overall_pass",
                      "human_claim_segmentation_complete"):
            value = row.get(field, "").strip().lower()
            if value not in BOOL_OR_BLANK:
                raise PostprocessError(f"{row['answer_id']}: {field} must be true, false, or blank")
            row[field] = value
    for row in claims:
        fact = row.get("human_fact_status", "").strip().lower()
        citation = row.get("human_citation_status", "").strip().lower()
        if fact and fact not in FACT_STATUS:
            raise PostprocessError(f"{row['claim_id']}: invalid human_fact_status {fact!r}")
        if citation and citation not in CITATION_STATUS:
            raise PostprocessError(f"{row['claim_id']}: invalid human_citation_status {citation!r}")
        row["human_fact_status"] = fact
        row["human_citation_status"] = citation

    claims_by_answer: dict[str, list[dict]] = defaultdict(list)
    for row in claims:
        claims_by_answer[row["answer_id"]].append(row)
    review_queue = []
    derived_rows = []
    for answer in answers:
        rows = claims_by_answer[answer["answer_id"]]
        factual = [row for row in rows if row["human_fact_status"] != "not_factual"]
        if any(not row["human_fact_status"] or row["human_fact_status"] == "unclear" for row in factual):
            facts = "unclear"
        elif any(row["human_fact_status"] in {"contradicted", "unsupported"} for row in factual):
            facts = "false"
        else:
            facts = "true"
        citation_values = []
        for row in rows:
            status = row["human_citation_status"]
            if status in {"correct", "not_required"}:
                citation_values.append("true")
            elif status in {"incorrect", "missing"}:
                citation_values.append("false")
            else:
                citation_values.append("unclear")
        citation = _tri_and(citation_values)
        answer_fact = _tri_and((facts, answer["human_answer_complete"] or "unclear"))
        overall = _tri_and((answer["operational_success"], answer_fact,
                            answer["human_handoff_appropriate"] or "unclear"))
        derived = {
            **answer,
            "derived_all_factual_claims_supported": facts,
            "derived_citation_binding_pass": citation,
            "derived_answer_fact_pass": answer_fact,
            "derived_overall_pass": overall,
        }
        derived_rows.append(derived)
        human = answer.get("human_overall_pass", "")
        if human and overall != "unclear" and human != overall:
            review_queue.append({
                "answer_id": answer["answer_id"], "task_id": answer["task_id"],
                "variant": answer["variant"], "human_overall_pass": human,
                "derived_overall_pass": overall, "reason": "manual/derived disagreement",
            })
        if answer["human_claim_segmentation_complete"] != "true":
            review_queue.append({
                "answer_id": answer["answer_id"], "task_id": answer["task_id"],
                "variant": answer["variant"], "human_overall_pass": human,
                "derived_overall_pass": overall, "reason": "claim segmentation incomplete or unreviewed",
            })

    eligible_answers = {
        row["answer_id"] for row in answers if row["human_claim_segmentation_complete"] == "true"
    }
    fact_rows = [row for row in claims if row["answer_id"] in eligible_answers and row["human_fact_status"]]
    citation_rows = [row for row in claims if row["answer_id"] in eligible_answers and row["human_citation_status"]]
    fact_unclear = [row for row in fact_rows if row["human_fact_status"] == "unclear"]
    citation_unclear = [row for row in citation_rows if row["human_citation_status"] == "unclear"]
    fact_effective = [row for row in fact_rows if row["human_fact_status"] != "unclear"]
    citation_effective = [row for row in citation_rows if row["human_citation_status"] != "unclear"]
    fact_matrix = Counter((row.get("auto_fact_status", ""), row["human_fact_status"]) for row in fact_effective)
    citation_matrix = Counter((row.get("auto_citation_status", ""), row["human_citation_status"])
                              for row in citation_effective)
    report = {
        "schema_version": 1,
        "run_name": RUN_NAME,
        "review_scope": {"task_groups": 32, "answers": 96, "claims": len(claims)},
        "human_review_status": "complete" if all(
            row["human_claim_segmentation_complete"] in {"true", "false"} and
            row["human_answer_complete"] in {"true", "false"} and
            row["human_handoff_appropriate"] in {"true", "false"} and
            row["human_overall_pass"] in {"true", "false"} for row in answers
        ) and all(row["human_fact_status"] and row["human_citation_status"] for row in claims) else "pending",
        "claim_fact_calibration": {
            "reviewed": len(fact_rows), "effective_sample_count": len(fact_effective),
            "unclear_rate": rate(len(fact_unclear), len(fact_rows)),
            "confusion_matrix": {f"auto={a}|human={h}": count for (a, h), count in sorted(fact_matrix.items())},
        },
        "claim_citation_calibration": {
            "reviewed": len(citation_rows), "effective_sample_count": len(citation_effective),
            "unclear_rate": rate(len(citation_unclear), len(citation_rows)),
            "confusion_matrix": {f"auto={a}|human={h}": count for (a, h), count in sorted(citation_matrix.items())},
        },
        "answer_level": {
            "derived_fact_pass": rate(sum(row["derived_answer_fact_pass"] == "true" for row in derived_rows), len(derived_rows)),
            "derived_overall_pass": rate(sum(row["derived_overall_pass"] == "true" for row in derived_rows), len(derived_rows)),
            "review_queue_count": len(review_queue),
        },
        "final_ready": False,
        "dpo_pairs_generated": 0,
        "warning": "Calibration sample rates must not be extrapolated to all 80 tasks or used to construct DPO pairs.",
    }
    return report, review_queue, derived_rows


def run_summarize(args: argparse.Namespace) -> dict:
    outputs = {
        "report": args.output_dir / f"{RUN_NAME}_human_calibration.json",
        "review_queue": args.output_dir / f"{RUN_NAME}_review_queue.csv",
        "derived_answers": args.output_dir / f"{RUN_NAME}_audit_32_answers_derived.csv",
    }
    for path in outputs.values():
        if path.exists():
            raise PostprocessError(f"refusing to overwrite: {path}")
    report, queue, derived = summarize_human_audit(args.answers, args.claims, args.audit_manifest)
    _write_json(outputs["report"], report)
    _write_csv(outputs["derived_answers"], derived)
    if queue:
        _write_csv(outputs["review_queue"], queue)
    else:
        # Preserve a versioned empty queue without violating _write_csv's safety guard.
        outputs["review_queue"].write_text(
            "answer_id,task_id,variant,human_overall_pass,derived_overall_pass,reason\n", encoding="utf-8"
        )
    return {"outputs": {key: str(path) for key, path in outputs.items()}, "status": report["human_review_status"]}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    analyze = sub.add_parser("analyze")
    analyze.add_argument("--manifest", type=Path, required=True)
    analyze.add_argument("--tasks", type=Path, required=True)
    analyze.add_argument("--report-dir", type=Path, required=True)
    analyze.add_argument("--base-store", type=Path, required=True)
    analyze.add_argument("--evidence-verify-store", type=Path, required=True)
    analyze.add_argument("--evidence-verify-repair-store", type=Path, required=True)
    analyze.add_argument("--pbs-log", type=Path)
    analyze.add_argument("--run-code-commit", required=True)
    analyze.add_argument("--result-commit", default=RESULT_COMMIT)
    analyze.add_argument("--postprocess-code-commit", required=True)
    analyze.add_argument("--output-dir", type=Path, required=True)
    summarize = sub.add_parser("summarize-audit")
    summarize.add_argument("--answers", type=Path, required=True)
    summarize.add_argument("--claims", type=Path, required=True)
    summarize.add_argument("--audit-manifest", type=Path, required=True)
    summarize.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        if args.command == "analyze":
            args.stores = {
                "base": args.base_store,
                "evidence_verify": args.evidence_verify_store,
                "evidence_verify_repair": args.evidence_verify_repair_store,
            }
            result = run_analysis(args)
        else:
            result = run_summarize(args)
    except PostprocessError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False, indent=2))
        raise SystemExit(1) from exc
    print(json.dumps({"ok": True, **result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
