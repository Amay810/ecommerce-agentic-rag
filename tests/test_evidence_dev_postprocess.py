from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

import pytest

from scripts.evidence_dev_postprocess import (
    PostprocessError,
    StoredTrajectory,
    _atomic_claims,
    _claim_id,
    _inventory_signature,
    build_audit_rows,
    conversion_errors,
    ensure_inventory_unchanged,
    mandatory_task_ids,
    open_readonly,
    rate,
    select_audit_tasks,
    summarize_human_audit,
    validate_rows,
)


VARIANTS = ("base", "evidence_verify", "evidence_verify_repair")


def _stored(task_id: str, variant: str, *, operational: bool = True, repair: bool = False,
            trajectory: dict | None = None, grade: dict | None = None) -> StoredTrajectory:
    base_grade = {
        "task_id": task_id,
        "operational_success": operational,
        "joint_success": operational,
        "repair_attempted": repair,
        "repair_succeeded": repair,
        "failure_type": None if operational else "wrong-tool",
        "handoff_expected": False,
        "handoff_observed": False,
        "hard_verification_pass": True,
        "citation_binding_pass": variant != "base",
    }
    base_grade.update(grade or {})
    body = {
        "trajectory_id": f"tr-{variant}-{task_id}",
        "task_id": task_id,
        "final_answer": "商品 P00001 价格为 400 元。[E1]",
        "tool_calls": [],
        "evidence_ledger": [],
        "evidence_conversion_spans": [],
        "verification_spans": [],
        "repair_spans": [],
    }
    body.update(trajectory or {})
    return StoredTrajectory(variant, body["trajectory_id"], task_id, 1, body, base_grade)


def _integrity_stored(task_id: str, variant: str) -> StoredTrajectory:
    trajectory = {
        "trajectory_id": f"tr-{variant}-{task_id}", "task_id": task_id,
        "messages": [], "model_calls": [{"llm": {"resolution": "parsed", "attempts": []}}],
        "tool_calls": [], "final_answer": "ok", "evidence_ledger": [],
        "evidence_conversion_spans": [], "verification_spans": [], "repair_spans": [],
    }
    grade = {
        "task_id": task_id, "operational_success": True, "policy_compliant": True,
        "terminal_state_match": True, "joint_success": True, "failure_type": None,
        "handoff_expected": False, "handoff_observed": False, "answer_fact_applicable": False,
        "hard_verification_pass": True, "answer_fact_pass": True, "citation_binding_pass": False,
        "required_evidence_coverage": None, "repair_attempted": False, "repair_succeeded": False,
        "repair_hard_recovery": False, "repair_diagnostic_improvement": False,
        "unsupported_high_risk_claims": [], "contradicted_claims": [],
        "omitted_required_facts": [], "citation_diagnostics": [], "latency_ms": 1,
    }
    return StoredTrajectory(variant, trajectory["trajectory_id"], task_id, 1, trajectory, grade)


def _valid_conversion() -> dict:
    return {
        "tool_calls": [{
            "call_id": "call-1", "name": "get_product", "arguments": {"product_id": "P00001"},
            "result": {"ok": True, "product": {"product_id": "P00001"}},
        }],
        "evidence_conversion_spans": [{
            "tool_call_id": "call-1", "tool_name": "get_product", "status": "converted",
            "evidence_ids": ["E1"], "source_item_count": 1, "evidence_item_count": 1,
        }],
        "evidence_ledger": [{
            "evidence_id": "E1", "tool_call_id": "call-1", "tool_name": "get_product",
            "field": "product.product_id", "value": "P00001",
        }],
    }


def test_conversion_allows_one_call_one_span_many_evidence() -> None:
    trajectory = _valid_conversion()
    trajectory["evidence_ledger"].append({
        "evidence_id": "E2", "tool_call_id": "call-1", "tool_name": "get_product",
        "field": "product.price", "value": 400,
    })
    trajectory["evidence_conversion_spans"][0].update(
        evidence_ids=["E1", "E2"], evidence_item_count=2,
    )
    assert conversion_errors(trajectory) == []


@pytest.mark.parametrize("mutation, expected", [
    (lambda t: t["evidence_conversion_spans"].append(dict(t["evidence_conversion_spans"][0])),
     "expected exactly one conversion span"),
    (lambda t: t["evidence_ledger"][0].update(tool_call_id="missing"), "orphan evidence owner"),
    (lambda t: t["evidence_conversion_spans"][0].update(evidence_item_count=2),
     "declared evidence count differs"),
    (lambda t: t["evidence_conversion_spans"][0].update(status="converter_missing"), "converter_missing"),
])
def test_conversion_corruption_fails_closed(mutation, expected: str) -> None:
    trajectory = _valid_conversion()
    mutation(trajectory)
    assert any(expected in error for error in conversion_errors(trajectory))


def test_successful_nonempty_read_requires_evidence() -> None:
    trajectory = _valid_conversion()
    trajectory["evidence_ledger"] = []
    trajectory["evidence_conversion_spans"][0].update(evidence_ids=[], evidence_item_count=0)
    assert any("successful nonempty read produced no evidence" in error for error in conversion_errors(trajectory))


def test_successful_write_can_legitimately_produce_zero_evidence() -> None:
    trajectory = {
        "tool_calls": [{
            "call_id": "write-1", "name": "create_return_request", "arguments": {},
            "result": {"ok": True},
        }],
        "evidence_conversion_spans": [{
            "tool_call_id": "write-1", "tool_name": "create_return_request", "status": "valid_empty",
            "evidence_ids": [], "source_item_count": 0, "evidence_item_count": 0,
        }],
        "evidence_ledger": [],
    }
    assert conversion_errors(trajectory) == []


def test_readonly_sqlite_cannot_be_mutated(tmp_path: Path) -> None:
    path = tmp_path / "store.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE sample(value INTEGER)")
    conn.commit()
    conn.close()
    before = path.read_bytes()
    readonly = open_readonly(path)
    with pytest.raises(sqlite3.OperationalError):
        readonly.execute("INSERT INTO sample VALUES(1)")
    readonly.close()
    assert path.read_bytes() == before


def test_inventory_change_is_fail_closed() -> None:
    before = [{"name": "a.sqlite", "size": 1, "mtime_ns": 1, "sha256": "a"}]
    after = [{"name": "a.sqlite", "size": 2, "mtime_ns": 2, "sha256": "b"}]
    with pytest.raises(PostprocessError, match="changed"):
        ensure_inventory_unchanged(before, after, "fixture")
    assert _inventory_signature(before) != _inventory_signature(after)


def test_every_rate_has_explicit_numerator_and_denominator() -> None:
    assert rate(56, 61) == {"numerator": 56, "denominator": 61, "rate": 56 / 61}
    assert rate(0, 0) == {"numerator": 0, "denominator": 0, "rate": None}


def test_integrity_detects_ids_generation_fallback_and_repeated_repair() -> None:
    expected = [f"task-{index:02d}" for index in range(80)]
    by_variant = {variant: [_integrity_stored(task_id, variant) for task_id in expected]
                  for variant in VARIANTS}
    damaged = by_variant["evidence_verify"][0]
    damaged.trajectory["model_calls"][0]["llm"].update(
        resolution="fallback_handoff",
        attempts=[{"parse_stage": "generation_error"}],
    )
    damaged.trajectory["repair_spans"] = [{}, {}]
    # A cross-variant trajectory collision is distinct from task pairing.
    base_first = by_variant["base"][0]
    repair_first = by_variant["evidence_verify_repair"][0]
    by_variant["evidence_verify_repair"][0] = StoredTrajectory(
        repair_first.variant, base_first.trajectory_id, repair_first.task_id, repair_first.seed,
        repair_first.trajectory, repair_first.grade,
    )
    checks, _ = validate_rows(by_variant, expected)
    failed = {row["name"] for row in checks if not row["passed"]}
    assert "evidence_verify:generation_errors_zero" in failed
    assert "evidence_verify:fallback_only_zero" in failed
    assert "evidence_verify:repair_at_most_once" in failed
    assert "trajectory_ids_unique_across_variants" in failed


def test_integrity_detects_missing_and_duplicate_task_ids() -> None:
    expected = [f"task-{index:02d}" for index in range(80)]
    by_variant = {variant: [_integrity_stored(task_id, variant) for task_id in expected]
                  for variant in VARIANTS}
    by_variant["base"][-1] = _integrity_stored(expected[0], "base-duplicate")
    checks, _ = validate_rows(by_variant, expected)
    failed = {row["name"] for row in checks if not row["passed"]}
    assert "base:80_rows_80_unique_tasks" in failed
    assert "base:task_ids_match_manifest" in failed


def test_current_report_fixture_has_mandatory_union_23() -> None:
    root = Path(__file__).parents[1]
    by_variant = {}
    for variant in VARIANTS:
        report = json.loads(
            (root / "docs" / f"evidence_phase_a_dev_v3_{variant}_report.json").read_text(encoding="utf-8")
        )
        by_variant[variant] = [
            _stored(row["task_id"], variant,
                    operational=bool(row.get("operational_success")),
                    repair=bool(row.get("repair_succeeded")), grade=row)
            for row in report["details"]
        ]
    mandatory, components = mandatory_task_ids(by_variant)
    assert len(components["operational_disagreement"]) == 13
    assert len(components["adopted_repair"]) == 14
    assert len(mandatory) == 23


def test_mandatory_over_32_refuses_to_truncate() -> None:
    tasks = {f"task-{index:02d}": {"task_id": f"task-{index:02d}", "category": "x", "split": "dev"}
             for index in range(33)}
    by_variant = {
        "base": [_stored(task_id, "base", operational=True) for task_id in tasks],
        "evidence_verify": [_stored(task_id, "evidence_verify", operational=False) for task_id in tasks],
        "evidence_verify_repair": [_stored(task_id, "evidence_verify_repair", operational=False) for task_id in tasks],
    }
    with pytest.raises(PostprocessError, match="refusing to truncate"):
        select_audit_tasks(by_variant, tasks)


def test_audit_selection_is_32_paired_groups_and_96_answers() -> None:
    tasks = {f"task-{index:02d}": {"task_id": f"task-{index:02d}", "category": f"c{index % 4}", "split": "dev",
                                      "user_goal": f"question {index}"}
             for index in range(40)}
    by_variant = {
        variant: [_stored(task_id, variant) for task_id in tasks] for variant in VARIANTS
    }
    selected, selection = select_audit_tasks(by_variant, tasks)
    answers, claims = build_audit_rows(selected, by_variant, tasks, selection)
    assert len(selected) == 32
    assert len(answers) == 96
    assert len({row["answer_id"] for row in answers}) == 96
    assert {row["variant"] for row in answers} == set(VARIANTS)
    assert set(row["task_id"] for row in answers) == set(selected)
    assert claims
    assert all(tasks[task_id]["split"] != "final" for task_id in selected)


def test_claim_ids_are_content_stable_and_citations_are_not_claims() -> None:
    first = _claim_id("A1", "0:10", "价格 400 元", "numeric_fact")
    second = _claim_id("A1", "0:10", " 价格   400 元 ", "numeric_fact")
    assert first == second
    claims = _atomic_claims("价格为 400 元。[E1]")
    assert all(not row["fact_key"].startswith("citation:") for row in claims)


def test_claim_splitter_keeps_full_date_and_does_not_emit_day_fragment() -> None:
    claims = _atomic_claims("订单于2020年11月2日送达。")
    assert any(row["fact_key"] == "date" and row["claim_text"] == "2020年11月2日" for row in claims)
    assert not any(row["fact_key"] == "numeric_fact" and row["claim_text"] == "2日" for row in claims)


def _write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_human_summary_excludes_unclear_and_incomplete_segmentation(tmp_path: Path) -> None:
    answers = []
    claims = []
    ids = []
    for index in range(96):
        aid = f"A{index:03d}"
        ids.append(aid)
        answers.append({
            "answer_id": aid, "task_id": f"T{index // 3:02d}", "variant": VARIANTS[index % 3],
            "operational_success": "true", "human_answer_complete": "true",
            "human_handoff_appropriate": "true", "human_overall_pass": "true",
            "human_claim_segmentation_complete": "false" if index == 1 else "true",
        })
        claims.append({
            "claim_id": f"C{index:03d}", "answer_id": aid, "auto_fact_status": "unclassified",
            "auto_citation_status": "no_automatic_failure",
            "human_fact_status": "unclear" if index == 0 else "supported",
            "human_citation_status": "unclear" if index == 0 else "correct",
        })
    answers_path, claims_path, manifest_path = tmp_path / "answers.csv", tmp_path / "claims.csv", tmp_path / "manifest.json"
    _write_rows(answers_path, answers)
    _write_rows(claims_path, claims)
    manifest_path.write_text(json.dumps({"answer_ids": ids}), encoding="utf-8")
    report, queue, derived = summarize_human_audit(answers_path, claims_path, manifest_path)
    assert report["claim_fact_calibration"]["effective_sample_count"] == 94
    assert report["claim_fact_calibration"]["unclear_rate"] == rate(1, 95)
    assert derived[0]["derived_answer_fact_pass"] == "unclear"
    assert any(row["answer_id"] == "A001" for row in queue)
