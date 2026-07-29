"""Freeze Phase-A closeout, verifier challenge data, and answer holdout IDs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


HOLDOUT_IDS = [
    "evidence_a_dev_compare_10",
    "evidence_a_dev_order_query_02", "evidence_a_dev_order_query_03",
    "evidence_a_dev_policy_01", "evidence_a_dev_policy_02",
    "evidence_a_dev_product_qa_03", "evidence_a_dev_product_qa_06",
    "evidence_a_dev_recommend_01", "evidence_a_dev_recommend_03",
    "evidence_a_dev_recovery_no_answer_02", "evidence_a_dev_recovery_no_answer_03",
    "evidence_a_dev_return_01", "evidence_a_dev_return_02", "evidence_a_dev_return_03",
    "evidence_a_dev_safety_02", "evidence_a_dev_safety_03",
]
SMOKE_IDS = [
    "evidence_a_dev_compare_01", "evidence_a_dev_compare_05",
    "evidence_a_dev_order_query_01", "evidence_a_dev_policy_03",
    "evidence_a_dev_policy_05", "evidence_a_dev_product_qa_04",
    "evidence_a_dev_product_qa_10", "evidence_a_dev_recommend_02",
    "evidence_a_dev_recommend_10", "evidence_a_dev_recovery_no_answer_01",
    "evidence_a_dev_return_05", "evidence_a_dev_safety_01",
]
FAMILIES = ("numeric_unit", "negation_polarity", "entity_binding", "budget_range",
            "state_date", "citation_binding")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _row(eid: str, source: str, field: str, value: Any, text: str | None = None) -> dict[str, Any]:
    return {"evidence_id": eid, "source_id": source, "tool_call_id": "challenge-call",
            "tool_name": "challenge_fixture", "field": field, "value": value,
            "text": str(value) if text is None else text}


def challenge_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family in FAMILIES:
        for index in range(1, 26):
            target = index <= 15
            product = f"P{index:05d}"
            price = 100 + index
            ledger = [
                _row("E1", f"product:{product}", "product.product_id", product),
                _row("E2", f"product:{product}", "product.price", price),
            ]
            user_messages: list[dict[str, Any]] = []
            citation_required = family == "citation_binding"
            if family == "numeric_unit":
                claim = f"{product} 的价格是 {price + 7 if target else price} 元" + ("。[E2]" if not target else "。")
                fact, citation = ("contradicted", "not_required") if target else ("supported", "correct")
            elif family == "negation_polarity":
                ledger.append(_row("E3", f"product:{product}", "product.discontinued", False, "未停产"))
                claim = f"{product} {'已停产' if target else '未停产'}" + ("。[E3]" if not target else "。")
                fact, citation = ("contradicted", "not_required") if target else ("supported", "correct")
            elif family == "entity_binding":
                other = f"P{index + 100:05d}"
                ledger.extend([
                    _row("E3", f"product:{other}", "product.product_id", other),
                    _row("E4", f"product:{other}", "product.price", price + 50),
                ])
                claim = f"{product} 的价格是 {price + 50 if target else price} 元" + ("。[E4]" if target else "。[E2]")
                fact, citation = ("contradicted", "incorrect") if target else ("supported", "correct")
            elif family == "budget_range":
                budget = price + 20
                user_messages = [{"role": "user", "content": f"预算不超过 {budget} 元"}]
                relation = "超过" if target else "不超过"
                claim = f"{product} 售价 {price} 元，{relation} {budget} 元预算。[E2]"
                fact, citation = ("contradicted", "correct") if target else ("supported", "correct")
            elif family == "state_date":
                order = f"O{index:06d}"
                ledger = [
                    _row("E1", f"order:{order}", "order.order_id", order),
                    _row("E2", f"order:{order}", "order.status", "delivered", "已送达"),
                    _row("E3", f"order:{order}", "order.delivered_at", "2026-07-01"),
                ]
                claim = (f"订单 {order} 已送达，送达日期为 2026-07-0{2 if target else 1}。"
                         f"[E2][E3]")
                fact, citation = ("contradicted", "incorrect") if target else ("supported", "correct")
            else:
                other = f"P{index + 100:05d}"
                ledger.extend([
                    _row("E3", f"product:{other}", "product.product_id", other),
                    _row("E4", f"product:{other}", "product.price", price + 50),
                ])
                claim = f"{product} 的价格是 {price} 元。[{'E4' if target else 'E2'}]"
                fact, citation = "supported", "incorrect" if target else "correct"
            # Reserve five target rows in each factual family for genuine
            # evidence absence.  These are not contradictions: the ledger says
            # nothing about battery capacity, so the only valid label is
            # ``unsupported`` (or an abstaining ``unknown`` prediction).
            if target and index > 10 and family != "citation_binding":
                claim = f"{product} 的电池容量为 {9000 + index} mAh。"
                fact, citation = "unsupported", "not_required"
            case_kind = "target_error" if target else (
                "supported_original" if index <= 18 else
                "supported_paraphrase" if index <= 21 else
                "equivalent_or_distractor_control"
            )
            rows.append({
                "challenge_id": f"vc1_{family}_{index:02d}", "family": family,
                "case_kind": case_kind, "claim_text": claim,
                "evidence_ledger": ledger, "user_messages": user_messages,
                "citation_required": citation_required,
                "gold_fact_status": fact, "gold_citation_status": citation,
                "human_fact_status": fact, "human_citation_status": citation,
                "human_review_status": "assistant_prefilled_pending_user_confirmation",
                "review_notes": "构造规则已逐项核对；需用户确认后才可用于正式准入。",
            })
    return rows


def load_tasks(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--audit-manifest", type=Path, required=True)
    parser.add_argument("--integrity", type=Path, required=True)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection-source-commit", required=True)
    args = parser.parse_args()

    tasks = load_tasks(args.tasks)
    dev = {row["task_id"]: row for row in tasks if row.get("split") == "dev"}
    audit = json.loads(args.audit_manifest.read_text(encoding="utf-8"))
    selected = set(audit["task_groups"])
    if len(dev) != 80 or len(selected) != 32 or selected & set(HOLDOUT_IDS):
        raise ValueError("holdout must be disjoint from the frozen 32-group calibration sample")
    if any(task_id not in dev for task_id in HOLDOUT_IDS):
        raise ValueError("holdout contains an unknown dev task")
    task_hash = sha256(args.tasks)
    holdout = {
        "schema_version": 1, "name": "answer_postprocess_holdout_v1",
        "selection_source_commit": args.selection_source_commit,
        "task_contract": "evidence_phase_a_tasks_v2", "task_contract_sha256": task_hash,
        "task_ids": HOLDOUT_IDS, "task_count": len(HOLDOUT_IDS),
        "selection_basis": "category, template/risk family, then task_id; frozen before new outputs",
        "category_counts": {category: sum(dev[x]["category"] == category for x in HOLDOUT_IDS)
                            for category in sorted({dev[x]["category"] for x in HOLDOUT_IDS})},
        "calibration_task_ids_sha256": stable_hash(sorted(selected)),
        "holdout_use": "evaluation_only_no_tuning",
    }
    if any(task_id not in selected for task_id in SMOKE_IDS):
        raise ValueError("smoke tasks must come only from the 32-group calibration sample")
    smoke = {
        "schema_version": 1, "name": "answer_postprocess_smoke_v1",
        "selection_source_commit": args.selection_source_commit,
        "task_contract_sha256": task_hash, "task_ids": SMOKE_IDS,
        "task_count": len(SMOKE_IDS), "holdout_disjoint": not bool(set(SMOKE_IDS) & set(HOLDOUT_IDS)),
        "selection_basis": "calibration-only coverage of compare/order/policy/product/recommend/no-answer/return/safety",
    }
    rows = challenge_rows()
    challenge_jsonl = args.output_dir / "verifier_challenge_dev_v1.jsonl"
    challenge_jsonl.parent.mkdir(parents=True, exist_ok=True)
    challenge_jsonl.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
                               encoding="utf-8")
    challenge_manifest = {
        "schema_version": 2, "name": "verifier_challenge_dev_v1",
        "dataset_role": "development", "row_count": len(rows),
        "families": {family: sum(row["family"] == family for row in rows) for family in FAMILIES},
        "target_error_per_family": 15, "control_per_family": 10,
        "all_rows_require_user_confirmation": True,
        "jsonl_sha256": sha256(challenge_jsonl),
        "admission_applicable": False,
        "reuse_policy": "development/regression only; permanently ineligible for NSCC admission",
    }
    integrity = json.loads(args.integrity.read_text(encoding="utf-8"))
    aggregate = json.loads(args.aggregate.read_text(encoding="utf-8"))
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    closeout = {
        "schema_version": 1, "run_name": "evidence_phase_a_dev_v3",
        "status": "negative_ablation_archived", "provenance_complete": bool(integrity.get("provenance_complete")),
        "official_archive_ready": True, "final_ready": False, "dpo_pairs_generated": 0,
        "human_review": calibration.get("review_scope"),
        "paired_sample_human_overall_pass": {
            "base": {"numerator": 26, "denominator": 32, "rate": 26 / 32},
            "evidence_verify": {"numerator": 16, "denominator": 32, "rate": 0.5},
            "evidence_verify_repair": {"numerator": 16, "denominator": 32, "rate": 0.5},
        },
        "fixed_findings": [
            "evidence did not improve initial-answer hard pass (56/61 base vs 55/61 evidence)",
            "100% accepted-answer hard pass must be reported with answer coverage and handoff rate",
            "repair attempted 44 times and produced one hard recovery",
            "product specification tasks require search_catalog then get_product for complete evidence",
            "rates from the targeted 32-group review must not be extrapolated to all 80 tasks",
        ],
        "aggregate_sha256": sha256(args.aggregate), "integrity_sha256": sha256(args.integrity),
        "calibration_sha256": sha256(args.calibration),
        "readme_claims_allowed": False,
    }
    write_json(args.output_dir / "answer_postprocess_holdout_v1_manifest.json", holdout)
    write_json(args.output_dir / "answer_postprocess_smoke_v1_manifest.json", smoke)
    write_json(args.output_dir / "verifier_challenge_dev_v1_manifest.json", challenge_manifest)
    write_json(args.output_dir / "evidence_phase_a_dev_v3_closeout.json", closeout)
    print(json.dumps({"holdout": len(HOLDOUT_IDS), "challenge": len(rows),
                      "provenance_complete": closeout["provenance_complete"]}))


if __name__ == "__main__":
    main()
