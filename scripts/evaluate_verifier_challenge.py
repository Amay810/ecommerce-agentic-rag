"""Evaluate a frozen, fully confirmed claim-level verifier challenge."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from ecommerce_rag.claim_verifier import classify_claim


def rate(numerator: int, denominator: int) -> dict[str, Any]:
    return {"numerator": numerator, "denominator": denominator,
            "rate": numerator / denominator if denominator else None}


def metric(rows: list[dict[str, Any]], label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    tp = sum(row["human_fact_status"] == label and row["predicted_fact_status"] == label for row in rows)
    fp = sum(row["human_fact_status"] != label and row["predicted_fact_status"] == label for row in rows)
    fn = sum(row["human_fact_status"] == label and row["predicted_fact_status"] != label for row in rows)
    return rate(tp, tp + fp), rate(tp, tp + fn)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenge-csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-pending", action="store_true")
    args = parser.parse_args()
    with args.challenge_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 150 or len({row["challenge_id"] for row in rows}) != 150:
        raise ValueError("challenge must contain exactly 150 unique rows")
    confirmed = all(row["human_review_status"] == "user_confirmed" for row in rows)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    recorded = (manifest.get("human_review_csv") or {}).get("sha256")
    actual = hashlib.sha256(args.challenge_csv.read_bytes()).hexdigest()
    if recorded != actual:
        raise ValueError("challenge CSV does not match its frozen manifest hash")
    if not confirmed and not args.allow_pending:
        raise ValueError("formal evaluation requires all 150 rows to be user_confirmed")
    evaluated = []
    for row in rows:
        ledger = json.loads(row["evidence_ledger_json"])
        messages = json.loads(row["user_messages_json"])
        result = classify_claim(row["claim_text"], ledger, user_messages=messages,
                                citation_required=row["citation_required"] == "true")
        evaluated.append({**row, "predicted_fact_status": result.fact_status,
                          "predicted_citation_status": result.citation_status,
                          "automatic_decision": result.automatic_decision})
    contradiction_precision, contradiction_recall = metric(evaluated, "contradicted")
    unsupported_precision, unsupported_recall = metric(evaluated, "unsupported")
    supported = [row for row in evaluated if row["human_fact_status"] == "supported"]
    supported_hard_fp = rate(sum(row["predicted_fact_status"] == "contradicted" for row in supported), len(supported))
    factual = [row for row in evaluated if row["human_fact_status"] != "not_factual"]
    unknown = rate(sum(row["predicted_fact_status"] == "unknown" for row in factual), len(factual))
    coverage = rate(sum(row["predicted_fact_status"] != "unknown" for row in factual), len(factual))
    citation_gold = [row for row in evaluated if row["human_citation_status"] in {"correct", "incorrect"}]
    citation_tp = sum(row["human_citation_status"] == "incorrect" and row["predicted_citation_status"] == "incorrect"
                      for row in citation_gold)
    citation_fp = sum(row["human_citation_status"] == "correct" and row["predicted_citation_status"] == "incorrect"
                      for row in citation_gold)
    citation_fn = sum(row["human_citation_status"] == "incorrect" and row["predicted_citation_status"] != "incorrect"
                      for row in citation_gold)
    critical_misses = [row["challenge_id"] for row in evaluated
                       if row["human_fact_status"] == "contradicted"
                       and row["predicted_fact_status"] != "contradicted"
                       and row["family"] in {"numeric_unit", "entity_binding", "state_date"}]
    admission = bool(
        confirmed
        and (contradiction_precision["rate"] or 0) >= 0.90
        and (contradiction_recall["rate"] or 0) >= 0.90
        and (unsupported_precision["rate"] or 0) >= 0.85
        and (unsupported_recall["rate"] or 0) >= 0.80
        and (supported_hard_fp["rate"] or 0) <= 0.05
        and not critical_misses
    )
    report = {
        "schema_version": 1, "challenge": "verifier_challenge_test_v1",
        "evaluation_status": "formal" if confirmed else "preliminary_pending_user_confirmation",
        "hard_gate_admitted": admission, "row_count": len(evaluated),
        "contradiction_precision": contradiction_precision, "contradiction_recall": contradiction_recall,
        "unsupported_precision": unsupported_precision, "unsupported_recall": unsupported_recall,
        "supported_hard_failure_false_positive_rate": supported_hard_fp,
        "unknown_rate": unknown, "automatic_decision_coverage": coverage,
        "citation_incorrect_precision": rate(citation_tp, citation_tp + citation_fp),
        "citation_incorrect_recall": rate(citation_tp, citation_tp + citation_fn),
        "critical_misses": critical_misses,
        "unsupported_is_diagnostic_only": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError("refusing to overwrite a versioned challenge evaluation")
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
