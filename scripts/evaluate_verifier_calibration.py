"""Evaluate claim verifier on the 634-row tuning/regression audit set."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
from typing import Any

from ecommerce_rag.claim_verifier import classify_claim


def rate(n: int, d: int) -> dict[str, Any]:
    return {"numerator": n, "denominator": d, "rate": n / d if d else None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--claims", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture-output", type=Path)
    args = parser.parse_args()
    with args.answers.open(encoding="utf-8", newline="") as handle:
        answers = {row["answer_id"]: row for row in csv.DictReader(handle)}
    with args.claims.open(encoding="utf-8", newline="") as handle:
        claims = list(csv.DictReader(handle))
    if len(answers) != 96 or len(claims) != 634:
        raise ValueError("expected the frozen 96-answer/634-claim calibration set")
    confusion: dict[str, int] = {}
    rows = []
    fixtures = []
    for claim in claims:
        answer = answers[claim["answer_id"]]
        result = classify_claim(
            claim["claim_text"], json.loads(answer["evidence_ledger_json"]),
            user_messages=[{"role": "user", "content": answer["user_prompt"]}],
            citation_required=bool(claim["cited_evidence_ids"]),
        )
        gold = claim["human_fact_status"]
        key = f"human={gold}|predicted={result.fact_status}"
        confusion[key] = confusion.get(key, 0) + 1
        rows.append((gold, result.fact_status))
        scope_exception = "配对的 get_product" in claim.get("review_notes", "")
        fixtures.append({
            "claim_id": claim["claim_id"], "answer_id": claim["answer_id"],
            "task_id": claim["task_id"], "variant": claim["variant"],
            "claim_text": claim["claim_text"], "evidence_ledger": json.loads(answer["evidence_ledger_json"]),
            "user_messages": [{"role": "user", "content": answer["user_prompt"]}],
            "human_fact_status": gold, "current_predicted_status": result.fact_status,
            "verifier_input_scope": "paired_external_evidence" if scope_exception else "trajectory_ledger",
            "expected_verifier_status": "unknown" if scope_exception else gold,
            "review_notes": claim.get("review_notes", ""),
        })
    factual = [(gold, pred) for gold, pred in rows if gold not in {"not_factual", "unclear"}]
    supported = [(gold, pred) for gold, pred in factual if gold == "supported"]
    hard_fp = sum(pred == "contradicted" for _, pred in supported)
    report = {
        "schema_version": 1, "dataset": "audit_32_claims_calibration_only",
        "row_count": len(rows), "effective_factual_claims": len(factual),
        "confusion_matrix": confusion,
        "unknown_rate": rate(sum(pred == "unknown" for _, pred in factual), len(factual)),
        "automatic_decision_coverage": rate(sum(pred != "unknown" for _, pred in factual), len(factual)),
        "supported_hard_failure_false_positive_rate": rate(hard_fp, len(supported)),
        "warning": "Tuning-set metrics are regression diagnostics and not generalization evidence.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.fixture_output:
        args.fixture_output.parent.mkdir(parents=True, exist_ok=True)
        payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in fixtures).encode("utf-8")
        if args.fixture_output.suffix == ".gz":
            args.fixture_output.write_bytes(gzip.compress(payload, mtime=0))
        else:
            args.fixture_output.write_bytes(payload)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
