"""Evaluate a versioned development or locked verifier challenge."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from ecommerce_rag.claim_verifier import (
    VERIFIER_CONFIG_V2,
    classify_claim,
    verifier_config_hash,
)


ROOT = Path(__file__).resolve().parents[1]
VERIFIER_SOURCE = ROOT / "ecommerce_rag" / "claim_verifier.py"
EVALUATOR_SOURCE = Path(__file__).resolve()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def source_commit(path: Path) -> str:
    return git_output("log", "-1", "--format=%H", "--", str(path.relative_to(ROOT)))


def worktree_dirty() -> bool:
    return bool(git_output("status", "--porcelain", "--untracked-files=no"))


def rate(numerator: int, denominator: int) -> dict[str, Any]:
    return {"numerator": numerator, "denominator": denominator,
            "rate": numerator / denominator if denominator else None}


def metric(rows: list[dict[str, Any]], label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    tp = sum(row["human_fact_status"] == label and row["predicted_fact_status"] == label for row in rows)
    fp = sum(row["human_fact_status"] != label and row["predicted_fact_status"] == label for row in rows)
    fn = sum(row["human_fact_status"] == label and row["predicted_fact_status"] != label for row in rows)
    return rate(tp, tp + fp), rate(tp, tp + fn)


def atomic_write_new(path: Path, payload: bytes) -> None:
    """Publish a completed report atomically and never overwrite an existing one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite versioned evaluation: {path}")
    lock = path.with_name(path.name + ".lock")
    lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    temporary: Path | None = None
    try:
        os.write(lock_fd, str(os.getpid()).encode("ascii"))
        os.fsync(lock_fd)
        os.close(lock_fd)
        lock_fd = -1
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
                                         delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise FileExistsError(f"evaluation appeared while lock was held: {path}")
        os.replace(temporary, path)
        temporary = None
    finally:
        if lock_fd >= 0:
            os.close(lock_fd)
        if temporary and temporary.exists():
            temporary.unlink()
        if lock.exists():
            lock.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenge-csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-role", choices=("development", "locked"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite a versioned challenge evaluation")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("dataset_role") != args.dataset_role:
        raise ValueError("dataset role does not match manifest")
    recorded = manifest.get("human_review_csv") or {}
    if recorded.get("sha256") != sha256(args.challenge_csv):
        raise ValueError("manifest CSV SHA-256 does not match the recomputed CSV SHA-256")
    if manifest.get("verifier_source_sha256") != sha256(VERIFIER_SOURCE):
        raise ValueError("verifier source differs from the frozen manifest")
    if manifest.get("verifier_config_hash") != verifier_config_hash():
        raise ValueError("verifier configuration differs from the frozen manifest")
    if manifest.get("evaluator_source_sha256") != sha256(EVALUATOR_SOURCE):
        raise ValueError("evaluator source differs from the frozen manifest")
    verifier_commit = source_commit(VERIFIER_SOURCE)
    evaluator_commit = source_commit(EVALUATOR_SOURCE)
    if manifest.get("verifier_code_commit") != verifier_commit:
        raise ValueError("verifier code commit differs from the frozen manifest")
    if manifest.get("evaluator_code_commit") != evaluator_commit:
        raise ValueError("evaluator code commit differs from the frozen manifest")
    dirty = worktree_dirty()
    if args.dataset_role == "locked" and dirty:
        raise ValueError("locked evaluation requires a clean worktree")

    with args.challenge_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 150 or len({row["challenge_id"] for row in rows}) != 150:
        raise ValueError("challenge must contain exactly 150 unique rows")
    if not all(row["human_review_status"] == "user_confirmed" for row in rows):
        raise ValueError("all 150 labels must be user_confirmed before evaluation")

    evaluated = []
    for row in rows:
        result = classify_claim(
            row["claim_text"], json.loads(row["evidence_ledger_json"]),
            user_messages=json.loads(row["user_messages_json"]),
            citation_required=row["citation_required"] == "true",
        )
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
    thresholds = VERIFIER_CONFIG_V2["admission_thresholds"]
    meets = bool(
        (contradiction_precision["rate"] or 0) >= thresholds["contradiction_precision"]
        and (contradiction_recall["rate"] or 0) >= thresholds["contradiction_recall"]
        and (unsupported_precision["rate"] or 0) >= thresholds["unsupported_precision"]
        and (unsupported_recall["rate"] or 0) >= thresholds["unsupported_recall"]
        and (supported_hard_fp["rate"] or 0) <= thresholds["supported_hard_failure_false_positive_rate_max"]
        and not critical_misses
    )
    manifest_hash = sha256(args.manifest)
    report = {
        "schema_version": 2, "challenge": manifest.get("name"), "dataset_role": args.dataset_role,
        "evaluation_status": "versioned_baseline" if args.dataset_role == "development" else "formal_locked",
        "admission_applicable": args.dataset_role == "locked",
        "hard_gate_admitted": meets if args.dataset_role == "locked" else False,
        "nscc_smoke_eligible": meets if args.dataset_role == "locked" else False,
        "meets_engineering_thresholds": meets, "row_count": len(evaluated),
        "contradiction_precision": contradiction_precision, "contradiction_recall": contradiction_recall,
        "unsupported_precision": unsupported_precision, "unsupported_recall": unsupported_recall,
        "supported_hard_failure_false_positive_rate": supported_hard_fp,
        "unknown_rate": unknown, "automatic_decision_coverage": coverage,
        "citation_incorrect_precision": rate(citation_tp, citation_tp + citation_fp),
        "citation_incorrect_recall": rate(citation_tp, citation_tp + citation_fn),
        "critical_misses": critical_misses, "unsupported_is_diagnostic_only": True,
        "provenance": {
            "csv_sha256": sha256(args.challenge_csv), "manifest_sha256": manifest_hash,
            "verifier_code_commit": verifier_commit, "verifier_source_sha256": sha256(VERIFIER_SOURCE),
            "verifier_config_hash": verifier_config_hash(), "evaluator_code_commit": evaluator_commit,
            "evaluator_source_sha256": sha256(EVALUATOR_SOURCE), "worktree_dirty_before_output": dirty,
        },
    }
    payload = (json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    atomic_write_new(args.output, payload)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
