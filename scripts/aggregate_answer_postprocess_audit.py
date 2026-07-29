"""Aggregate the preregistered blinded terminal-grounding paired audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from ecommerce_rag.answer_postprocess import stable_hash
from scripts.prepare_answer_postprocess_audit import HUMAN_FIELDS, SEED, immutable_hash


BOOTSTRAP_SAMPLES = 10_000
ALLOWED_LABELS = frozenset({"true", "false", "unclear"})


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def unique_rows(path: Path, key: str, expected: int) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    result = {str(row[key]): row for row in rows}
    if len(rows) != expected or len(result) != expected:
        raise ValueError(f"{path}: expected {expected} unique rows by {key}")
    return result


def rate(numerator: int, denominator: int) -> dict[str, Any]:
    return {"numerator": numerator, "denominator": denominator,
            "rate": numerator / denominator if denominator else None}


def percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def paired_bootstrap(pairs: list[tuple[int, int]]) -> dict[str, Any]:
    rng = random.Random(SEED)
    deltas: list[float] = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sample = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        deltas.append(mean(grounded - base for base, grounded in sample))
    return {
        "seed": SEED,
        "samples": BOOTSTRAP_SAMPLES,
        "lower_95": percentile(deltas, 0.025),
        "upper_95": percentile(deltas, 0.975),
    }


def nearest_rank_p95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def numeric_summary(values: list[float]) -> dict[str, Any]:
    return {"count": len(values), "mean": mean(values) if values else None,
            "p95_nearest_rank": nearest_rank_p95(values)}


def verifier_diagnostics(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    fact: Counter[str] = Counter()
    citation: Counter[str] = Counter()
    claims = 0
    for row in rows.values():
        for result in row.get("verification") or []:
            claims += 1
            fact[str(result.get("fact_status", "missing"))] += 1
            citation[str(result.get("citation_status", "missing"))] += 1
    return {"claim_count": claims, "fact_status_counts": dict(sorted(fact.items())),
            "citation_status_counts": dict(sorted(citation.items()))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--package-manifest", type=Path, required=True)
    parser.add_argument("--shadow", type=Path, required=True)
    parser.add_argument("--terminal-grounded", type=Path, required=True)
    parser.add_argument("--shadow-gate", type=Path, required=True)
    parser.add_argument("--terminal-grounded-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite audit aggregate: {args.output}")

    manifest = json.loads(args.package_manifest.read_text(encoding="utf-8"))
    if sha256(args.mapping) != manifest["mapping_file_sha256"]:
        raise ValueError("blinded mapping differs from the frozen package manifest")
    if sha256(args.shadow) != manifest["shadow_sidecar_sha256"]:
        raise ValueError("shadow sidecar differs from the frozen package manifest")
    if sha256(args.terminal_grounded) != manifest["terminal_grounded_sidecar_sha256"]:
        raise ValueError("terminal-grounded sidecar differs from the frozen package manifest")

    reviews = unique_rows(args.review, "review_id", 80)
    mappings = unique_rows(args.mapping, "review_id", 80)
    if set(reviews) != set(mappings):
        raise ValueError("review and mapping ids differ")
    expected_hashes = manifest["immutable_review_row_hashes"]
    if set(expected_hashes) != set(reviews):
        raise ValueError("package manifest review ids differ")
    for review_id, row in reviews.items():
        if immutable_hash(row) != expected_hashes[review_id]:
            raise ValueError(f"{review_id}: immutable blinded review content changed")
        for field in HUMAN_FIELDS - {"review_notes"}:
            if row.get(field) not in ALLOWED_LABELS:
                raise ValueError(f"{review_id}: invalid or missing {field}")
        if not isinstance(row.get("review_notes"), str):
            raise ValueError(f"{review_id}: review_notes must be text")

    shadow = unique_rows(args.shadow, "task_id", 80)
    grounded = unique_rows(args.terminal_grounded, "task_id", 80)
    shadow_gate = json.loads(args.shadow_gate.read_text(encoding="utf-8"))
    grounded_gate = json.loads(args.terminal_grounded_gate.read_text(encoding="utf-8"))
    immutability_passed = shadow_gate.get("passed") is True and grounded_gate.get("passed") is True
    if not immutability_passed:
        raise ValueError("both structural sidecar gates must pass before aggregation")

    paired: dict[str, dict[str, dict[str, Any]]] = {}
    for review_id, mapping in mappings.items():
        task_id, variant = mapping["task_id"], mapping["variant"]
        if variant not in {"base", "terminal_grounded"}:
            raise ValueError(f"{review_id}: invalid hidden variant")
        paired.setdefault(task_id, {})[variant] = reviews[review_id]
    if len(paired) != 40 or any(set(rows) != {"base", "terminal_grounded"} for rows in paired.values()):
        raise ValueError("mapping must resolve to exactly 40 complete pairs")

    fact_pairs: list[tuple[int, int]] = []
    sensitivity_pairs: list[tuple[int, int]] = []
    complete_pairs: list[tuple[int, int]] = []
    contradiction_pairs: list[tuple[int, int]] = []
    for rows in paired.values():
        base, terminal = rows["base"], rows["terminal_grounded"]
        fact_pairs.append((int(base["fact_pass"] == "true"), int(terminal["fact_pass"] == "true")))
        if base["fact_pass"] != "unclear" and terminal["fact_pass"] != "unclear":
            sensitivity_pairs.append((int(base["fact_pass"] == "true"), int(terminal["fact_pass"] == "true")))
        complete_pairs.append((int(base["answer_complete"] == "true"),
                               int(terminal["answer_complete"] == "true")))
        contradiction_pairs.append((int(base["contradiction_present"] != "false"),
                                    int(terminal["contradiction_present"] != "false")))

    fact_base = sum(base for base, _ in fact_pairs)
    fact_grounded = sum(grounded_value for _, grounded_value in fact_pairs)
    complete_base = sum(base for base, _ in complete_pairs)
    complete_grounded = sum(grounded_value for _, grounded_value in complete_pairs)
    contradiction_base = sum(base for base, _ in contradiction_pairs)
    contradiction_grounded = sum(grounded_value for _, grounded_value in contradiction_pairs)
    fact_delta = (fact_grounded - fact_base) / len(fact_pairs)
    complete_delta = (complete_grounded - complete_base) / len(complete_pairs)
    bootstrap = paired_bootstrap(fact_pairs)

    eligible = [row for row in grounded.values() if row.get("eligible")]
    prompt_tokens = [float(row["prompt_tokens"]) for row in eligible if row.get("prompt_tokens") is not None]
    completion_tokens = [float(row["completion_tokens"]) for row in eligible
                         if row.get("completion_tokens") is not None]
    latency = [float(row["latency_ms"]) for row in eligible if row.get("latency_ms") is not None]
    positive = bool(
        fact_delta > 0
        and bootstrap["lower_95"] is not None
        and bootstrap["lower_95"] > 0
        and complete_delta >= -0.05
        and contradiction_grounded <= contradiction_base
        and immutability_passed
    )
    report = {
        "schema_version": 1,
        "name": "answer_postprocess_blind_audit_v1_aggregate",
        "status": "positive" if positive else "negative_or_inconclusive",
        "primary_human_fact_pass": {
            "base": rate(fact_base, 40),
            "terminal_grounded": rate(fact_grounded, 40),
            "paired_difference": fact_delta,
            "unclear_policy": "count_as_failure",
            "paired_bootstrap_95_ci": bootstrap,
            "discordant_pairs": {
                "base_only_pass": sum(base == 1 and terminal == 0 for base, terminal in fact_pairs),
                "grounded_only_pass": sum(base == 0 and terminal == 1 for base, terminal in fact_pairs),
            },
        },
        "fact_pass_sensitivity_excluding_unclear_pairs": {
            "pairs": len(sensitivity_pairs),
            "base": rate(sum(base for base, _ in sensitivity_pairs), len(sensitivity_pairs)),
            "terminal_grounded": rate(sum(value for _, value in sensitivity_pairs), len(sensitivity_pairs)),
        },
        "answer_completeness": {
            "base": rate(complete_base, 40),
            "terminal_grounded": rate(complete_grounded, 40),
            "paired_difference": complete_delta,
            "omission_rate_base": rate(40 - complete_base, 40),
            "omission_rate_terminal_grounded": rate(40 - complete_grounded, 40),
        },
        "contradictions": {
            "unclear_policy": "count_as_present",
            "base": rate(contradiction_base, 40),
            "terminal_grounded": rate(contradiction_grounded, 40),
            "new_in_terminal_grounded": sum(base == 0 and terminal == 1
                                              for base, terminal in contradiction_pairs),
        },
        "all_80_answer_coverage": {
            "eligible": rate(len(eligible), 80),
            "pass_through": rate(80 - len(eligible), 80),
            "changed_eligible": rate(sum(bool(row.get("changed")) for row in eligible), len(eligible)),
            "nonempty_terminal_grounded_eligible": rate(
                sum(bool(str(row.get("final_answer") or "").strip()) for row in eligible), len(eligible)
            ),
        },
        "incremental_generation_cost": {
            "prompt_tokens": numeric_summary(prompt_tokens),
            "completion_tokens": numeric_summary(completion_tokens),
            "latency_ms": numeric_summary(latency),
        },
        "verifier_diagnostics_not_used_for_decision": {
            "shadow": verifier_diagnostics(shadow),
            "terminal_grounded": verifier_diagnostics(grounded),
        },
        "decision_checks": {
            "fact_delta_positive": fact_delta > 0,
            "fact_ci_lower_above_zero": bootstrap["lower_95"] is not None and bootstrap["lower_95"] > 0,
            "completeness_drop_at_most_5pp": complete_delta >= -0.05,
            "contradictions_do_not_increase": contradiction_grounded <= contradiction_base,
            "trajectory_immutability_passed": immutability_passed,
            "positive_result": positive,
        },
        "provenance": {
            "review_sha256": sha256(args.review),
            "mapping_sha256": sha256(args.mapping),
            "package_manifest_sha256": sha256(args.package_manifest),
            "shadow_sidecar_sha256": sha256(args.shadow),
            "terminal_grounded_sidecar_sha256": sha256(args.terminal_grounded),
            "shadow_gate_sha256": sha256(args.shadow_gate),
            "terminal_grounded_gate_sha256": sha256(args.terminal_grounded_gate),
            "aggregation_config_sha256": stable_hash({
                "seed": SEED, "bootstrap_samples": BOOTSTRAP_SAMPLES,
                "unclear_fact": "failure", "unclear_contradiction": "present",
                "max_completeness_drop": 0.05,
            }),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps({"status": report["status"], "paired_difference": fact_delta}, ensure_ascii=False))


if __name__ == "__main__":
    main()
