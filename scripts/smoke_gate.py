# -*- coding: utf-8 -*-
"""Decide whether a smoke run is good enough to justify the full evaluation.

A non-zero parse rate is far too weak a bar — one success in ten would pass it.
These are hard checks: any failure means the next run would produce another
batch of unattributable trajectories, which is exactly what happened last time.

``strict_envelope_parse_rate`` is a warning, not a gate. A model that wraps its
JSON in a markdown fence is still usable; blocking on cosmetics would burn a
cluster slot to fix something the parser already recovers from.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"


#: Every quality signal the gate reasons about. Checked for presence separately,
#: so a renamed or dropped metric fails the gate instead of quietly skipping it.
REQUIRED_QUALITY_KEYS = (
    "effective_action_parse_rate",
    "generation_error_rate",
    "fallback_only_trajectory_rate",
    "illegal_tool_rate",
)


def _check(name: str, ok: bool, detail: str) -> dict[str, Any]:
    return {"check": name, "status": PASS if ok else FAIL, "detail": detail}


def _zero_metric(name: str, quality: dict[str, Any]) -> dict[str, Any]:
    """A metric that must be present and exactly zero.

    ``(quality.get(name) or 0) == 0`` reads as a zero check but passes when the
    field is absent, so a renamed or missing metric would silently clear a gate
    whose entire purpose is to fail closed.
    """
    value = quality.get(name)
    if value is None:
        return _check(name + "_is_zero", False, f"{name} missing from the diagnosis — cannot verify")
    return _check(name + "_is_zero", value == 0, f"{name}={value}")


def evaluate(manifest: dict, report: dict, diagnosis: dict, min_parse_rate: float) -> dict[str, Any]:
    quality = diagnosis.get("quality") or {}
    scenarios = manifest["scenarios"]
    expected_ids = {meta["task_id"]: scenario for scenario, meta in scenarios.items()}
    details = {row.get("task_id"): row for row in report.get("details", [])}

    absent = [key for key in REQUIRED_QUALITY_KEYS if quality.get(key) is None]
    checks = [
        _check("instrumented", bool(diagnosis.get("instrumented")),
               "no LLM trace in the store" if not diagnosis.get("instrumented") else "trace present"),
        _check("diagnosis_has_required_metrics", not absent,
               f"missing: {', '.join(absent)}" if absent else "all present"),
        _check("trajectory_count", diagnosis.get("trajectories") == len(expected_ids),
               f"expected {len(expected_ids)}, got {diagnosis.get('trajectories')}"),
        _zero_metric("generation_error_rate", quality),
        _zero_metric("fallback_only_trajectory_rate", quality),
        _zero_metric("illegal_tool_rate", quality),
    ]

    parse_rate = quality.get("effective_action_parse_rate")
    checks.append(_check(
        "effective_action_parse_rate", parse_rate is not None and parse_rate >= min_parse_rate,
        f"{parse_rate} (threshold {min_parse_rate})"))

    missing = sorted(set(expected_ids) - set(details))
    checks.append(_check("all_scenarios_ran", not missing,
                         f"missing task ids: {', '.join(missing)}" if missing else "all present"))

    for task_id, scenario in sorted(expected_ids.items(), key=lambda kv: kv[1]):
        row = details.get(task_id)
        if row is None:
            continue
        problems = []
        if not row.get("success"):
            problems.append(f"failure_type={row.get('failure_type')}")
        if not row.get("terminal_state_match"):
            problems.append(f"terminal_state mismatch: {row.get('state_diff')}")
        if not row.get("policy_compliant"):
            problems.append("forbidden tool used")
        if scenarios[scenario].get("handoff_expected") and not row.get("handoff_observed"):
            problems.append("expected a handoff, none observed")
        if scenarios[scenario].get("expected_tool_sequence") and row.get("tool_sequence_match") is not True:
            problems.append(f"tool sequence mismatch: {scenarios[scenario]['expected_tool_sequence']}")
        checks.append(_check(f"scenario:{scenario}", not problems, "; ".join(problems) or "ok"))

    strict = quality.get("strict_envelope_parse_rate")
    warnings = []
    if strict is not None and parse_rate is not None and strict < parse_rate:
        warnings.append({
            "check": "strict_envelope_parse_rate", "status": WARN,
            "detail": (f"{strict} of {parse_rate} parsed cleanly; "
                       f"violations={diagnosis.get('envelope_violations')}")})
    if (quality.get("truncation_rate") or 0) > 0:
        warnings.append({"check": "truncation_rate", "status": WARN,
                         "detail": f"{quality['truncation_rate']} — consider raising max_new_tokens"})

    failed = [c for c in checks if c["status"] == FAIL]
    return {"passed": not failed, "checks": checks, "warnings": warnings,
            "failed_checks": [c["check"] for c in failed],
            "verdict": ("smoke passed; the full evaluation is justified" if not failed else
                        "smoke failed; fix the reported checks before spending a full run")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True, help="harness run report")
    parser.add_argument("--diagnosis", type=Path, required=True, help="diagnose_llm_trace output")
    parser.add_argument("--min-parse-rate", type=float, default=0.9)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    for path in (args.manifest, args.report, args.diagnosis):
        if not path.exists():
            print(json.dumps({"passed": False, "verdict": f"missing required input: {path}"}, ensure_ascii=False))
            sys.exit(2)

    result = evaluate(json.loads(args.manifest.read_text(encoding="utf-8")),
                      json.loads(args.report.read_text(encoding="utf-8")),
                      json.loads(args.diagnosis.read_text(encoding="utf-8")),
                      args.min_parse_rate)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    for row in result["checks"] + result["warnings"]:
        print(f"  [{row['status']:4}] {row['check']:36} {row['detail']}")
    print(f"\n{result['verdict']}")
    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
