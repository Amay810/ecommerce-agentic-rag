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


def _check(name: str, ok: bool, detail: str) -> dict[str, Any]:
    return {"check": name, "status": PASS if ok else FAIL, "detail": detail}


def evaluate(manifest: dict, report: dict, diagnosis: dict, min_parse_rate: float) -> dict[str, Any]:
    quality = diagnosis.get("quality") or {}
    scenarios = manifest["scenarios"]
    expected_ids = {meta["task_id"]: scenario for scenario, meta in scenarios.items()}
    details = {row.get("task_id"): row for row in report.get("details", [])}

    checks = [
        _check("instrumented", bool(diagnosis.get("instrumented")),
               "no LLM trace in the store" if not diagnosis.get("instrumented") else "trace present"),
        _check("trajectory_count", diagnosis.get("trajectories") == len(expected_ids),
               f"expected {len(expected_ids)}, got {diagnosis.get('trajectories')}"),
        _check("generation_error_rate_is_zero", (quality.get("generation_error_rate") or 0) == 0,
               f"generation_error_rate={quality.get('generation_error_rate')}"),
        _check("no_fallback_only_trajectories", (quality.get("fallback_only_trajectory_rate") or 0) == 0,
               f"fallback_only_trajectory_rate={quality.get('fallback_only_trajectory_rate')}"),
        _check("illegal_tool_rate_is_zero", (quality.get("illegal_tool_rate") or 0) == 0,
               f"illegal_tool_rate={quality.get('illegal_tool_rate')}"),
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
