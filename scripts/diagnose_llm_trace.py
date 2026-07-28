# -*- coding: utf-8 -*-
"""Turn recorded LLM traces into an attribution report.

The first 360-trajectory run reported ``model_action_parse_failure`` on every
step and there was no way to tell whether the cause was the prompt, the chat
template, the token budget, the output format or the parser — none of it was
kept. ``LLMPolicy`` now records every generation; this script aggregates it.

It also computes the quality signals a trajectory-count gate cannot see:
effective action-parse rate, non-fallback tool-call rate, and truncation rate.
"""

from __future__ import annotations

import argparse
import collections
import json
import sqlite3
from pathlib import Path
from typing import Any

#: Reasons the policy emits when it could not produce a usable action at all.
FALLBACK_REASONS = {"model_action_parse_failure", "model_generation_error"}


def load_trajectories(store: Path) -> list[dict[str, Any]]:
    conn = sqlite3.connect(store)
    try:
        rows = conn.execute("SELECT trajectory_json FROM trajectories").fetchall()
    finally:
        conn.close()
    return [json.loads(row[0]) for row in rows]


def _stat(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)
    return {"min": ordered[0], "median": ordered[len(ordered) // 2], "max": ordered[-1],
            "mean": round(sum(ordered) / len(ordered), 2)}


def build_report(trajectories: list[dict[str, Any]], samples_per_stage: int = 3) -> dict[str, Any]:
    resolutions = collections.Counter()
    stages = collections.Counter()
    finish_reasons = collections.Counter()
    violation_counter = collections.Counter()
    generators: list[dict] = []
    prompt_tokens: list[float] = []
    completion_tokens: list[float] = []
    truncated = total_generations = 0
    samples: dict[str, list[dict]] = collections.defaultdict(list)

    steps_with_llm = 0
    for trajectory in trajectories:
        for call in trajectory.get("model_calls", []):
            trace = call.get("llm")
            if not trace:
                continue
            trace_items = [("initial", trace)]
            if isinstance(trace.get("repair_llm"), dict):
                trace_items.append(("repair", trace["repair_llm"]))
            for trace_phase, trace_item in trace_items:
                steps_with_llm += 1
                resolutions[trace_item.get("resolution", "unknown")] += 1
                if trace_item.get("generator") and trace_item["generator"] not in generators:
                    generators.append(trace_item["generator"])
                for violation in trace_item.get("envelope_violations", []):
                    # keep the field list out of the key so the histogram stays readable
                    violation_counter[violation.split(":", 1)[0]] += 1
                for attempt in trace_item.get("attempts", []):
                    total_generations += 1
                    if attempt.get("parse_stage") == "generation_error":
                        # no text was produced; counting a null finish_reason here would
                        # pollute the histogram used to spot truncation
                        finish_reasons["<generation_error>"] += 1
                    else:
                        finish_reasons[str(attempt.get("finish_reason"))] += 1
                    if attempt.get("truncated"):
                        truncated += 1
                    if attempt.get("prompt_tokens") is not None:
                        prompt_tokens.append(attempt["prompt_tokens"])
                    if attempt.get("completion_tokens") is not None:
                        completion_tokens.append(attempt["completion_tokens"])
                    stage = attempt.get("parse_stage")
                    if stage:
                        stages[stage] += 1
                        if len(samples[stage]) < samples_per_stage:
                            samples[stage].append({
                                "trajectory_id": trajectory.get("trajectory_id"),
                                "trace_phase": trace_phase,
                                "attempt": attempt.get("attempt"),
                                "parse_error": attempt.get("parse_error"),
                                "finish_reason": attempt.get("finish_reason"),
                                "truncated": attempt.get("truncated"),
                                "raw_output": attempt.get("raw_output"),
                            })

    # Trajectory-level quality: a run that only ever escalates on a parse failure
    # is not a model evaluation, however many rows it produced.
    fallback_only = 0
    real_tool_calls = 0
    with_real_tool_call = 0
    for trajectory in trajectories:
        names = [call.get("name") for call in trajectory.get("tool_calls", [])]
        reasons = {call.get("arguments", {}).get("reason") for call in trajectory.get("tool_calls", [])
                   if call.get("name") == "escalate_to_human"}
        if names and set(names) == {"escalate_to_human"} and reasons and reasons <= FALLBACK_REASONS:
            fallback_only += 1
        real = sum(1 for name in names if name != "escalate_to_human")
        real_tool_calls += real
        with_real_tool_call += int(real > 0)

    n = len(trajectories) or 1
    # "parsed" is strictly compliant; "parsed_with_violations" was recoverable but
    # broke the one-object-and-nothing-else contract. Reporting only their sum
    # would let protocol violations count as clean output.
    strict = resolutions.get("parsed", 0)
    recovered = resolutions.get("parsed_with_violations", 0)
    illegal_tool = stages.get("unknown_tool", 0)
    return {
        "trajectories": len(trajectories),
        "steps_with_llm_trace": steps_with_llm,
        "instrumented": steps_with_llm > 0,
        "quality": {
            # usable action produced, whether or not the envelope was clean
            "effective_action_parse_rate": round((strict + recovered) / steps_with_llm, 4) if steps_with_llm else None,
            # subset that also obeyed "exactly one JSON object and nothing else"
            "strict_envelope_parse_rate": round(strict / steps_with_llm, 4) if steps_with_llm else None,
            "recovered_parse_rate": round(recovered / steps_with_llm, 4) if steps_with_llm else None,
            # share of generations that named a tool which was not offered
            "illegal_tool_rate": round(illegal_tool / total_generations, 4) if total_generations else None,
            "generation_error_rate": round(stages.get("generation_error", 0) / total_generations, 4) if total_generations else None,
            "fallback_only_trajectory_rate": round(fallback_only / n, 4),
            # a mean, not a rate: a multi-step trajectory contributes more than one
            "avg_non_fallback_tool_calls": round(real_tool_calls / n, 4),
            "trajectories_with_real_tool_call_rate": round(with_real_tool_call / n, 4),
            "truncation_rate": round(truncated / total_generations, 4) if total_generations else None,
        },
        "envelope_violations": dict(violation_counter.most_common()),
        "resolutions": dict(resolutions),
        "parse_stages": dict(stages.most_common()),
        "finish_reasons": dict(finish_reasons.most_common()),
        "generations": total_generations,
        "prompt_tokens": _stat(prompt_tokens),
        "completion_tokens": _stat(completion_tokens),
        "generators": generators,
        "failure_samples": {stage: rows for stage, rows in samples.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--samples-per-stage", type=int, default=3)
    args = parser.parse_args()

    trajectories = load_trajectories(args.store)
    report = build_report(trajectories, args.samples_per_stage)
    if not report["instrumented"]:
        report["notice"] = ("No LLM trace found. This store predates the observability change; "
                            "re-run with the current LLMPolicy to attribute failures.")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    printable = {k: v for k, v in report.items() if k != "failure_samples"}
    print(json.dumps(printable, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
