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

FALLBACK_REASON = "model_action_parse_failure"


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
            steps_with_llm += 1
            resolutions[trace.get("resolution", "unknown")] += 1
            if trace.get("generator") and trace["generator"] not in generators:
                generators.append(trace["generator"])
            for attempt in trace.get("attempts", []):
                total_generations += 1
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
    for trajectory in trajectories:
        names = [call.get("name") for call in trajectory.get("tool_calls", [])]
        reasons = {call.get("arguments", {}).get("reason") for call in trajectory.get("tool_calls", [])
                   if call.get("name") == "escalate_to_human"}
        if names and set(names) == {"escalate_to_human"} and reasons == {FALLBACK_REASON}:
            fallback_only += 1
        real_tool_calls += sum(1 for name in names if name != "escalate_to_human")

    n = len(trajectories) or 1
    parsed = resolutions.get("parsed", 0)
    return {
        "trajectories": len(trajectories),
        "steps_with_llm_trace": steps_with_llm,
        "instrumented": steps_with_llm > 0,
        "quality": {
            "effective_action_parse_rate": round(parsed / steps_with_llm, 4) if steps_with_llm else None,
            "fallback_only_trajectory_rate": round(fallback_only / n, 4),
            "non_fallback_tool_call_rate": round(real_tool_calls / n, 4),
            "truncation_rate": round(truncated / total_generations, 4) if total_generations else None,
        },
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
