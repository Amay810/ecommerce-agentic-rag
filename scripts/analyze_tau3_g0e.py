"""Recompute the read-only G0-E pre-GRPO audit from a tau3 results artifact."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median, quantiles
from typing import Any, Iterable


def _p90(values: Iterable[float]) -> float | None:
    values = list(values)
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    return float(quantiles(values, n=10, method="inclusive")[8])


def _distribution(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(mean(values)) if values else 0.0,
        "median": float(median(values)) if values else 0.0,
        "p90": _p90(values) or 0.0,
    }


def _timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _message_usage(message: dict[str, Any]) -> dict[str, int]:
    usage = message.get("usage") or {}
    raw_usage = (message.get("raw_data") or {}).get("usage") or {}
    prompt = int(usage.get("prompt_tokens") or raw_usage.get("prompt_tokens") or 0)
    completion = int(
        usage.get("completion_tokens")
        or raw_usage.get("completion_tokens")
        or 0
    )
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "prompt_cache_hit_tokens": int(raw_usage.get("prompt_cache_hit_tokens") or 0),
        "prompt_cache_miss_tokens": int(
            raw_usage.get("prompt_cache_miss_tokens") or 0
        ),
    }


def _cost_usd(usage: dict[str, int], pricing: dict[str, Any] | None) -> float | None:
    if pricing is None:
        return None
    return (
        usage["prompt_cache_hit_tokens"] * pricing["input_cache_hit_usd_per_1m"]
        + usage["prompt_cache_miss_tokens"] * pricing["input_cache_miss_usd_per_1m"]
        + usage["completion_tokens"] * pricing["output_usd_per_1m"]
    ) / 1_000_000


def _usage_summary(
    rows: list[dict[str, Any]], pricing: dict[str, Any] | None
) -> dict[str, Any]:
    fields = (
        "calls",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "prompt_cache_hit_tokens",
        "prompt_cache_miss_tokens",
    )
    aggregate = {field: sum(int(row[field]) for row in rows) for field in fields}
    costs = [_cost_usd(row, pricing) for row in rows if row["calls"] > 0]
    clean_costs = [value for value in costs if value is not None]
    return {
        **aggregate,
        "estimated_cost_usd": sum(clean_costs) if clean_costs else None,
        "per_simulation": {
            field: _distribution([float(row[field]) for row in rows])
            for field in fields
        },
        "per_simulation_cost_usd": _distribution(clean_costs)
        if clean_costs
        else None,
    }


def analyze(
    payload: dict[str, Any], *, pricing: dict[str, Any] | None = None
) -> dict[str, Any]:
    simulations = payload.get("simulations") or []
    by_task: defaultdict[str, list[float]] = defaultdict(list)
    for simulation in simulations:
        reward = (simulation.get("reward_info") or {}).get("reward")
        if reward not in (0, 0.0, 1, 1.0):
            raise ValueError(f"non-binary terminal reward: {reward!r}")
        by_task[str(simulation.get("task_id"))].append(float(reward))

    group_sizes = Counter(len(values) for values in by_task.values())
    if len(by_task) != 74 or group_sizes != Counter({8: 74}):
        raise ValueError(f"expected 74 groups of 8, got {dict(group_sizes)}")

    per_task: dict[str, dict[str, Any]] = {}
    classes = Counter()
    for task_id, rewards in sorted(by_task.items(), key=lambda item: int(item[0])):
        successes = int(sum(rewards))
        if successes == 0:
            task_class = "0/8"
        elif successes == 8:
            task_class = "8/8"
        else:
            task_class = "mixed"
        classes[task_class] += 1
        per_task[task_id] = {
            "nonzero_trials": successes,
            "class": task_class,
            "mean_reward": successes / 8,
        }

    all_zero = classes["0/8"]
    mixed = classes["mixed"]
    all_one = classes["8/8"]
    if all_zero + mixed + all_one != 74 or len(simulations) != 592:
        raise ValueError("G0-E group/simulation totals are inconsistent")

    request = (payload.get("tau3_experiment") or {}).get("requested") or {}
    command = (payload.get("tau3_experiment") or {}).get("command") or []
    summary = payload.get("experiment_summary") or {}

    starts = [_timestamp(s["start_time"]) for s in simulations if s.get("start_time")]
    ends = [_timestamp(s["end_time"]) for s in simulations if s.get("end_time")]
    artifact_span = (max(ends) - min(starts)).total_seconds() if starts and ends else None
    artifact_throughput = len(simulations) / artifact_span if artifact_span else None
    step_seconds = artifact_span * 16 / len(simulations) if artifact_span else None

    usage_rows: list[dict[str, Any]] = []
    observed_models = Counter()
    for simulation in simulations:
        row = {field: 0 for field in (
            "calls",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "prompt_cache_hit_tokens",
            "prompt_cache_miss_tokens",
        )}
        for message in simulation.get("messages") or []:
            raw_data = message.get("raw_data") or {}
            model = raw_data.get("model")
            if not model or not str(model).lower().startswith("deepseek"):
                continue
            observed_models[str(model)] += 1
            usage = _message_usage(message)
            row["calls"] += 1
            for field in usage:
                row[field] += usage[field]
        usage_rows.append(row)

    deepseek_pricing = pricing
    user_usage = _usage_summary(usage_rows, deepseek_pricing)
    judge_eligible = sum(
        bool((simulation.get("reward_info") or {}).get("nl_assertions"))
        for simulation in simulations
    )

    runtime = {
        "evidence_source": (
            "results.json: experiment_summary.wall_clock_seconds plus "
            "simulations[*].start_time/end_time"
        ),
        "g0e_elapsed_seconds": None,
        "elapsed_reason": (
            "Exact watchdog launch-to-final-exit time is not committed. The "
            "artifact first-to-last simulation span is reported separately; "
            "experiment_summary.wall_clock_seconds is one wrapper invocation."
        ),
        "wrapper_invocation_elapsed_seconds": summary.get("wall_clock_seconds"),
        "artifact_span_start": min(starts).isoformat() if starts else None,
        "artifact_span_end": max(ends).isoformat() if ends else None,
        "artifact_simulation_span_seconds": artifact_span,
        "observed_rollouts_per_second_artifact_span": artifact_throughput,
        "estimated_rollout_seconds_per_p2k8_step": step_seconds,
        "estimated_9_step_rollout_seconds": step_seconds * 9 if step_seconds else None,
        "estimated_18_step_rollout_seconds": step_seconds * 18 if step_seconds else None,
        "estimate_scope": "rollout-only empirical estimate using artifact span; includes resume gaps",
        "training_update_overhead": "not measured",
        "max_concurrency": request.get("max_concurrency"),
        "gpu_count": None,
        "agent_serving_configuration": request.get("agent_model"),
        "user_simulator_concurrency": None,
        "watchdog": {
            "auto_resume_flag_in_command": "--auto-resume" in command,
            "committed_watchdog_log": False,
        },
    }

    deepseek = {
        "source": "results.json simulations[*].messages[*].raw_data.usage",
        "requested_model": request.get("user_model"),
        "observed_provider_models": dict(observed_models),
        "user_simulator": {
            **user_usage,
            "scope": "observable user-simulator messages only",
        },
        "nl_judge": {
            "calls": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "estimated_cost_usd": None,
            "judge_eligible_simulations": judge_eligible,
            "reason": "NL-assertion results exist, but judge provider messages and usage are not recorded in the artifact.",
        },
        "combined": {
            **user_usage,
            "scope": "observable DeepSeek user-simulator usage only; NL judge excluded",
        },
        "pricing": pricing,
    }

    return {
        "artifact": None,
        "protocol": payload.get("tau3_experiment", {}).get("protocol"),
        "provenance": {
            "tau2_source": (payload.get("tau3_experiment") or {}).get("tau2_source"),
            "requested": request,
            "checks_passed": (payload.get("tau3_experiment") or {}).get("checks_passed"),
        },
        "native_info": {
            "agent_implementation": (payload.get("info", {}).get("agent_info") or {}).get("implementation"),
            "agent_llm": (payload.get("info", {}).get("agent_info") or {}).get("llm"),
            "agent_llm_args": (payload.get("info", {}).get("agent_info") or {}).get("llm_args"),
            "user_implementation": (payload.get("info", {}).get("user_info") or {}).get("implementation"),
            "user_llm": (payload.get("info", {}).get("user_info") or {}).get("llm"),
            "user_llm_args": (payload.get("info", {}).get("user_info") or {}).get("llm_args"),
            "seed": payload.get("info", {}).get("seed"),
            "num_trials": payload.get("info", {}).get("num_trials"),
            "max_steps": payload.get("info", {}).get("max_steps"),
        },
        "experiment_summary": summary,
        "group_variance": {
            "tasks": len(by_task),
            "simulations": len(simulations),
            "classes": {"mixed": mixed, "0/8": all_zero, "8/8": all_one},
            "usable_mixed_groups": mixed,
            "non_zero_variance_groups": mixed,
            "non_all_zero_groups": mixed + all_one,
        },
        "s0_reference_classes": {"0/8": 21, "mixed": 37, "8/8": 16},
        "runtime": runtime,
        "deepseek_usage": deepseek,
        "pilot_projection": {
            "P": 2,
            "K": 8,
            "rollouts_9_steps": 144,
            "rollouts_18_steps": 288,
            "user_simulator_cost_9_steps_usd": (
                user_usage["estimated_cost_usd"] * 144 / len(simulations)
                if user_usage["estimated_cost_usd"] is not None
                else None
            ),
            "user_simulator_cost_18_steps_usd": (
                user_usage["estimated_cost_usd"] * 288 / len(simulations)
                if user_usage["estimated_cost_usd"] is not None
                else None
            ),
            "cost_scope": "user simulator only; NL judge unavailable",
        },
        "per_task": per_task,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pricing-retrieved-at")
    parser.add_argument("--pricing-source")
    parser.add_argument("--input-cache-hit-usd-per-1m", type=float)
    parser.add_argument("--input-cache-miss-usd-per-1m", type=float)
    parser.add_argument("--output-usd-per-1m", type=float)
    args = parser.parse_args()

    prices = (
        args.input_cache_hit_usd_per_1m,
        args.input_cache_miss_usd_per_1m,
        args.output_usd_per_1m,
    )
    if any(value is not None for value in prices) and not all(value is not None for value in prices):
        parser.error("provide all three DeepSeek prices or none")
    pricing = None
    if all(value is not None for value in prices):
        pricing = {
            "model": "deepseek-v4-flash",
            "input_cache_hit_usd_per_1m": args.input_cache_hit_usd_per_1m,
            "input_cache_miss_usd_per_1m": args.input_cache_miss_usd_per_1m,
            "output_usd_per_1m": args.output_usd_per_1m,
            "retrieved_at": args.pricing_retrieved_at,
            "source": args.pricing_source,
        }

    payload = json.loads(args.results.read_text(encoding="utf-8"))
    report = analyze(payload, pricing=pricing)
    report["artifact"] = str(args.results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["group_variance"], ensure_ascii=False, indent=2))
    print(json.dumps(report["runtime"], ensure_ascii=False, indent=2))
    print(json.dumps(report["deepseek_usage"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
