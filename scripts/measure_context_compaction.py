"""Measure provider-facing history compaction on saved multi-turn trajectories."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from ecommerce_rag.context_compaction import compact_history


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    return sorted(values)[max(0, int(len(values) * fraction) - 1)]


def _history(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    pending_tool: str | None = None
    for message in messages:
        role = message.get("role")
        if role == "assistant" and message.get("tool_calls"):
            calls = message.get("tool_calls") or []
            pending_tool = (calls[0].get("name") or (calls[0].get("function") or {}).get("name")) if calls else None
            history.append({"role": "assistant", "content": message.get("content") or ""})
        elif role == "tool":
            history.append({"role": "tool", "name": pending_tool, "content": message.get("content") or ""})
            pending_tool = None
        elif role in {"user", "assistant"}:
            history.append({"role": role, "content": message.get("content") or ""})
    return history


def measure(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    simulations = payload.get("simulations") or []
    rows: list[dict[str, Any]] = []
    for simulation in simulations:
        source_messages = simulation.get("messages") or []
        history = _history(source_messages)
        _, stats = compact_history(history)
        prompt_tokens = completion_tokens = 0
        generation_seconds = 0.0
        agent_calls = 0
        cumulative_raw_history_chars = cumulative_compact_history_chars = 0
        for index, message in enumerate(source_messages):
            if message.get("role") != "assistant" or not message.get("usage"):
                continue
            prefix = _history(source_messages[:index])
            _, prefix_stats = compact_history(prefix)
            cumulative_raw_history_chars += prefix_stats.raw_chars
            cumulative_compact_history_chars += prefix_stats.compact_chars
            usage = message["usage"] or {}
            prompt_tokens += usage.get("prompt_tokens") or 0
            completion_tokens += usage.get("completion_tokens") or 0
            generation_seconds += message.get("generation_time_seconds") or 0.0
            agent_calls += 1
        rows.append({
            "task_id": str(simulation.get("task_id")),
            "trial": simulation.get("trial"),
            **stats.to_dict(),
            "recorded_prompt_tokens": prompt_tokens,
            "recorded_completion_tokens": completion_tokens,
            "agent_calls": agent_calls,
            "agent_generation_seconds": generation_seconds,
            "cumulative_raw_history_chars": cumulative_raw_history_chars,
            "cumulative_compact_history_chars": cumulative_compact_history_chars,
        })
    reductions = [row["reduction_ratio"] for row in rows]
    return {
        "source": str(path),
        "simulations": len(rows),
        "measurement_boundary": (
            "Character reduction is exact for stored message content. Recorded prompt tokens and latency "
            "are the uncompressed historical baseline; a live paired run is required for post-compaction tokens, latency, and success."
        ),
        "summary": {
            "raw_chars_total": sum(row["raw_chars"] for row in rows),
            "compact_chars_total": sum(row["compact_chars"] for row in rows),
            "character_reduction_ratio": (
                1.0 - sum(row["compact_chars"] for row in rows) / sum(row["raw_chars"] for row in rows)
                if rows and sum(row["raw_chars"] for row in rows) else 0.0
            ),
            "per_simulation_reduction_mean": statistics.mean(reductions) if reductions else 0.0,
            "per_simulation_reduction_p50": statistics.median(reductions) if reductions else 0.0,
            "per_simulation_reduction_p95": _percentile(reductions, 0.95),
            "recorded_prompt_tokens_total": sum(row["recorded_prompt_tokens"] for row in rows),
            "recorded_prompt_tokens_mean": statistics.mean(
                [row["recorded_prompt_tokens"] for row in rows]) if rows else 0.0,
            "recorded_agent_generation_seconds_mean": statistics.mean(
                [row["agent_generation_seconds"] for row in rows]) if rows else 0.0,
            "cumulative_history_chars_total": sum(row["cumulative_raw_history_chars"] for row in rows),
            "cumulative_compact_history_chars_total": sum(
                row["cumulative_compact_history_chars"] for row in rows),
            "cumulative_history_character_reduction_ratio": (
                1.0 - sum(row["cumulative_compact_history_chars"] for row in rows)
                / sum(row["cumulative_raw_history_chars"] for row in rows)
                if rows and sum(row["cumulative_raw_history_chars"] for row in rows) else 0.0
            ),
        },
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = measure(args.results)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
