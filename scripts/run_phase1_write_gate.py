"""Phase 1 Base measurement: ask_user vs premature write, split by missing class."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from ecommerce_rag.harness import HarnessRunner, RulePolicy
from ecommerce_rag.legacy_closure_benchmark import clone_database
from ecommerce_rag.phase1_write_gate import (
    DEFAULT_REPEATS,
    AlwaysAskPolicy,
    AlwaysWritePolicy,
    aggregate_repeats,
    build_phase1_probes,
    go_nogo,
    prepare_phase1_database,
    score_trajectory,
    summarize_repeat,
    validate_probe_catalog,
)


def _policy(name: str):
    if name == "always_ask":
        return AlwaysAskPolicy()
    if name == "always_write":
        return AlwaysWritePolicy(confirmed=False)
    if name == "rule":
        return RulePolicy()
    if name == "native":
        from ecommerce_rag.native_tool_policy import NativeToolPolicy
        return NativeToolPolicy.from_env()
    raise ValueError(f"unknown policy: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--policy",
        choices=("native", "rule", "always_ask", "always_write"),
        default="native",
    )
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    probes = build_phase1_probes()
    catalog = validate_probe_catalog(probes)
    if args.check_only:
        print(json.dumps({"ok": True, **catalog, "repeats": args.repeats}, ensure_ascii=False, indent=2))
        return

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    policy = _policy(args.policy)
    generator = getattr(policy, "generator_meta", {"policy": args.policy})
    pristine = prepare_phase1_database(probes, args.output_dir / "pristine.sqlite")

    records: list[dict[str, Any]] = []
    repeat_summaries: list[dict[str, Any]] = []
    for repeat in range(args.repeats):
        repeat_rows: list[dict[str, Any]] = []
        for probe in probes:
            spec = replace(probe.to_task_spec(), seed=probe.seed + repeat * 1_000_003)
            database = clone_database(
                pristine,
                args.output_dir / "db" / f"r{repeat}" / f"{probe.task_id}.sqlite",
            )
            trajectory, _grade = HarnessRunner(
                database,
                policy=policy,
                max_steps=6,
            ).run(spec)
            row = score_trajectory(probe, trajectory, repeat=repeat)
            row["seed"] = spec.seed
            records.append(row)
            repeat_rows.append(row)
        repeat_summaries.append(summarize_repeat(repeat_rows))

    aggregate = aggregate_repeats(repeat_summaries)
    decisions = go_nogo(aggregate)
    summary = {
        "catalog": catalog,
        "policy": args.policy,
        "generator": generator,
        "runtime": "system-v1" if args.policy == "native" else args.policy,
        "repeats": args.repeats,
        "per_repeat": repeat_summaries,
        "aggregate": aggregate,
        "go_nogo": decisions,
        "note": (
            "confirmation_required and verification_code_required are reported "
            "separately; do not merge them into one missing-precondition rate."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output_dir / "records.jsonl").open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "catalog": catalog,
        "go_nogo": decisions,
        "aggregate": aggregate,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
