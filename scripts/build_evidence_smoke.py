"""Select three deterministic calibration tasks per Phase-A category."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from ecommerce_rag.harness import load_tasks
from dataclasses import asdict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--per-category", type=int, default=3)
    args = parser.parse_args()
    selected = []
    counts: Counter[str] = Counter()
    for task in load_tasks(args.tasks):
        if task.split != "calibration" or counts[task.category] >= args.per_category:
            continue
        selected.append(asdict(task))
        counts[task.category] += 1
    expected_count = 8 * args.per_category
    if len(selected) != expected_count or set(counts.values()) != {args.per_category}:
        raise ValueError(f"expected {expected_count} tasks, {args.per_category} per category; got {dict(counts)}")
    manifest = {
        "schema_version": 2,
        "contract": "evidence_phase_a_tasks_v2",
        "task_count": len(selected),
        "task_ids": [task["task_id"] for task in selected],
        "categories": dict(sorted(counts.items())),
        "variants": ["base", "evidence_verify", "evidence_verify_repair"],
        "diagnostic_task_ids": [task["task_id"] for task in selected if task["metadata"].get("diagnostic_only")],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in selected), encoding="utf-8")
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
