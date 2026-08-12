# -*- coding: utf-8 -*-
"""Compile the formal M1 structured Retail task set from τ³ DB entities."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ecommerce_rag.retail_task_compiler import compile_m1_dataset, coverage_from_blueprints
from ecommerce_rag.retail_task_compiler.contamination import load_test_signatures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tau-root",
        type=Path,
        default=Path(r"E:\cv_codex\external\tau2-bench"),
    )
    parser.add_argument(
        "--signatures",
        type=Path,
        default=Path("docs/tau3_retail_test40_structure_signatures.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/compiled_retail_m1"),
    )
    parser.add_argument("--instances-per-structure", type=int, default=5)
    args = parser.parse_args(argv)

    signatures = load_test_signatures(args.signatures)
    db_path = args.tau_root / "data" / "tau2" / "domains" / "retail" / "db.json"
    payload = compile_m1_dataset(
        db_path=db_path,
        test_signatures=signatures,
        instances_per_structure=args.instances_per_structure,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks_path = args.output_dir / "tasks.json"
    split_path = args.output_dir / "split_tasks.json"
    report_path = args.output_dir / "compiler_report.json"
    coverage_path = args.output_dir / "coverage.json"
    contamination_path = args.output_dir / "contamination_report.json"

    tasks_path.write_text(
        json.dumps(payload["tasks"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    split_path.write_text(
        json.dumps(payload["split_tasks"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    blueprints = [item["blueprint"] for item in payload["instances"]]
    coverage = coverage_from_blueprints(blueprints).to_dict()
    coverage_path.write_text(
        json.dumps(coverage, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    contamination_path.write_text(
        json.dumps(
            {
                "test40_contamination_count": payload["contamination_count"],
                "rejected_structures": payload["rejected_structures"],
                "accepted_instances": payload["accepted_instances"],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    report = {
        key: value
        for key, value in payload.items()
        if key not in {"tasks", "instances"}
    }
    report["tasks_path"] = str(tasks_path)
    report["coverage"] = coverage
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "structure_count": payload["structure_count"],
        "behavior_family_count": payload["behavior_family_count"],
        "accepted_instances": payload["accepted_instances"],
        "split_counts": payload["split_counts"],
        "contamination_count": payload["contamination_count"],
        "rejected_structures": len(payload["rejected_structures"]),
        "output_dir": str(args.output_dir),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if payload["accepted_instances"] < 250:
        return 2
    if payload["contamination_count"] != 0:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
