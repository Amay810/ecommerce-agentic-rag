"""Run Phase-A variants while loading the LLM and retriever only once."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from ecommerce_rag.domain import TaskSpec
from ecommerce_rag.evidence_policy import EvidenceGroundedPolicy
from ecommerce_rag.harness import HarnessRunner, TrajectoryStore, load_tasks, summarize
from ecommerce_rag.hybrid_retriever import HybridRetriever
from ecommerce_rag.llm_policy import LLMPolicy


VARIANTS = ("base", "evidence_verify", "evidence_verify_repair")


def _resource_metrics(trajectory) -> dict:
    prompt_tokens = completion_tokens = generations = 0
    for call in trajectory.model_calls:
        trace = call.get("llm") or {}
        traces = [trace]
        if isinstance(trace.get("repair_llm"), dict):
            traces.append(trace["repair_llm"])
        for item in traces:
            for attempt in item.get("attempts", []):
                generations += 1
                prompt_tokens += int(attempt.get("prompt_tokens") or 0)
                completion_tokens += int(attempt.get("completion_tokens") or 0)
    return {
        "model_generations": generations,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "tool_calls": len(trajectory.tool_calls),
        "verification_spans": len(trajectory.verification_spans),
        "repair_spans": len(trajectory.repair_spans),
    }


def _refuse_existing(paths: list[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing Phase-A artifacts: {existing}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--split", choices=("calibration", "dev"))
    parser.add_argument("--store-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()

    tasks = load_tasks(args.tasks)
    if args.split:
        tasks = [task for task in tasks if task.split == args.split]
    if not tasks:
        raise ValueError("no tasks selected")
    stores = {variant: args.store_dir / f"{args.run_name}_{variant}.sqlite" for variant in VARIANTS}
    reports = {variant: args.report_dir / f"{args.run_name}_{variant}_report.json" for variant in VARIANTS}
    _refuse_existing([*stores.values(), *reports.values()])
    args.store_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    base = LLMPolicy.from_env()
    policies = {
        "base": base,
        "evidence_verify": EvidenceGroundedPolicy(
            base.generate, generator_meta=base.generator_meta, repair=False),
        "evidence_verify_repair": EvidenceGroundedPolicy(
            base.generate, generator_meta=base.generator_meta, repair=True),
    }
    retriever = HybridRetriever(args.index)
    run_manifest = {
        "schema_version": 1, "run_name": args.run_name, "task_count": len(tasks),
        "task_ids": [task.task_id for task in tasks], "variants": list(VARIANTS),
        "generator": base.generator_meta,
    }

    for variant in VARIANTS:
        runner = HarnessRunner(args.db, retriever, policies[variant])
        store = TrajectoryStore(stores[variant])
        results, details = [], []
        for task in tasks:
            repeated = TaskSpec(**asdict(task))
            trajectory, result = runner.run(repeated)
            store.save(trajectory, result)
            results.append(result)
            details.append({"trajectory_id": trajectory.trajectory_id, **result.to_dict(),
                            **_resource_metrics(trajectory)})
        report = {
            "schema_version": 1, "variant": variant, "policy": type(policies[variant]).__name__,
            "task_ids": run_manifest["task_ids"], "summary": summarize(results, repeats=1),
            "details": details,
        }
        reports[variant].write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"variant": variant, **report["summary"]}, ensure_ascii=False))

    manifest_path = args.report_dir / f"{args.run_name}_run_manifest.json"
    manifest_path.write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
