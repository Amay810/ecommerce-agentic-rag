"""Run shadow or terminal-grounded processing over a frozen base store."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any
from urllib.parse import quote

from ecommerce_rag.answer_postprocess import AnswerPostprocessor, PostprocessResult, stable_hash
from ecommerce_rag.llm_policy import LLMPolicy


def open_readonly(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{quote(str(path.resolve()))}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=ON")
    return conn


def load_rows(path: Path) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    conn = open_readonly(path)
    try:
        rows = conn.execute("SELECT trajectory_json, grade_json FROM trajectories ORDER BY task_id").fetchall()
        return [(json.loads(trajectory), json.loads(grade)) for trajectory, grade in rows]
    finally:
        conn.close()


def passthrough(mode: str, draft: str, reason: str) -> PostprocessResult:
    return PostprocessResult(mode, False, reason, draft, draft, False)


def ratio(n: int, d: int) -> dict[str, Any]:
    return {"numerator": n, "denominator": d, "rate": n / d if d else None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-store", type=Path, required=True)
    parser.add_argument("--mode", choices=("shadow", "terminal_grounded"), required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=80)
    parser.add_argument("--task-manifest", type=Path)
    args = parser.parse_args()
    if args.output_jsonl.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite answer-postprocess artifacts")
    rows = load_rows(args.base_store)
    if args.task_manifest:
        manifest = json.loads(args.task_manifest.read_text(encoding="utf-8"))
        wanted = set(manifest["task_ids"])
        rows = [row for row in rows if row[0]["task_id"] in wanted]
        if {row[0]["task_id"] for row in rows} != wanted:
            raise ValueError("task manifest does not exactly match the frozen base store")
    if len(rows) != args.expected_count:
        raise ValueError(f"expected {args.expected_count} frozen trajectories, got {len(rows)}")

    generation_config: dict[str, Any] = {}
    generator = None
    if args.mode == "terminal_grounded":
        policy = LLMPolicy.from_env()
        generator = policy.generate
        generation_config = {
            **policy.generator_meta, "model_family": "Qwen3-4B-Instruct-2507",
            "enable_thinking": False, "do_sample": False, "temperature": 0,
            "max_new_tokens": 512, "chat_template": "model_default",
        }
        if not str(policy.generator_meta.get("model", "")).endswith("Qwen3-4B-Instruct-2507"):
            raise ValueError("terminal-grounded v1 requires Qwen3-4B-Instruct-2507")
    processor = AnswerPostprocessor(generator, generation_config=generation_config)
    output_rows = []
    failures = []
    for trajectory, grade in rows:
        reason = processor.eligibility(trajectory, grade)
        draft = str(trajectory.get("final_answer") or "")
        if reason:
            result = passthrough(args.mode, draft, reason)
        else:
            result = processor.process(draft, trajectory.get("evidence_ledger") or [],
                                       trajectory.get("messages") or [], args.mode)
        if result.error:
            failures.append({"task_id": trajectory["task_id"], "error": result.error})
        record = {
            "schema_version": 1, "mode": args.mode,
            "task_id": trajectory["task_id"], "source_trajectory_id": trajectory["trajectory_id"],
            "action_sequence_sha256": stable_hash(trajectory.get("actions") or []),
            "tool_calls_sha256": stable_hash(trajectory.get("tool_calls") or []),
            "terminal_state_sha256": stable_hash(trajectory.get("final_state") or {}),
            "evidence_ledger_sha256": stable_hash(trajectory.get("evidence_ledger") or []),
            "generation_config": generation_config if args.mode == "terminal_grounded" else None,
            **result.to_dict(),
        }
        output_rows.append(record)
    if failures:
        raise RuntimeError(f"generation failed closed for {len(failures)} rows: {failures[:3]}")
    eligible = [row for row in output_rows if row["eligible"]]
    report = {
        "schema_version": 1, "mode": args.mode, "source_store": args.base_store.name,
        "all_tasks": len(output_rows), "eligible": ratio(len(eligible), len(output_rows)),
        "pass_through": ratio(len(output_rows) - len(eligible), len(output_rows)),
        "ineligible_reasons": {reason: sum(row["ineligible_reason"] == reason for row in output_rows)
                               for reason in sorted({row["ineligible_reason"] for row in output_rows
                                                     if row["ineligible_reason"]})},
        "changed_answers_eligible": ratio(sum(row["changed"] for row in eligible), len(eligible)),
        "generation_config_hash": stable_hash(generation_config) if generation_config else None,
        "action_mutation_allowed": False, "handoff_mutation_allowed": False,
    }
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                                         for row in output_rows), encoding="utf-8")
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
