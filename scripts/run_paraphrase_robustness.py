# -*- coding: utf-8 -*-
"""Run every task in three fixed phrasings and report language robustness.

This is deliberately **not** ``pass^3``.  The harness's repeat loop varies only
``seed``, and nothing downstream consumes randomness, so its three runs are
identical.  Here the database seed, the user goal and the expected terminal
state are held fixed and only the user's wording changes, across three
hand-authored registers (see ``ecommerce_rag.paraphrase``).

The headline number is ``all_three_pass``: the fraction of tasks solved in all
three registers.  It is the worst-of-3 success rate by construction — a task
counts only if its weakest phrasing also passes — so both are reported once,
under a name that says what was actually varied.
"""

from __future__ import annotations

import argparse
import collections
import json
from dataclasses import asdict
from pathlib import Path

from ecommerce_rag.domain import TaskSpec
from ecommerce_rag.harness import HarnessRunner, OraclePolicy, RulePolicy, TrajectoryStore, load_tasks
from ecommerce_rag.paraphrase import PHRASINGS, paraphrase


def _first_tool(trajectory) -> str | None:
    return trajectory.tool_calls[0].name if trajectory.tool_calls else None


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def build_report(runs: dict[str, dict[str, dict]], policy_name: str) -> dict:
    """``runs[task_id][phrasing] -> record``."""
    tasks = sorted(runs)
    by_phrasing = {ph: [runs[t][ph] for t in tasks] for ph in PHRASINGS}

    success_by_phrasing = {ph: _rate(sum(r["success"] for r in rows), len(rows)) for ph, rows in by_phrasing.items()}
    all_three = [t for t in tasks if all(runs[t][ph]["success"] for ph in PHRASINGS)]
    flipped = [t for t in tasks if len({runs[t][ph]["first_tool"] for ph in PHRASINGS}) > 1]

    per_category: dict[str, dict] = {}
    for task_id in tasks:
        category = runs[task_id]["template"]["category"]
        bucket = per_category.setdefault(category, {ph: [0, 0] for ph in PHRASINGS} | {"tasks": 0, "all_three": 0})
        bucket["tasks"] += 1
        bucket["all_three"] += int(all(runs[task_id][ph]["success"] for ph in PHRASINGS))
        for ph in PHRASINGS:
            bucket[ph][0] += int(runs[task_id][ph]["success"])
            bucket[ph][1] += 1
    per_category = {
        category: {
            "tasks": bucket["tasks"],
            "all_three_pass": _rate(bucket["all_three"], bucket["tasks"]),
            **{ph: _rate(bucket[ph][0], bucket[ph][1]) for ph in PHRASINGS},
        }
        for category, bucket in sorted(per_category.items())
    }

    # A regression is a task the template wording solves and a paraphrase does not.
    regressions = []
    for task_id in tasks:
        template = runs[task_id]["template"]
        if not template["success"]:
            continue
        for ph in PHRASINGS[1:]:
            row = runs[task_id][ph]
            if not row["success"]:
                regressions.append({
                    "task_id": task_id, "category": template["category"], "phrasing": ph,
                    "request": row["request"],
                    "template_first_tool": template["first_tool"], "first_tool": row["first_tool"],
                    "failure_type": row["failure_type"], "final_answer": row["final_answer"][:120],
                })

    return {
        "experiment": "paraphrase_robustness",
        "policy": policy_name,
        "phrasings": list(PHRASINGS),
        "note": (
            "Three fixed, hand-authored rephrasings of the same task; database seed, goal and "
            "expected terminal state are unchanged. These are not independent stochastic samples, "
            "so this must not be reported as pass^3."
        ),
        "summary": {
            "tasks": len(tasks),
            "trajectories": len(tasks) * len(PHRASINGS),
            "success_by_phrasing": success_by_phrasing,
            "all_three_pass": _rate(len(all_three), len(tasks)),
            "all_three_pass_is_worst_of_3": True,
            "degradation_vs_template": {
                ph: round(success_by_phrasing[ph] - success_by_phrasing["template"], 4) for ph in PHRASINGS[1:]
            },
            "routing_flip_rate": _rate(len(flipped), len(tasks)),
            "first_tool_by_phrasing": {
                ph: dict(collections.Counter(str(r["first_tool"]) for r in rows).most_common())
                for ph, rows in by_phrasing.items()
            },
            "failure_taxonomy_by_phrasing": {
                ph: dict(collections.Counter(r["failure_type"] for r in rows if r["failure_type"]).most_common())
                for ph, rows in by_phrasing.items()
            },
            "policy_compliance_by_phrasing": {
                ph: _rate(sum(r["policy_compliant"] for r in rows), len(rows)) for ph, rows in by_phrasing.items()
            },
            "terminal_state_accuracy_by_phrasing": {
                ph: _rate(sum(r["terminal_state_match"] for r in rows), len(rows)) for ph, rows in by_phrasing.items()
            },
        },
        "per_category": per_category,
        "regressions": regressions,
        "per_task": {t: {ph: runs[t][ph] for ph in PHRASINGS} for t in tasks},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--store", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--index")
    parser.add_argument("--policy", choices=("rule", "oracle"), default="rule")
    parser.add_argument("--split", choices=("dev", "locked"))
    args = parser.parse_args()

    retriever = None
    if args.index:
        from ecommerce_rag.hybrid_retriever import HybridRetriever
        retriever = HybridRetriever(Path(args.index))

    policy = OraclePolicy() if args.policy == "oracle" else RulePolicy()
    runner, store = HarnessRunner(args.db, retriever, policy), TrajectoryStore(args.store)

    tasks = load_tasks(args.tasks)
    if args.split:
        tasks = [task for task in tasks if task.split == args.split]

    runs: dict[str, dict[str, dict]] = {}
    for index, task in enumerate(tasks, 1):
        for phrasing in PHRASINGS:
            # Only the wording moves: seed, goal, metadata and expected state are held fixed.
            variant = TaskSpec(**{**asdict(task), "user_goal": paraphrase(task, phrasing)})
            trajectory, result = runner.run(variant)
            store.save(trajectory, result)
            runs.setdefault(task.task_id, {})[phrasing] = {
                "category": task.category, "split": task.split, "request": variant.user_goal,
                "trajectory_id": trajectory.trajectory_id, "success": result.success,
                "first_tool": _first_tool(trajectory),
                "tool_sequence": [call.name for call in trajectory.tool_calls],
                "failure_type": result.failure_type, "policy_compliant": result.policy_compliant,
                "terminal_state_match": result.terminal_state_match, "tool_f1": result.tool_f1,
                "handoff_observed": result.handoff_observed, "final_answer": trajectory.final_answer,
            }
        if index % 20 == 0:
            print(f"  {index}/{len(tasks)} tasks")

    report = build_report(runs, args.policy)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
