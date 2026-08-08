"""Audit pinned tau3 task sets: split sizes, reward bases, judge exposure, tool mix.

P0 provenance artifact. Reads only; writes a JSON report next to the docs so the
numbers used in planning can be regenerated instead of quoted from memory.

Usage:
    python scripts/audit_tau3_retail_split.py --tau2-root E:/cv_codex/external/tau2-bench
    python scripts/audit_tau3_retail_split.py --domain retail airline telecom
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

DEFAULT_TAU2_ROOT = Path("E:/cv_codex/external/tau2-bench")
DOMAINS_SUBPATH = Path("data/tau2/domains")


def load_domain(tau2_root: Path, domain: str) -> tuple[list[dict], dict[str, list[str]]]:
    domain_dir = tau2_root / DOMAINS_SUBPATH / domain
    tasks = json.loads((domain_dir / "tasks.json").read_text(encoding="utf-8"))
    splits = json.loads((domain_dir / "split_tasks.json").read_text(encoding="utf-8"))
    return tasks, splits


def summarize(tasks: list[dict], task_ids: list[str] | None) -> dict:
    if task_ids is not None:
        wanted = set(task_ids)
        subset = [t for t in tasks if t["id"] in wanted]
    else:
        subset = tasks

    reward_bases = Counter()
    write_actions = Counter()
    judge_gated = 0
    db_gated = 0
    action_counts = []

    for task in subset:
        criteria = task.get("evaluation_criteria") or {}
        basis = tuple(sorted(criteria.get("reward_basis") or []))
        reward_bases[basis] += 1

        if "DB" in basis:
            db_gated += 1
        # NL_ASSERTION in the basis is inert when the task carries no assertions:
        # NLAssertionsEvaluator short-circuits to reward 1.0 and never calls the LLM.
        if "NL_ASSERTION" in basis and criteria.get("nl_assertions"):
            judge_gated += 1

        actions = criteria.get("actions") or []
        action_counts.append(len(actions))
        for action in actions:
            write_actions[action["name"]] += 1

    return {
        "num_tasks": len(subset),
        "reward_bases": {"+".join(k) or "(none)": v for k, v in sorted(reward_bases.items())},
        "db_gated_tasks": db_gated,
        "judge_gated_tasks": judge_gated,
        "judge_free_tasks": len(subset) - judge_gated,
        "reference_actions_total": sum(action_counts),
        "reference_actions_mean": round(sum(action_counts) / len(subset), 2) if subset else 0.0,
        "reference_actions_max": max(action_counts) if action_counts else 0,
        "action_name_histogram": dict(write_actions.most_common()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau2-root", type=Path, default=DEFAULT_TAU2_ROOT)
    parser.add_argument("--domain", nargs="+", default=["retail", "airline", "telecom"])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/tau3_split_audit.json"),
    )
    args = parser.parse_args()

    report = {"tau2_root": str(args.tau2_root), "domains": {}}
    for domain in args.domain:
        tasks, splits = load_domain(args.tau2_root, domain)
        report["domains"][domain] = {
            "all": summarize(tasks, None),
            "splits": {name: summarize(tasks, ids) for name, ids in splits.items()},
            "split_sizes": {name: len(ids) for name, ids in splits.items()},
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for domain, data in report["domains"].items():
        print(f"{domain}: {data['split_sizes']}")
        for split, stats in data["splits"].items():
            print(
                f"  {split:6s} n={stats['num_tasks']:4d} "
                f"judge_gated={stats['judge_gated_tasks']:4d} "
                f"ref_actions_mean={stats['reference_actions_mean']}"
            )
    print(f"\nFull report: {args.out}")


if __name__ == "__main__":
    main()
