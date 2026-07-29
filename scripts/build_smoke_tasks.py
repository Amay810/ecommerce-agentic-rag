# -*- coding: utf-8 -*-
"""Select a small covering smoke set from the validated task file.

Picking existing tasks rather than authoring new ones keeps the expected terminal
states, metadata and user-simulator behaviour that the full benchmark already
exercises; only the size changes. Selection is deterministic — first match per
scenario in file order — so the smoke set is reproducible.

Scenarios, chosen so a pass means the whole loop works end to end:

    retrieval        a read tool that goes through the retriever
    recommend        retrieval carrying a constraint
    compare          multi-argument tool call
    policy           routing to a non-order read tool
    order_query      identity check, single write-free tool
    return_allowed   multi-turn: eligibility, then explicit confirmation, then write
    return_blocked   eligibility fails; the database must not change
    safety           prompt injection; must refuse and escalate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _is_return_allowed(task: dict) -> bool:
    for state in (task.get("expected_state") or {}).values():
        if state.get("return_status") == "requested":
            return True
    return False


SELECTORS: dict[str, callable] = {
    "retrieval": lambda t: t["category"] == "product_qa",
    "recommend": lambda t: t["category"] == "recommend",
    "compare": lambda t: t["category"] == "compare",
    "policy": lambda t: t["category"] == "policy",
    "order_query": lambda t: t["category"] == "order_query",
    "return_allowed": lambda t: t["category"] == "return" and _is_return_allowed(t),
    "return_blocked": lambda t: t["category"] == "return" and not _is_return_allowed(t),
    "safety": lambda t: t["category"] == "safety",
}


def select(tasks: list[dict], split: str | None = None) -> dict[str, dict]:
    pool = [t for t in tasks if split is None or t.get("split") == split]
    chosen: dict[str, dict] = {}
    for scenario, matches in SELECTORS.items():
        picked = next((t for t in pool if matches(t)), None)
        if picked is None:
            raise SystemExit(f"no task matches smoke scenario {scenario!r}")
        chosen[scenario] = picked
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, default=Path("ecommerce_rag/data/harness_tasks_v2.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("ecommerce_rag/data/harness_smoke.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("ecommerce_rag/data/harness_smoke_manifest.json"))
    parser.add_argument("--split", default="dev", help="draw from this split; smoke must not consume locked")
    args = parser.parse_args()

    tasks = [json.loads(x) for x in args.tasks.read_text(encoding="utf-8").splitlines() if x.strip()]
    chosen = select(tasks, args.split)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(task, ensure_ascii=False) + "\n" for task in chosen.values()), encoding="utf-8")
    args.manifest.write_text(json.dumps({
        "source": str(args.tasks), "split": args.split, "count": len(chosen),
        "scenarios": {scenario: {"task_id": task["task_id"], "category": task["category"],
                                 "expects_write": _is_return_allowed(task),
                                 "handoff_expected": bool((task.get("metadata") or {}).get("handoff_expected"))}
                      for scenario, task in chosen.items()},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"written": str(args.output), "scenarios": list(chosen)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
