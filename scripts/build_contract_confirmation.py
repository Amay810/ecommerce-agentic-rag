"""Build the seven-task confirmation set from the immutable v2 benchmark."""

import argparse
import copy
import json
from pathlib import Path


TASK_IDS = (
    "v2_dev_policy_02",
    "v2_dev_policy_07",
    "v2_locked_policy_12",
    "v2_locked_policy_17",
    "v2_dev_product_qa_02",
    "v2_dev_product_qa_07",
    "v2_locked_product_qa_14",
)


def select(tasks: list[dict]) -> list[dict]:
    by_id = {task["task_id"]: task for task in tasks}
    missing = [task_id for task_id in TASK_IDS if task_id not in by_id]
    if missing:
        raise ValueError(f"missing confirmation tasks: {', '.join(missing)}")
    selected = [copy.deepcopy(by_id[task_id]) for task_id in TASK_IDS]
    for task in selected:
        if task["category"] == "product_qa":
            task["allowed_tools"] = ["search_catalog", "get_product"]
            task["expected_tool_sequence"] = ["search_catalog", "get_product"]
            task.setdefault("answer_expectations", {})["required_fact_keys"] = ["product.evidence.*"]
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    chosen = select([json.loads(line) for line in args.tasks.read_text(encoding="utf-8").splitlines() if line.strip()])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(task, ensure_ascii=False) + "\n" for task in chosen), encoding="utf-8")
    args.manifest.write_text(json.dumps({
        "schema_version": 3,
        "contract": "contract_confirmation_v3",
        "source": str(args.tasks).replace("\\", "/"),
        "count": len(chosen),
        "scenarios": {
            task["task_id"]: {
                "task_id": task["task_id"],
                "category": task["category"],
                "handoff_expected": bool(task.get("metadata", {}).get("handoff_expected")),
                "expected_tool_sequence": task.get("expected_tool_sequence", []),
            }
            for task in chosen
        },
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
