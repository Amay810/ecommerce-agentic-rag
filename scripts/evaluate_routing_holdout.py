"""Evaluate a public routing holdout without exposing TaskSpec gold fields."""

import argparse
import json
from collections import Counter
from pathlib import Path

from ecommerce_rag.domain import AgentObservation
from ecommerce_rag.harness import RulePolicy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    policy = RulePolicy()
    details = []
    for row in rows:
        observation = AgentObservation(current_message=row["message"],
                                       history=[{"role": "user", "content": row["message"]}],
                                       session={"user_id": "U0001"}, tool_schemas=[])
        action = policy.act(observation)
        details.append({**row, "observed_tool": action.tool_name, "action_type": action.action_type,
                        "correct": action.tool_name == row["expected_tool"]})
    failures = [row for row in details if not row["correct"]]
    summary = {"rows": len(rows), "accuracy": sum(row["correct"] for row in details) / max(1, len(rows)),
               "failures": len(failures), "failure_kinds": dict(Counter(row["kind"] for row in failures))}
    report = {"summary": summary, "details": details}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
