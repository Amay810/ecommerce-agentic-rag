"""Emit deterministic process-audit JSONL for τ³ Retail results."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from ecommerce_rag.process_audit import audit_contract_from_task, audit_simulation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--tasks",
        type=Path,
        help="Optional compiled task file; enables structure- and target-aware auditing.",
    )
    args = parser.parse_args()

    payload = json.loads(args.results.read_text(encoding="utf-8"))
    simulations = payload.get("simulations") or []
    contracts = {}
    if args.tasks:
        task_payload = json.loads(args.tasks.read_text(encoding="utf-8"))
        task_rows = task_payload.get("tasks") if isinstance(task_payload, dict) else task_payload
        for task in task_rows or []:
            contract = audit_contract_from_task(task)
            if contract is not None:
                contracts[str(task.get("id") or "")] = contract
    args.output.parent.mkdir(parents=True, exist_ok=True)
    violations: Counter[str] = Counter()
    compliant = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for simulation in simulations:
            result = audit_simulation(simulation, contracts.get(str(simulation.get("task_id") or "")))
            handle.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
            compliant += result.process_compliant
            violations.update(result.violations)
    summary = {
        "audit_version": "tau3-retail-process-v2",
        "total": len(simulations),
        "process_compliant": compliant,
        "noncompliant": len(simulations) - compliant,
        "violations": dict(violations.most_common()),
        "source_results": str(args.results),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
