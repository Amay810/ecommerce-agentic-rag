"""Merge τ³ train + compiled_retail teacher results into one verified M1 dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ecommerce_rag.verified_sft import (
    DatasetBuildConfig,
    build_verified_dataset,
    default_system_prompt,
    file_sha256,
    load_process_audits,
)


def load_tau3_contract(tau_root: Path) -> tuple[str, list[dict], set[str]]:
    sys.path.insert(0, str(tau_root / "src"))
    from tau2.environment.toolkit import get_tool_signatures
    from tau2.registry import registry

    environment = registry.get_env_constructor("retail")()
    signatures = get_tool_signatures(environment.tools)
    tools = [
        {"type": "function", "function": value.model_dump(exclude_none=True)}
        for value in signatures.values()
    ]
    split_path = tau_root / "data" / "tau2" / "domains" / "retail" / "split_tasks.json"
    splits = json.loads(split_path.read_text(encoding="utf-8"))
    return environment.get_policy(), tools, {str(value) for value in splits["train"]}


def _load_results(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tau3-results", type=Path, required=True)
    parser.add_argument("--compiled-results", type=Path, required=True)
    parser.add_argument("--tau3-audit", type=Path, required=True)
    parser.add_argument("--compiled-audit", type=Path, required=True)
    parser.add_argument("--compiled-splits", type=Path, required=True)
    parser.add_argument(
        "--compiled-tasks",
        type=Path,
        default=Path("data/compiled_retail_m1/tasks.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tau-root", type=Path, default=Path(r"E:\cv_codex\external\tau2-bench"))
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument(
        "--teacher-usage-rights",
        choices=("approved_open_weights", "approved_terms"),
        required=True,
    )
    parser.add_argument("--max-per-task", type=int, default=3)
    parser.add_argument("--max-per-structure", type=int, default=8)
    args = parser.parse_args()

    policy, tools, train_ids = load_tau3_contract(args.tau_root)
    compiled_splits = json.loads(args.compiled_splits.read_text(encoding="utf-8"))
    compiled_tasks = json.loads(args.compiled_tasks.read_text(encoding="utf-8"))
    compiled_train_ids = {str(value) for value in compiled_splits.get("train") or []}
    allowed = set(train_ids) | compiled_train_ids
    task_structures = {
        str(task["id"]): {
            "structure_id": str((task.get("provenance") or {})["structure_id"]),
            "signature_hash": str(
                (task.get("provenance") or {})["structure_signature_hash"]
            ),
            "behavior_family": str((task.get("description") or {}).get("purpose") or ""),
        }
        for task in compiled_tasks
    }
    preassigned_task_splits = {
        str(task_id): split
        for split in ("train", "dev", "held_out")
        for task_id in (compiled_splits.get(split) or [])
    }

    tau3_payload = _load_results(args.tau3_results)
    compiled_payload = _load_results(args.compiled_results)
    for simulation in compiled_payload.get("simulations") or []:
        simulation.setdefault("provenance", {})["source"] = "compiled_retail"
    merged = {
        "simulations": list(tau3_payload.get("simulations") or [])
        + list(compiled_payload.get("simulations") or []),
        "source_label": "tau3_retail_train+compiled_retail",
    }
    audits = load_process_audits(args.tau3_audit)
    audits.update(load_process_audits(args.compiled_audit))
    config = DatasetBuildConfig(
        source_split="train",
        teacher_model=args.teacher_model,
        teacher_usage_rights=args.teacher_usage_rights,
        max_per_task=args.max_per_task,
        max_per_structure=args.max_per_structure,
        require_process_audit=True,
        reserved_task_ids=frozenset(
            str(value) for value in (compiled_splits.get("held_out") or [])
        ),
        task_structures=task_structures,
        preassigned_task_splits=preassigned_task_splits,
    )
    manifest = build_verified_dataset(
        payload=merged,
        source_results=args.tau3_results,
        output_dir=args.output_dir,
        system_prompt=default_system_prompt(policy, config),
        tools=tools,
        allowed_task_ids=allowed,
        process_audits=audits,
        config=config,
    )
    manifest["compiled_results"] = str(args.compiled_results)
    manifest["compiled_results_sha256"] = file_sha256(args.compiled_results)
    manifest["tau3_results_sha256"] = file_sha256(args.tau3_results)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
