"""Create an immutable, blinded 40-pair answer-postprocessing review package."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any
from urllib.parse import quote

from ecommerce_rag.answer_postprocess import stable_hash


HUMAN_FIELDS = frozenset({"fact_pass", "answer_complete", "contradiction_present", "review_notes"})
SEED = 20260818


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sidecar_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    by_task = {row["task_id"]: row for row in rows}
    if len(rows) != 80 or len(by_task) != 80:
        raise ValueError(f"{path}: expected 80 unique task rows")
    return by_task


def base_rows(path: Path) -> dict[str, dict[str, Any]]:
    conn = sqlite3.connect(f"file:{quote(str(path.resolve()))}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=ON")
    try:
        rows = conn.execute("SELECT task_id, trajectory_json FROM trajectories").fetchall()
    finally:
        conn.close()
    result = {task_id: json.loads(payload) for task_id, payload in rows}
    if len(rows) != 80 or len(result) != 80:
        raise ValueError(f"{path}: expected 80 unique base trajectories")
    return result


def opaque_id(task_id: str, variant: str) -> str:
    digest = hashlib.sha256(f"answer-post-audit:{SEED}:{task_id}:{variant}".encode()).hexdigest()
    return f"R{digest[:15].upper()}"


def display_rank(review_id: str) -> str:
    return hashlib.sha256(f"answer-post-display:{SEED}:{review_id}".encode()).hexdigest()


def immutable_hash(row: dict[str, Any]) -> str:
    return stable_hash({key: value for key, value in row.items() if key not in HUMAN_FIELDS})


def validate_sidecar(task_id: str, row: dict[str, Any], trajectory: dict[str, Any]) -> None:
    expected = {
        "source_trajectory_id": trajectory["trajectory_id"],
        "action_sequence_sha256": stable_hash(trajectory.get("actions") or []),
        "tool_calls_sha256": stable_hash(trajectory.get("tool_calls") or []),
        "terminal_state_sha256": stable_hash(trajectory.get("final_state") or {}),
        "evidence_ledger_sha256": stable_hash(trajectory.get("evidence_ledger") or []),
    }
    failures = [key for key, value in expected.items() if row.get(key) != value]
    if failures:
        raise ValueError(f"{task_id}: sidecar immutability mismatch: {', '.join(failures)}")
    if row.get("error") or row.get("truncated"):
        raise ValueError(f"{task_id}: generation error or truncation in sidecar")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--base-store", type=Path, required=True)
    parser.add_argument("--shadow", type=Path, required=True)
    parser.add_argument("--terminal-grounded", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output_paths = {
        "review": args.output_dir / "answer_postprocess_blind_audit_v1_review.jsonl",
        "mapping": args.output_dir / "answer_postprocess_blind_audit_v1_mapping.jsonl",
        "manifest": args.output_dir / "answer_postprocess_blind_audit_v1_package_manifest.json",
    }
    existing = [str(path) for path in output_paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite blinded audit artifacts: {existing}")

    selection = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
    selected = [row["task_id"] for row in selection["selected_tasks"]]
    if len(selected) != 40 or len(set(selected)) != 40:
        raise ValueError("selection manifest must contain 40 unique tasks")
    tasks = {row["task_id"]: row for row in read_jsonl(args.tasks)}
    base = base_rows(args.base_store)
    shadow = sidecar_rows(args.shadow)
    grounded = sidecar_rows(args.terminal_grounded)
    if set(base) != set(shadow) or set(base) != set(grounded):
        raise ValueError("base and paired sidecars must cover the same 80 tasks")

    review_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    for task_id in selected:
        trajectory, shadow_row, grounded_row = base[task_id], shadow[task_id], grounded[task_id]
        validate_sidecar(task_id, shadow_row, trajectory)
        validate_sidecar(task_id, grounded_row, trajectory)
        draft = str(trajectory.get("final_answer") or "")
        if shadow_row.get("final_answer") != draft or shadow_row.get("draft_answer") != draft:
            raise ValueError(f"{task_id}: shadow answer differs from the frozen draft")
        if grounded_row.get("draft_answer") != draft:
            raise ValueError(f"{task_id}: grounded sidecar uses a different frozen draft")
        if shadow_row.get("eligible") != grounded_row.get("eligible"):
            raise ValueError(f"{task_id}: diagnostic processing changed eligibility")
        variants = (("base", draft), ("terminal_grounded", grounded_row.get("final_answer", "")))
        for variant, answer in variants:
            review_id = opaque_id(task_id, variant)
            review_rows.append({
                "review_id": review_id,
                "user_goal": tasks[task_id]["user_goal"],
                "evidence": trajectory.get("evidence_ledger") or [],
                "answer": answer,
                "fact_pass": "",
                "answer_complete": "",
                "contradiction_present": "",
                "review_notes": "",
            })
            mapping_rows.append({
                "review_id": review_id,
                "task_id": task_id,
                "variant": variant,
                "eligible": bool(grounded_row.get("eligible")),
                "ineligible_reason": grounded_row.get("ineligible_reason"),
            })
    if len(review_rows) != 80 or len({row["review_id"] for row in review_rows}) != 80:
        raise ValueError("blinded audit must contain 80 unique answer records")
    review_rows.sort(key=lambda row: display_rank(row["review_id"]))
    mapping_rows.sort(key=lambda row: row["review_id"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_paths["review"].write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in review_rows),
        encoding="utf-8",
    )
    output_paths["mapping"].write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in mapping_rows),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "name": "answer_postprocess_blind_audit_v1_package",
        "selection_manifest_sha256": sha256(args.selection_manifest),
        "task_contract_sha256": sha256(args.tasks),
        "base_store_sha256": sha256(args.base_store),
        "shadow_sidecar_sha256": sha256(args.shadow),
        "terminal_grounded_sidecar_sha256": sha256(args.terminal_grounded),
        "review_file_sha256_before_labels": sha256(output_paths["review"]),
        "mapping_file_sha256": sha256(output_paths["mapping"]),
        "task_groups": 40,
        "blinded_answers": 80,
        "display_seed": SEED,
        "human_fields": sorted(HUMAN_FIELDS),
        "allowed_labels": ["true", "false", "unclear"],
        "variant_hidden_from_review": True,
        "pairing_hidden_from_review": True,
        "automatic_verifier_hidden_from_review": True,
        "gold_expectations_hidden_from_review": True,
        "immutable_review_row_hashes": {
            row["review_id"]: immutable_hash(row) for row in review_rows
        },
    }
    output_paths["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"task_groups": 40, "blinded_answers": 80}, ensure_ascii=False))


if __name__ == "__main__":
    main()
