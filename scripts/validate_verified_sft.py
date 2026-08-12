"""Fail-closed validation for a versioned verified-ecommerce SFT dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


ALLOWED_ROLES = {"system", "user", "assistant", "tool_call", "tool_response"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--min-train-records", type=int, default=400)
    parser.add_argument("--max-train-records", type=int, default=1200)
    args = parser.parse_args()

    manifest = json.loads((args.dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("source_split") != "train":
        raise SystemExit("source_split must be train")
    if not manifest.get("require_process_audit"):
        raise SystemExit("formal dataset must require process audit")
    if manifest.get("teacher_usage_rights") not in {
        "approved_open_weights",
        "approved_terms",
    }:
        raise SystemExit("teacher usage-rights gate not satisfied")

    signatures: dict[str, set[str]] = {}
    tasks: dict[str, set[str]] = {}
    counts: Counter[str] = Counter()
    for split in ("train", "dev", "held_out"):
        signatures[split] = set()
        tasks[split] = set()
        path = args.dataset_dir / f"{split}.jsonl"
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                messages = record.get("messages") or []
                if not messages or messages[0].get("role") != "system":
                    raise SystemExit(f"{path}:{line_number}: system message must be first")
                roles = {message.get("role") for message in messages}
                if not roles <= ALLOWED_ROLES:
                    raise SystemExit(f"{path}:{line_number}: invalid roles {roles}")
                if "tool_call" not in roles or "tool_response" not in roles:
                    raise SystemExit(f"{path}:{line_number}: missing real tool interaction")
                tools = json.loads(record.get("tools") or "[]")
                if not isinstance(tools, list) or not tools:
                    raise SystemExit(f"{path}:{line_number}: tools must be a non-empty list")
                provenance = record.get("provenance") or {}
                if provenance.get("split") != split:
                    raise SystemExit(f"{path}:{line_number}: split provenance mismatch")
                if provenance.get("official_reward") != 1.0:
                    raise SystemExit(f"{path}:{line_number}: official reward is not 1")
                if not provenance.get("process_audit_version"):
                    raise SystemExit(f"{path}:{line_number}: process audit missing")
                signature = (provenance.get("structure") or {}).get("signature_hash")
                if not signature:
                    raise SystemExit(f"{path}:{line_number}: structure signature missing")
                signatures[split].add(signature)
                tasks[split].add(str(provenance.get("task_id")))
                counts[split] += 1

    if not (args.min_train_records <= counts["train"] <= args.max_train_records):
        raise SystemExit(
            f"train records {counts['train']} outside "
            f"[{args.min_train_records}, {args.max_train_records}]"
        )
    if not counts["dev"] or not counts["held_out"]:
        raise SystemExit("dev and held_out must both be non-empty")
    if signatures["train"] & signatures["dev"]:
        raise SystemExit("train/dev structure leakage")
    if signatures["train"] & signatures["held_out"]:
        raise SystemExit("train/held_out structure leakage")
    if signatures["dev"] & signatures["held_out"]:
        raise SystemExit("dev/held_out structure leakage")
    if tasks["train"] & tasks["dev"]:
        raise SystemExit("train/dev task leakage")
    if tasks["train"] & tasks["held_out"]:
        raise SystemExit("train/held_out task leakage")
    if tasks["dev"] & tasks["held_out"]:
        raise SystemExit("dev/held_out task leakage")
    print(json.dumps({"valid": True, "counts": counts}, default=dict, indent=2))


if __name__ == "__main__":
    main()
