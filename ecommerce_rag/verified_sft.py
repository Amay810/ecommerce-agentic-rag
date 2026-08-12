"""Build versioned SFT datasets from environment-verified Agent trajectories."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .agent_runtime import RuntimeConfig, build_system_prompt


ACCEPTED_TERMINATIONS = {"user_stop", "agent_stop"}
WRITE_TOOLS = {
    "cancel_pending_order",
    "modify_pending_order_address",
    "modify_pending_order_items",
    "modify_pending_order_payment",
    "modify_user_address",
    "return_delivered_order_items",
    "exchange_delivered_order_items",
}


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tool_calls(messages: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            function = call.get("function") or call
            calls.append(
                {
                    "message_index": index,
                    "name": str(function.get("name") or ""),
                    "arguments": function.get("arguments") or {},
                }
            )
    return calls


def tool_path(messages: Iterable[dict[str, Any]]) -> list[str]:
    return [call["name"] for call in _tool_calls(messages) if call["name"]]


def behavior_family(path: list[str]) -> str:
    writes = [name for name in path if name in WRITE_TOOLS]
    if writes:
        return writes[-1]
    if "transfer_to_human_agents" in path:
        return "handoff"
    return "read_or_clarify"


def structure_signature(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Entity-agnostic trajectory structure used for dedup and split isolation."""

    path = tool_path(messages)
    payload = {
        "behavior_family": behavior_family(path),
        "tool_path": path,
        "has_write": any(name in WRITE_TOOLS for name in path),
        "has_handoff": "transfer_to_human_agents" in path,
        "tool_count": len(path),
    }
    return {**payload, "signature_hash": canonical_hash(payload)}


def split_for_signature(signature_hash: str) -> str:
    """Assign whole structures, never individual paraphrases, to one split."""

    bucket = int(signature_hash[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "dev"
    return "held_out"


def assign_structure_splits(signature_hashes: Iterable[str]) -> dict[str, str]:
    """Deterministically allocate complete structures with non-empty eval splits."""

    unique = sorted(
        set(signature_hashes),
        key=lambda value: hashlib.sha256(f"split-v1:{value}".encode()).hexdigest(),
    )
    count = len(unique)
    if count < 3:
        return {value: "train" for value in unique}
    eval_count = max(1, round(count * 0.1))
    while (2 * eval_count) >= count:
        eval_count -= 1
    dev = set(unique[:eval_count])
    held_out = set(unique[eval_count : 2 * eval_count])
    return {
        value: "dev" if value in dev else "held_out" if value in held_out else "train"
        for value in unique
    }


def assign_isolated_splits(records: list[dict[str, Any]]) -> dict[str, str]:
    """Keep both task ids and structural signatures isolated across splits.

    A task can produce different tool paths, and a path can occur in multiple
    tasks. Treating the resulting task/structure graph as connected components
    prevents either form of leakage.
    """

    parent: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        if parent[value] != value:
            parent[value] = find(parent[value])
        return parent[value]

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for record in records:
        provenance = record["provenance"]
        union(
            f"task:{provenance['task_id']}",
            f"structure:{provenance['structure']['signature_hash']}",
        )

    components: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        root = find(f"task:{record['provenance']['task_id']}")
        components[root].append(record)
    ordered = sorted(
        components.values(),
        key=lambda group: canonical_hash(
            sorted(record["provenance"]["simulation_id"] for record in group)
        ),
    )
    if len(ordered) < 3:
        return {
            record["provenance"]["simulation_id"]: "train"
            for group in ordered
            for record in group
        }

    target = max(1, round(len(records) * 0.1))
    dev_index = min(range(len(ordered)), key=lambda index: (abs(len(ordered[index]) - target), index))
    remaining = [index for index in range(len(ordered)) if index != dev_index]
    held_index = min(
        remaining, key=lambda index: (abs(len(ordered[index]) - target), index)
    )
    allocation: dict[str, str] = {}
    for index, group in enumerate(ordered):
        split = "dev" if index == dev_index else "held_out" if index == held_index else "train"
        for record in group:
            allocation[record["provenance"]["simulation_id"]] = split
    return allocation


def normalize_json_content(value: Any) -> str:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def convert_messages(raw_messages: list[dict[str, Any]]) -> tuple[list[dict], str | None]:
    """Convert τ³ messages to the ms-swift Hermes Agent format."""

    converted: list[dict[str, Any]] = []
    for message in raw_messages:
        role = message.get("role")
        if role == "user":
            if message.get("tool_calls"):
                return [], "user_requestor_tool_call"
            converted.append({"role": "user", "content": message.get("content") or ""})
        elif role == "assistant":
            content = message.get("content") or ""
            calls = message.get("tool_calls") or []
            if content and calls:
                return [], "assistant_content_and_tool_call"
            if content:
                converted.append({"role": "assistant", "content": content})
            for call in calls:
                function = call.get("function") or call
                arguments = function.get("arguments") or {}
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        return [], "tool_arguments_json"
                if not isinstance(arguments, dict):
                    return [], "tool_arguments_not_object"
                converted.append(
                    {
                        "role": "tool_call",
                        "content": json.dumps(
                            {"name": function.get("name"), "arguments": arguments},
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    }
                )
            if not content and not calls:
                return [], "empty_assistant_message"
        elif role == "tool":
            if message.get("error"):
                return [], "tool_error_in_trajectory"
            converted.append(
                {"role": "tool_response", "content": normalize_json_content(message.get("content"))}
            )
        elif role == "system":
            continue
        else:
            return [], f"unknown_role:{role}"
    return converted, None


@dataclass(frozen=True)
class ProcessAudit:
    simulation_id: str
    process_compliant: bool
    violations: tuple[str, ...] = ()
    reviewer: str = ""
    audit_version: str = "process-audit-v1"


def load_process_audits(path: Path | None) -> dict[str, ProcessAudit]:
    if path is None:
        return {}
    audits: dict[str, ProcessAudit] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            simulation_id = str(row.get("simulation_id") or "")
            if not simulation_id:
                raise ValueError(f"audit line {line_number} missing simulation_id")
            if simulation_id in audits:
                raise ValueError(f"duplicate process audit: {simulation_id}")
            audits[simulation_id] = ProcessAudit(
                simulation_id=simulation_id,
                process_compliant=bool(row.get("process_compliant")),
                violations=tuple(str(x) for x in row.get("violations") or []),
                reviewer=str(row.get("reviewer") or ""),
                audit_version=str(row.get("audit_version") or "process-audit-v1"),
            )
    return audits


@dataclass(frozen=True)
class DatasetBuildConfig:
    source_split: str
    teacher_model: str
    teacher_usage_rights: str
    runtime_version: str = "system-v1"
    prompt_version: str = "ecommerce-native-v1"
    max_per_task: int = 3
    max_per_structure: int = 8
    require_process_audit: bool = True
    reserved_task_ids: frozenset[str] = field(default_factory=frozenset)
    task_structures: dict[str, dict[str, Any]] = field(default_factory=dict)
    preassigned_task_splits: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.source_split != "train":
            raise ValueError("formal SFT data must come from the train split")
        if self.teacher_usage_rights not in {"approved_open_weights", "approved_terms"}:
            raise ValueError("teacher usage rights must pass the training-data terms gate")
        if not self.teacher_model.strip():
            raise ValueError("teacher_model is required")


def build_verified_dataset(
    *,
    payload: dict[str, Any],
    source_results: Path,
    output_dir: Path,
    system_prompt: str,
    tools: list[dict[str, Any]],
    allowed_task_ids: set[str],
    process_audits: dict[str, ProcessAudit],
    config: DatasetBuildConfig,
) -> dict[str, Any]:
    """Filter, split, write, and manifest one versioned trajectory dataset."""

    simulations = payload.get("simulations") or []
    if not simulations:
        raise ValueError("results contain no simulations")
    rejections: Counter[str] = Counter()
    candidates: list[dict[str, Any]] = []
    per_task: Counter[str] = Counter()
    per_structure: Counter[str] = Counter()
    seen_simulations: set[str] = set()

    for simulation in simulations:
        simulation_id = str(simulation.get("id") or "")
        task_id = str(simulation.get("task_id") or "")
        if not simulation_id or simulation_id in seen_simulations:
            rejections["missing_or_duplicate_simulation_id"] += 1
            continue
        seen_simulations.add(simulation_id)
        if task_id not in allowed_task_ids:
            rejections["task_not_in_frozen_train_split"] += 1
            continue
        if task_id in config.reserved_task_ids:
            rejections["reserved_task"] += 1
            continue
        reward_info = simulation.get("reward_info") or {}
        if float(reward_info.get("reward") or 0.0) != 1.0:
            rejections["official_reward_not_1"] += 1
            continue
        db_check = reward_info.get("db_check")
        if db_check is not None and not db_check.get("db_match", False):
            rejections["db_mismatch"] += 1
            continue
        if simulation.get("termination_reason") not in ACCEPTED_TERMINATIONS:
            rejections[f"termination:{simulation.get('termination_reason')}"] += 1
            continue
        audit = process_audits.get(simulation_id)
        if config.require_process_audit and audit is None:
            rejections["missing_process_audit"] += 1
            continue
        if audit is not None and not audit.process_compliant:
            rejections["process_noncompliant"] += 1
            for violation in audit.violations:
                rejections[f"violation:{violation}"] += 1
            continue
        messages, reason = convert_messages(simulation.get("messages") or [])
        if reason:
            rejections[reason] += 1
            continue
        if not any(message["role"] == "tool_call" for message in messages):
            rejections["no_tool_calls"] += 1
            continue
        observed_signature = structure_signature(simulation.get("messages") or [])
        frozen_structure = config.task_structures.get(task_id)
        signature = dict(frozen_structure) if frozen_structure else observed_signature
        if "signature_hash" not in signature:
            raise ValueError(f"frozen structure missing signature_hash: {task_id}")
        signature_hash = signature["signature_hash"]
        if per_task[task_id] >= config.max_per_task:
            rejections["per_task_cap"] += 1
            continue
        if per_structure[signature_hash] >= config.max_per_structure:
            rejections["per_structure_cap"] += 1
            continue
        per_task[task_id] += 1
        per_structure[signature_hash] += 1
        candidates.append(
            {
                "messages": [{"role": "system", "content": system_prompt}, *messages],
                "tools": json.dumps(tools, ensure_ascii=False, separators=(",", ":")),
                "provenance": {
                    "source": str(
                        (simulation.get("provenance") or {}).get("source")
                        or payload.get("source_label")
                        or "tau3_retail_train"
                    ),
                    "source_results": str(source_results),
                    "simulation_id": simulation_id,
                    "task_id": task_id,
                    "trial": simulation.get("trial"),
                    "seed": simulation.get("seed"),
                    "official_reward": 1.0,
                    "db_match": True if db_check is None else db_check.get("db_match"),
                    "termination_reason": simulation.get("termination_reason"),
                    "teacher_model": config.teacher_model,
                    "teacher_usage_rights": config.teacher_usage_rights,
                    "runtime_version": config.runtime_version,
                    "prompt_version": config.prompt_version,
                    "process_audit_version": audit.audit_version if audit else None,
                    "process_audit_reviewer": audit.reviewer if audit else None,
                    "structure": signature,
                    "observed_tool_path": observed_signature,
                    "split": None,
                },
            }
        )

    unassigned = [
        record
        for record in candidates
        if record["provenance"]["task_id"] not in config.preassigned_task_splits
    ]
    simulation_splits = assign_isolated_splits(unassigned)
    for record in candidates:
        simulation_id = record["provenance"]["simulation_id"]
        task_id = record["provenance"]["task_id"]
        record["provenance"]["split"] = config.preassigned_task_splits.get(
            task_id, simulation_splits.get(simulation_id, "train")
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    split_counts: Counter[str] = Counter()
    split_structures: dict[str, set[str]] = defaultdict(set)
    split_tasks: dict[str, set[str]] = defaultdict(set)
    for split in ("train", "dev", "held_out"):
        path = output_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for record in candidates:
                if record["provenance"]["split"] != split:
                    continue
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                split_counts[split] += 1
                split_structures[split].add(
                    record["provenance"]["structure"]["signature_hash"]
                )
                split_tasks[split].add(record["provenance"]["task_id"])

    all_structure_sets = list(split_structures.values())
    for index, left in enumerate(all_structure_sets):
        for right in all_structure_sets[index + 1 :]:
            if left & right:
                raise AssertionError("structure leakage across dataset splits")
    all_task_sets = list(split_tasks.values())
    for index, left in enumerate(all_task_sets):
        for right in all_task_sets[index + 1 :]:
            if left & right:
                raise AssertionError("task leakage across dataset splits")

    manifest = {
        "dataset_version": "verified-ecommerce-sft-v1",
        "source_results": str(source_results),
        "source_sha256": file_sha256(source_results),
        "source_split": config.source_split,
        "teacher_model": config.teacher_model,
        "teacher_usage_rights": config.teacher_usage_rights,
        "runtime_version": config.runtime_version,
        "prompt_version": config.prompt_version,
        "require_process_audit": config.require_process_audit,
        "total_simulations": len(simulations),
        "accepted": len(candidates),
        "tasks_covered": len(per_task),
        "unique_structures": len(per_structure),
        "split_counts": dict(split_counts),
        "split_structure_counts": {
            name: len(values) for name, values in split_structures.items()
        },
        "split_task_counts": {name: len(values) for name, values in split_tasks.items()},
        "rejections": dict(rejections.most_common()),
        "config_hash": canonical_hash(
            {
                "config": {
                    **config.__dict__,
                    "reserved_task_ids": sorted(config.reserved_task_ids),
                },
                "system_prompt": system_prompt,
                "tools": tools,
            }
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def default_system_prompt(domain_policy: str, config: DatasetBuildConfig) -> str:
    return build_system_prompt(
        domain_policy,
        RuntimeConfig(
            runtime_version=config.runtime_version,
            prompt_version=config.prompt_version,
            compact_context=False,
        ),
    )
