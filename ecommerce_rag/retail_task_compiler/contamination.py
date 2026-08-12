# -*- coding: utf-8 -*-
"""Structural signatures and τ³ test-40 contamination checks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .blueprint import TaskBlueprint, canonical_hash, validate_blueprint
from .constants import RETAIL_WRITE_TOOLS, WRITE_TOOL_TO_FAMILY


def _tool_path_from_actions(actions: Sequence[Mapping[str, Any]]) -> list[str]:
    return [str(step["name"]) for step in actions]


def infer_task_family(tool_path: Sequence[str]) -> str:
    writes = [name for name in tool_path if name in RETAIL_WRITE_TOOLS]
    if not writes:
        if "transfer_to_human_agents" in tool_path:
            return "handoff"
        return "read_only"
    families = [WRITE_TOOL_TO_FAMILY.get(name, name) for name in writes]
    # Multi-write official tasks keep a stable joined family label.
    return "+".join(dict.fromkeys(families))


def structure_signature(
    *,
    task_family: str,
    tool_path: Sequence[str],
    state_predicates: Sequence[str],
    required_effect_kinds: Sequence[str],
    initial_order_state: str | None = None,
    user_order_relationship: str | None = None,
    clarification_or_confirmation: str | None = None,
    allowed_state_change: str | None = None,
    forbidden_state_change: str | None = None,
    expected_termination: str | None = None,
) -> dict[str, Any]:
    payload = {
        "task_family": task_family,
        "tool_path": list(tool_path),
        "state_predicates": sorted(set(state_predicates)),
        "required_effect_kinds": sorted(set(required_effect_kinds)),
    }
    optional = {
        "initial_order_state": initial_order_state,
        "user_order_relationship": user_order_relationship,
        "clarification_or_confirmation": clarification_or_confirmation,
        "allowed_state_change": allowed_state_change,
        "forbidden_state_change": forbidden_state_change,
        "expected_termination": expected_termination,
    }
    for key, value in optional.items():
        if value is not None:
            payload[key] = value
    payload["signature_hash"] = canonical_hash(
        {key: value for key, value in payload.items() if key != "signature_hash"}
    )
    return payload


def signature_from_tau_task(task: Mapping[str, Any]) -> dict[str, Any]:
    actions = (task.get("evaluation_criteria") or {}).get("actions") or []
    tool_path = _tool_path_from_actions(actions)
    family = infer_task_family(tool_path)
    effect_kinds = [name for name in tool_path if name in RETAIL_WRITE_TOOLS]
    if "transfer_to_human_agents" in tool_path:
        effect_kinds.append("handoff")
    predicates: list[str] = []
    if "cancel_pending_order" in tool_path or any(
        name.startswith("modify_pending_") for name in tool_path
    ):
        predicates.append("order_status=pending")
    if "return_delivered_order_items" in tool_path or (
        "exchange_delivered_order_items" in tool_path
    ):
        predicates.append("order_status=delivered")
    return structure_signature(
        task_family=family,
        tool_path=tool_path,
        state_predicates=predicates,
        required_effect_kinds=effect_kinds,
    )


def signature_from_blueprint(blueprint: TaskBlueprint | Mapping[str, Any]) -> dict[str, Any]:
    bp = validate_blueprint(blueprint)
    # Primary reference path defines the structural identity for contamination.
    primary = bp.reference_tool_paths[0]
    tool_path = [step["name"] for step in primary]
    effect_kinds = [
        str(effect.get("kind") or effect.get("op") or effect.get("type") or "effect")
        for effect in bp.required_effects
    ]
    predicates = [
        str(pred)
        for pred in (bp.initial_state.get("predicates") or [])
    ]
    family = bp.task_family or infer_task_family(tool_path)
    structure_meta = bp.initial_state.get("structure") or {}
    return structure_signature(
        task_family=family,
        tool_path=tool_path,
        state_predicates=predicates,
        required_effect_kinds=effect_kinds,
        initial_order_state=structure_meta.get("initial_order_state"),
        user_order_relationship=structure_meta.get("user_order_relationship"),
        clarification_or_confirmation=(
            f"{structure_meta.get('ambiguity_type')}|{structure_meta.get('confirmation_requirement')}"
            if structure_meta
            else None
        ),
        allowed_state_change=structure_meta.get("allowed_state_change"),
        forbidden_state_change=structure_meta.get("forbidden_state_change"),
        expected_termination=structure_meta.get("expected_termination"),
    )


@dataclass(frozen=True)
class ContaminationReport:
    contaminated: bool
    matched_task_ids: tuple[str, ...]
    blueprint_signature_hash: str
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "contaminated": self.contaminated,
            "matched_task_ids": list(self.matched_task_ids),
            "blueprint_signature_hash": self.blueprint_signature_hash,
            "reason": self.reason,
        }


def _gray_zone_similar(entry: Mapping[str, Any], bp_sig: Mapping[str, Any]) -> bool:
    """Fail closed on ambiguous near-collisions with frozen test signatures."""
    same_family = entry.get("task_family") == bp_sig.get("task_family")
    same_state = sorted(entry.get("state_predicates") or []) == sorted(
        bp_sig.get("state_predicates") or []
    )
    same_effects = sorted(entry.get("required_effect_kinds") or []) == sorted(
        bp_sig.get("required_effect_kinds") or []
    )
    entry_path = list(entry.get("tool_path") or [])
    bp_path = list(bp_sig.get("tool_path") or [])
    # Same write tool sequence ignoring duplicated reads is a gray zone.
    def writes(path: Sequence[str]) -> list[str]:
        return [
            name
            for name in path
            if name in RETAIL_WRITE_TOOLS or name == "transfer_to_human_agents"
        ]

    same_writes = writes(entry_path) == writes(bp_path) and bool(writes(bp_path))
    return bool(same_family and same_state and same_effects and same_writes)


def check_contamination(
    blueprint: TaskBlueprint | Mapping[str, Any],
    test_signatures: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> ContaminationReport:
    """Reject blueprints whose structure signature collides with frozen test 40."""
    bp_sig = signature_from_blueprint(blueprint)
    bp_hash = bp_sig["signature_hash"]
    entries: list[Mapping[str, Any]]
    if isinstance(test_signatures, Mapping) and "signatures" in test_signatures:
        entries = list(test_signatures["signatures"])
    else:
        entries = list(test_signatures)  # type: ignore[arg-type]

    matched: list[str] = []
    reasons: list[str] = []
    for entry in entries:
        if entry.get("signature_hash") == bp_hash:
            matched.append(str(entry.get("task_id")))
            reasons.append("signature_hash")
            continue
        same_path = list(entry.get("tool_path") or []) == list(bp_sig["tool_path"])
        same_family = entry.get("task_family") == bp_sig.get("task_family")
        same_state = sorted(entry.get("state_predicates") or []) == sorted(
            bp_sig.get("state_predicates") or []
        )
        # Tool-path identity alone is insufficient: refusal/read structures may
        # share auth/read prefixes with unrelated test families.
        if same_path and same_family and same_state:
            matched.append(str(entry.get("task_id")))
            reasons.append("tool_path_family_state")
            continue
        if _gray_zone_similar(entry, bp_sig):
            matched.append(str(entry.get("task_id")))
            reasons.append("gray_zone_similar_write_structure")

    if matched:
        return ContaminationReport(
            contaminated=True,
            matched_task_ids=tuple(dict.fromkeys(matched)),
            blueprint_signature_hash=bp_hash,
            reason=(
                "structure collision with tau3 retail test: "
                + ",".join(dict.fromkeys(reasons))
            ),
        )
    return ContaminationReport(
        contaminated=False,
        matched_task_ids=(),
        blueprint_signature_hash=bp_hash,
    )


def extract_test_signatures(
    *,
    tasks: Sequence[Mapping[str, Any]],
    test_ids: Iterable[str],
    tau2_commit: str,
) -> dict[str, Any]:
    by_id = {str(task["id"]): task for task in tasks}
    signatures = []
    for task_id in test_ids:
        task = by_id[str(task_id)]
        sig = signature_from_tau_task(task)
        signatures.append({"task_id": str(task_id), **sig})
    return {
        "schema_version": "1.0",
        "split": "test",
        "expected_count": 40,
        "tau2_commit": tau2_commit,
        "count": len(signatures),
        "signatures": signatures,
    }


def load_test_signatures(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("count") or 0) != 40:
        raise ValueError(
            f"frozen test signature count drift: {payload.get('count')} != 40"
        )
    return payload
