# -*- coding: utf-8 -*-
"""Machine-readable Task Blueprint schema and validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

from .constants import GENERATOR_VERSION, RETAIL_ALL_TOOLS

ENVIRONMENTS = frozenset({"tau3_retail", "native_retail"})
BEHAVIOR_PROFILES = frozenset(
    {
        "cooperative",
        "incomplete",
        "impatient",
        "digressive",
        "unsupported_request",
        "goal_shift",
    }
)

REQUIRED_FIELDS = (
    "task_id",
    "environment",
    "source_policy_version",
    "tool_graph_hash",
    "db_snapshot_hash",
    "initial_state",
    "user_goal",
    "private_user_facts",
    "disclosure_schedule",
    "required_effects",
    "forbidden_effects",
    "acceptable_terminal_conditions",
    "reference_tool_paths",
    "behavior_profile",
    "generator_version",
    "generator_prompt_hash",
)


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ToolStep:
    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "arguments": dict(self.arguments)}


@dataclass(frozen=True)
class TaskBlueprint:
    task_id: str
    environment: str
    source_policy_version: str
    tool_graph_hash: str
    db_snapshot_hash: str
    initial_state: Mapping[str, Any]
    user_goal: Mapping[str, Any]
    private_user_facts: Mapping[str, Any]
    disclosure_schedule: Sequence[Mapping[str, Any]]
    required_effects: Sequence[Mapping[str, Any]]
    forbidden_effects: Sequence[Mapping[str, Any]]
    acceptable_terminal_conditions: Sequence[Mapping[str, Any]]
    reference_tool_paths: Sequence[Sequence[Mapping[str, Any]]]
    behavior_profile: str
    generator_version: str
    generator_prompt_hash: str
    task_family: str = ""
    outcome_class: str = "success"  # success | impossible | unsafe | handoff
    composition_split: str = "seen"  # seen | held_out
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload

    def blueprint_hash(self) -> str:
        # Hash excludes free-form provenance notes that may be appended later.
        core = {key: getattr(self, key) for key in REQUIRED_FIELDS}
        core["task_family"] = self.task_family
        core["outcome_class"] = self.outcome_class
        core["composition_split"] = self.composition_split
        return canonical_hash(core)


def validate_blueprint(blueprint: TaskBlueprint | Mapping[str, Any]) -> TaskBlueprint:
    """Fail closed when a blueprint is incomplete or internally inconsistent."""
    if isinstance(blueprint, TaskBlueprint):
        data = blueprint.to_dict()
    else:
        data = dict(blueprint)

    missing = [name for name in REQUIRED_FIELDS if name not in data]
    if missing:
        raise ValueError(f"blueprint missing required fields: {missing}")

    environment = data["environment"]
    if environment not in ENVIRONMENTS:
        raise ValueError(f"unsupported environment: {environment}")

    behavior = data["behavior_profile"]
    if behavior not in BEHAVIOR_PROFILES:
        raise ValueError(f"unsupported behavior_profile: {behavior}")

    if not str(data["task_id"]).strip():
        raise ValueError("task_id must be non-empty")
    if not str(data["tool_graph_hash"]).strip():
        raise ValueError("tool_graph_hash must be non-empty")
    if not str(data["db_snapshot_hash"]).strip():
        raise ValueError("db_snapshot_hash must be non-empty")
    if not str(data["generator_version"]).strip():
        raise ValueError("generator_version must be non-empty")
    if data.get("generator_version") != GENERATOR_VERSION and not str(
        data["generator_version"]
    ).startswith("retail_task_compiler."):
        raise ValueError(
            f"generator_version must be project-owned; got {data['generator_version']!r}"
        )

    paths = data["reference_tool_paths"]
    if not isinstance(paths, list) or not paths:
        raise ValueError("reference_tool_paths must be a non-empty list of paths")
    for path in paths:
        if not isinstance(path, list) or not path:
            raise ValueError("each reference tool path must be a non-empty list")
        for step in path:
            name = step.get("name")
            if name not in RETAIL_ALL_TOOLS and environment == "tau3_retail":
                raise ValueError(f"unknown τ³ retail tool in reference path: {name}")
            if "arguments" not in step or not isinstance(step["arguments"], Mapping):
                raise ValueError(f"tool step {name!r} requires an arguments object")

    if not isinstance(data["required_effects"], list):
        raise ValueError("required_effects must be a list")
    if not isinstance(data["forbidden_effects"], list):
        raise ValueError("forbidden_effects must be a list")
    if not isinstance(data["acceptable_terminal_conditions"], list) or not data[
        "acceptable_terminal_conditions"
    ]:
        raise ValueError("acceptable_terminal_conditions must be a non-empty list")

    outcome = data.get("outcome_class", "success")
    if outcome not in {"success", "impossible", "unsafe", "handoff"}:
        raise ValueError(f"unsupported outcome_class: {outcome}")

    return TaskBlueprint(
        task_id=str(data["task_id"]),
        environment=environment,
        source_policy_version=str(data["source_policy_version"]),
        tool_graph_hash=str(data["tool_graph_hash"]),
        db_snapshot_hash=str(data["db_snapshot_hash"]),
        initial_state=dict(data["initial_state"]),
        user_goal=dict(data["user_goal"]),
        private_user_facts=dict(data["private_user_facts"]),
        disclosure_schedule=list(data["disclosure_schedule"]),
        required_effects=list(data["required_effects"]),
        forbidden_effects=list(data["forbidden_effects"]),
        acceptable_terminal_conditions=list(data["acceptable_terminal_conditions"]),
        reference_tool_paths=[
            [dict(step) for step in path] for path in data["reference_tool_paths"]
        ],
        behavior_profile=behavior,
        generator_version=str(data["generator_version"]),
        generator_prompt_hash=str(data["generator_prompt_hash"]),
        task_family=str(data.get("task_family") or ""),
        outcome_class=outcome,
        composition_split=str(data.get("composition_split") or "seen"),
        provenance=dict(data.get("provenance") or {}),
    )
