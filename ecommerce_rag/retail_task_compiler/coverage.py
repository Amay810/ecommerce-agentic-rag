# -*- coding: utf-8 -*-
"""Coverage axes for compiled blueprints (not entity-swap counts)."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .blueprint import TaskBlueprint, validate_blueprint


COVERAGE_AXES = (
    "task_family",
    "tool_path_length",
    "read_write_handoff",
    "initial_state_predicate",
    "required_clarification",
    "user_behavior",
    "outcome_class",
    "composition_split",
)


@dataclass(frozen=True)
class CoverageReport:
    totals: Mapping[str, int]
    axes: Mapping[str, Mapping[str, int]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "totals": dict(self.totals),
            "axes": {key: dict(value) for key, value in self.axes.items()},
        }


def _path_mode(tool_path: list[str]) -> str:
    has_write = any(
        name.startswith(("cancel_", "exchange_", "modify_", "return_"))
        for name in tool_path
    )
    has_handoff = "transfer_to_human_agents" in tool_path
    if has_handoff and has_write:
        return "write+handoff"
    if has_handoff:
        return "handoff"
    if has_write:
        return "write"
    return "read"


def coverage_from_blueprints(
    blueprints: Iterable[TaskBlueprint | Mapping[str, Any]],
) -> CoverageReport:
    axes = {name: Counter() for name in COVERAGE_AXES}
    total = 0
    for raw in blueprints:
        bp = validate_blueprint(raw)
        total += 1
        path = [step["name"] for step in bp.reference_tool_paths[0]]
        axes["task_family"][bp.task_family or "unknown"] += 1
        axes["tool_path_length"][str(len(path))] += 1
        axes["read_write_handoff"][_path_mode(path)] += 1
        predicates = bp.initial_state.get("predicates") or ["none"]
        for pred in predicates:
            axes["initial_state_predicate"][str(pred)] += 1
        needs_clarify = "yes" if bp.disclosure_schedule else "no"
        axes["required_clarification"][needs_clarify] += 1
        axes["user_behavior"][bp.behavior_profile] += 1
        axes["outcome_class"][bp.outcome_class] += 1
        axes["composition_split"][bp.composition_split] += 1
    return CoverageReport(totals={"blueprints": total}, axes=axes)
