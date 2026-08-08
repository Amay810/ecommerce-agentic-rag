# -*- coding: utf-8 -*-
"""Deterministic double-replay verifier for reference tool paths."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from .blueprint import TaskBlueprint, validate_blueprint


class EnvironmentExecutor(Protocol):
    """Minimal reset/execute/snapshot surface required by the compiler."""

    def reset(self, initial_state: Mapping[str, Any]) -> None: ...

    def execute(self, name: str, arguments: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def snapshot(self) -> Mapping[str, Any]: ...

    def diff_effects(
        self, before: Mapping[str, Any], after: Mapping[str, Any]
    ) -> list[dict[str, Any]]: ...


@dataclass(frozen=True)
class ReplayReport:
    accepted: bool
    reasons: tuple[str, ...]
    first_effects: tuple[dict[str, Any], ...]
    second_effects: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reasons": list(self.reasons),
            "first_effects": list(self.first_effects),
            "second_effects": list(self.second_effects),
        }


def _effect_key(effect: Mapping[str, Any]) -> str:
    return (
        f"{effect.get('kind')}|{effect.get('entity')}|{effect.get('field')}|"
        f"{effect.get('before')}|{effect.get('after')}"
    )


def _effects_match_required(
    observed: Sequence[Mapping[str, Any]],
    required: Sequence[Mapping[str, Any]],
) -> list[str]:
    reasons: list[str] = []
    for req in required:
        partial = [
            item
            for item in observed
            if item.get("kind") == req.get("kind")
            and item.get("entity") == req.get("entity")
            and item.get("field") == req.get("field")
            and item.get("after") == req.get("after")
        ]
        if not partial:
            reasons.append(f"missing required effect: {req}")
    return reasons


def _effects_hit_forbidden(
    observed: Sequence[Mapping[str, Any]],
    forbidden: Sequence[Mapping[str, Any]],
) -> list[str]:
    reasons: list[str] = []
    for bad in forbidden:
        for item in observed:
            same_kind = item.get("kind") == bad.get("kind")
            same_entity = bad.get("entity") in (None, item.get("entity"))
            same_field = bad.get("field") in (None, item.get("field"))
            same_after = bad.get("after") in (None, item.get("after"))
            if same_kind and same_entity and same_field and same_after:
                reasons.append(f"forbidden effect observed: {item}")
    return reasons


def replay_reference_path_twice(
    blueprint: TaskBlueprint | Mapping[str, Any],
    executor: EnvironmentExecutor,
    *,
    path_index: int = 0,
) -> ReplayReport:
    """Reset → execute → reset → execute; both terminal diffs must match."""
    bp = validate_blueprint(blueprint)
    path = bp.reference_tool_paths[path_index]
    reasons: list[str] = []

    def run_once() -> tuple[list[dict[str, Any]], Mapping[str, Any]]:
        executor.reset(copy.deepcopy(dict(bp.initial_state)))
        before = copy.deepcopy(executor.snapshot())
        for step in path:
            executor.execute(step["name"], dict(step["arguments"]))
        after = copy.deepcopy(executor.snapshot())
        effects = list(executor.diff_effects(before, after))
        return effects, after

    first_effects, first_snap = run_once()
    second_effects, second_snap = run_once()

    if first_snap != second_snap:
        reasons.append("terminal snapshots differ across deterministic replays")
    if [_effect_key(e) for e in first_effects] != [
        _effect_key(e) for e in second_effects
    ]:
        reasons.append("effect diffs differ across deterministic replays")

    reasons.extend(_effects_match_required(first_effects, bp.required_effects))
    reasons.extend(_effects_hit_forbidden(first_effects, bp.forbidden_effects))

    terminal_ok = False
    for condition in bp.acceptable_terminal_conditions:
        if all(first_snap.get(key) == value for key, value in condition.items()):
            terminal_ok = True
            break
    if not terminal_ok and bp.acceptable_terminal_conditions:
        # Allow nested order-status terminal checks used by cancel_pending v0.
        nested_ok = False
        for condition in bp.acceptable_terminal_conditions:
            order_id = condition.get("order_id")
            status = condition.get("order_status")
            if order_id and status:
                orders = {
                    order["order_id"]: order for order in first_snap.get("orders", [])
                }
                if orders.get(order_id, {}).get("status") == status:
                    nested_ok = True
                    break
        if not nested_ok:
            reasons.append("terminal snapshot matches no acceptable_terminal_conditions")

    return ReplayReport(
        accepted=not reasons,
        reasons=tuple(reasons),
        first_effects=tuple(first_effects),
        second_effects=tuple(second_effects),
    )


class CancelPendingMockExecutor:
    """In-memory executor for cancel_pending unit tests (not the live τ³ DB)."""

    def __init__(self) -> None:
        self._state: dict[str, Any] = {}

    def reset(self, initial_state: Mapping[str, Any]) -> None:
        self._state = copy.deepcopy(dict(initial_state.get("db") or {}))

    def execute(self, name: str, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        if name in {"find_user_id_by_email", "find_user_id_by_name_zip"}:
            user_id = self._state["users"][0]["user_id"]
            return {"user_id": user_id}
        if name == "get_user_details":
            return dict(self._state["users"][0])
        if name == "get_order_details":
            order_id = arguments["order_id"]
            order = next(o for o in self._state["orders"] if o["order_id"] == order_id)
            return dict(order)
        if name == "cancel_pending_order":
            order_id = arguments["order_id"]
            reason = arguments["reason"]
            if reason not in {"no longer needed", "ordered by mistake"}:
                raise ValueError("Invalid reason")
            for order in self._state["orders"]:
                if order["order_id"] == order_id:
                    if order["status"] != "pending":
                        raise ValueError("Non-pending order cannot be cancelled")
                    order["status"] = "cancelled"
                    order["cancel_reason"] = reason
                    return dict(order)
            raise ValueError("Order not found")
        raise ValueError(f"unsupported mock tool: {name}")

    def snapshot(self) -> Mapping[str, Any]:
        return copy.deepcopy(self._state)

    def diff_effects(
        self, before: Mapping[str, Any], after: Mapping[str, Any]
    ) -> list[dict[str, Any]]:
        effects: list[dict[str, Any]] = []
        before_orders = {o["order_id"]: o for o in before.get("orders", [])}
        after_orders = {o["order_id"]: o for o in after.get("orders", [])}
        for order_id, after_order in after_orders.items():
            before_order = before_orders[order_id]
            if before_order.get("status") != after_order.get("status"):
                effects.append(
                    {
                        "kind": "order_status_change",
                        "entity": order_id,
                        "field": "status",
                        "before": before_order.get("status"),
                        "after": after_order.get("status"),
                    }
                )
        return effects
