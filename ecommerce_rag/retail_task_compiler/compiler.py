# -*- coding: utf-8 -*-
"""Retail Task Compiler v0: first slice = cancel_pending only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .blueprint import TaskBlueprint, canonical_hash, validate_blueprint
from .constants import GENERATOR_VERSION_V0 as GENERATOR_VERSION, SOURCE_POLICY_VERSION, TAU2_COMMIT
from .contamination import ContaminationReport, check_contamination
from .replay import (
    CancelPendingMockExecutor,
    EnvironmentExecutor,
    ReplayReport,
    replay_reference_path_twice,
)
from .tool_graph import ToolDependencyGraph, load_retail_tool_graph


CANCEL_REASONS = ("no longer needed", "ordered by mistake")


@dataclass(frozen=True)
class CompileResult:
    blueprint: TaskBlueprint
    contamination: ContaminationReport
    replay: ReplayReport | None

    @property
    def accepted(self) -> bool:
        if self.contamination.contaminated:
            return False
        if self.replay is not None and not self.replay.accepted:
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "blueprint": self.blueprint.to_dict(),
            "blueprint_hash": self.blueprint.blueprint_hash(),
            "contamination": self.contamination.to_dict(),
            "replay": None if self.replay is None else self.replay.to_dict(),
        }


class RetailTaskCompiler:
    """Compile executable blueprints; NL surface forms are out of scope for v0."""

    def __init__(
        self,
        *,
        tool_graph: ToolDependencyGraph | None = None,
        test_signatures: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
        executor: EnvironmentExecutor | None = None,
    ) -> None:
        self.tool_graph = tool_graph or load_retail_tool_graph()
        self.test_signatures = test_signatures or {"signatures": []}
        self.executor = executor

    def compile_cancel_pending(
        self,
        *,
        task_id: str,
        user: Mapping[str, Any],
        order: Mapping[str, Any],
        auth_mode: str = "email",
        reason: str = "no longer needed",
        behavior_profile: str = "cooperative",
        outcome_class: str = "success",
        composition_split: str = "seen",
        db_snapshot_hash: str,
        run_replay: bool = True,
    ) -> CompileResult:
        if reason not in CANCEL_REASONS:
            raise ValueError(f"invalid cancel reason: {reason}")
        if auth_mode not in {"email", "name_zip"}:
            raise ValueError(f"unsupported auth_mode: {auth_mode}")
        if order.get("status") != "pending" and outcome_class == "success":
            raise ValueError("success cancel_pending requires a pending order entity")

        if auth_mode == "email":
            auth_step = {
                "name": "find_user_id_by_email",
                "arguments": {"email": user["email"]},
            }
        else:
            auth_step = {
                "name": "find_user_id_by_name_zip",
                "arguments": {
                    "first_name": user["first_name"],
                    "last_name": user["last_name"],
                    "zip": user["zip"],
                },
            }

        path = [
            auth_step,
            {"name": "get_user_details", "arguments": {"user_id": user["user_id"]}},
            {
                "name": "get_order_details",
                "arguments": {"order_id": order["order_id"]},
            },
            {
                "name": "cancel_pending_order",
                "arguments": {"order_id": order["order_id"], "reason": reason},
            },
        ]
        tool_names = [step["name"] for step in path]
        self.tool_graph.assert_path_allowed(tool_names)

        prompt_payload = {
            "family": "cancel_pending",
            "auth_mode": auth_mode,
            "reason": reason,
            "behavior_profile": behavior_profile,
            "outcome_class": outcome_class,
        }
        blueprint = validate_blueprint(
            {
                "task_id": task_id,
                "environment": "tau3_retail",
                "source_policy_version": SOURCE_POLICY_VERSION,
                "tool_graph_hash": self.tool_graph.graph_hash(),
                "db_snapshot_hash": db_snapshot_hash,
                "initial_state": {
                    "predicates": ["order_status=pending"],
                    "db": {
                        "users": [dict(user)],
                        "orders": [dict(order)],
                    },
                },
                "user_goal": {
                    "op": "cancel_pending_order",
                    "order_id": order["order_id"],
                    "reason": reason,
                },
                "private_user_facts": {
                    "user_id": user["user_id"],
                    "email": user.get("email"),
                    "first_name": user.get("first_name"),
                    "last_name": user.get("last_name"),
                    "zip": user.get("zip"),
                    "order_id": order["order_id"],
                },
                "disclosure_schedule": [
                    {"turn": 0, "reveal": ["identity"]},
                    {"turn": 1, "reveal": ["order_id", "cancel_reason"]},
                ],
                "required_effects": [
                    {
                        "kind": "order_status_change",
                        "entity": order["order_id"],
                        "field": "status",
                        "before": "pending",
                        "after": "cancelled",
                    }
                ],
                "forbidden_effects": [
                    {
                        "kind": "order_status_change",
                        "entity": order["order_id"],
                        "field": "status",
                        "after": "delivered",
                    }
                ],
                "acceptable_terminal_conditions": [
                    {
                        "order_id": order["order_id"],
                        "order_status": "cancelled",
                    }
                ],
                "reference_tool_paths": [path],
                "behavior_profile": behavior_profile,
                "generator_version": GENERATOR_VERSION,
                "generator_prompt_hash": canonical_hash(prompt_payload),
                "task_family": "cancel_pending",
                "outcome_class": outcome_class,
                "composition_split": composition_split,
                "provenance": {
                    "tau2_commit": TAU2_COMMIT,
                    "compiler": GENERATOR_VERSION,
                    "note": (
                        "NL utterances are intentionally absent; "
                        "LLM may only rewrite surface forms later."
                    ),
                },
            }
        )

        contamination = check_contamination(blueprint, self.test_signatures)
        replay: ReplayReport | None = None
        if run_replay:
            executor = self.executor or CancelPendingMockExecutor()
            replay = replay_reference_path_twice(blueprint, executor)

        return CompileResult(
            blueprint=blueprint, contamination=contamination, replay=replay
        )


def compile_cancel_pending_v0(**kwargs: Any) -> CompileResult:
    """Convenience wrapper around ``RetailTaskCompiler.compile_cancel_pending``."""
    compiler = RetailTaskCompiler(
        test_signatures=kwargs.pop("test_signatures", None),
        executor=kwargs.pop("executor", None),
    )
    return compiler.compile_cancel_pending(**kwargs)
