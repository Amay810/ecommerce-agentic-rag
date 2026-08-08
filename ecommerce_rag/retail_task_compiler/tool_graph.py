# -*- coding: utf-8 -*-
"""Versioned τ³ Retail tool dependency graph with provenance.

Edges are not invented by an LLM. Each edge cites at least one of:
policy text, tools.py precondition, schema I/O, or a human-reviewed rule id.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .blueprint import canonical_hash
from .constants import TOOL_GRAPH_VERSION


@dataclass(frozen=True)
class Provenance:
    kind: str  # policy | tools_impl | schema | human_reviewed
    locator: str
    note: str = ""

    def to_dict(self) -> dict[str, str]:
        payload = {"kind": self.kind, "locator": self.locator}
        if self.note:
            payload["note"] = self.note
        return payload


@dataclass(frozen=True)
class GraphEdge:
    source: str
    target: str
    relation: str
    provenance: tuple[Provenance, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "provenance": [item.to_dict() for item in self.provenance],
        }


@dataclass(frozen=True)
class ToolDependencyGraph:
    version: str
    nodes: tuple[str, ...]
    edges: tuple[GraphEdge, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "nodes": list(self.nodes),
            "edges": [edge.to_dict() for edge in self.edges],
        }

    def graph_hash(self) -> str:
        return canonical_hash(self.to_dict())

    def assert_path_allowed(self, tool_names: Sequence[str]) -> None:
        """Reject reference paths that skip required write preconditions."""
        names = list(tool_names)
        if not names:
            raise ValueError("empty tool path")
        write_ops = {
            "cancel_pending_order",
            "exchange_delivered_order_items",
            "modify_pending_order_address",
            "modify_pending_order_items",
            "modify_pending_order_payment",
            "modify_user_address",
            "return_delivered_order_items",
        }
        auth_tools = {"find_user_id_by_email", "find_user_id_by_name_zip"}
        if any(name in write_ops for name in names):
            if not any(name in auth_tools for name in names):
                raise ValueError(
                    "write path missing authentication tool "
                    "(policy: authenticate before profile/order writes)"
                )
            if "get_order_details" not in names and any(
                name.startswith(("cancel_", "exchange_", "modify_pending_", "return_"))
                for name in names
            ):
                raise ValueError(
                    "order-mutating path missing get_order_details "
                    "(tools/policy: check status before write)"
                )


def _p(kind: str, locator: str, note: str = "") -> Provenance:
    return Provenance(kind=kind, locator=locator, note=note)


def load_retail_tool_graph() -> ToolDependencyGraph:
    """Return the frozen v0 graph for τ³ Retail cancel/modify/return/exchange."""
    nodes = (
        "find_user_id_by_email",
        "find_user_id_by_name_zip",
        "predicate:user_authenticated",
        "get_user_details",
        "get_order_details",
        "get_product_details",
        "get_item_details",
        "list_all_product_types",
        "predicate:order_status_pending",
        "predicate:order_status_delivered",
        "predicate:user_confirms_write",
        "cancel_pending_order",
        "modify_pending_order_address",
        "modify_pending_order_items",
        "modify_pending_order_payment",
        "modify_user_address",
        "return_delivered_order_items",
        "exchange_delivered_order_items",
        "transfer_to_human_agents",
        "calculate",
    )
    edges = (
        GraphEdge(
            "find_user_id_by_email",
            "predicate:user_authenticated",
            "establishes",
            (
                _p("policy", "policy.md:authenticate via email or name+zip"),
                _p("tools_impl", "tools.py:find_user_id_by_email"),
            ),
        ),
        GraphEdge(
            "find_user_id_by_name_zip",
            "predicate:user_authenticated",
            "establishes",
            (
                _p("policy", "policy.md:authenticate via email or name+zip"),
                _p("tools_impl", "tools.py:find_user_id_by_name_zip"),
            ),
        ),
        GraphEdge(
            "predicate:user_authenticated",
            "get_user_details",
            "enables",
            (_p("policy", "policy.md:provide profile information after auth"),),
        ),
        GraphEdge(
            "predicate:user_authenticated",
            "get_order_details",
            "enables",
            (_p("policy", "policy.md:look up order info after auth"),),
        ),
        GraphEdge(
            "get_order_details",
            "predicate:order_status_pending",
            "may_establish",
            (
                _p("tools_impl", "tools.py:Order.status"),
                _p("policy", "policy.md:Cancel/Modify pending order"),
            ),
        ),
        GraphEdge(
            "get_order_details",
            "predicate:order_status_delivered",
            "may_establish",
            (
                _p("tools_impl", "tools.py:Order.status"),
                _p("policy", "policy.md:Return/Exchange delivered order"),
            ),
        ),
        GraphEdge(
            "predicate:order_status_pending",
            "predicate:user_confirms_write",
            "requires_before",
            (
                _p(
                    "policy",
                    "policy.md:obtain explicit user confirmation before DB updates",
                ),
            ),
        ),
        GraphEdge(
            "predicate:order_status_delivered",
            "predicate:user_confirms_write",
            "requires_before",
            (
                _p(
                    "policy",
                    "policy.md:obtain explicit user confirmation before DB updates",
                ),
            ),
        ),
        GraphEdge(
            "predicate:user_confirms_write",
            "cancel_pending_order",
            "enables",
            (
                _p("policy", "policy.md:Cancel pending order"),
                _p("tools_impl", "tools.py:cancel_pending_order status==pending"),
            ),
        ),
        GraphEdge(
            "predicate:user_confirms_write",
            "modify_pending_order_address",
            "enables",
            (
                _p("policy", "policy.md:Modify pending order"),
                _p("tools_impl", "tools.py:modify_pending_order_address"),
            ),
        ),
        GraphEdge(
            "predicate:user_confirms_write",
            "modify_pending_order_items",
            "enables",
            (
                _p("policy", "policy.md:Modify items"),
                _p("tools_impl", "tools.py:modify_pending_order_items"),
            ),
        ),
        GraphEdge(
            "predicate:user_confirms_write",
            "modify_pending_order_payment",
            "enables",
            (
                _p("policy", "policy.md:Modify payment"),
                _p("tools_impl", "tools.py:modify_pending_order_payment"),
            ),
        ),
        GraphEdge(
            "predicate:user_confirms_write",
            "return_delivered_order_items",
            "enables",
            (
                _p("policy", "policy.md:Return delivered order"),
                _p("tools_impl", "tools.py:return_delivered_order_items"),
            ),
        ),
        GraphEdge(
            "predicate:user_confirms_write",
            "exchange_delivered_order_items",
            "enables",
            (
                _p("policy", "policy.md:Exchange delivered order"),
                _p("tools_impl", "tools.py:exchange_delivered_order_items"),
            ),
        ),
        GraphEdge(
            "predicate:user_authenticated",
            "modify_user_address",
            "enables",
            (
                _p("policy", "policy.md:modify default user address"),
                _p("tools_impl", "tools.py:modify_user_address"),
                _p(
                    "human_reviewed",
                    "RTC-WRITE-CONFIRM-001",
                    "still requires explicit confirmation before call",
                ),
            ),
        ),
        GraphEdge(
            "predicate:user_authenticated",
            "transfer_to_human_agents",
            "enables",
            (
                _p("policy", "policy.md:transfer iff request cannot be handled"),
                _p("tools_impl", "tools.py:transfer_to_human_agents"),
            ),
        ),
    )
    for edge in edges:
        if not edge.provenance:
            raise ValueError(f"edge without provenance: {edge}")
        for item in edge.provenance:
            if item.kind not in {"policy", "tools_impl", "schema", "human_reviewed"}:
                raise ValueError(f"invalid provenance kind: {item.kind}")
    return ToolDependencyGraph(version=TOOL_GRAPH_VERSION, nodes=nodes, edges=edges)


def edges_for_target(graph: ToolDependencyGraph, target: str) -> list[GraphEdge]:
    return [edge for edge in graph.edges if edge.target == target]


def validate_edge_batch(edges: Iterable[Mapping[str, Any]]) -> None:
    """Reject candidate edges that lack auditable provenance."""
    for raw in edges:
        provenance = raw.get("provenance") or []
        if not provenance:
            raise ValueError(f"candidate edge rejected: missing provenance: {raw}")
