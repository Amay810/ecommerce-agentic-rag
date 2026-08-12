# -*- coding: utf-8 -*-
"""Pinned versions used by the Retail Task Compiler."""

from __future__ import annotations

GENERATOR_VERSION = "retail_task_compiler.v1.m1_structures"
GENERATOR_VERSION_V0 = "retail_task_compiler.v0.cancel_pending"
SOURCE_POLICY_VERSION = "tau3_retail.v1.0.1.policy.md"
TAU2_COMMIT = "fc0055dc4e0a316c3f83133267fbd6faaa770992"
TOOL_GRAPH_VERSION = "retail_tool_graph.v0"

# Tools present in pinned τ³ Retail tools.py (think is commented out upstream).
RETAIL_READ_TOOLS = frozenset(
    {
        "find_user_id_by_email",
        "find_user_id_by_name_zip",
        "get_order_details",
        "get_product_details",
        "get_item_details",
        "get_user_details",
        "list_all_product_types",
    }
)
RETAIL_WRITE_TOOLS = frozenset(
    {
        "cancel_pending_order",
        "exchange_delivered_order_items",
        "modify_pending_order_address",
        "modify_pending_order_items",
        "modify_pending_order_payment",
        "modify_user_address",
        "return_delivered_order_items",
    }
)
RETAIL_GENERIC_TOOLS = frozenset({"calculate", "transfer_to_human_agents"})
RETAIL_ALL_TOOLS = RETAIL_READ_TOOLS | RETAIL_WRITE_TOOLS | RETAIL_GENERIC_TOOLS

WRITE_TOOL_TO_FAMILY = {
    "cancel_pending_order": "cancel_pending",
    "exchange_delivered_order_items": "exchange_delivered",
    "modify_pending_order_address": "modify_pending_address",
    "modify_pending_order_items": "modify_pending_items",
    "modify_pending_order_payment": "modify_pending_payment",
    "modify_user_address": "modify_user_address",
    "return_delivered_order_items": "return_delivered",
    "transfer_to_human_agents": "handoff",
}
