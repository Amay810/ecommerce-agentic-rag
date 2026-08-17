# -*- coding: utf-8 -*-
"""Pinned versions used by the Retail Task Compiler."""

from __future__ import annotations

from ecommerce_rag.retail_protocol import (
    RETAIL_ALL_TOOLS,
    RETAIL_GENERIC_TOOLS,
    RETAIL_READ_TOOLS,
    RETAIL_WRITE_TOOLS,
)

GENERATOR_VERSION = "retail_task_compiler.v1.m1_structures"
GENERATOR_VERSION_V0 = "retail_task_compiler.v0.cancel_pending"
SOURCE_POLICY_VERSION = "tau3_retail.v1.0.1.policy.md"
TAU2_COMMIT = "fc0055dc4e0a316c3f83133267fbd6faaa770992"
TOOL_GRAPH_VERSION = "retail_tool_graph.v0"

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
