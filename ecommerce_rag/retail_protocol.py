# -*- coding: utf-8 -*-
"""Retail tool-surface facts shared by runtime, audit, and evaluation.

These names are protocol/runtime facts for pinned τ³ Retail plus the local
return-write surface. They are not dataset-compiler artifacts.
"""

from __future__ import annotations

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
