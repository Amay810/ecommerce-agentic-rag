# -*- coding: utf-8 -*-
"""Typed tool contracts — the single source of truth for the action protocol.

The parameter schemas are plain JSON Schema, so one definition is shown to the
policy inside ``AgentObservation``, validates arguments before a tool executes,
and can back a native tool-calling API or a constrained decoder. Providers wrap
these differently (``function.parameters``, ``input_schema``, …), so the outer
envelope still needs a small adapter — the parameter schema itself is reusable
as is.

``tests/test_tool_schema.py`` asserts every schema agrees with the corresponding
``RetailTools`` method: parameter names, the required set, Python type
annotations and declared defaults, so the contract cannot silently drift.
"""

from __future__ import annotations

from typing import Any

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "search_catalog",
        "description": "Search the product catalogue. Use for product questions and recommendations.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural-language product query."},
                "top_k": {"type": "integer", "default": 5, "description": "Number of results."},
                "category": {"type": ["string", "null"], "default": None, "description": "Optional category filter."},
                "max_price": {"type": ["number", "null"], "default": None, "description": "Optional budget ceiling."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_product",
        "description": "Fetch one product card by id.",
        "parameters": {
            "type": "object",
            "properties": {"product_id": {"type": "string"}},
            "required": ["product_id"],
        },
    },
    {
        "name": "compare_products",
        "description": "Compare two or more products by id.",
        "parameters": {
            "type": "object",
            "properties": {"product_ids": {"type": "array", "items": {"type": "string"}}},
            "required": ["product_ids"],
        },
    },
    {
        "name": "get_policy",
        "description": "Fetch an after-sales policy document.",
        "parameters": {
            "type": "object",
            "properties": {"policy_type": {"type": "string", "description": "e.g. 退换货 / 保修 / 物流 / 发票 / 退款"}},
            "required": ["policy_type"],
        },
    },
    {
        "name": "get_order",
        "description": "Read one order. Requires the caller's six-digit verification code.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "user_id": {"type": "string"},
                "verification_code": {"type": "string", "description": "Six digits, supplied by the user."},
            },
            "required": ["order_id", "user_id", "verification_code"],
        },
    },
    {
        "name": "check_return_eligibility",
        "description": "Check whether an order may be returned. Read-only.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "user_id": {"type": "string"},
                "verification_code": {"type": "string"},
            },
            "required": ["order_id", "user_id", "verification_code"],
        },
    },
    {
        "name": "create_return_request",
        "description": "WRITE. Only after eligibility passes and the user explicitly confirms.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "user_id": {"type": "string"},
                "verification_code": {"type": "string"},
                "confirmed": {"type": "boolean", "description": "True only if the user said so in this conversation."},
            },
            "required": ["order_id", "user_id", "verification_code", "confirmed"],
        },
    },
    {
        "name": "escalate_to_human",
        "description": "Hand the conversation to a human agent.",
        "parameters": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "reason": {"type": "string"},
                "order_id": {"type": ["string", "null"], "default": None},
            },
            "required": ["user_id", "reason"],
        },
    },
]

SCHEMA_BY_NAME: dict[str, dict[str, Any]] = {schema["name"]: schema for schema in TOOL_SCHEMAS}

_PYTHON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
}


class ToolArgumentError(ValueError):
    """Raised when arguments do not satisfy a tool's schema."""


def _type_ok(value: Any, spec: Any) -> bool:
    names = spec if isinstance(spec, list) else [spec]
    for name in names:
        if name == "null":
            if value is None:
                return True
            continue
        expected = _PYTHON_TYPES.get(name)
        if not expected:
            continue
        # bool is a subclass of int; keep integer/number strict so True is not a count
        if bool not in expected and isinstance(value, bool):
            continue
        if isinstance(value, expected):
            return True
    return False


def validate_arguments(tool_name: str, arguments: dict[str, Any]) -> None:
    """Raise :class:`ToolArgumentError` unless ``arguments`` satisfies the schema."""
    schema = SCHEMA_BY_NAME.get(tool_name)
    if schema is None:
        raise ToolArgumentError(f"unknown tool: {tool_name}")
    if not isinstance(arguments, dict):
        raise ToolArgumentError("arguments must be an object")

    parameters = schema["parameters"]
    properties: dict[str, Any] = parameters["properties"]

    missing = [name for name in parameters.get("required", []) if name not in arguments]
    if missing:
        raise ToolArgumentError(f"{tool_name}: missing required argument(s): {', '.join(missing)}")

    unknown = [name for name in arguments if name not in properties]
    if unknown:
        raise ToolArgumentError(f"{tool_name}: unknown argument(s): {', '.join(sorted(unknown))}")

    for name, value in arguments.items():
        spec = properties[name]
        if not _type_ok(value, spec.get("type")):
            raise ToolArgumentError(
                f"{tool_name}.{name}: expected {spec.get('type')}, got {type(value).__name__}"
            )
        if spec.get("type") == "array":
            item_type = (spec.get("items") or {}).get("type")
            bad = [x for x in value if item_type and not _type_ok(x, item_type)]
            if bad:
                raise ToolArgumentError(f"{tool_name}.{name}: items must be {item_type}")


def prompt_block() -> str:
    """Compact, deterministic rendering of the tool contract for a prompt."""
    lines = []
    for schema in TOOL_SCHEMAS:
        parameters = schema["parameters"]
        required = set(parameters.get("required", []))
        fields = ", ".join(
            f"{name}: {spec.get('type')}{'' if name in required else ' (optional)'}"
            for name, spec in parameters["properties"].items()
        )
        lines.append(f"- {schema['name']}({fields}) — {schema['description']}")
    return "\n".join(lines)
