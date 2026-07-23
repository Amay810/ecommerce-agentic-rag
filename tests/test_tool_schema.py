# -*- coding: utf-8 -*-
"""The tool contract must never drift from the implementation."""

import inspect
import unittest

from ecommerce_rag.tool_schema import (
    SCHEMA_BY_NAME,
    TOOL_SCHEMAS,
    ToolArgumentError,
    prompt_block,
    validate_arguments,
)
from ecommerce_rag.tools import READ_TOOLS, WRITE_TOOLS, RetailTools


def _signature(name: str) -> inspect.Signature:
    return inspect.signature(getattr(RetailTools, name))


class SchemaMatchesImplementationTests(unittest.TestCase):
    def test_every_schema_has_a_method(self):
        for schema in TOOL_SCHEMAS:
            self.assertTrue(callable(getattr(RetailTools, schema["name"], None)),
                            f"{schema['name']} has a schema but no implementation")

    def test_every_declared_tool_has_a_schema(self):
        for name in READ_TOOLS | WRITE_TOOLS:
            self.assertIn(name, SCHEMA_BY_NAME, f"{name} is a declared tool with no schema")

    def test_schema_properties_match_the_signature(self):
        for schema in TOOL_SCHEMAS:
            params = [p for p in _signature(schema["name"]).parameters if p != "self"]
            self.assertEqual(sorted(schema["parameters"]["properties"]), sorted(params),
                             f"{schema['name']}: schema properties differ from the signature")

    def test_required_matches_parameters_without_defaults(self):
        for schema in TOOL_SCHEMAS:
            signature = _signature(schema["name"])
            mandatory = sorted(name for name, p in signature.parameters.items()
                               if name != "self" and p.default is inspect.Parameter.empty)
            self.assertEqual(sorted(schema["parameters"].get("required", [])), mandatory,
                             f"{schema['name']}: required set differs from the signature")


class ValidateArgumentsTests(unittest.TestCase):
    def test_accepts_a_well_formed_call(self):
        validate_arguments("get_order", {"order_id": "O000001", "user_id": "U0001", "verification_code": "123456"})
        validate_arguments("search_catalog", {"query": "耳机", "top_k": 3, "max_price": 199.5, "category": None})

    def test_missing_required_argument_is_reported_by_name(self):
        with self.assertRaises(ToolArgumentError) as ctx:
            validate_arguments("get_order", {"order_id": "O000001"})
        self.assertIn("user_id", str(ctx.exception))
        self.assertIn("verification_code", str(ctx.exception))

    def test_unknown_argument_is_rejected(self):
        with self.assertRaises(ToolArgumentError) as ctx:
            validate_arguments("get_policy", {"policy_type": "退换货", "locale": "zh"})
        self.assertIn("locale", str(ctx.exception))

    def test_wrong_type_is_rejected(self):
        # the failure mode that matters: a model emitting "true" instead of true
        with self.assertRaises(ToolArgumentError):
            validate_arguments("create_return_request", {"order_id": "O1", "user_id": "U1",
                                                         "verification_code": "123456", "confirmed": "true"})

    def test_boolean_is_not_accepted_as_a_number(self):
        with self.assertRaises(ToolArgumentError):
            validate_arguments("search_catalog", {"query": "x", "top_k": True})

    def test_nullable_field_accepts_none_but_not_a_wrong_type(self):
        validate_arguments("escalate_to_human", {"user_id": "U1", "reason": "r", "order_id": None})
        with self.assertRaises(ToolArgumentError):
            validate_arguments("escalate_to_human", {"user_id": "U1", "reason": "r", "order_id": 12})

    def test_array_item_type_is_enforced(self):
        validate_arguments("compare_products", {"product_ids": ["P00001", "P00002"]})
        with self.assertRaises(ToolArgumentError):
            validate_arguments("compare_products", {"product_ids": [1, 2]})

    def test_unknown_tool_is_rejected(self):
        with self.assertRaises(ToolArgumentError):
            validate_arguments("drop_database", {})


class PromptBlockTests(unittest.TestCase):
    def test_prompt_block_lists_every_tool_and_is_deterministic(self):
        block = prompt_block()
        for schema in TOOL_SCHEMAS:
            self.assertIn(schema["name"], block)
        self.assertEqual(block, prompt_block())


if __name__ == "__main__":
    unittest.main()
