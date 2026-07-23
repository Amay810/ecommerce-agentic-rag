# -*- coding: utf-8 -*-
"""Action-protocol parsing and the observability trace behind it."""

import unittest

from ecommerce_rag.domain import AgentObservation
from ecommerce_rag.harness import TOOL_SCHEMAS
from ecommerce_rag.llm_policy import Generation, LLMPolicy

GOOD = '{"action_type":"tool_call","tool_name":"search_catalog","arguments":{"query":"camera"}}'


def _observation(message: str = "camera", history=None) -> AgentObservation:
    return AgentObservation(message, {"user_id": "U0001"},
                            history if history is not None else [{"role": "user", "content": message}],
                            TOOL_SCHEMAS)


def _scripted(*outputs):
    """A generator that replays fixed outputs, ignoring the prompt."""
    stream = iter(outputs)

    def generate(system, user):
        return next(stream)
    return generate


class ActionParsingTests(unittest.TestCase):
    def test_invalid_json_is_retried_once(self):
        policy = LLMPolicy(_scripted("not json", GOOD))
        action = policy.act(_observation())
        self.assertEqual(action.tool_name, "search_catalog")
        self.assertEqual(policy.retry_count, 1)

    def test_unknown_tool_fails_closed_to_handoff(self):
        policy = LLMPolicy(_scripted(*['{"action_type":"tool_call","tool_name":"delete_order","arguments":{}}'] * 2))
        action = policy.act(_observation("x", []))
        self.assertEqual(action.action_type, "handoff")
        self.assertEqual(action.arguments["reason"], "model_action_parse_failure")

    def test_prose_wrapped_json_is_accepted(self):
        raw = f"Sure! Here is the action:\n```json\n{GOOD}\n```\nHope that helps."
        action = LLMPolicy(_scripted(raw)).act(_observation())
        self.assertEqual(action.tool_name, "search_catalog")

    def test_only_the_first_balanced_object_is_read(self):
        # A greedy {.*} would span both objects and fail to decode, hiding the cause.
        action = LLMPolicy(_scripted(GOOD + '\n{"action_type":"handoff"}')).act(_observation())
        self.assertEqual(action.tool_name, "search_catalog")

    def test_arguments_are_validated_against_the_tool_schema(self):
        # right tool, wrong argument type — must not reach the tool layer
        bad = ('{"action_type":"tool_call","tool_name":"create_return_request","arguments":'
               '{"order_id":"O1","user_id":"U1","verification_code":"123456","confirmed":"yes"}}')
        policy = LLMPolicy(_scripted(bad, bad))
        action = policy.act(_observation())
        self.assertEqual(action.action_type, "handoff")
        self.assertEqual(policy.last_trace["final_stage"], "schema_violation")

    def test_missing_required_argument_is_a_schema_violation(self):
        bad = '{"action_type":"tool_call","tool_name":"get_order","arguments":{"order_id":"O1"}}'
        policy = LLMPolicy(_scripted(bad, bad))
        policy.act(_observation())
        self.assertEqual(policy.last_trace["final_stage"], "schema_violation")


class ObservabilityTraceTests(unittest.TestCase):
    def test_successful_call_records_the_raw_output(self):
        policy = LLMPolicy(_scripted(GOOD))
        policy.act(_observation())
        trace = policy.last_trace
        self.assertEqual(trace["resolution"], "parsed")
        self.assertEqual(len(trace["attempts"]), 1)
        attempt = trace["attempts"][0]
        self.assertEqual(attempt["raw_output"], GOOD)
        self.assertTrue(attempt["parse_ok"])
        self.assertGreater(attempt["system_chars"], 0)
        self.assertGreater(attempt["user_chars"], 0)

    def test_every_failed_attempt_keeps_its_raw_output_and_stage(self):
        policy = LLMPolicy(_scripted("not json", "still not json"))
        policy.act(_observation())
        trace = policy.last_trace
        self.assertEqual(trace["resolution"], "fallback_handoff")
        self.assertEqual([a["raw_output"] for a in trace["attempts"]], ["not json", "still not json"])
        self.assertEqual([a["parse_stage"] for a in trace["attempts"]], ["no_json_object", "no_json_object"])

    def test_truncated_generation_is_distinguishable_from_bad_json(self):
        # the failure mode a token budget produces: the object never closes
        truncated = Generation('{"action_type":"tool_call","tool_name":"search_catalog","arguments":{"query":"cam',
                               finish_reason="length", completion_tokens=512, truncated=True)
        policy = LLMPolicy(_scripted(truncated, truncated))
        policy.act(_observation())
        attempt = policy.last_trace["attempts"][0]
        self.assertEqual(attempt["parse_stage"], "unbalanced_json")
        self.assertTrue(attempt["truncated"])
        self.assertEqual(attempt["finish_reason"], "length")

    def test_empty_output_is_its_own_stage(self):
        policy = LLMPolicy(_scripted(Generation("", finish_reason="stop"), Generation("  ")))
        policy.act(_observation())
        self.assertEqual([a["parse_stage"] for a in policy.last_trace["attempts"]],
                         ["empty_output", "empty_output"])

    def test_generator_metadata_is_carried_into_the_trace(self):
        meta = {"backend": "local", "model": "Qwen3-4B", "enable_thinking_supported": False}
        policy = LLMPolicy(_scripted("nope", "nope"), generator_meta=meta)
        policy.act(_observation())
        self.assertEqual(policy.last_trace["generator"], meta)

    def test_retry_prompt_reports_the_previous_error(self):
        seen = []

        def generate(system, user):
            seen.append(user)
            return "not json"
        LLMPolicy(generate).act(_observation())
        self.assertNotIn("could not be parsed", seen[0])
        self.assertIn("no_json_object", seen[1])


if __name__ == "__main__":
    unittest.main()
