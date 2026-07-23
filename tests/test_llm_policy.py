# -*- coding: utf-8 -*-
"""Action-protocol parsing and the observability trace behind it."""

import unittest

from ecommerce_rag.domain import AgentObservation
from ecommerce_rag.harness import TOOL_SCHEMAS
from ecommerce_rag.llm_policy import Generation, LLMPolicy

#: A fully specified action: all five envelope fields, nothing around it.
GOOD = ('{"action_type":"tool_call","tool_name":"search_catalog","arguments":{"query":"camera"},'
        '"content":"","requires_user_response":false}')


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

    def test_only_the_first_balanced_object_is_read_but_the_extra_is_recorded(self):
        # A greedy {.*} would span both objects and fail to decode, hiding the cause.
        # The protocol says exactly one object, so recovering must not look clean.
        policy = LLMPolicy(_scripted(GOOD + '\n{"action_type":"handoff"}'))
        action = policy.act(_observation())
        self.assertEqual(action.tool_name, "search_catalog")
        self.assertEqual(policy.last_trace["resolution"], "parsed_with_violations")
        self.assertIn("content_outside_json_object", policy.last_trace["envelope_violations"])


class ActionEnvelopeTests(unittest.TestCase):
    """Envelope fields are type-checked, never coerced."""

    def _reject(self, payload: str) -> str:
        policy = LLMPolicy(_scripted(payload, payload))
        action = policy.act(_observation())
        self.assertEqual(action.action_type, "handoff")
        return policy.last_trace["final_stage"]

    def test_string_false_is_not_coerced_to_true(self):
        # bool("false") is True, which silently flips a multi-turn interaction
        stage = self._reject('{"action_type":"final_answer","arguments":{},"content":"hi",'
                             '"requires_user_response":"false"}')
        self.assertEqual(stage, "bad_requires_user_response_type")

    def test_numeric_content_is_not_coerced_to_string(self):
        stage = self._reject('{"action_type":"final_answer","arguments":{},"content":7,'
                             '"requires_user_response":false}')
        self.assertEqual(stage, "bad_content_type")

    def test_non_tool_action_may_not_carry_a_tool_name(self):
        stage = self._reject('{"action_type":"final_answer","tool_name":123,"arguments":{},'
                             '"content":"x","requires_user_response":false}')
        self.assertEqual(stage, "tool_name_on_non_tool_action")

    def test_list_arguments_are_rejected_rather_than_emptied(self):
        # `value.get("arguments") or {}` turned [] into {} before the isinstance
        # check could ever run, making that whole branch unreachable
        stage = self._reject('{"action_type":"tool_call","tool_name":"search_catalog",'
                             '"arguments":[],"content":"","requires_user_response":false}')
        self.assertEqual(stage, "arguments_not_object")

    def test_null_optional_fields_are_accepted(self):
        policy = LLMPolicy(_scripted('{"action_type":"final_answer","tool_name":null,"arguments":null,'
                                     '"content":null,"requires_user_response":null}'))
        action = policy.act(_observation())
        self.assertEqual(action.action_type, "final_answer")
        self.assertEqual(action.content, "")
        self.assertFalse(action.requires_user_response)

    def test_missing_envelope_fields_are_recorded_not_silently_defaulted(self):
        # The protocol names five fields; filling them in for the model would hide
        # a real format deviation behind a clean-looking parse.
        policy = LLMPolicy(_scripted('{"action_type":"final_answer"}'))
        action = policy.act(_observation())
        self.assertEqual(action.action_type, "final_answer")
        self.assertEqual(policy.last_trace["resolution"], "parsed_with_violations")
        violation = next(v for v in policy.last_trace["envelope_violations"]
                         if v.startswith("missing_envelope_field"))
        for field in ("tool_name", "arguments", "content", "requires_user_response"):
            self.assertIn(field, violation)

    def test_markdown_fence_is_a_violation_not_noise(self):
        policy = LLMPolicy(_scripted('```json\n{"action_type":"final_answer","tool_name":null,'
                                     '"arguments":{},"content":"x","requires_user_response":false}\n```'))
        action = policy.act(_observation())
        self.assertEqual(action.action_type, "final_answer")
        self.assertIn("markdown_fence", policy.last_trace["envelope_violations"])
        # the fence alone must not also be reported as stray prose
        self.assertNotIn("content_outside_json_object", policy.last_trace["envelope_violations"])

    def test_bare_language_tag_without_a_fence_is_stray_prose(self):
        # the language tag is fence syntax; without a fence it is just text, and
        # discounting it would let non-compliant output count as strict
        policy = LLMPolicy(_scripted('json\n{"action_type":"final_answer","tool_name":null,'
                                     '"arguments":{},"content":"x","requires_user_response":false}'))
        policy.act(_observation())
        self.assertEqual(policy.last_trace["resolution"], "parsed_with_violations")
        self.assertIn("content_outside_json_object", policy.last_trace["envelope_violations"])
        self.assertNotIn("markdown_fence", policy.last_trace["envelope_violations"])

    def test_a_fully_specified_object_has_no_violations(self):
        policy = LLMPolicy(_scripted('{"action_type":"final_answer","tool_name":null,"arguments":{},'
                                     '"content":"done","requires_user_response":false}'))
        policy.act(_observation())
        self.assertEqual(policy.last_trace["resolution"], "parsed")
        self.assertEqual(policy.last_trace["envelope_violations"], [])

    def test_unknown_envelope_field_is_recoverable_but_recorded(self):
        policy = LLMPolicy(_scripted('{"action_type":"final_answer","tool_name":null,"arguments":{},'
                                     '"content":"x","requires_user_response":false,"reasoning":"..."}'))
        policy.act(_observation())
        self.assertEqual(policy.last_trace["resolution"], "parsed_with_violations")
        self.assertIn("unknown_envelope_field:reasoning", policy.last_trace["envelope_violations"])


class GenerationErrorTests(unittest.TestCase):
    """A backend that raises is the one path that would bypass all instrumentation."""

    @staticmethod
    def _raising(exc: Exception):
        def generate(system, user):
            raise exc
        return generate

    def test_backend_exception_is_recorded_not_propagated(self):
        policy = LLMPolicy(self._raising(TypeError("apply_chat_template() got an unexpected keyword")))
        action = policy.act(_observation())
        self.assertEqual(action.action_type, "handoff")
        attempts = policy.last_trace["attempts"]
        self.assertEqual([a["parse_stage"] for a in attempts], ["generation_error", "generation_error"])
        self.assertEqual(attempts[0]["generation_error_type"], "TypeError")
        self.assertIn("apply_chat_template", attempts[0]["parse_error"])

    def test_generation_failure_has_its_own_fallback_reason(self):
        # must be distinguishable from "the model returned unusable text"
        policy = LLMPolicy(self._raising(RuntimeError("CUDA out of memory")))
        action = policy.act(_observation())
        self.assertEqual(action.arguments["reason"], "model_generation_error")
        self.assertEqual(policy.last_trace["fallback_reason"], "model_generation_error")

    def test_transient_failure_then_success_still_yields_an_action(self):
        calls = {"n": 0}

        def generate(system, user):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("transient")
            return GOOD
        policy = LLMPolicy(generate)
        action = policy.act(_observation())
        self.assertEqual(action.tool_name, "search_catalog")
        self.assertEqual(policy.last_trace["attempts"][0]["parse_stage"], "generation_error")

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
