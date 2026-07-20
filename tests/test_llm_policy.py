import unittest

from ecommerce_rag.domain import AgentObservation
from ecommerce_rag.harness import TOOL_SCHEMAS
from ecommerce_rag.llm_policy import LLMPolicy


class LLMPolicyTests(unittest.TestCase):
    def test_invalid_json_is_retried_once(self):
        outputs = iter(["not json", '{"action_type":"tool_call","tool_name":"search_catalog","arguments":{"query":"camera"}}'])
        policy = LLMPolicy(lambda _: next(outputs))
        action = policy.act(AgentObservation("camera", {"user_id":"U0001"}, [{"role":"user","content":"camera"}], TOOL_SCHEMAS))
        self.assertEqual(action.tool_name, "search_catalog")
        self.assertEqual(policy.retry_count, 1)

    def test_unknown_tool_fails_closed_to_handoff(self):
        policy = LLMPolicy(lambda _: '{"action_type":"tool_call","tool_name":"delete_order","arguments":{}}')
        action = policy.act(AgentObservation("x", {"user_id":"U0001"}, [], TOOL_SCHEMAS))
        self.assertEqual(action.action_type, "handoff")
        self.assertEqual(action.arguments["reason"], "model_action_parse_failure")


if __name__ == "__main__": unittest.main()
