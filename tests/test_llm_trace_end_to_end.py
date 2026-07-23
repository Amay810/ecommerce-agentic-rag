# -*- coding: utf-8 -*-
"""The LLM trace must survive the whole loop and land in the trajectory.

Recording raw output inside the policy is useless if the harness drops it, so
this drives a real ``HarnessRunner`` with a scripted generator: no model is
needed, but the tool layer, the user simulator, the database and the grader all
run for real.
"""

import tempfile
import unittest
from pathlib import Path

from ecommerce_rag.domain import TaskSpec
from ecommerce_rag.harness import HarnessRunner
from ecommerce_rag.llm_policy import Generation, LLMPolicy
from ecommerce_rag.orders import connect, seed_database
from scripts.diagnose_llm_trace import build_report


def _order(db):
    conn = connect(db)
    try:
        row = dict(conn.execute("SELECT * FROM orders LIMIT 1").fetchone())
        code = conn.execute("SELECT verification_code FROM users WHERE user_id=?", (row["user_id"],)).fetchone()[0]
        return row, code
    finally:
        conn.close()


def _scripted(*outputs):
    stream = iter(outputs)

    def generate(system, user):
        try:
            return next(stream)
        except StopIteration:  # keep failing rather than crashing the run
            return "no more output"
    return generate


class LLMTraceEndToEndTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.db = Path(self._dir.name) / "env.db"
        seed_database(self.db, users=20, orders=100)
        self.order, self.code = _order(self.db)

    def tearDown(self):
        self._dir.cleanup()

    def _task(self, category="order_query"):
        return TaskSpec(
            task_id="t_trace", category=category, user_id=self.order["user_id"],
            user_goal=f"查一下订单 {self.order['order_id']} 到哪了", seed=1,
            allowed_tools=["get_order"],
            metadata={"order_id": self.order["order_id"], "verification_code": self.code,
                      "user_behavior": {"verification_code": self.code, "disclose_verification": True}},
        )

    def test_successful_tool_call_is_executed_and_traced(self):
        ask = ('{"action_type":"final_answer","tool_name":null,"arguments":{},'
               '"content":"请提供六位身份验证码。","requires_user_response":true}')
        call = ('{"action_type":"tool_call","tool_name":"get_order","arguments":'
                f'{{"order_id":"{self.order["order_id"]}","user_id":"{self.order["user_id"]}",'
                f'"verification_code":"{self.code}"}},"content":"","requires_user_response":false}}')
        done = '{"action_type":"final_answer","tool_name":null,"arguments":{},"content":"订单已查询。","requires_user_response":false}'

        policy = LLMPolicy(_scripted(ask, call, done))
        trajectory, result = HarnessRunner(self.db, None, policy).run(self._task())

        # the tool really ran against the database, not a mocked result
        self.assertEqual([c.name for c in trajectory.tool_calls], ["get_order"])
        self.assertTrue(trajectory.tool_calls[0].result.get("ok"))
        self.assertTrue(result.success)

        traces = [call.get("llm") for call in trajectory.model_calls]
        self.assertTrue(all(traces), "every model call must carry its trace")
        self.assertEqual([t["resolution"] for t in traces], ["parsed"] * 3)
        self.assertEqual(traces[1]["attempts"][0]["raw_output"], call)

    def test_parse_failure_keeps_raw_output_in_the_trajectory(self):
        policy = LLMPolicy(_scripted(Generation("I think we should look it up.", finish_reason="stop"),
                                     Generation("Still prose.", finish_reason="stop")))
        trajectory, _ = HarnessRunner(self.db, None, policy).run(self._task())

        trace = trajectory.model_calls[0]["llm"]
        self.assertEqual(trace["resolution"], "fallback_handoff")
        self.assertEqual(trace["final_stage"], "no_json_object")
        self.assertEqual([a["raw_output"] for a in trace["attempts"]],
                         ["I think we should look it up.", "Still prose."])

    def test_diagnose_report_attributes_the_failure(self):
        truncated = Generation('{"action_type":"tool_call","tool_name":"get_order","arguments":{"order_id":"O0',
                               finish_reason="length", completion_tokens=512, truncated=True)
        policy = LLMPolicy(_scripted(truncated, truncated), generator_meta={"backend": "local", "model": "fake"})
        trajectory, _ = HarnessRunner(self.db, None, policy).run(self._task())

        report = build_report([trajectory.to_dict()])
        self.assertTrue(report["instrumented"])
        self.assertEqual(report["quality"]["effective_action_parse_rate"], 0.0)
        self.assertEqual(report["quality"]["fallback_only_trajectory_rate"], 1.0)
        self.assertEqual(report["quality"]["avg_non_fallback_tool_calls"], 0.0)
        self.assertEqual(report["quality"]["trajectories_with_real_tool_call_rate"], 0.0)
        self.assertEqual(report["quality"]["truncation_rate"], 1.0)
        self.assertEqual(report["parse_stages"], {"unbalanced_json": 2})
        self.assertEqual(report["finish_reasons"], {"length": 2})
        # the sample is the thing that was missing last time
        self.assertIn("unbalanced_json", report["failure_samples"])
        self.assertIn('"action_type"', report["failure_samples"]["unbalanced_json"][0]["raw_output"])

    def test_handoff_reaches_the_tool_and_succeeds(self):
        # Live smoke: every handoff failed because the contract said to send {}
        # while escalate_to_human requires a reason. Same shape, now end to end.
        policy = LLMPolicy(_scripted(
            '{"action_type":"handoff","tool_name":null,"arguments":'
            '{"reason":"identity_verification_failed","order_id":"' + self.order["order_id"] + '"},'
            '"content":"已为您转接人工客服。","requires_user_response":false}'))
        trajectory, _ = HarnessRunner(self.db, None, policy).run(self._task())

        self.assertEqual([c.name for c in trajectory.tool_calls], ["escalate_to_human"])
        call = trajectory.tool_calls[0]
        self.assertTrue(call.result.get("ok"), call.result)
        self.assertIsNone(call.error)
        self.assertEqual(call.arguments["reason"], "identity_verification_failed")

    def test_policy_supplied_user_id_cannot_override_the_session(self):
        # A handoff on someone else's behalf must be impossible even if the parser
        # is bypassed, so the harness injects identity last.
        from ecommerce_rag.domain import AgentAction

        class Impersonating:
            privileged = False

            def act(self, observation):
                return AgentAction.handoff("x", user_id="U9999")

        trajectory, _ = HarnessRunner(self.db, None, Impersonating()).run(self._task())
        self.assertEqual(trajectory.tool_calls[0].arguments["user_id"], self.order["user_id"])

    def test_order_tool_with_a_blank_code_is_refused_before_execution(self):
        from ecommerce_rag.tools import RetailTools

        tools = RetailTools(self.db)
        result = tools.call("get_order", order_id=self.order["order_id"],
                            user_id=self.order["user_id"], verification_code="")
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "verification_code_required")
        self.assertEqual(tools.guardrails[0]["tool"], "get_order")
        self.assertTrue(tools.guardrails[0]["blocked"])

    def test_guard_does_not_coerce_the_verification_code(self):
        from ecommerce_rag.tools import RetailTools

        # Each of these previously cleared the guard: strip() rescued the padded
        # form, str() rescued the integer, and `\d` matches full-width digits.
        for code in (" " + self.code + " ", int(self.code), "１２３４５６", None, ["123456"]):
            tools = RetailTools(self.db)
            result = tools.call("get_order", order_id=self.order["order_id"],
                                user_id=self.order["user_id"], verification_code=code)
            self.assertEqual(result["error"], "verification_code_required", f"{code!r} slipped through")
            self.assertEqual(tools.guardrails[0]["reason"], "verification_code_required")

    def test_write_tool_with_a_blank_code_cannot_change_the_database(self):
        from ecommerce_rag.tools import RetailTools

        tools = RetailTools(self.db)
        before = _order(self.db)[0]["return_status"]
        result = tools.call("create_return_request", order_id=self.order["order_id"],
                            user_id=self.order["user_id"], verification_code="", confirmed=True)
        self.assertFalse(result["ok"])
        self.assertFalse(result["changed"])
        self.assertEqual(_order(self.db)[0]["return_status"], before)

    def test_old_store_without_traces_is_reported_as_uninstrumented(self):
        report = build_report([{"trajectory_id": "old", "model_calls": [{"step": 0}], "tool_calls": []}])
        self.assertFalse(report["instrumented"])
        self.assertIsNone(report["quality"]["effective_action_parse_rate"])


if __name__ == "__main__":
    unittest.main()
