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
        self.assertEqual(report["quality"]["non_fallback_tool_call_rate"], 0.0)
        self.assertEqual(report["quality"]["truncation_rate"], 1.0)
        self.assertEqual(report["parse_stages"], {"unbalanced_json": 2})
        self.assertEqual(report["finish_reasons"], {"length": 2})
        # the sample is the thing that was missing last time
        self.assertIn("unbalanced_json", report["failure_samples"])
        self.assertIn('"action_type"', report["failure_samples"]["unbalanced_json"][0]["raw_output"])

    def test_old_store_without_traces_is_reported_as_uninstrumented(self):
        report = build_report([{"trajectory_id": "old", "model_calls": [{"step": 0}], "tool_calls": []}])
        self.assertFalse(report["instrumented"])
        self.assertIsNone(report["quality"]["effective_action_parse_rate"])


if __name__ == "__main__":
    unittest.main()
