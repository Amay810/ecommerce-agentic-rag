import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from ecommerce_rag.domain import TaskSpec
from ecommerce_rag.domain import AgentAction
from ecommerce_rag.harness import HarnessRunner, RulePolicy
from ecommerce_rag.orders import connect, seed_database
from ecommerce_rag.tools import RetailTools


def _eligible(db):
    conn = connect(db)
    try:
        order = dict(conn.execute("SELECT * FROM orders WHERE status='delivered' AND quality_issue=1 LIMIT 1").fetchone())
        code = conn.execute("SELECT verification_code FROM users WHERE user_id=?", (order["user_id"],)).fetchone()[0]
        return order, code
    finally:
        conn.close()


class HarnessToolTests(unittest.TestCase):
 def test_policy_observation_excludes_hidden_gold(self):
    class SpyPolicy:
        privileged = False
        observed = None
        def act(self, observation):
            self.observed = asdict(observation)
            return AgentAction.answer("done")
    with tempfile.TemporaryDirectory() as d:
        db = Path(d) / "env.db"; seed_database(db, users=20, orders=100)
        policy = SpyPolicy()
        task = TaskSpec("hidden_01", "secret_category", "U0001", "hello", 7,
                        gold_doc_ids=["secret_doc"], allowed_tools=[], expected_state={},
                        metadata={"answer": "secret", "verification_code": "123456"}, split="locked")
        _, result = HarnessRunner(db, policy=policy).run(task)
        payload = str(policy.observed)
        assert result.leakage_checked
        for secret in ("secret_category", "secret_doc", "secret", "123456", "allowed_tools", "expected_state"):
            assert secret not in payload

 def test_rule_policy_gets_verification_and_confirmation_from_user_simulator(self):
    with tempfile.TemporaryDirectory() as d:
        db = Path(d) / "env.db"; seed_database(db, users=20, orders=100)
        order, code = _eligible(db)
        task = TaskSpec("return_hidden", "return", order["user_id"], f"订单 {order['order_id']} 想退货", 8,
                        allowed_tools=["check_return_eligibility", "create_return_request"],
                        expected_state={order["order_id"]: {"return_status": "requested"}},
                        initial_state={order["order_id"]: {"return_status": None, "version": 0}},
                        metadata={"order_id": order["order_id"], "verification_code": code,
                                  "user_behavior": {"verification_code": code, "confirmation": True}}, split="locked")
        trajectory, result = HarnessRunner(db, policy=RulePolicy()).run(task)
        assert result.success and result.leakage_checked
        assert len(trajectory.user_simulator_spans) == 2
        assert [c.name for c in trajectory.tool_calls] == ["check_return_eligibility", "create_return_request"]

 def test_illegal_return_is_blocked_without_state_change(self):
    with tempfile.TemporaryDirectory() as d:
        db = Path(d) / "env.db"; seed_database(db, users=20, orders=100)
        order, _ = _eligible(db); tools = RetailTools(db)
        result = tools.call("create_return_request", order_id=order["order_id"], user_id=order["user_id"], verification_code="wrong", confirmed=True)
        assert result["changed"] is False
        conn = connect(db)
        try: assert conn.execute("SELECT return_status FROM orders WHERE order_id=?", (order["order_id"],)).fetchone()[0] is None
        finally: conn.close()


 def test_write_requires_confirmation_and_is_idempotent(self):
    with tempfile.TemporaryDirectory() as d:
        db = Path(d) / "env.db"; seed_database(db, users=20, orders=100)
        order, code = _eligible(db); tools = RetailTools(db)
        blocked = tools.call("create_return_request", order_id=order["order_id"], user_id=order["user_id"], verification_code=code, confirmed=False)
        assert blocked["changed"] is False and blocked["error"] == "confirmation_required"
        ok = tools.call("create_return_request", order_id=order["order_id"], user_id=order["user_id"], verification_code=code, confirmed=True)
        again = tools.call("create_return_request", order_id=order["order_id"], user_id=order["user_id"], verification_code=code, confirmed=True)
        assert ok["changed"] is True and again["changed"] is False


 def test_harness_terminal_state_is_deterministic(self):
    with tempfile.TemporaryDirectory() as d:
        db = Path(d) / "env.db"; seed_database(db, users=20, orders=100)
        order, code = _eligible(db)
        task = TaskSpec("return_01", "return", order["user_id"], "退货", 7,
                        allowed_tools=["check_return_eligibility", "create_return_request"],
                        expected_state={order["order_id"]: {"return_status": "requested"}},
                        initial_state={order["order_id"]: {"return_status": None, "version": 0}},
                        metadata={"order_id": order["order_id"], "verification_code": code, "confirmed": True})
        runner = HarnessRunner(db)
        _, first = runner.run(task); _, second = runner.run(task)
        assert first.success and second.success and first.reward == second.reward
