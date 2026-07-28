import unittest
import tempfile
from pathlib import Path

from ecommerce_rag.domain import AgentObservation, Trajectory
from ecommerce_rag.evidence import evidence_from_tool_call, verify_answer
from ecommerce_rag.evidence_policy import EvidenceGroundedPolicy
from ecommerce_rag.harness import HarnessRunner, grade
from ecommerce_rag.domain import TaskSpec
from ecommerce_rag.orders import seed_database
from ecommerce_rag.tool_schema import TOOL_SCHEMAS


def _ledger(*rows):
    return [
        {
            "evidence_id": f"E{i}", "source_id": source, "tool_call_id": "c1",
            "tool_name": "test", "kind": "field", "field": field,
            "value": value, "text": text, "confidence": 1.0, "updated_at": None,
        }
        for i, (source, field, value, text) in enumerate(rows, 1)
    ]


class EvidenceLedgerTests(unittest.TestCase):
    def test_policy_result_is_normalized_to_atomic_rows(self):
        rows = evidence_from_tool_call(
            "get_policy", {"policy_type": "return"},
            {"ok": True, "policies": [{"doc_id": "policy:POL001", "title": "七天无理由退货",
                                           "text": "签收后7天内可申请。", "updated_at": "2026-06-01"}]},
            "call-1",
        )
        self.assertEqual([row["evidence_id"] for row in rows], ["E1", "E2", "E3"])
        self.assertEqual({row["source_id"] for row in rows}, {"policy:POL001"})
        self.assertIn("policy.text", {row["field"] for row in rows})

    def test_failed_tool_produces_no_evidence(self):
        self.assertEqual(evidence_from_tool_call("get_order", {}, {"ok": False}, "c1"), [])

    def test_order_sources_are_keyed_by_order_not_the_ordered_product(self):
        rows = evidence_from_tool_call(
            "get_order", {}, {"ok": True, "order": {
                "order_id": "O000001", "product_id": "P00072", "status": "delivered",
            }}, "c1")
        self.assertEqual({row["source_id"] for row in rows}, {"order:O000001"})


class DeterministicVerifierTests(unittest.TestCase):
    def test_discontinued_contradiction_is_not_hidden_by_semantic_similarity(self):
        ledger = _ledger(("product:P00072", "product.discontinued", False, "P00072 未停产"))
        result = verify_answer("P00072 已停产。[E1]", ledger, require_citations=True)
        self.assertFalse(result["answer_fact_pass"])
        self.assertEqual(result["contradicted_claims"][0]["field"], "product.discontinued")

    def test_wrong_policy_duration_is_a_contradiction(self):
        ledger = _ledger(("policy:POL001", "policy.text", "签收后7天内可申请退货", "签收后7天内可申请退货"))
        result = verify_answer("退货期限是30天。[E1]", ledger, require_citations=True)
        self.assertFalse(result["answer_fact_pass"])
        self.assertTrue(any(row["claim"] == "30天" for row in result["contradicted_claims"]))

    def test_order_state_mismatch_is_rejected(self):
        ledger = _ledger(("order:O000001", "order.order_id", "O000001", "O000001"),
                         ("order:O000001", "order.status", "delivered", "delivered"))
        result = verify_answer("订单 O000001 的状态是 cancelled。[E1][E2]", ledger, require_citations=True)
        self.assertFalse(result["answer_fact_pass"])
        self.assertTrue(result["contradicted_claims"])

    def test_negative_eligibility_phrase_does_not_emit_its_positive_substring(self):
        ledger = _ledger(("order:O000001", "return.eligible", False, "false"))
        result = verify_answer("该订单当前不符合退货条件。", ledger)
        self.assertTrue(result["answer_fact_pass"])

    def test_wrong_citation_is_reported_separately(self):
        ledger = _ledger(("policy:POL001", "policy.text", "7天", "退货期为7天"),
                         ("policy:POL002", "policy.text", "15天", "保修换货期为15天"))
        result = verify_answer("退货期为7天。[E2]", ledger, require_citations=True)
        self.assertTrue(result["answer_fact_pass"])
        self.assertFalse(result["citation_binding_pass"])

    def test_naked_citation_does_not_satisfy_binding_or_required_coverage(self):
        ledger = _ledger(("policy:POL004", "policy.text", "订单完成后可申请电子发票",
                          "订单完成后可在订单详情页申请电子发票"))
        result = verify_answer("政策已经查到了。[E1]", ledger, require_citations=True,
                               expectations={"required_fact_keys": ["policy.text"]})
        self.assertFalse(result["citation_binding_pass"])
        self.assertFalse(result["answer_fact_pass"])
        self.assertEqual(result["required_evidence_coverage"], 0.0)

    def test_required_fact_omission_uses_hidden_expectation_only_in_grading(self):
        ledger = _ledger(("order:O000001", "order.status", "delivered", "delivered"))
        result = verify_answer("订单信息已查询。[E1]", ledger,
                               expectations={"required_fact_keys": ["order.status"]})
        self.assertFalse(result["answer_fact_pass"])
        self.assertEqual(result["omitted_required_facts"], ["order.status"])

    def test_base_does_not_need_evidence_citation_for_joint_fact_check(self):
        ledger = _ledger(("order:O000001", "order.status", "delivered", "delivered"))
        result = verify_answer("订单状态是 delivered。", ledger, require_citations=False)
        self.assertTrue(result["answer_fact_pass"])
        self.assertTrue(result["citation_binding_pass"])

    def test_freshness_is_diagnostic_not_a_phase_a_hard_failure(self):
        ledger = _ledger(("policy:POL001", "policy.text", "7天", "退货期限是7天"))
        ledger[0]["updated_at"] = "2020-01-01"
        result = verify_answer("退货期限是7天。[E1]", ledger, require_citations=True)
        self.assertEqual(result["freshness"]["status"], "stale")
        self.assertTrue(result["answer_fact_pass"])

    def test_joint_success_does_not_fail_base_only_for_missing_evidence_tags(self):
        ledger = _ledger(("product:P00001", "product.price", 10, "10"))
        task = TaskSpec("base-fairness", "product_qa", "U0001", "多少钱", 1)
        trajectory = Trajectory("tr", task.task_id, 1, final_answer="价格是10元。", evidence_ledger=ledger)
        result = grade(task, trajectory, leakage_checked=True)
        self.assertTrue(result.success)
        self.assertTrue(result.answer_fact_pass)
        self.assertFalse(result.citation_binding_pass)
        self.assertTrue(result.joint_success)


class EvidencePolicyTests(unittest.TestCase):
    @staticmethod
    def observation(ledger, *, step=1):
        return AgentObservation(
            current_message="tool result", session={"user_id": "U0001"},
            history=[{"role": "user", "content": "退货期限多久？"}],
            tool_schemas=TOOL_SCHEMAS, step=step, evidence_ledger=ledger,
        )

    def test_verify_variant_fails_closed_without_repair(self):
        raw = ('{"action_type":"final_answer","tool_name":null,"arguments":{},'
               '"content":"退货期限是30天。[E1]","requires_user_response":false}')
        policy = EvidenceGroundedPolicy(lambda _s, _u: raw, repair=False)
        ledger = _ledger(("policy:POL001", "policy.text", "7天", "退货期限是7天"))
        action = policy.act(self.observation(ledger))
        self.assertEqual(action.action_type, "handoff")
        self.assertEqual(len(policy.last_verification_spans), 1)
        self.assertEqual(policy.last_repair_spans, [])

    def test_harness_records_ledger_verification_and_successful_repair(self):
        class Retriever:
            chunks = [{
                "doc_id": "product:P00001", "product_id": "P00001", "title": "Example",
                "category": "test", "price": 10, "inventory": "available",
                "updated_at": "2026-07-20", "text": "Example costs 10元",
            }]

            def search(self, *_args, **_kwargs):
                return list(self.chunks)

        outputs = iter([
            ('{"action_type":"tool_call","tool_name":"search_catalog",'
             '"arguments":{"query":"Example","top_k":5},"content":"",'
             '"requires_user_response":false}'),
            ('{"action_type":"final_answer","tool_name":null,"arguments":{},'
             '"content":"价格是30元。[E4]","requires_user_response":false}'),
            ('{"action_type":"final_answer","tool_name":null,"arguments":{},'
             '"content":"价格是10元。[E4]","requires_user_response":false}'),
        ])
        policy = EvidenceGroundedPolicy(lambda _s, _u: next(outputs), repair=True)
        task = TaskSpec(
            "evidence_e2e", "product_qa", "U0001", "Example 多少钱？", 1,
            gold_doc_ids=["product:P00001"], allowed_tools=["search_catalog"],
            answer_expectations={"required_fact_keys": ["product.price"]},
        )
        with tempfile.TemporaryDirectory() as directory:
            db = Path(directory) / "env.db"
            seed_database(db, users=5, orders=20)
            trajectory, result = HarnessRunner(db, retriever=Retriever(), policy=policy).run(task)
        self.assertGreaterEqual(len(trajectory.evidence_ledger), 4)
        self.assertEqual(len(trajectory.verification_spans), 2)
        self.assertEqual(len(trajectory.repair_spans), 1)
        self.assertTrue(result.repair_attempted)
        self.assertTrue(result.repair_succeeded, trajectory.repair_spans)
        self.assertTrue(result.answer_fact_pass)
        self.assertTrue(result.citation_binding_pass)
        self.assertTrue(result.joint_success)

    def test_repair_runs_exactly_once_and_can_recover(self):
        outputs = iter([
            ('{"action_type":"final_answer","tool_name":null,"arguments":{},'
             '"content":"退货期限是30天。[E1]","requires_user_response":false}'),
            ('{"action_type":"final_answer","tool_name":null,"arguments":{},'
             '"content":"退货期限是7天。[E1]","requires_user_response":false}'),
        ])
        policy = EvidenceGroundedPolicy(lambda _s, _u: next(outputs), repair=True)
        ledger = _ledger(("policy:POL001", "policy.text", "7天", "退货期限是7天"))
        action = policy.act(self.observation(ledger))
        self.assertEqual(action.action_type, "final_answer", (policy.last_verification_spans, policy.last_repair_spans))
        self.assertEqual(len(policy.last_verification_spans), 2)
        self.assertEqual(len(policy.last_repair_spans), 1)
        self.assertTrue(policy.last_repair_spans[0]["passed"])

    def test_intermediate_user_question_is_not_verified(self):
        raw = ('{"action_type":"final_answer","tool_name":null,"arguments":{},'
               '"content":"请提供验证码。","requires_user_response":true}')
        policy = EvidenceGroundedPolicy(lambda _s, _u: raw, repair=True)
        action = policy.act(self.observation([]))
        self.assertTrue(action.requires_user_response)
        self.assertEqual(policy.last_verification_spans, [])
        self.assertEqual(policy.last_repair_spans, [])


if __name__ == "__main__":
    unittest.main()
