import unittest
from ecommerce_rag.domain import Trajectory
from ecommerce_rag.evidence import convert_tool_call_to_evidence, evidence_from_tool_call, extract_user_context, verify_answer
from ecommerce_rag.harness import HarnessRunner, grade
from ecommerce_rag.domain import TaskSpec


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

    def test_conversion_span_distinguishes_converted_empty_and_failed(self):
        rows, converted = convert_tool_call_to_evidence(
            "search_catalog", {"query": "x"},
            {"ok": True, "items": [{"product_id": "P00001", "title": "x"}]}, "c1")
        self.assertTrue(rows)
        self.assertEqual(converted["status"], "converted")
        rows, empty = convert_tool_call_to_evidence("search_catalog", {"query": "x"},
                                                    {"ok": True, "items": []}, "c2")
        self.assertEqual(rows, [])
        self.assertEqual(empty["status"], "valid_empty")
        rows, failed = convert_tool_call_to_evidence("search_catalog", {"query": "x"},
                                                     {"ok": False, "error": "x"}, "c3")
        self.assertEqual(rows, [])
        self.assertEqual(failed["status"], "tool_failed")


class DeterministicVerifierTests(unittest.TestCase):
    def test_user_budget_is_supported_only_in_constraint_context(self):
        ledger = _ledger(("product:P1", "product.title", "耳机", "耳机"))
        constrained = verify_answer("预算是400元。", ledger,
                                    user_messages=[{"role": "user", "content": "预算不超过400元"}])
        guessed = verify_answer("价格是400元。", ledger,
                                user_messages=[{"role": "user", "content": "这个商品是不是400元？"}])
        self.assertEqual(constrained["unsupported_high_risk_claims"], [])
        self.assertTrue(constrained["hard_verification_pass"])
        self.assertTrue(guessed["unsupported_high_risk_claims"])

    def test_user_context_never_reads_task_metadata_or_gold(self):
        context = extract_user_context([{"role": "user", "content": "订单 O000001 应该已送达了吧？"}])
        self.assertEqual(context["identifiers"], ["O000001"])
        self.assertNotIn("delivered", str(context))

    def test_complete_chinese_date_does_not_emit_day_fragment(self):
        ledger = _ledger(("order:O1", "order.delivered_at", "2020-11-02", "2020-11-02"))
        result = verify_answer("送达日期是2020年11月2日。[E1]", ledger, require_citations=True)
        self.assertEqual(result["unsupported_high_risk_claims"], [])

    def test_days_since_delivery_supports_chinese_day_unit(self):
        ledger = _ledger(("order:O1", "return.days_since_delivery", 81, "81"))
        result = verify_answer("已送达81天。[E1]", ledger, require_citations=True)
        self.assertFalse(any(row["claim"] == "81天" for row in result["unsupported_high_risk_claims"]))

    def test_requested_chinese_alias_is_supported(self):
        ledger = _ledger(("order:O1", "return.return_status", "requested", "requested"))
        result = verify_answer("退货申请已成功提交。[E1]", ledger, require_citations=True)
        self.assertTrue(result["hard_verification_pass"])
        self.assertEqual(result["unsupported_high_risk_claims"], [])

    def test_english_attribute_value_can_bind(self):
        ledger = _ledger(("product:P1", "product.evidence.1", "Product Dimensions 6.8 x 9.5 x 9.2 inches",
                          "Product Dimensions 6.8 x 9.5 x 9.2 inches"))
        result = verify_answer("Product dimensions are 6.8 x 9.5 x 9.2 inches. [E1]", ledger,
                               require_citations=True)
        self.assertTrue(result["citation_binding_pass"], result)

    def test_available_without_inventory_context_is_not_a_state_claim(self):
        ledger = _ledger(("product:P1", "product.category", "watch band", "watch band"))
        result = verify_answer("This item is available in the watch band category. [E1]", ledger)
        self.assertFalse(any(row["claim"] == "available" for row in result["unsupported_high_risk_claims"]))

    def test_missing_citation_and_coverage_are_diagnostic_only(self):
        ledger = _ledger(("policy:POL1", "policy.text", "签收后7天内可退货", "签收后7天内可退货"))
        result = verify_answer("签收后7天内可退货。", ledger, require_citations=True,
                               expectations={"required_fact_keys": ["policy.text"]})
        self.assertTrue(result["hard_verification_pass"])
        self.assertTrue(result["citation_diagnostics"])

    def test_nonexistent_evidence_id_remains_hard_failure(self):
        result = verify_answer("退货期限是7天。[E999]", [], require_citations=True)
        self.assertFalse(result["hard_verification_pass"])
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

    def test_wrong_structured_price_is_a_hard_contradiction(self):
        ledger = _ledger(("product:P1", "product.price", 10, "10"))
        result = verify_answer("价格是30元。[E1]", ledger, require_citations=True)
        self.assertFalse(result["hard_verification_pass"])
        self.assertEqual(result["contradicted_claims"][0]["field"], "product.price")

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
        self.assertFalse(result["answer_fact_pass"])
        self.assertFalse(result["citation_binding_pass"])
        self.assertTrue(result["citation_oppositions"])

    def test_naked_citation_does_not_satisfy_binding_or_required_coverage(self):
        ledger = _ledger(("policy:POL004", "policy.text", "订单完成后可申请电子发票",
                          "订单完成后可在订单详情页申请电子发票"))
        result = verify_answer("政策已经查到了。[E1]", ledger, require_citations=True,
                               expectations={"required_fact_keys": ["policy.text"]})
        self.assertFalse(result["citation_binding_pass"])
        self.assertTrue(result["answer_fact_pass"])
        self.assertEqual(result["required_evidence_coverage"], 0.0)

    def test_required_fact_omission_uses_hidden_expectation_only_in_grading(self):
        ledger = _ledger(("order:O000001", "order.status", "delivered", "delivered"))
        result = verify_answer("订单信息已查询。[E1]", ledger,
                               expectations={"required_fact_keys": ["order.status"]})
        self.assertTrue(result["answer_fact_pass"])
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


if __name__ == "__main__":
    unittest.main()
