import unittest

from scripts.build_retrieval_audit_evidence import build_html, candidate_checks, parse_constraints


class RetrievalAuditEvidenceTests(unittest.TestCase):
    def test_budget_constraints_are_structured(self):
        row = {"audit_scope": "budget", "question": "预算不超过 14.74，想买 StarTech 的数码，关键词 StarTech.com 25ft"}
        parsed = parse_constraints(row, {"constraints": {"max_price": 14.74}})
        self.assertEqual(parsed["brand"], "StarTech")
        self.assertEqual(parsed["category_alias"], "数码")
        self.assertEqual(parsed["keyword"], "StarTech.com 25ft")

    def test_candidate_reports_each_failed_constraint(self):
        facts = {"title": "Other Cable", "price": 20.0, "brand": "Other", "category": "Electronics", "attributes": {}}
        constraints = {"max_price": 14.74, "brand": "StarTech", "category_alias": "数码", "keyword": "StarTech.com 25ft"}
        checks = candidate_checks(facts, constraints)
        self.assertFalse(checks["budget"])
        self.assertFalse(checks["brand"])
        self.assertTrue(checks["category"])
        self.assertFalse(checks["keyword_pass"])

    def test_no_answer_model_code_must_be_present(self):
        facts = {"title": "Ordinary Toaster", "price": 30, "brand": "Acme", "category": "Home & Kitchen", "attributes": {}}
        checks = candidate_checks(facts, {"required_model_code": "ZX-9000"})
        self.assertFalse(checks["model_code"])

    def test_html_panel_is_self_contained(self):
        payload = {"metadata": {"version": "test"}, "cases": []}
        page = build_html(payload)
        self.assertIn("Retrieval Human Adjudication", page)
        self.assertIn("导出审核 CSV", page)
        self.assertIn("最终 gold IDs", page)
        self.assertIn('id="payload"', page)


if __name__ == "__main__":
    unittest.main()
