import tempfile
import unittest
from pathlib import Path

from ecommerce_rag.tools import POLICY_CATEGORIES, RetailTools


class RecordingRetriever:
    def __init__(self):
        self.calls = []

    def search(self, query, **kwargs):
        self.calls.append((query, kwargs))
        category = kwargs["category"]
        doc_id = {
            "退换货": "policy:POL001", "售后保修": "policy:POL002", "物流": "policy:POL003",
            "发票": "policy:POL004", "退款": "policy:POL005",
        }[category]
        return [{"doc_id": doc_id, "title": category, "text": category}]


class PolicyToolTests(unittest.TestCase):
    def test_each_canonical_type_filters_to_its_exact_policy_category(self):
        retriever = RecordingRetriever()
        with tempfile.TemporaryDirectory() as d:
            tools = RetailTools(Path(d) / "unused.sqlite", retriever=retriever)
            expected = {
                "return": "policy:POL001", "warranty": "policy:POL002", "shipping": "policy:POL003",
                "invoice": "policy:POL004", "refund": "policy:POL005",
            }
            for policy_type, doc_id in expected.items():
                result = tools.get_policy(policy_type)
                self.assertEqual(result["policies"][0]["doc_id"], doc_id)
                query, kwargs = retriever.calls[-1]
                self.assertEqual(query, POLICY_CATEGORIES[policy_type])
                self.assertEqual(kwargs["category"], POLICY_CATEGORIES[policy_type])

    def test_chinese_alias_remains_supported_by_the_implementation(self):
        retriever = RecordingRetriever()
        with tempfile.TemporaryDirectory() as d:
            result = RetailTools(Path(d) / "unused.sqlite", retriever=retriever).get_policy("保修")
        self.assertEqual(result["policies"][0]["doc_id"], "policy:POL002")


if __name__ == "__main__":
    unittest.main()
