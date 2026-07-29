import unittest
from ecommerce_rag.verifier import citation_check

class CitationTests(unittest.TestCase):
    def test_numeric_fact_requires_citation(self):
        self.assertFalse(citation_check("价格是 99 元。", 1)["ok"])
        self.assertTrue(citation_check("价格是 99 元 [资料1]。", 1)["ok"])

    def test_out_of_range_citation_is_rejected(self):
        result = citation_check("价格是 99 元 [资料2]。", 1)
        self.assertFalse(result["ok"])
        self.assertEqual(result["invalid_indices"], [2])

if __name__ == "__main__": unittest.main()
