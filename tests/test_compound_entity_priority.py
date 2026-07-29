import unittest
from types import SimpleNamespace

from ecommerce_rag.agent import CustomerSupportAgent


def chunk(doc_id, title, dense_sim):
    return {"doc_id": doc_id, "chunk_id": f"{doc_id}:desc:0", "title": title,
            "source_type": "product", "text": title, "dense_sim": dense_sim}


class FakeRetriever:
    def __init__(self):
        self.p1 = chunk("product:P001", "Air Pro 2", .70)
        self.p9 = chunk("product:P009", "QuietMax H900", .68)
        self.p8 = chunk("product:P008", "WindMax H1", .66)
        self.chunks = [self.p1, self.p9, self.p8]

    def search(self, query, source_type=None):
        if query == "Air Pro 2":
            return [self.p1, self.p8, self.p9]
        if query == "QuietMax H900":
            return [self.p9, self.p8, self.p1]
        return [self.p8, self.p1, self.p9]


class CompoundEntityPriorityTests(unittest.TestCase):
    def test_explicit_entities_precede_fused_distractor(self):
        agent = CustomerSupportAgent.__new__(CustomerSupportAgent)
        agent.retriever = FakeRetriever()
        route = SimpleNamespace(source_type="product")
        result, _ = agent._retrieve_compound(
            "Air Pro 2 vs QuietMax H900", ["Air Pro 2", "QuietMax H900"], route, []
        )
        self.assertEqual([row["doc_id"] for row in result[:2]], ["product:P001", "product:P009"])


if __name__ == "__main__":
    unittest.main()
