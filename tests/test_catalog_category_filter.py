import json
import unittest
from pathlib import Path

import numpy as np

from ecommerce_rag.hybrid_retriever import HybridRetriever
from ecommerce_rag.tools import RetailTools


ROOT = Path(__file__).parents[1]


class _OneVectorModel:
    def encode(self, *_args, **_kwargs):
        return np.asarray([[1.0]], dtype="float32")


class CatalogCategoryFilterTests(unittest.TestCase):
    def test_lowercase_accessories_filter_returns_p00014_for_full_model_query(self):
        chunk = next(
            json.loads(line)
            for line in (ROOT / "ecommerce_rag" / "index_5000" / "chunks.jsonl").read_text(
                encoding="utf-8").splitlines()
            if '"product_id": "P00014"' in line
        )
        self.assertIn("Accessories", chunk["category"])

        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever.model = _OneVectorModel()
        retriever.chunks = [chunk]
        retriever.by_id = {chunk["chunk_id"]: chunk}
        retriever.parents = {chunk["doc_id"]: chunk["text"]}
        retriever.embeddings = np.asarray([[1.0]], dtype="float32")
        retriever.faiss_index = None
        retriever.reranker = None
        retriever.dense_backend = "numpy-test"
        retriever._bm25_ranking = lambda _query, _k: [chunk["chunk_id"]]

        tools = RetailTools(ROOT / "ecommerce_rag" / "data" / "agent_env_v2.db", retriever)
        result = tools.search_catalog(
            "eF0rCity Leather Case with Stand for 7-Inch Samsung Galaxy Tab 2, "
            "White/Black Zebra (PSAMGLXTLC26)",
            category="accessories",
        )
        self.assertTrue(result["ok"])
        self.assertEqual([item["product_id"] for item in result["items"]], ["P00014"])


if __name__ == "__main__":
    unittest.main()
