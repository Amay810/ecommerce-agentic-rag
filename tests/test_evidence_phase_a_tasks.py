import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.generate_evidence_tasks import build, write_audit


ROOT = Path(__file__).parents[1]
PRODUCTS = ROOT / "ecommerce_rag" / "data" / "amazon_products_5k.jsonl"


class EvidencePhaseATaskTests(unittest.TestCase):
    def test_generator_is_deterministic_and_splits_are_isolated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, manifest1 = build(root / "a.db", PRODUCTS, 20260728)
            second, manifest2 = build(root / "b.db", PRODUCTS, 20260728)
        payload1 = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in first)
        payload2 = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in second)
        self.assertEqual(hashlib.sha256(payload1.encode()).digest(), hashlib.sha256(payload2.encode()).digest())
        self.assertEqual(manifest1, manifest2)
        self.assertEqual(manifest1["task_count"], 240)
        self.assertEqual(manifest1["splits"], {"calibration": 160, "dev": 80})
        self.assertTrue(all(not values for values in manifest1["cross_split_overlap"].values()))

    def test_audit_has_exactly_four_rows_per_category(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks, _ = build(root / "env.db", PRODUCTS, 20260728)
            audit = root / "audit.csv"
            write_audit(tasks, audit)
            rows = audit.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(rows), 33)
        self.assertIn("human_answer_fact_pass", rows[0])
        self.assertIn("human_contradiction_present", rows[0])
        self.assertIn("required_fact_keys", rows[0])


if __name__ == "__main__":
    unittest.main()
