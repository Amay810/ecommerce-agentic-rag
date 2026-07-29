import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.generate_evidence_tasks import build, write_audit


ROOT = Path(__file__).parents[1]


def write_product_fixture(path: Path) -> Path:
    """Write the minimum corpus shape required by the Phase-A generator."""
    rows = [
        {
            "id": f"P{index:05d}",
            "title": f"Fixture product {index}",
            "category": "Electronics/Accessories",
            "category_aliases_zh": ["数码配件"],
            "price": 100 + index % 500,
        }
        for index in range(1500)
    ]
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


class EvidencePhaseATaskTests(unittest.TestCase):
    def test_generator_is_deterministic_and_splits_are_isolated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            products = write_product_fixture(root / "products.jsonl")
            first, manifest1 = build(root / "a.db", products, 20260728)
            second, manifest2 = build(root / "b.db", products, 20260728)
        payload1 = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in first)
        payload2 = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in second)
        self.assertEqual(hashlib.sha256(payload1.encode()).digest(), hashlib.sha256(payload2.encode()).digest())
        self.assertEqual(manifest1, manifest2)
        self.assertEqual(manifest1["task_count"], 240)
        self.assertEqual(manifest1["splits"], {"calibration": 160, "dev": 80})
        self.assertTrue(all(not values for values in manifest1["cross_split_overlap"].values()))
        product_tasks = [task for task in first if task["category"] == "product_qa"]
        self.assertTrue(all(task["expected_tool_sequence"] == ["search_catalog", "get_product"]
                            for task in product_tasks))
        no_answer = [task for task in first if task["category"] == "recovery_no_answer"]
        self.assertTrue(all(task["metadata"]["diagnostic_only"] for task in no_answer))

    def test_audit_has_exactly_four_rows_per_category(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            products = write_product_fixture(root / "products.jsonl")
            tasks, _ = build(root / "env.db", products, 20260728)
            audit = root / "audit.csv"
            write_audit(tasks, audit)
            rows = audit.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(rows), 33)
        self.assertIn("human_answer_fact_pass", rows[0])
        self.assertIn("human_contradiction_present", rows[0])
        self.assertIn("required_fact_keys", rows[0])


if __name__ == "__main__":
    unittest.main()
