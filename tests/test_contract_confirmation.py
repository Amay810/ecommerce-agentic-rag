import json
import unittest
from pathlib import Path

from scripts.build_contract_confirmation import TASK_IDS, select


class ContractConfirmationTests(unittest.TestCase):
    def test_selects_the_seven_unique_original_failure_tasks(self):
        path = Path(__file__).parents[1] / "ecommerce_rag" / "data" / "harness_tasks_v2.jsonl"
        tasks = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        selected = select(tasks)
        self.assertEqual([task["task_id"] for task in selected], list(TASK_IDS))
        self.assertEqual(len({task["task_id"] for task in selected}), 7)
        self.assertEqual({task["category"] for task in selected}, {"policy", "product_qa"})
        products = [task for task in selected if task["category"] == "product_qa"]
        self.assertTrue(all(task["allowed_tools"] == ["search_catalog", "get_product"] for task in products))
        self.assertTrue(all(task["expected_tool_sequence"] == ["search_catalog", "get_product"] for task in products))
        self.assertTrue(all(task["answer_expectations"]["required_fact_keys"] == ["product.evidence.*"] for task in products))


if __name__ == "__main__":
    unittest.main()
