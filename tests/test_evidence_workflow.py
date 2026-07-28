import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from scripts.evidence_smoke_gate import assess
class EvidenceSmokeGateTests(unittest.TestCase):
    @staticmethod
    def write_store(path: Path, *, variant: str, repairs=0, generation_error=False):
        trajectory = {
            "trajectory_id": f"tr-{variant}", "task_id": "t1",
            "retrievals": [{"result": {"items": [{"doc_id": "product:P00001"}]}}],
            "evidence_ledger": [] if variant == "base" else [{"evidence_id": "E1"}],
            "repair_spans": [{} for _ in range(repairs)],
            "model_calls": [{"llm": {
                "resolution": "parsed", "attempts": [{
                    "parse_stage": "generation_error" if generation_error else None,
                }],
            }}],
        }
        grade = {"illegal_state_change": False}
        conn = sqlite3.connect(path)
        try:
            conn.execute("CREATE TABLE trajectories(trajectory_id TEXT PRIMARY KEY, task_id TEXT, seed INTEGER, trajectory_json TEXT, grade_json TEXT)")
            conn.execute("INSERT INTO trajectories VALUES(?,?,?,?,?)",
                         (trajectory["trajectory_id"], "t1", 1, json.dumps(trajectory), json.dumps(grade)))
            conn.commit()
        finally:
            conn.close()

    def test_clean_three_variant_smoke_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"task_ids": ["t1"]}), encoding="utf-8")
            for variant in ("base", "evidence_verify", "evidence_verify_repair"):
                self.write_store(root / f"run_{variant}.sqlite", variant=variant)
            result = assess(manifest, root, "run")
        self.assertTrue(result["passed"])

    def test_generation_error_and_second_repair_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"task_ids": ["t1"]}), encoding="utf-8")
            self.write_store(root / "run_base.sqlite", variant="base")
            self.write_store(root / "run_evidence_verify.sqlite", variant="evidence_verify", generation_error=True)
            self.write_store(root / "run_evidence_verify_repair.sqlite", variant="evidence_verify_repair", repairs=2)
            result = assess(manifest, root, "run")
        failed = {row["name"] for row in result["checks"] if not row["passed"]}
        self.assertIn("evidence_verify:generation_errors_zero", failed)
        self.assertIn("evidence_verify_repair:repair_at_most_once", failed)
        self.assertFalse(result["passed"])


if __name__ == "__main__":
    unittest.main()
