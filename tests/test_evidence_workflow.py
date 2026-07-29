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
            "tool_calls": [{"name": "search_catalog", "call_id": "c1",
                            "result": {"ok": True, "items": [{"doc_id": "product:P00001"}]}}],
            "evidence_ledger": [{"evidence_id": "E1", "tool_call_id": "c1"}],
            "evidence_conversion_spans": [{"tool_call_id": "c1", "tool_name": "search_catalog",
                "status": "converted", "evidence_ids": ["E1"], "source_item_count": 1,
                "evidence_item_count": 1}],
            "repair_spans": [{} for _ in range(repairs)],
            "model_calls": [{"llm": {
                "resolution": "parsed", "attempts": [{
                    "parse_stage": "generation_error" if generation_error else None,
                }],
            }}],
        }
        grade = {"illegal_state_change": False, "success": True, "operational_success": True,
                 "policy_compliant": True, "terminal_state_match": True,
                 "hard_verification_pass": True, "answer_fact_pass": True,
                 "joint_success": True, "citation_diagnostics": [],
                 "repair_attempted": repairs > 0, "repair_succeeded": False}
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

    def test_converter_missing_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"task_ids": ["t1"]}), encoding="utf-8")
            for variant in ("base", "evidence_verify", "evidence_verify_repair"):
                path = root / f"run_{variant}.sqlite"
                self.write_store(path, variant=variant)
                conn = sqlite3.connect(path)
                try:
                    row = conn.execute("SELECT trajectory_json FROM trajectories").fetchone()
                    trajectory = json.loads(row[0])
                    trajectory["evidence_conversion_spans"][0]["status"] = "converter_missing"
                    conn.execute("UPDATE trajectories SET trajectory_json=?", (json.dumps(trajectory),))
                    conn.commit()
                finally:
                    conn.close()
            result = assess(manifest, root, "run")
        self.assertFalse(result["passed"])
        self.assertTrue(any(not row["passed"] and "evidence_conversion" in row["name"]
                            for row in result["checks"]))


if __name__ == "__main__":
    unittest.main()
