# -*- coding: utf-8 -*-
"""Stdlib-only tests for the fail-closed RL gate's human-audit accounting.

Regression guard: a CSV that has 40 template rows but no human verdicts must
report "not started", not "humans agreed with the grader 0% of the time".
"""

import csv
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from ecommerce_rag import rl_gate

AUDIT_FIELDS = ["trajectory_id", "grader_success", "human_success", "human_policy_compliant", "review_notes"]


def _write_tasks(path: Path, count: int) -> None:
    lines = [json.dumps({"task_id": f"t{i}"}) for i in range(count)]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_store(path: Path, count: int, success: bool = False) -> None:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE trajectories(trajectory_id TEXT PRIMARY KEY, task_id TEXT, seed INTEGER,"
                 " trajectory_json TEXT, grade_json TEXT)")
    for i in range(count):
        trajectory = {"policy_name": "LLMPolicy", "observations": [{"current_message": "hi", "step": 0}]}
        grade = {"success": success, "policy_compliant": True, "terminal_state_match": True}
        conn.execute("INSERT INTO trajectories VALUES (?,?,?,?,?)",
                     (f"tr{i}", f"t{i}", 0, json.dumps(trajectory), json.dumps(grade)))
    conn.commit()
    conn.close()


def _write_audit(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in AUDIT_FIELDS})


class RLGateAuditAccountingTests(unittest.TestCase):
    def test_blank_human_cells_are_not_counted_as_reviewed(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            _write_tasks(d / "tasks.jsonl", 120)
            _write_store(d / "store.db", 360)
            # 40 template rows, grader verdict filled, human verdict blank
            _write_audit(d / "audit.csv", [{"trajectory_id": f"tr{i}", "grader_success": "false"} for i in range(40)])

            result = rl_gate.assess(str(d / "tasks.jsonl"), str(d / "store.db"), str(d / "audit.csv"))

            self.assertEqual(result["human_audit_rows"], 40)      # the template exists
            self.assertEqual(result["human_audit_reviewed"], 0)   # nobody adjudicated it
            self.assertEqual(result["human_audit_status"], "not_started")
            # the critical part: absence of judgement must not read as disagreement
            self.assertIsNone(result["human_reward_agreement"])

    def test_complete_sidecar_drives_both_human_agreements(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            _write_tasks(d / "tasks.jsonl", 120)
            _write_store(d / "store.db", 360, success=False)
            sidecar = d / "grades.jsonl"
            sidecar.write_text("".join(json.dumps({"trajectory_id": f"tr{i}", "grade": {
                "success": i != 0, "policy_compliant": i != 1, "terminal_state_match": True,
            }}) + "\n" for i in range(360)), encoding="utf-8")
            rows = []
            for i in range(40):
                rows.append({"trajectory_id": f"tr{i}", "grader_success": "false",
                             "human_success": str(i != 0).lower(),
                             "human_policy_compliant": str(i != 1).lower()})
            _write_audit(d / "audit.csv", rows)

            result = rl_gate.assess(d / "tasks.jsonl", d / "store.db", d / "audit.csv", grades=sidecar)

            self.assertTrue(result["checks"]["grade_sidecar_complete"])
            self.assertEqual(result["human_success_agreement"], 1.0)
            self.assertEqual(result["human_policy_agreement"], 1.0)

    def test_incomplete_or_duplicate_sidecar_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            _write_tasks(d / "tasks.jsonl", 120)
            _write_store(d / "store.db", 360)
            row = json.dumps({"trajectory_id": "tr0", "grade": {
                "success": False, "policy_compliant": True, "terminal_state_match": True}})
            (d / "grades.jsonl").write_text(row + "\n" + row + "\n", encoding="utf-8")

            result = rl_gate.assess(d / "tasks.jsonl", d / "store.db", grades=d / "grades.jsonl")

            self.assertFalse(result["checks"]["grade_sidecar_complete"])
            self.assertFalse(result["eligible"])
            self.assertTrue(any("duplicate" in error for error in result["grade_sidecar_errors"]))
            self.assertTrue(any("missing" in error for error in result["grade_sidecar_errors"]))
            self.assertFalse(result["checks"]["human_audit_at_least_40"])
            self.assertFalse(result["checks"]["human_reward_agreement_at_least_90pct"])
            self.assertFalse(result["eligible"])

    def test_partial_review_counts_only_judged_rows(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            _write_tasks(d / "tasks.jsonl", 120)
            _write_store(d / "store.db", 360)
            rows = [{"trajectory_id": f"tr{i}", "grader_success": "false"} for i in range(40)]
            # 4 rows fully adjudicated: 3 agree with the grader, 1 disagrees
            for i, verdict in enumerate(["false", "false", "false", "true"]):
                rows[i]["human_success"] = verdict
                rows[i]["human_policy_compliant"] = "true"
            _write_audit(d / "audit.csv", rows)

            result = rl_gate.assess(str(d / "tasks.jsonl"), str(d / "store.db"), str(d / "audit.csv"))

            self.assertEqual(result["human_audit_reviewed"], 4)
            self.assertEqual(result["human_audit_status"], "partial")
            self.assertAlmostEqual(result["human_reward_agreement"], 0.75)
            self.assertFalse(result["checks"]["human_audit_at_least_40"])

    def test_full_agreement_is_reported_as_complete(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            _write_tasks(d / "tasks.jsonl", 120)
            _write_store(d / "store.db", 360)
            rows = [{"trajectory_id": f"tr{i}", "grader_success": "false", "human_success": "false",
                     "human_policy_compliant": "true"} for i in range(40)]
            _write_audit(d / "audit.csv", rows)

            result = rl_gate.assess(str(d / "tasks.jsonl"), str(d / "store.db"), str(d / "audit.csv"))

            self.assertEqual(result["human_audit_reviewed"], 40)
            self.assertEqual(result["human_audit_status"], "complete")
            self.assertEqual(result["human_reward_agreement"], 1.0)
            self.assertTrue(result["checks"]["human_audit_at_least_40"])
            self.assertTrue(result["checks"]["human_reward_agreement_at_least_90pct"])

    def test_non_boolean_verdicts_are_rejected_and_surfaced(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            _write_tasks(d / "tasks.jsonl", 120)
            _write_store(d / "store.db", 360)
            # a reviewer who typed free text instead of true/false
            rows = [{"trajectory_id": f"tr{i}", "grader_success": "false",
                     "human_success": verdict, "human_policy_compliant": "yes"}
                    for i, verdict in enumerate(["pending", "yes", "OK", "1", "n/a"])]
            _write_audit(d / "audit.csv", rows)

            result = rl_gate.assess(str(d / "tasks.jsonl"), str(d / "store.db"), str(d / "audit.csv"))

            self.assertEqual(result["human_audit_reviewed"], 0)
            self.assertIsNone(result["human_reward_agreement"])
            # the mistyped sheet must not look identical to an untouched one
            self.assertEqual(result["human_audit_malformed_rows"], 5)

    def test_row_missing_policy_verdict_is_not_fully_reviewed(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            _write_tasks(d / "tasks.jsonl", 120)
            _write_store(d / "store.db", 360)
            # HUMAN_AUDIT_GUIDE requires both verdicts; only one is filled here
            rows = [{"trajectory_id": "tr0", "grader_success": "false", "human_success": "false"},
                    {"trajectory_id": "tr1", "grader_success": "false", "human_success": "false",
                     "human_policy_compliant": "true"}]
            _write_audit(d / "audit.csv", rows)

            result = rl_gate.assess(str(d / "tasks.jsonl"), str(d / "store.db"), str(d / "audit.csv"))

            self.assertEqual(result["human_audit_reviewed"], 1)
            self.assertEqual(result["human_audit_malformed_rows"], 1)
            self.assertEqual(result["human_audit_status"], "partial")

    def test_missing_audit_file_reports_not_started(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            _write_tasks(d / "tasks.jsonl", 120)
            _write_store(d / "store.db", 360)

            result = rl_gate.assess(str(d / "tasks.jsonl"), str(d / "store.db"), str(d / "missing.csv"))

            self.assertEqual(result["human_audit_rows"], 0)
            self.assertEqual(result["human_audit_status"], "not_started")
            self.assertIsNone(result["human_reward_agreement"])

    def test_utf8_bom_on_csv_header_is_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            _write_tasks(d / "tasks.jsonl", 120)
            _write_store(d / "store.db", 360)
            with open(d / "audit.csv", "w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS)
                writer.writeheader()
                writer.writerow({"trajectory_id": "tr0", "grader_success": "false",
                                 "human_success": "false", "human_policy_compliant": "true"})

            result = rl_gate.assess(d / "tasks.jsonl", d / "store.db", d / "audit.csv")

            self.assertEqual(result["human_audit_reviewed"], 1)
            self.assertEqual(result["human_audit_linkage_errors"], [])


if __name__ == "__main__":
    unittest.main()
