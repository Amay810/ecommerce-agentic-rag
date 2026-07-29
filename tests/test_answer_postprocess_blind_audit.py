from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from ecommerce_rag.answer_postprocess import AnswerPostprocessor, stable_hash
from ecommerce_rag.llm_policy import Generation
from scripts.aggregate_answer_postprocess_audit import main as aggregate_main, paired_bootstrap
from scripts.answer_postprocess_gate import main as sidecar_gate_main
from scripts.freeze_answer_postprocess_audit import TARGETS, select_tasks
from scripts.prepare_answer_postprocess_audit import main as prepare_main
from scripts.run_answer_postprocess import VERIFIER_GATE_APPLIED, VERIFIER_ROLE


ROOT = Path(__file__).resolve().parents[1]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


class DiagnosticOnlyProtocolTests(unittest.TestCase):
    def test_pbs_jobs_do_not_depend_on_locked_verifier_admission(self) -> None:
        for name in ("run_answer_postprocess_smoke_v1.pbs", "run_answer_postprocess_dev_v1.pbs",
                     "run_answer_postprocess_smoke_v2.pbs", "run_answer_postprocess_dev_v2.pbs"):
            text = (ROOT / "nscc" / name).read_text(encoding="utf-8")
            self.assertNotIn("check_verifier_admission", text)
            self.assertNotIn("verifier_challenge_locked_v2_report", text)

    def test_run_contract_declares_diagnostic_only_verifier(self) -> None:
        self.assertEqual(VERIFIER_ROLE, "diagnostic_only")
        self.assertIs(VERIFIER_GATE_APPLIED, False)

    def test_verifier_diagnostic_cannot_replace_generated_answer(self) -> None:
        class HardFailure:
            def to_dict(self) -> dict:
                return {"fact_status": "contradicted", "hard_failure": True}

        processor = AnswerPostprocessor(
            lambda _system, _user: Generation("grounded answer", "stop", 2, 2, False),
            generation_config={"temperature": 0},
        )
        with patch("ecommerce_rag.answer_postprocess.classify_claim", return_value=HardFailure()):
            result = processor.process(
                "draft", [{"evidence_id": "E1", "field": "x", "value": "y", "text": "y"}],
                [], "terminal_grounded",
            )
        self.assertEqual(result.final_answer, "grounded answer")
        self.assertTrue(result.changed)
        self.assertEqual(result.verification[0]["fact_status"], "contradicted")

    def test_frozen_selection_matches_preregistered_contract(self) -> None:
        manifest = json.loads(
            (ROOT / "docs" / "answer_postprocess_blind_audit_v1_manifest.json").read_text(encoding="utf-8")
        )
        selected = manifest["selected_tasks"]
        self.assertEqual(len(selected), 40)
        self.assertEqual(len({row["task_id"] for row in selected}), 40)
        self.assertEqual(Counter(row["category"] for row in selected), Counter(TARGETS))
        self.assertFalse({row["task_id"] for row in selected} & set(manifest["excluded_smoke_task_ids"]))
        self.assertTrue(set(manifest["fact_applicable_holdout_task_ids"]).issubset(
            {row["task_id"] for row in selected}
        ))

    def test_selection_algorithm_reproduces_frozen_manifest(self) -> None:
        tasks = {row["task_id"]: row for row in (
            json.loads(line) for line in
            (ROOT / "ecommerce_rag" / "data" / "evidence_phase_a_tasks_v2.jsonl").read_text(
                encoding="utf-8"
            ).splitlines() if line.strip()
        )}
        report = json.loads(
            (ROOT / "docs" / "evidence_phase_a_dev_v3_base_report.json").read_text(encoding="utf-8")
        )
        grades = {row["task_id"]: row for row in report["details"]}
        smoke = json.loads(
            (ROOT / "docs" / "answer_postprocess_smoke_v1_manifest.json").read_text(encoding="utf-8")
        )["task_ids"]
        holdout = json.loads(
            (ROOT / "docs" / "answer_postprocess_holdout_v1_manifest.json").read_text(encoding="utf-8")
        )["task_ids"]
        expected = [row["task_id"] for row in json.loads(
            (ROOT / "docs" / "answer_postprocess_blind_audit_v1_manifest.json").read_text(encoding="utf-8")
        )["selected_tasks"]]
        self.assertEqual(select_tasks(tasks, grades, set(smoke), holdout), expected)


class BlindAuditWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.tasks_path = self.root / "tasks.jsonl"
        self.store_path = self.root / "base.sqlite"
        self.selection_path = self.root / "selection.json"
        self.shadow_path = self.root / "shadow.jsonl"
        self.grounded_path = self.root / "grounded.jsonl"
        self.output_dir = self.root / "audit"
        tasks, shadow, grounded = [], [], []
        conn = sqlite3.connect(self.store_path)
        conn.execute("CREATE TABLE trajectories(task_id TEXT, trajectory_json TEXT)")
        for index in range(80):
            task_id = f"task-{index:02d}"
            evidence = [{"evidence_id": "E1", "field": "product.price", "value": index,
                         "text": str(index)}]
            trajectory = {
                "trajectory_id": f"tr-{index:02d}", "task_id": task_id,
                "actions": [{"action_type": "final_answer", "content": f"draft {index}"}],
                "tool_calls": [], "final_state": {}, "evidence_ledger": evidence,
                "final_answer": f"draft {index}",
            }
            conn.execute("INSERT INTO trajectories VALUES(?, ?)",
                         (task_id, json.dumps(trajectory, ensure_ascii=False)))
            tasks.append({"task_id": task_id, "user_goal": f"question {index}"})
            common = {
                "schema_version": 1, "task_id": task_id,
                "source_trajectory_id": trajectory["trajectory_id"],
                "action_sequence_sha256": stable_hash(trajectory["actions"]),
                "tool_calls_sha256": stable_hash([]),
                "terminal_state_sha256": stable_hash({}),
                "evidence_ledger_sha256": stable_hash(evidence),
                "eligible": True, "ineligible_reason": None, "error": None, "truncated": False,
                "prompt_tokens": 10, "completion_tokens": 4, "latency_ms": 5.0,
                "generation_config_hash": "config", "verification": [],
            }
            shadow.append({**common, "mode": "shadow", "draft_answer": f"draft {index}",
                           "final_answer": f"draft {index}", "changed": False})
            grounded.append({**common, "mode": "terminal_grounded", "draft_answer": f"draft {index}",
                             "final_answer": f"grounded {index} [E1]", "changed": True})
        conn.commit()
        conn.close()
        write_jsonl(self.tasks_path, tasks)
        write_jsonl(self.shadow_path, shadow)
        write_jsonl(self.grounded_path, grounded)
        self.selection_path.write_text(json.dumps({
            "selected_tasks": [{"task_id": f"task-{index:02d}"} for index in range(40)]
        }), encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def prepare(self) -> None:
        argv = [
            "prepare", "--selection-manifest", str(self.selection_path), "--tasks", str(self.tasks_path),
            "--base-store", str(self.store_path), "--shadow", str(self.shadow_path),
            "--terminal-grounded", str(self.grounded_path), "--output-dir", str(self.output_dir),
        ]
        with patch.object(sys, "argv", argv):
            prepare_main()

    def paths(self) -> tuple[Path, Path, Path]:
        return (
            self.output_dir / "answer_postprocess_blind_audit_v1_review.jsonl",
            self.output_dir / "answer_postprocess_blind_audit_v1_mapping.jsonl",
            self.output_dir / "answer_postprocess_blind_audit_v1_package_manifest.json",
        )

    def test_package_contains_80_blinded_unpaired_answers(self) -> None:
        self.prepare()
        review_path, mapping_path, _ = self.paths()
        reviews, mappings = read_rows(review_path), read_rows(mapping_path)
        self.assertEqual(len(reviews), 80)
        self.assertEqual(len({row["review_id"] for row in reviews}), 80)
        for row in reviews:
            self.assertFalse({"variant", "task_id", "verification", "gold", "pair_id"} & set(row))
            self.assertEqual(row["fact_pass"], "")
        self.assertEqual(Counter(row["variant"] for row in mappings),
                         Counter({"base": 40, "terminal_grounded": 40}))

    def test_sidecar_immutability_mismatch_is_rejected(self) -> None:
        rows = read_rows(self.grounded_path)
        rows[0]["terminal_state_sha256"] = "tampered"
        write_jsonl(self.grounded_path, rows)
        with self.assertRaisesRegex(ValueError, "immutability mismatch"):
            self.prepare()

    def test_existing_package_is_never_overwritten(self) -> None:
        self.prepare()
        with self.assertRaises(FileExistsError):
            self.prepare()

    def test_aggregate_validates_blind_rows_and_produces_paired_statistics(self) -> None:
        self.prepare()
        review_path, mapping_path, package_path = self.paths()
        mappings = {row["review_id"]: row for row in read_rows(mapping_path)}
        reviews = read_rows(review_path)
        for row in reviews:
            variant = mappings[row["review_id"]]["variant"]
            row["fact_pass"] = "true" if variant == "terminal_grounded" else "false"
            row["answer_complete"] = "true"
            row["contradiction_present"] = "false"
            row["review_notes"] = ""
        write_jsonl(review_path, reviews)
        shadow_gate, grounded_gate = self.root / "shadow_gate.json", self.root / "grounded_gate.json"
        shadow_gate.write_text('{"passed":true}\n', encoding="utf-8")
        grounded_gate.write_text('{"passed":true}\n', encoding="utf-8")
        output = self.root / "aggregate.json"
        argv = [
            "aggregate", "--review", str(review_path), "--mapping", str(mapping_path),
            "--package-manifest", str(package_path), "--shadow", str(self.shadow_path),
            "--terminal-grounded", str(self.grounded_path), "--shadow-gate", str(shadow_gate),
            "--terminal-grounded-gate", str(grounded_gate), "--output", str(output),
        ]
        with patch.object(sys, "argv", argv):
            aggregate_main()
        report = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "positive")
        self.assertEqual(report["primary_human_fact_pass"]["base"]["denominator"], 40)
        self.assertEqual(report["primary_human_fact_pass"]["paired_difference"], 1.0)
        self.assertEqual(report["primary_human_fact_pass"]["paired_bootstrap_95_ci"]["lower_95"], 1.0)

    def test_aggregate_rejects_immutable_answer_edit(self) -> None:
        self.prepare()
        review_path, mapping_path, package_path = self.paths()
        reviews = read_rows(review_path)
        reviews[0]["answer"] = "changed after blinding"
        for row in reviews:
            row["fact_pass"] = "true"
            row["answer_complete"] = "true"
            row["contradiction_present"] = "false"
        write_jsonl(review_path, reviews)
        gates = self.root / "gate.json"
        gates.write_text('{"passed":true}\n', encoding="utf-8")
        argv = [
            "aggregate", "--review", str(review_path), "--mapping", str(mapping_path),
            "--package-manifest", str(package_path), "--shadow", str(self.shadow_path),
            "--terminal-grounded", str(self.grounded_path), "--shadow-gate", str(gates),
            "--terminal-grounded-gate", str(gates), "--output", str(self.root / "bad.json"),
        ]
        with patch.object(sys, "argv", argv), self.assertRaisesRegex(ValueError, "immutable"):
            aggregate_main()

    def test_bootstrap_is_seeded_and_reproducible(self) -> None:
        pairs = [(0, 1), (1, 1), (1, 0), (0, 1)]
        self.assertEqual(paired_bootstrap(pairs), paired_bootstrap(pairs))
        result = paired_bootstrap(pairs)
        self.assertEqual(result["samples"], 10_000)
        self.assertEqual(result["seed"], 20260818)

    def test_sidecar_gate_accepts_diagnostic_only_report(self) -> None:
        report = self.root / "report.json"
        report.write_text(json.dumps({
            "mode": "terminal_grounded", "generation_config_hash": "config",
            "verifier_code_commit": "commit", "verifier_config_hash": "verifier",
            "verifier_role": "diagnostic_only", "verifier_gate_applied": False,
        }), encoding="utf-8")
        output = self.root / "sidecar_gate.json"
        argv = [
            "gate", "--base-store", str(self.store_path), "--sidecar", str(self.grounded_path),
            "--mode", "terminal_grounded", "--expected-count", "80",
            "--run-report", str(report), "--output", str(output),
        ]
        with patch.object(sys, "argv", argv), self.assertRaises(SystemExit) as stopped:
            sidecar_gate_main()
        self.assertEqual(stopped.exception.code, 0)
        result = json.loads(output.read_text(encoding="utf-8"))
        self.assertTrue(result["passed"])
        self.assertTrue(next(row for row in result["checks"]
                             if row["name"] == "diagnostic_only_run_contract")["passed"])

    def test_sidecar_gate_rejects_smoke_to_dev_configuration_drift(self) -> None:
        report = self.root / "report.json"
        reference = self.root / "reference.json"
        common = {
            "mode": "terminal_grounded", "verifier_code_commit": "commit",
            "verifier_config_hash": "verifier", "verifier_role": "diagnostic_only",
            "verifier_gate_applied": False, "model": "model", "max_new_tokens": 1024,
            "grounding_prompt_sha256": "prompt",
        }
        report.write_text(json.dumps({**common, "generation_config_hash": "config"}), encoding="utf-8")
        reference.write_text(json.dumps({**common, "generation_config_hash": "different",
                                         "grounding_prompt_sha256": "different-prompt"}), encoding="utf-8")
        output = self.root / "drift_gate.json"
        argv = [
            "gate", "--base-store", str(self.store_path), "--sidecar", str(self.grounded_path),
            "--mode", "terminal_grounded", "--expected-count", "80", "--run-report", str(report),
            "--reference-run-report", str(reference), "--output", str(output),
        ]
        with patch.object(sys, "argv", argv), self.assertRaisesRegex(SystemExit, "1"):
            sidecar_gate_main()
        result = json.loads(output.read_text(encoding="utf-8"))
        self.assertFalse(result["passed"])
        self.assertFalse(next(row for row in result["checks"]
                              if row["name"] == "frozen_smoke_to_dev_configuration")["passed"])
        drift = next(row for row in result["checks"]
                     if row["name"] == "frozen_smoke_to_dev_configuration")["detail"]
        self.assertEqual(set(drift), {"generation_config_hash", "grounding_prompt_sha256"})

    def test_sidecar_gate_rejects_changed_ineligible_answer(self) -> None:
        rows = read_rows(self.grounded_path)
        rows[0].update(eligible=False, ineligible_reason="no_evidence", final_answer="changed anyway")
        write_jsonl(self.grounded_path, rows)
        output = self.root / "ineligible_gate.json"
        argv = [
            "gate", "--base-store", str(self.store_path), "--sidecar", str(self.grounded_path),
            "--mode", "terminal_grounded", "--expected-count", "80", "--output", str(output),
        ]
        with patch.object(sys, "argv", argv), self.assertRaisesRegex(SystemExit, "1"):
            sidecar_gate_main()
        result = json.loads(output.read_text(encoding="utf-8"))
        mismatches = next(row for row in result["checks"]
                          if row["name"] == "action_and_terminal_immutability")["detail"]
        self.assertIn("ineligible_answer_pass_through", {row["field"] for row in mismatches})


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    unittest.main()
