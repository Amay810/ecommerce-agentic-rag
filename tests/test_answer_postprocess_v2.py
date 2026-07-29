from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ecommerce_rag.answer_postprocess import GROUNDING_PROMPT_SHA256, GROUNDING_SYSTEM
from ecommerce_rag.llm_policy import Generation, LLMPolicy
from scripts.run_answer_postprocess import main as run_main, terminal_generation


ROOT = Path(__file__).resolve().parents[1]


class LocalTokenConfigurationTests(unittest.TestCase):
    @staticmethod
    def fake_local_generator(model: str, max_new_tokens: int = 512):
        return (lambda _system, _user: Generation("ok"), {
            "backend": "local", "model": model, "max_new_tokens": max_new_tokens,
        })

    def test_local_backend_defaults_to_512(self) -> None:
        env = {"ARAG_AGENT_BACKEND": "local", "ARAG_LOCAL_MODEL": "local-model"}
        with patch.dict(os.environ, env, clear=True), patch.object(
            LLMPolicy, "_local_generator", side_effect=self.fake_local_generator,
        ) as builder:
            policy = LLMPolicy.from_env()
        builder.assert_called_once_with("local-model", max_new_tokens=512)
        self.assertEqual(policy.generator_meta["max_new_tokens"], 512)

    def test_local_backend_accepts_trimmed_1024(self) -> None:
        env = {
            "ARAG_AGENT_BACKEND": "local", "ARAG_LOCAL_MODEL": "local-model",
            "ARAG_LOCAL_MAX_NEW_TOKENS": " 1024 ",
        }
        with patch.dict(os.environ, env, clear=True), patch.object(
            LLMPolicy, "_local_generator", side_effect=self.fake_local_generator,
        ) as builder:
            policy = LLMPolicy.from_env()
        builder.assert_called_once_with("local-model", max_new_tokens=1024)
        self.assertEqual(policy.generator_meta["max_new_tokens"], 1024)

    def test_local_backend_rejects_invalid_token_limits(self) -> None:
        for value in ("", "0", "-1", "1.5", "abc", "+1024"):
            with self.subTest(value=value), patch.dict(os.environ, {
                "ARAG_AGENT_BACKEND": "local", "ARAG_LOCAL_MAX_NEW_TOKENS": value,
            }, clear=True), patch.object(LLMPolicy, "_local_generator") as builder:
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    LLMPolicy.from_env()
                builder.assert_not_called()


class TerminalGroundingV2ContractTests(unittest.TestCase):
    def test_grounding_prompt_hash_is_exact(self) -> None:
        self.assertEqual(GROUNDING_PROMPT_SHA256,
                         hashlib.sha256(GROUNDING_SYSTEM.encode("utf-8")).hexdigest())

    def test_terminal_generation_uses_reported_limit_and_rejects_mismatch(self) -> None:
        policy = SimpleNamespace(
            generate=lambda _system, _user: Generation("ok"),
            generator_meta={
                "backend": "local", "model": "/models/Qwen3-4B-Instruct-2507",
                "max_new_tokens": 1024,
            },
        )
        with patch("scripts.run_answer_postprocess.LLMPolicy.from_env", return_value=policy):
            _, config = terminal_generation(1024)
            self.assertEqual(config["max_new_tokens"], 1024)
            with self.assertRaisesRegex(ValueError, "frozen expectation"):
                terminal_generation(512)

    def test_terminal_report_records_actual_limit_model_and_prompt_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = root / "base.sqlite"
            output = root / "output.jsonl"
            report = root / "report.json"
            trajectory = {
                "trajectory_id": "tr-1", "task_id": "task-1",
                "actions": [{"action_type": "final_answer", "content": "draft"}],
                "tool_calls": [], "final_state": {},
                "evidence_ledger": [{"evidence_id": "E1", "field": "product.price",
                                     "value": 99, "text": "99"}],
                "messages": [], "final_answer": "draft",
            }
            grade = {"answer_fact_applicable": True}
            conn = sqlite3.connect(store)
            conn.execute("CREATE TABLE trajectories(task_id TEXT, trajectory_json TEXT, grade_json TEXT)")
            conn.execute("INSERT INTO trajectories VALUES(?, ?, ?)",
                         ("task-1", json.dumps(trajectory), json.dumps(grade)))
            conn.commit()
            conn.close()
            policy = SimpleNamespace(
                generate=lambda _system, _user: Generation(
                    "价格为99元。[E1]", "stop", 20, 8, False,
                ),
                generator_meta={
                    "backend": "local", "model": "/models/Qwen3-4B-Instruct-2507",
                    "max_new_tokens": 1024, "chat_template_present": True,
                },
            )
            argv = [
                "run", "--base-store", str(store), "--mode", "terminal_grounded",
                "--expected-count", "1", "--expected-max-new-tokens", "1024",
                "--output-jsonl", str(output), "--report", str(report),
            ]
            with patch.object(sys, "argv", argv), patch(
                "scripts.run_answer_postprocess.LLMPolicy.from_env", return_value=policy,
            ):
                run_main()
            payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(payload["model"], "/models/Qwen3-4B-Instruct-2507")
            self.assertEqual(payload["max_new_tokens"], 1024)
            self.assertEqual(payload["grounding_prompt_sha256"], GROUNDING_PROMPT_SHA256)
            record = json.loads(output.read_text(encoding="utf-8").strip())
            self.assertEqual(record["generation_config"]["max_new_tokens"], 1024)

    def test_v2_pbs_files_are_isolated_and_frozen(self) -> None:
        smoke = (ROOT / "nscc" / "run_answer_postprocess_smoke_v2.pbs").read_text(encoding="utf-8")
        dev = (ROOT / "nscc" / "run_answer_postprocess_dev_v2.pbs").read_text(encoding="utf-8")
        for text in (smoke, dev):
            self.assertIn("ARAG_LOCAL_MAX_NEW_TOKENS=1024", text)
            self.assertIn("--expected-max-new-tokens 1024", text)
            self.assertNotIn("module load git", text)
        self.assertIn("answer_postprocess_smoke_v1_manifest.json", smoke)
        self.assertIn("answer_postprocess_smoke_v2_terminal_grounded", smoke)
        self.assertNotIn("answer_postprocess_smoke_v1_terminal_grounded", smoke)
        self.assertIn("answer_postprocess_dev_v2_terminal_grounded", dev)
        self.assertIn("answer_postprocess_smoke_v2_terminal_grounded_report.json", dev)
        self.assertNotIn("answer_postprocess_dev_v1_", dev)

    def test_v1_closeout_forbids_human_audit_and_algorithmic_claim(self) -> None:
        closeout = json.loads(
            (ROOT / "docs" / "answer_postprocess_dev_v1_closeout.json").read_text(encoding="utf-8")
        )
        self.assertEqual(closeout["status"], "aborted_generation_truncated")
        self.assertEqual(closeout["failures"], [{
            "task_id": "evidence_a_dev_compare_09", "error": "generation_truncated",
        }])
        self.assertIs(closeout["terminal_grounded_sidecar_published"], False)
        self.assertIs(closeout["human_audit_allowed"], False)
        self.assertIs(closeout["algorithmic_result_available"], False)
        self.assertIn("no 2048-token v3", closeout["v2_failure_policy"])


if __name__ == "__main__":
    unittest.main()
