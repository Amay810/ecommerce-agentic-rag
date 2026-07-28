# -*- coding: utf-8 -*-
"""The gate decides whether to spend a full cluster run; it must not be lenient."""

import unittest
import copy

from scripts.smoke_gate import evaluate

MANIFEST = {
    "count": 2,
    "scenarios": {
        "order_query": {"task_id": "t_order", "category": "order_query", "handoff_expected": False},
        "safety": {"task_id": "t_safety", "category": "safety", "handoff_expected": True},
    },
}


def _details(**overrides):
    rows = {
        "t_order": {"task_id": "t_order", "success": True, "terminal_state_match": True,
                    "policy_compliant": True, "handoff_observed": False, "failure_type": None},
        "t_safety": {"task_id": "t_safety", "success": True, "terminal_state_match": True,
                     "policy_compliant": True, "handoff_observed": True, "failure_type": None},
    }
    for task_id, patch in overrides.items():
        rows[task_id].update(patch)
    return {"details": list(rows.values())}


def _diagnosis(**overrides):
    quality = {"effective_action_parse_rate": 1.0, "strict_envelope_parse_rate": 1.0,
               "illegal_tool_rate": 0.0, "generation_error_rate": 0.0,
               "fallback_only_trajectory_rate": 0.0, "truncation_rate": 0.0}
    quality.update(overrides.pop("quality", {}))
    diagnosis = {"instrumented": True, "trajectories": 2, "quality": quality, "envelope_violations": {}}
    diagnosis.update(overrides)
    return diagnosis


def _run(report=None, diagnosis=None, min_parse_rate=0.9):
    return evaluate(MANIFEST, report or _details(), diagnosis or _diagnosis(), min_parse_rate)


class SmokeGateTests(unittest.TestCase):
    def test_a_clean_run_passes(self):
        result = _run()
        self.assertTrue(result["passed"], result["failed_checks"])
        self.assertEqual(result["warnings"], [])

    def test_uninstrumented_store_is_blocked(self):
        result = _run(diagnosis=_diagnosis(instrumented=False))
        self.assertFalse(result["passed"])
        self.assertIn("instrumented", result["failed_checks"])

    def test_any_generation_error_blocks(self):
        result = _run(diagnosis=_diagnosis(quality={"generation_error_rate": 0.05}))
        self.assertIn("generation_error_rate_is_zero", result["failed_checks"])

    def test_any_fallback_only_trajectory_blocks(self):
        result = _run(diagnosis=_diagnosis(quality={"fallback_only_trajectory_rate": 0.125}))
        self.assertIn("fallback_only_trajectory_rate_is_zero", result["failed_checks"])

    def test_any_illegal_tool_blocks(self):
        result = _run(diagnosis=_diagnosis(quality={"illegal_tool_rate": 0.01}))
        self.assertIn("illegal_tool_rate_is_zero", result["failed_checks"])

    def test_a_single_success_does_not_clear_the_parse_rate_bar(self):
        # the bar this gate exists to replace: "> 0" would have let 1-in-10 through
        result = _run(diagnosis=_diagnosis(quality={"effective_action_parse_rate": 0.1}))
        self.assertFalse(result["passed"])
        self.assertIn("effective_action_parse_rate", result["failed_checks"])

    def test_trajectory_count_must_match_the_manifest(self):
        result = _run(diagnosis=_diagnosis(trajectories=1))
        self.assertIn("trajectory_count", result["failed_checks"])

    def test_a_failed_scenario_blocks_even_when_rates_are_perfect(self):
        result = _run(report=_details(t_order={"success": False, "failure_type": "wrong-tool"}))
        self.assertFalse(result["passed"])
        self.assertIn("scenario:order_query", result["failed_checks"])

    def test_terminal_state_mismatch_blocks(self):
        result = _run(report=_details(t_order={"terminal_state_match": False, "state_diff": {"x": 1}}))
        self.assertIn("scenario:order_query", result["failed_checks"])

    def test_safety_scenario_requires_an_observed_handoff(self):
        result = _run(report=_details(t_safety={"handoff_observed": False}))
        self.assertIn("scenario:safety", result["failed_checks"])

    def test_forbidden_tool_use_blocks(self):
        result = _run(report=_details(t_safety={"policy_compliant": False}))
        self.assertIn("scenario:safety", result["failed_checks"])

    def test_missing_scenario_is_reported(self):
        result = _run(report={"details": [_details()["details"][0]]})
        self.assertIn("all_scenarios_ran", result["failed_checks"])

    def test_sequence_failure_reports_expected_raw_successful_and_empty_calls(self):
        manifest = copy.deepcopy(MANIFEST)
        manifest["scenarios"]["order_query"]["expected_tool_sequence"] = ["search_catalog", "get_product"]
        report = _details(t_order={
            "success": False, "failure_type": "missing-required-tool", "tool_sequence_match": False,
            "raw_observed_tool_sequence": ["search_catalog"], "successful_tool_sequence": ["search_catalog"],
            "failed_or_empty_tool_calls": [{"tool_name": "search_catalog", "status": "empty_result"}],
        })
        result = evaluate(manifest, report, _diagnosis(), 0.9)
        row = next(check for check in result["checks"] if check["check"] == "scenario:order_query")
        for marker in ("expected=", "raw_observed=", "successful=", "failed_or_empty="):
            self.assertIn(marker, row["detail"])

class MissingMetricTests(unittest.TestCase):
    """A gate that exists to fail closed must not pass when it cannot see a metric.

    `(quality.get(x) or 0) == 0` reads as a zero check but treats an absent field
    as zero, so a renamed metric would silently clear the gate. This project has
    already renamed one quality metric once.
    """

    @staticmethod
    def _without(*keys):
        quality = {"effective_action_parse_rate": 1.0, "strict_envelope_parse_rate": 1.0,
                   "illegal_tool_rate": 0.0, "generation_error_rate": 0.0,
                   "fallback_only_trajectory_rate": 0.0, "truncation_rate": 0.0}
        for key in keys:
            quality.pop(key)
        return {"instrumented": True, "trajectories": 2, "quality": quality, "envelope_violations": {}}

    def test_missing_generation_error_rate_blocks(self):
        result = _run(diagnosis=self._without("generation_error_rate"))
        self.assertFalse(result["passed"])
        self.assertIn("generation_error_rate_is_zero", result["failed_checks"])
        self.assertIn("diagnosis_has_required_metrics", result["failed_checks"])

    def test_missing_fallback_only_rate_blocks(self):
        result = _run(diagnosis=self._without("fallback_only_trajectory_rate"))
        self.assertFalse(result["passed"])
        self.assertIn("fallback_only_trajectory_rate_is_zero", result["failed_checks"])

    def test_missing_illegal_tool_rate_blocks(self):
        result = _run(diagnosis=self._without("illegal_tool_rate"))
        self.assertFalse(result["passed"])
        self.assertIn("illegal_tool_rate_is_zero", result["failed_checks"])

    def test_missing_parse_rate_blocks(self):
        result = _run(diagnosis=self._without("effective_action_parse_rate"))
        self.assertFalse(result["passed"])
        self.assertIn("effective_action_parse_rate", result["failed_checks"])

    def test_an_empty_quality_block_blocks_everything(self):
        result = _run(diagnosis={"instrumented": True, "trajectories": 2, "quality": {}})
        self.assertFalse(result["passed"])
        for name in ("diagnosis_has_required_metrics", "generation_error_rate_is_zero",
                     "fallback_only_trajectory_rate_is_zero", "illegal_tool_rate_is_zero",
                     "effective_action_parse_rate"):
            self.assertIn(name, result["failed_checks"])

    def test_a_renamed_metric_is_caught_by_the_completeness_check(self):
        diagnosis = self._without("illegal_tool_rate")
        diagnosis["quality"]["illegal_tool_ratio"] = 0.0  # plausible rename
        result = _run(diagnosis=diagnosis)
        self.assertFalse(result["passed"])
        self.assertIn("diagnosis_has_required_metrics", result["failed_checks"])


class WarningTests(unittest.TestCase):
    def test_envelope_violations_warn_but_do_not_block(self):
        result = _run(diagnosis=_diagnosis(
            quality={"strict_envelope_parse_rate": 0.5},
            envelope_violations={"markdown_fence": 4}))
        self.assertTrue(result["passed"], result["failed_checks"])
        self.assertEqual([w["check"] for w in result["warnings"]], ["strict_envelope_parse_rate"])

    def test_truncation_warns_but_does_not_block(self):
        result = _run(diagnosis=_diagnosis(quality={"truncation_rate": 0.2}))
        self.assertTrue(result["passed"], result["failed_checks"])
        self.assertIn("truncation_rate", [w["check"] for w in result["warnings"]])


if __name__ == "__main__":
    unittest.main()
