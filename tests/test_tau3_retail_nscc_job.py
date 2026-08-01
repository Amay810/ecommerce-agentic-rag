"""Static protocol checks for the frozen NSCC Base job."""

from pathlib import Path


JOB = Path(__file__).parents[1] / "nscc" / "run_tau3_retail_base_v1.pbs"


def test_base_job_is_frozen_test_only_qwen_arm():
    text = JOB.read_text(encoding="utf-8")
    assert "--phase base" in text
    assert "--pass-k 3" in text
    assert 'VLLM_MODEL_NAME="Qwen3-4B-Instruct-2507"' in text
    assert "hosted_vllm/${VLLM_MODEL_NAME}" in text
    assert "--enable-auto-tool-choice" in text
    assert "--tool-call-parser hermes" in text
    assert "Action Constraint" not in text


def test_base_job_requires_frozen_external_models_and_commit():
    text = JOB.read_text(encoding="utf-8")
    assert "TAU3_USER_MODEL:?" in text
    assert "TAU3_NL_ASSERTIONS_MODEL:?" in text
    assert "fc0055dc4e0a316c3f83133267fbd6faaa770992" in text
    assert "experiment_summary']['valid'] is True" in text
