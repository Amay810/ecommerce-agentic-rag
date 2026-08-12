from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_teacher_job_is_train_only_native_and_strictly_audited():
    text = (ROOT / "nscc" / "run_tau3_teacher_rollout_v1.pbs").read_text(
        encoding="utf-8"
    )
    assert "--phase teacher" in text
    assert "--agent-name ecommerce_native" in text
    assert "--compaction off" in text
    assert "approved_open_weights" in text
    assert "scripts.audit_tau3_process" in text
    assert "scripts.build_verified_ecommerce_sft" in text
    assert "fc0055dc4e0a316c3f83133267fbd6faaa770992" in text


def test_m1_job_has_one_frozen_lora_config_and_data_gate():
    text = (ROOT / "nscc" / "run_ecommerce_m1_lora_v1.pbs").read_text(
        encoding="utf-8"
    )
    assert "scripts.validate_verified_sft" in text
    assert "--min-train-records 400" in text
    assert "--max-train-records 1200" in text
    assert "--tuner_type lora" in text
    assert "--lora_rank 16" in text
    assert "--num_train_epochs 2" in text
    assert "NPROC_PER_NODE=4" in text


def test_m1_service_uses_named_lora_with_native_fc():
    text = (ROOT / "nscc" / "serve_ecommerce_m1_v1.pbs").read_text(
        encoding="utf-8"
    )
    assert "Qwen3-4B-Ecommerce-M1" in text
    assert "--enable-lora" in text
    assert "--lora-modules" in text
    assert "--enable-auto-tool-choice" in text
    assert "--tool-call-parser hermes" in text
