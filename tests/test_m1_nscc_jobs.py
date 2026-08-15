from pathlib import Path


ROOT = Path(__file__).parents[1]


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
