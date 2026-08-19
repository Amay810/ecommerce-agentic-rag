"""Unit checks for the read-only G0-E analyzer."""

from __future__ import annotations

from scripts.analyze_tau3_g0e import analyze


PRICING = {
    "model": "deepseek-v4-flash",
    "input_cache_hit_usd_per_1m": 0.0028,
    "input_cache_miss_usd_per_1m": 0.14,
    "output_usd_per_1m": 0.28,
}


def _payload(*, include_deepseek: bool = True, non_binary: bool = False) -> dict:
    simulations = []
    for task_id in range(74):
        rewards = [0] * 8 if task_id == 0 else [1] * 8
        if 1 < task_id:
            rewards = [0, 1] + [0] * 6
        for trial, reward in enumerate(rewards):
            if non_binary and task_id == 0 and trial == 0:
                reward = 0.5
            messages = [
                {
                    "role": "assistant",
                    "raw_data": {
                        "model": "Qwen3-4B",
                        "usage": {"prompt_tokens": 100, "completion_tokens": 5},
                    },
                    "usage": {"prompt_tokens": 100, "completion_tokens": 5},
                }
            ]
            if include_deepseek:
                messages.append(
                    {
                        "role": "user",
                        "raw_data": {
                            "model": "deepseek-v4-flash",
                            "usage": {
                                "prompt_tokens": 10,
                                "completion_tokens": 2,
                                "prompt_cache_hit_tokens": 8,
                                "prompt_cache_miss_tokens": 2,
                            },
                        },
                        "usage": {"prompt_tokens": 10, "completion_tokens": 2},
                    }
                )
            simulations.append(
                {
                    "task_id": str(task_id),
                    "trial": trial,
                    "reward_info": {"reward": reward, "nl_assertions": []},
                    "messages": messages,
                }
            )
    return {"simulations": simulations, "experiment_summary": {}}


def test_group_classes_distinguish_variance_from_non_all_zero():
    report = analyze(_payload())
    assert report["group_variance"]["classes"] == {"mixed": 72, "0/8": 1, "8/8": 1}
    assert report["group_variance"]["non_zero_variance_groups"] == 72
    assert report["group_variance"]["non_all_zero_groups"] == 73


def test_usage_excludes_qwen_and_missing_usage_has_no_cost():
    report = analyze(_payload(), pricing=PRICING)
    usage = report["deepseek_usage"]["user_simulator"]
    assert usage["calls"] == 592
    assert usage["prompt_tokens"] == 5920
    assert usage["completion_tokens"] == 1184
    assert usage["estimated_cost_usd"] > 0

    missing = analyze(_payload(include_deepseek=False), pricing=PRICING)
    assert missing["deepseek_usage"]["user_simulator"]["estimated_cost_usd"] is None


def test_non_binary_reward_is_rejected():
    try:
        analyze(_payload(non_binary=True))
    except ValueError as exc:
        assert "non-binary terminal reward" in str(exc)
    else:
        raise AssertionError("non-binary reward was accepted")
