"""Protocol tests for the frozen tau3 Retail v1 runner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ecommerce_rag.tau3_retail_v1 import annotate_results, build_tau2_command


def test_smoke_is_train_only_and_five_tasks():
    command = build_tau2_command(
        tau_python=Path("python"),
        launcher_script=Path("launcher.py"),
        phase="smoke",
        agent_model="deepseek/deepseek-chat",
        user_model="deepseek/deepseek-chat",
        pass_k=1,
        save_to="tau3_retail_v1_smoke",
    )
    assert command[command.index("--task-split-name") + 1] == "train"
    assert command[command.index("--num-tasks") + 1] == "5"
    assert command[command.index("--max-steps") + 1] == "200"
    assert "--auto-resume" in command


@pytest.mark.parametrize("phase", ["base", "sft"])
def test_formal_arms_are_full_test_split_without_constraint(phase):
    command = build_tau2_command(
        tau_python=Path("python"),
        launcher_script=Path("launcher.py"),
        phase=phase,
        agent_model="openai/Qwen3-4B-Instruct-2507",
        user_model="deepseek/deepseek-chat",
        pass_k=2,
        save_to=f"tau3_retail_v1_{phase}",
    )
    assert command[command.index("--task-split-name") + 1] == "test"
    assert "--num-tasks" not in command
    assert "constraint" not in " ".join(command).lower()


def test_unsafe_short_episode_is_rejected():
    with pytest.raises(ValueError, match="truncate"):
        build_tau2_command(
            tau_python=Path("python"),
            launcher_script=Path("launcher.py"),
            phase="base",
            agent_model="agent",
            user_model="user",
            pass_k=1,
            save_to="out",
            max_steps=8,
        )


def test_results_embed_required_provenance_and_cost(tmp_path):
    result_path = tmp_path / "results.json"
    result_path.write_text(
        json.dumps(
            {
                "simulations": [
                    {
                        "reward_info": {
                            "reward": 1,
                            "db_check": {"db_match": True},
                            "action_checks": [
                                {"tool_type": "write", "action_match": True}
                            ],
                        },
                        "messages": [
                            {
                                "role": "assistant",
                                "usage": {"prompt_tokens": 10, "completion_tokens": 2},
                            },
                            {
                                "role": "user",
                                "usage": {"prompt_tokens": 8, "completion_tokens": 3},
                            },
                        ],
                    }
                ]
                * 5
            }
        ),
        encoding="utf-8",
    )
    summary = annotate_results(
        result_path,
        phase="smoke",
        agent_model="agent-v1",
        user_model="user-v1",
        nl_assertions_model="judge-v1",
        pass_k=1,
        wall_clock_seconds=4.0,
    )
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["tau2_commit"]
    assert payload["user_simulator_model"] == "user-v1"
    assert payload["nl_assertions_model"] == "judge-v1"
    assert payload["agent_model"] == "agent-v1"
    assert payload["pass_k"] == 1
    assert payload["action_constraint"] is False
    assert summary["valid"] is True
    assert summary["write_action_checks"] == 5
    assert summary["failed_write_action_checks"] == 0
    assert summary["db_mismatches"] == 0
    assert summary["total_tokens"] == 115
    assert summary["mean_tokens"] == 23
    assert summary["mean_wall_clock_seconds"] == 0.8


def test_incomplete_results_are_rejected(tmp_path):
    result_path = tmp_path / "results.json"
    result_path.write_text(json.dumps({"simulations": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete"):
        annotate_results(
            result_path,
            phase="base",
            agent_model="agent-v1",
            user_model="user-v1",
            nl_assertions_model="judge-v1",
            pass_k=1,
            wall_clock_seconds=4.0,
        )
