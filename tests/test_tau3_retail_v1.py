"""Protocol tests for the frozen tau3 Retail v1 runner.

The central property under test is that the annotation cannot prove itself: a
results file is accepted only when tau2's *own* ``info`` block matches the
requested configuration, and a missing native field is a failure rather than a
silent pass.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

import ecommerce_rag.tau3_retail_v1 as tau3_module
from ecommerce_rag.tau3_retail_v1 import (
    TAU2_COMMIT,
    annotate_results,
    build_tau2_command,
    command_from_requested,
    requested_config,
    verify_tau2_source,
)
from scripts.run_tau3_retail_v1 import _configure_provider_environment, main


# --------------------------------------------------------------------------
# command construction
# --------------------------------------------------------------------------


def test_smoke_is_train_only_and_names_its_tasks():
    command = build_tau2_command(
        tau_python=Path("python"),
        launcher_script=Path("launcher.py"),
        phase="smoke",
        agent_name="llm_agent",
        agent_model="deepseek/deepseek-chat",
        user_model="deepseek/deepseek-chat",
        agent_temperature=0.0,
        user_temperature=0.0,
        seed=300,
        pass_k=1,
        max_steps=200,
        save_to="tau3_retail_v1_smoke",
        task_ids=["0", "1"],
    )
    assert command[command.index("--task-split-name") + 1] == "train"
    assert command[command.index("--task-ids") + 1 :] == ["0", "1"]
    assert command[command.index("--max-steps") + 1] == "200"
    assert "--auto-resume" in command
    # The 5-task fallback is gone: a smoke run states which tasks it ran.
    assert "--num-tasks" not in command


def test_smoke_without_task_ids_is_rejected_instead_of_defaulting_to_five():
    with pytest.raises(ValueError, match="no default task count"):
        build_tau2_command(
            tau_python=Path("python"),
            launcher_script=Path("launcher.py"),
            phase="smoke",
            agent_name="llm_agent",
            agent_model="deepseek/deepseek-chat",
            user_model="deepseek/deepseek-chat",
            agent_temperature=0.0,
            user_temperature=0.0,
            seed=300,
            pass_k=1,
            max_steps=200,
            save_to="tau3_retail_v1_smoke",
        )


def test_product_runtime_agent_can_be_selected_without_changing_tau_backend():
    command = build_tau2_command(
        tau_python=Path("python"),
        launcher_script=Path("launcher.py"),
        phase="base",
        agent_name="ecommerce_native",
        agent_model="hosted_vllm/Qwen3-4B-Instruct-2507",
        user_model="deepseek/deepseek-chat",
        agent_temperature=0.0,
        user_temperature=0.0,
        seed=300,
        pass_k=1,
        max_steps=200,
        save_to="system_v1",
    )
    assert command[command.index("--agent") + 1] == "ecommerce_native"
    assert command[command.index("--domain") + 1] == "retail"
    assert command[command.index("--task-split-name") + 1] == "test"


@pytest.mark.parametrize("phase", ["base", "sft"])
def test_formal_arms_are_full_test_split_without_constraint(phase):
    command = build_tau2_command(
        tau_python=Path("python"),
        launcher_script=Path("launcher.py"),
        phase=phase,
        agent_name="llm_agent",
        agent_model="openai/Qwen3-4B-Instruct-2507",
        user_model="deepseek/deepseek-chat",
        agent_temperature=0.0,
        user_temperature=0.0,
        seed=300,
        pass_k=2,
        max_steps=200,
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
            agent_name="llm_agent",
            agent_model="agent",
            user_model="user",
            agent_temperature=0.0,
            user_temperature=0.0,
            seed=300,
            pass_k=1,
            max_steps=8,
            save_to="out",
        )


def test_teacher_arm_uses_all_frozen_train_tasks():
    command = build_tau2_command(
        tau_python=Path("python"),
        launcher_script=Path("launcher.py"),
        phase="teacher",
        agent_name="ecommerce_native",
        agent_model="hosted_vllm/Qwen3-Teacher",
        user_model="deepseek/deepseek-chat",
        agent_temperature=0.8,
        user_temperature=0.0,
        seed=1234,
        pass_k=3,
        max_steps=200,
        save_to="teacher_rollout",
    )
    assert command[command.index("--task-split-name") + 1] == "train"
    assert "--num-tasks" not in command
    assert command[command.index("--num-trials") + 1] == "3"


def test_temperature_and_seed_are_emitted_in_the_pinned_cli_format():
    """tau2 parses --agent-llm-args/--user-llm-args with json.loads."""
    command = build_tau2_command(
        tau_python=Path("python"),
        launcher_script=Path("launcher.py"),
        phase="teacher",
        agent_name="llm_agent",
        agent_model="agent",
        user_model="user",
        agent_temperature=0.8,
        user_temperature=0.0,
        seed=4242,
        pass_k=1,
        max_steps=200,
        save_to="out",
    )
    agent_args = json.loads(command[command.index("--agent-llm-args") + 1])
    user_args = json.loads(command[command.index("--user-llm-args") + 1])
    assert agent_args == {"temperature": 0.8}
    assert user_args == {"temperature": 0.0}
    assert command[command.index("--seed") + 1] == "4242"


# --------------------------------------------------------------------------
# results verification
# --------------------------------------------------------------------------


def _requested(**overrides):
    base = dict(
        phase="smoke",
        agent_name="llm_agent",
        agent_model="hosted_vllm/Qwen3-4B-Instruct-2507",
        user_model="deepseek/deepseek-chat",
        nl_assertions_model="deepseek/deepseek-chat",
        agent_temperature=0.0,
        user_temperature=0.0,
        seed=300,
        pass_k=1,
        max_steps=200,
        compaction="off",
        save_to="smoke",
        task_ids=["0", "1"],
    )
    base.update(overrides)
    return requested_config(**base)


def _command(requested):
    return command_from_requested(
        tau_python=Path("python"),
        launcher_script=Path("launcher.py"),
        requested=requested,
    )


def _native_info(**overrides):
    """A faithful subset of what tau2 v1.0.1 writes into results.json info."""
    info = {
        "git_commit": "084fd225e8d397b445af886e7e6211226ca9ee46",
        "num_trials": 1,
        "max_steps": 200,
        "max_errors": 10,
        "seed": 300,
        "agent_info": {
            "implementation": "llm_agent",
            "llm": "hosted_vllm/Qwen3-4B-Instruct-2507",
            "llm_args": {"temperature": 0.0},
        },
        "user_info": {
            "implementation": "user_simulator",
            "llm": "deepseek/deepseek-chat",
            "llm_args": {"temperature": 0.0},
        },
    }
    info.update(overrides)
    return info


def _simulation():
    return {
        "reward_info": {
            "reward": 1,
            "db_check": {"db_match": True},
            "action_checks": [{"tool_type": "write", "action_match": True}],
        },
        "messages": [
            {"role": "assistant", "usage": {"prompt_tokens": 10, "completion_tokens": 2}},
            {"role": "user", "usage": {"prompt_tokens": 8, "completion_tokens": 3}},
        ],
    }


def _write_results(tmp_path, *, info=None, simulations=2, extra=None):
    payload = {"simulations": [_simulation() for _ in range(simulations)]}
    if info is not None:
        payload["info"] = info
    if extra:
        payload.update(extra)
    result_path = tmp_path / "results.json"
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    return result_path


def _annotate(result_path, requested=None):
    requested = requested or _requested()
    return annotate_results(
        result_path,
        requested=requested,
        command=_command(requested),
        tau2_source={"commit": TAU2_COMMIT, "verified_by": "snapshot"},
        wall_clock_seconds=4.0,
    )


def test_annotation_records_native_observed_config_separately_from_requested(tmp_path):
    result_path = _write_results(tmp_path, info=_native_info())
    summary = _annotate(result_path)

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    block = payload["tau3_experiment"]

    # tau2's own record survives untouched.
    assert payload["info"] == _native_info()

    # The four layers stay distinguishable.
    assert block["requested"]["agent_model"] == "hosted_vllm/Qwen3-4B-Instruct-2507"
    assert block["command"][block["command"].index("--seed") + 1] == "300"
    assert block["native_observed"]["agent_llm"] == "hosted_vllm/Qwen3-4B-Instruct-2507"
    assert block["native_observed"]["agent_implementation"] == "llm_agent"
    assert block["native_observed"]["agent_temperature"] == 0.0
    assert block["native_observed"]["user_temperature"] == 0.0
    assert block["native_observed"]["seed"] == 300
    assert block["native_observed"]["num_trials"] == 1
    assert block["tau2_source"]["commit"] == TAU2_COMMIT
    assert block["requested"]["compaction"] == "off"
    assert set(block["checks_passed"]) >= {
        "agent_name",
        "agent_model",
        "user_model",
        "agent_temperature",
        "user_temperature",
        "seed",
        "num_trials",
    }

    assert summary["valid"] is True
    assert summary["write_action_checks"] == 2
    assert summary["failed_write_action_checks"] == 0
    assert summary["db_mismatches"] == 0
    assert summary["total_tokens"] == 46
    assert summary["mean_tokens"] == 23
    assert summary["mean_wall_clock_seconds"] == 2.0


def test_missing_info_is_rejected(tmp_path):
    result_path = _write_results(tmp_path, info=None)
    with pytest.raises(ValueError, match="native info is missing"):
        _annotate(result_path)


def test_missing_agent_info_is_rejected(tmp_path):
    info = _native_info()
    del info["agent_info"]
    result_path = _write_results(tmp_path, info=info)
    with pytest.raises(ValueError, match="native info.agent_info is missing"):
        _annotate(result_path)


def test_missing_agent_llm_args_is_rejected(tmp_path):
    info = _native_info()
    del info["agent_info"]["llm_args"]
    result_path = _write_results(tmp_path, info=info)
    with pytest.raises(ValueError, match="native info.agent_info.llm_args is missing"):
        _annotate(result_path)


def test_missing_user_llm_args_is_rejected(tmp_path):
    info = _native_info()
    del info["user_info"]["llm_args"]
    result_path = _write_results(tmp_path, info=info)
    with pytest.raises(ValueError, match="native info.user_info.llm_args is missing"):
        _annotate(result_path)


def test_missing_seed_is_rejected(tmp_path):
    info = _native_info()
    del info["seed"]
    result_path = _write_results(tmp_path, info=info)
    with pytest.raises(ValueError, match="native info.seed is missing"):
        _annotate(result_path)


def test_agent_temperature_mismatch_is_rejected(tmp_path):
    info = _native_info()
    info["agent_info"]["llm_args"] = {"temperature": 0.8}
    result_path = _write_results(tmp_path, info=info)
    with pytest.raises(ValueError, match="native agent_temperature does not match"):
        _annotate(result_path)


def test_user_temperature_mismatch_is_rejected(tmp_path):
    info = _native_info()
    info["user_info"]["llm_args"] = {"temperature": 1.0}
    result_path = _write_results(tmp_path, info=info)
    with pytest.raises(ValueError, match="native user_temperature does not match"):
        _annotate(result_path)


def test_seed_mismatch_is_rejected(tmp_path):
    result_path = _write_results(tmp_path, info=_native_info(seed=1))
    with pytest.raises(ValueError, match="native seed does not match"):
        _annotate(result_path)


def test_num_trials_mismatch_is_rejected(tmp_path):
    result_path = _write_results(tmp_path, info=_native_info(num_trials=4))
    with pytest.raises(ValueError, match="native num_trials does not match"):
        _annotate(result_path)


def test_agent_name_mismatch_is_rejected(tmp_path):
    info = _native_info()
    info["agent_info"]["implementation"] = "ecommerce_native"
    result_path = _write_results(tmp_path, info=info)
    with pytest.raises(ValueError, match="native agent_name does not match"):
        _annotate(result_path)


def test_requested_fields_written_at_top_level_cannot_launder_a_mismatch(tmp_path):
    """The regression this whole rewrite exists for.

    A results file that already carries the *requested* values as top-level keys
    -- exactly what the previous annotate_results wrote back -- must still fail
    when tau2's own info block disagrees.
    """
    info = _native_info(seed=999)
    info["agent_info"]["llm"] = "openai/some-other-model"
    info["agent_info"]["llm_args"] = {"temperature": 1.5}
    result_path = _write_results(
        tmp_path,
        info=info,
        extra={
            "agent_model": "hosted_vllm/Qwen3-4B-Instruct-2507",
            "user_simulator_model": "deepseek/deepseek-chat",
            "pass_k": 1,
            "tau2_commit": TAU2_COMMIT,
        },
    )
    with pytest.raises(ValueError, match="does not match requested"):
        _annotate(result_path)


def test_incomplete_results_are_rejected(tmp_path):
    result_path = _write_results(tmp_path, info=_native_info(), simulations=1)
    with pytest.raises(ValueError, match="incomplete"):
        _annotate(result_path)


def test_nothing_is_written_when_verification_fails(tmp_path):
    result_path = _write_results(tmp_path, info=_native_info(seed=1))
    before = result_path.read_text(encoding="utf-8")
    with pytest.raises(ValueError):
        _annotate(result_path)
    assert result_path.read_text(encoding="utf-8") == before


# --------------------------------------------------------------------------
# CLI surface
# --------------------------------------------------------------------------


@pytest.fixture
def fake_tau_root(tmp_path):
    root = tmp_path / "tau2"
    retail = root / "data" / "tau2" / "domains" / "retail"
    retail.mkdir(parents=True)
    splits = {
        "train": [str(i) for i in range(74)],
        "test": [str(i) for i in range(74, 114)],
        "base": [str(i) for i in range(114)],
    }
    retail.joinpath("split_tasks.json").write_text(json.dumps(splits), encoding="utf-8")
    retail.joinpath("tasks.json").write_text(
        json.dumps([{"id": str(i)} for i in range(114)]), encoding="utf-8"
    )
    root.joinpath("SNAPSHOT.json").write_text(
        json.dumps({"upstream_commit": TAU2_COMMIT}), encoding="utf-8"
    )
    venv_python = root / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("", encoding="utf-8")
    return root


def _cli(fake_tau_root, *extra):
    return [
        "run_tau3_retail_v1.py",
        "--tau-root",
        str(fake_tau_root),
        "--phase",
        "smoke",
        "--agent-model",
        "hosted_vllm/Qwen3-4B-Instruct-2507",
        "--user-model",
        "deepseek/deepseek-chat",
        "--nl-assertions-model",
        "deepseek/deepseek-chat",
        "--save-to",
        "smoke",
        "--task-ids",
        "0",
        "1",
        *extra,
    ]


def test_check_only_manifest_carries_real_temperature_and_seed(
    fake_tau_root, monkeypatch, capsys
):
    monkeypatch.setattr(
        sys,
        "argv",
        _cli(
            fake_tau_root,
            "--agent-name",
            "llm_agent",
            "--agent-temperature",
            "0.8",
            "--user-temperature",
            "0.0",
            "--seed",
            "4242",
            "--pass-k",
            "1",
            "--check-only",
        ),
    )
    main()
    manifest = json.loads(capsys.readouterr().out)

    assert manifest["requested"]["agent_name"] == "llm_agent"
    assert manifest["requested"]["agent_temperature"] == 0.8
    assert manifest["requested"]["user_temperature"] == 0.0
    assert manifest["requested"]["seed"] == 4242
    assert manifest["requested"]["compaction"] == "off"
    assert manifest["tau2_source"]["verified_by"] == "snapshot"

    command = manifest["command"]
    assert json.loads(command[command.index("--agent-llm-args") + 1]) == {
        "temperature": 0.8
    }
    assert json.loads(command[command.index("--user-llm-args") + 1]) == {
        "temperature": 0.0
    }
    assert command[command.index("--seed") + 1] == "4242"
    assert command[command.index("--agent") + 1] == "llm_agent"


@pytest.mark.parametrize(
    "omitted",
    [
        ("--agent-name", "llm_agent"),
        ("--agent-temperature", "0.0"),
        ("--user-temperature", "0.0"),
        ("--seed", "300"),
        ("--pass-k", "1"),
    ],
)
def test_cli_fails_when_an_experiment_control_variable_is_omitted(
    fake_tau_root, monkeypatch, omitted
):
    full = {
        "--agent-name": "llm_agent",
        "--agent-temperature": "0.0",
        "--user-temperature": "0.0",
        "--seed": "300",
        "--pass-k": "1",
    }
    del full[omitted[0]]
    extra = [token for pair in full.items() for token in pair] + ["--check-only"]
    monkeypatch.setattr(sys, "argv", _cli(fake_tau_root, *extra))
    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 2


def test_hosted_vllm_and_deepseek_use_separate_provider_environment(monkeypatch):
    monkeypatch.setenv("TAU3_AGENT_BASE_URL", "http://127.0.0.1:8123/v1")
    monkeypatch.setenv("TAU3_USER_API_KEY", "test-only-key")
    environment = {}
    _configure_provider_environment(
        environment,
        "hosted_vllm/Qwen3-4B-Instruct-2507",
        "deepseek/deepseek-chat",
        "deepseek/deepseek-chat",
    )
    assert environment["HOSTED_VLLM_API_BASE"] == "http://127.0.0.1:8123/v1"
    assert environment["HOSTED_VLLM_API_KEY"] == "local-vllm"
    assert environment["DEEPSEEK_API_KEY"] == "test-only-key"
    assert environment["NO_PROXY"] == "127.0.0.1,localhost"
    assert environment["no_proxy"] == "127.0.0.1,localhost"
    assert "OPENAI_API_KEY" not in environment


# --------------------------------------------------------------------------
# frozen Base arm: the agent under evaluation is Qwen served by vLLM
# --------------------------------------------------------------------------


def test_frozen_base_arm_command_carries_every_control_variable():
    """The Base arm evaluates llm_agent backed by hosted_vllm Qwen3-4B."""
    command = build_tau2_command(
        tau_python=Path("python"),
        launcher_script=Path("launcher.py"),
        phase="base",
        agent_name="llm_agent",
        agent_model="hosted_vllm/Qwen3-4B-Instruct-2507",
        user_model="deepseek/deepseek-chat",
        agent_temperature=0.0,
        user_temperature=0.0,
        seed=300,
        pass_k=3,
        max_steps=200,
        save_to="tau3_retail_v1_base",
    )
    assert command[command.index("--agent") + 1] == "llm_agent"
    assert (
        command[command.index("--agent-llm") + 1]
        == "hosted_vllm/Qwen3-4B-Instruct-2507"
    )
    assert json.loads(command[command.index("--agent-llm-args") + 1]) == {
        "temperature": 0.0
    }
    assert json.loads(command[command.index("--user-llm-args") + 1]) == {
        "temperature": 0.0
    }
    assert command[command.index("--seed") + 1] == "300"
    assert command[command.index("--num-trials") + 1] == "3"
    assert command[command.index("--max-steps") + 1] == "200"


def test_agent_model_mismatch_is_rejected(tmp_path):
    info = _native_info()
    info["agent_info"]["llm"] = "hosted_vllm/some-other-checkpoint"
    result_path = _write_results(tmp_path, info=info)
    with pytest.raises(ValueError, match="native agent_model does not match"):
        _annotate(result_path)


def test_user_model_mismatch_is_rejected(tmp_path):
    info = _native_info()
    info["user_info"]["llm"] = "openai/gpt-4.1-2025-04-14"
    result_path = _write_results(tmp_path, info=info)
    with pytest.raises(ValueError, match="native user_model does not match"):
        _annotate(result_path)


def test_max_steps_mismatch_is_rejected(tmp_path):
    result_path = _write_results(tmp_path, info=_native_info(max_steps=50))
    with pytest.raises(ValueError, match="native max_steps does not match"):
        _annotate(result_path)


def test_missing_agent_implementation_is_rejected(tmp_path):
    info = _native_info()
    del info["agent_info"]["implementation"]
    result_path = _write_results(tmp_path, info=info)
    with pytest.raises(
        ValueError, match="native info.agent_info.implementation is missing"
    ):
        _annotate(result_path)


def test_missing_agent_llm_is_rejected(tmp_path):
    info = _native_info()
    del info["agent_info"]["llm"]
    result_path = _write_results(tmp_path, info=info)
    with pytest.raises(ValueError, match="native info.agent_info.llm is missing"):
        _annotate(result_path)


def test_missing_num_trials_is_rejected(tmp_path):
    info = _native_info()
    del info["num_trials"]
    result_path = _write_results(tmp_path, info=info)
    with pytest.raises(ValueError, match="native info.num_trials is missing"):
        _annotate(result_path)


# --------------------------------------------------------------------------
# tau2 source identity
# --------------------------------------------------------------------------


def test_snapshot_upstream_commit_is_the_authority(tmp_path):
    tmp_path.joinpath("SNAPSHOT.json").write_text(
        json.dumps({"upstream_commit": TAU2_COMMIT}), encoding="utf-8"
    )
    source = verify_tau2_source(tmp_path)
    assert source == {
        "commit": TAU2_COMMIT,
        "verified_by": "snapshot",
        "source": str(tmp_path / "SNAPSHOT.json"),
    }


def test_snapshot_upstream_commit_mismatch_is_rejected(tmp_path):
    tmp_path.joinpath("SNAPSHOT.json").write_text(
        json.dumps({"upstream_commit": "0" * 40}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="tau2 snapshot commit drift"):
        verify_tau2_source(tmp_path)


def test_snapshot_without_upstream_commit_is_rejected(tmp_path):
    tmp_path.joinpath("SNAPSHOT.json").write_text(
        json.dumps({"upstream_tag": "v1.0.1"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="missing upstream_commit"):
        verify_tau2_source(tmp_path)


def test_no_snapshot_and_no_own_git_fails_without_crawling_to_parent(
    tmp_path, monkeypatch
):
    """A snapshot vendored inside another repository must not inherit its HEAD.

    ``git rev-parse HEAD`` does not fail in a directory without ``.git`` -- it
    walks up. So git is never consulted unless tau_root is itself a checkout.
    """
    parent = tmp_path / "enclosing_repo"
    (parent / ".git").mkdir(parents=True)
    tau_root = parent / "vendor" / "tau2-bench-fc0055dc"
    tau_root.mkdir(parents=True)

    def _explode(repository):
        raise AssertionError("git_head must not run when tau_root has no .git")

    monkeypatch.setattr(tau3_module, "git_head", _explode)

    with pytest.raises(ValueError, match="not itself a git checkout"):
        verify_tau2_source(tau_root)


def test_own_git_checkout_is_verified_against_the_pinned_commit(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(tau3_module, "git_head", lambda repository: "1" * 40)
    with pytest.raises(ValueError, match="tau2 commit drift"):
        verify_tau2_source(tmp_path)

    monkeypatch.setattr(tau3_module, "git_head", lambda repository: TAU2_COMMIT)
    assert verify_tau2_source(tmp_path)["verified_by"] == "git"


def test_gitfile_worktree_counts_as_its_own_checkout(tmp_path, monkeypatch):
    (tmp_path / ".git").write_text("gitdir: /elsewhere/.git/worktrees/w", encoding="utf-8")
    monkeypatch.setattr(tau3_module, "git_head", lambda repository: TAU2_COMMIT)
    assert verify_tau2_source(tmp_path)["verified_by"] == "git"


@pytest.mark.parametrize(
    "path_to_drop",
    [
        ("info",),
        ("info", "agent_info"),
        ("info", "agent_info", "implementation"),
        ("info", "agent_info", "llm"),
        ("info", "agent_info", "llm_args"),
        ("info", "agent_info", "llm_args", "temperature"),
        ("info", "user_info"),
        ("info", "user_info", "llm"),
        ("info", "user_info", "llm_args"),
        ("info", "user_info", "llm_args", "temperature"),
        ("info", "seed"),
        ("info", "num_trials"),
        ("info", "max_steps"),
    ],
)
def test_every_required_native_field_fails_closed_when_absent(tmp_path, path_to_drop):
    payload = {
        "info": _native_info(),
        "simulations": [_simulation(), _simulation()],
    }
    container = payload
    for key in path_to_drop[:-1]:
        container = container[key]
    del container[path_to_drop[-1]]

    result_path = tmp_path / "results.json"
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    before = result_path.read_text(encoding="utf-8")

    expected = "native " + ".".join(path_to_drop) + " is missing"
    with pytest.raises(ValueError, match=re.escape(expected)):
        _annotate(result_path)
    assert result_path.read_text(encoding="utf-8") == before
