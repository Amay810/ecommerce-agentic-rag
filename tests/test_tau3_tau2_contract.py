"""Regression checks against the vendored tau2 evaluator boundary.

Run with the archive source on PYTHONPATH when the CPU-side tau2 dependencies
are available.  The evaluator itself is patched in this test so no provider
or network call is made.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from ecommerce_rag.grpo.reward_adapter import OfficialTerminalRewardAdapter


def _tau2_src() -> Path:
    root = Path(
        os.environ.get("TAU_ROOT", "/home/may/ecommerce-agentic-rag-archive")
    )
    direct = root / "src" / "tau2"
    nested = root / "vendor" / "tau2-bench-fc0055dc" / "src" / "tau2"
    source = direct if direct.is_dir() else nested
    if not source.is_dir():
        pytest.skip("vendored tau2 snapshot is not available")
    sys.path.insert(0, str(source.parent))
    return source


def test_pinned_agent_gym_reward_contract(monkeypatch):
    _tau2_src()
    gym_agent = pytest.importorskip("tau2.gym.gym_agent")
    from tau2.evaluator.evaluator import EvaluationType

    calls = []

    def fake_evaluate_simulation(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(reward=0.0, model_dump_json=lambda **_: "{}")

    env = gym_agent.AgentGymEnv.__new__(gym_agent.AgentGymEnv)
    env._simulation_run = object()
    env.solo_mode = False
    env.domain = "retail"
    env.task_id = "contract-test"
    env._get_task = lambda: object()
    monkeypatch.setattr(gym_agent, "evaluate_simulation", fake_evaluate_simulation)

    reward, _ = env._get_reward()

    assert reward == 0.0
    assert calls and calls[0]["evaluation_type"] is EvaluationType.ALL
    assert calls[0]["domain"] == "retail"


@pytest.mark.parametrize("termination_reason", ["MAX_STEPS", "AGENT_ERROR"])
def test_complete_policy_termination_is_frozen_zero_and_valid_grpo_negative(
    termination_reason,
):
    _tau2_src()
    from tau2.data_model.simulation import SimulationRun, TerminationReason
    from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation

    simulation = SimulationRun.model_construct(
        id="valid-terminal",
        task_id="17",
        start_time="2026-01-01T00:00:00Z",
        end_time="2026-01-01T00:00:01Z",
        duration=1.0,
        termination_reason=getattr(TerminationReason, termination_reason),
    )
    reward_info = evaluate_simulation(
        simulation=simulation,
        task=object(),
        evaluation_type=EvaluationType.ALL,
        solo_mode=False,
        domain="retail",
    )

    assert reward_info.reward == 0.0
    accepted = OfficialTerminalRewardAdapter().from_step(
        terminated=True,
        reward=reward_info.reward,
        info={
            "tau2_simulation_run_present": True,
            "tau2_simulation_run_complete": True,
            "tau2_official_evaluator_succeeded": True,
        },
    )
    assert accepted.value == 0.0


def test_project_adapter_uses_the_frozen_track_a_judge(monkeypatch):
    _tau2_src()
    nl_module = pytest.importorskip("tau2.evaluator.evaluator_nl_assertions")
    from ecommerce_rag.grpo.tau3_env_adapter import _configure_frozen_nl_judge

    monkeypatch.setattr(nl_module, "DEFAULT_LLM_NL_ASSERTIONS", "wrong/default")
    _configure_frozen_nl_judge()
    assert nl_module.DEFAULT_LLM_NL_ASSERTIONS == "deepseek/deepseek-chat"


def test_pinned_retail_split_has_nl_assertion_coverage():
    source = _tau2_src()
    tasks_path = (
        source.parent.parent
        / "data"
        / "tau2"
        / "domains"
        / "retail"
        / "tasks.json"
    )
    if not tasks_path.exists():
        pytest.skip("vendored tau2 retail task data is not available")
    tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
    with_nl_basis = 0
    with_nonempty_assertions = 0
    task_items = tasks.values() if isinstance(tasks, dict) else tasks
    for task in task_items:
        criteria = task.get("evaluation_criteria") or {}
        if "NL_ASSERTION" in (criteria.get("reward_basis") or []):
            with_nl_basis += 1
            if criteria.get("nl_assertions"):
                with_nonempty_assertions += 1
    assert len(tasks) == 114
    assert with_nl_basis == 112
    assert with_nonempty_assertions == 40
