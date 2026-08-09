"""Launch tau2 with a private action-name plan and the frozen NL judge."""

from __future__ import annotations

import os

from tau2.agent.llm_agent import LLMGTAgent
from tau2.evaluator import evaluator_nl_assertions
from tau2.registry import registry


def create_hint_agent(tools, domain_policy, **kwargs):
    return LLMGTAgent(
        tools=tools,
        domain_policy=domain_policy,
        task=kwargs.get("task"),
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
        provide_function_args=False,
    )


judge_model = os.environ.get("TAU3_NL_ASSERTIONS_MODEL")
if not judge_model:
    raise RuntimeError("TAU3_NL_ASSERTIONS_MODEL must be set")
evaluator_nl_assertions.DEFAULT_LLM_NL_ASSERTIONS = judge_model

registry.register_agent_factory(
    create_hint_agent,
    "llm_agent_gt_no_args",
    task_filter=LLMGTAgent.check_valid_task,
)

from tau2.cli import main  # noqa: E402


main()
