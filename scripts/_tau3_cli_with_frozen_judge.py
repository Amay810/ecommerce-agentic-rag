"""Launch tau2 after selecting the NL-assertion judge from a process variable."""

from __future__ import annotations

import os

from ecommerce_rag.tau3_agent_adapter import register_tau3_agent
from tau2.cli import main
from tau2.evaluator import evaluator_nl_assertions


judge_model = os.environ.get("TAU3_NL_ASSERTIONS_MODEL")
if not judge_model:
    raise RuntimeError("TAU3_NL_ASSERTIONS_MODEL must be set")
evaluator_nl_assertions.DEFAULT_LLM_NL_ASSERTIONS = judge_model
register_tau3_agent()
main()
