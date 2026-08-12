"""Launch tau2 after selecting the NL-assertion judge from a process variable."""

from __future__ import annotations

import os
from pathlib import Path

from ecommerce_rag.tau3_agent_adapter import register_tau3_agent
from tau2.cli import main
from tau2.evaluator import evaluator_nl_assertions


judge_model = os.environ.get("TAU3_NL_ASSERTIONS_MODEL")
if not judge_model:
    raise RuntimeError("TAU3_NL_ASSERTIONS_MODEL must be set")
evaluator_nl_assertions.DEFAULT_LLM_NL_ASSERTIONS = judge_model
register_tau3_agent()

compiled_tasks = os.environ.get("ERAG_COMPILED_RETAIL_TASKS")
compiled_splits = os.environ.get("ERAG_COMPILED_RETAIL_SPLITS")
if compiled_tasks and compiled_splits:
    from ecommerce_rag.compiled_retail_registry import register_compiled_retail

    register_compiled_retail(
        tasks_path=Path(compiled_tasks),
        split_path=Path(compiled_splits),
    )

main()
