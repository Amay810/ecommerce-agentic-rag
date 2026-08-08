# -*- coding: utf-8 -*-
"""Retail Task Compiler: blueprint-first verified task generation for τ³ Retail.

This package is a project-owned component. The pinned τ³ ``v1.0.1`` tree does
not ship a Retail task generator; only Telecom has ``create_tasks.py``.
"""

from .blueprint import (
    BEHAVIOR_PROFILES,
    ENVIRONMENTS,
    TaskBlueprint,
    validate_blueprint,
)
from .compiler import RetailTaskCompiler, compile_cancel_pending_v0
from .contamination import (
    ContaminationReport,
    check_contamination,
    structure_signature,
)
from .coverage import CoverageReport, coverage_from_blueprints
from .replay import ReplayReport, replay_reference_path_twice
from .tool_graph import ToolDependencyGraph, load_retail_tool_graph

__all__ = [
    "BEHAVIOR_PROFILES",
    "ENVIRONMENTS",
    "ContaminationReport",
    "CoverageReport",
    "ReplayReport",
    "RetailTaskCompiler",
    "TaskBlueprint",
    "ToolDependencyGraph",
    "check_contamination",
    "compile_cancel_pending_v0",
    "coverage_from_blueprints",
    "load_retail_tool_graph",
    "replay_reference_path_twice",
    "structure_signature",
    "validate_blueprint",
]
