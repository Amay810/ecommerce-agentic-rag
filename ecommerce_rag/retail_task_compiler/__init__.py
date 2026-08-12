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
from .grounding import compile_m1_dataset
from .replay import ReplayReport, replay_reference_path_twice
from .structures import (
    BEHAVIOR_FAMILIES,
    StructureSpec,
    assign_structure_splits,
    m1_structure_catalog,
)
from .tool_graph import ToolDependencyGraph, load_retail_tool_graph

__all__ = [
    "BEHAVIOR_FAMILIES",
    "BEHAVIOR_PROFILES",
    "ENVIRONMENTS",
    "ContaminationReport",
    "CoverageReport",
    "ReplayReport",
    "RetailTaskCompiler",
    "StructureSpec",
    "TaskBlueprint",
    "ToolDependencyGraph",
    "assign_structure_splits",
    "check_contamination",
    "compile_cancel_pending_v0",
    "compile_m1_dataset",
    "coverage_from_blueprints",
    "load_retail_tool_graph",
    "m1_structure_catalog",
    "replay_reference_path_twice",
    "structure_signature",
    "validate_blueprint",
]
