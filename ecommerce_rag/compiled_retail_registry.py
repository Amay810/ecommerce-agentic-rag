# -*- coding: utf-8 -*-
"""Register compiled_retail tasks onto the pinned τ³ retail environment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


def load_compiled_tasks(tasks_path: Path) -> list[dict[str, Any]]:
    return json.loads(tasks_path.read_text(encoding="utf-8"))


def load_compiled_splits(split_path: Path) -> dict[str, list[str]]:
    return json.loads(split_path.read_text(encoding="utf-8"))


def register_compiled_retail(
    *,
    tasks_path: Path,
    split_path: Path,
    task_set_name: str = "compiled_retail",
) -> None:
    """Register a project-owned task set that reuses the official retail env."""
    from tau2.data_model.tasks import Task
    from tau2.domains.retail.environment import get_environment as retail_get_environment
    from tau2.registry import registry

    tasks = [Task.model_validate(item) for item in load_compiled_tasks(tasks_path)]
    splits = load_compiled_splits(split_path)
    by_id = {str(task.id): task for task in tasks}

    def get_tasks(task_split_name: Optional[str] = "base") -> list[Task]:
        if task_split_name is None:
            return list(tasks)
        if task_split_name not in splits:
            raise ValueError(
                f"Invalid compiled_retail split {task_split_name}; "
                f"valid={sorted(splits)}"
            )
        return [by_id[task_id] for task_id in splits[task_split_name] if task_id in by_id]

    def get_task_splits() -> dict[str, list[str]]:
        return {key: list(value) for key, value in splits.items()}

    # Domain may already be registered as retail; only add the task set name.
    try:
        registry.get_env_constructor("retail")
    except Exception:
        registry.register_domain(retail_get_environment, "retail")
    registry.register_tasks(
        get_tasks,
        task_set_name,
        get_task_splits=get_task_splits,
    )
