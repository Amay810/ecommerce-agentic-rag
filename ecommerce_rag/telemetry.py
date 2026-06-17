# -*- coding: utf-8 -*-
"""JSONL telemetry for evaluation and operations feedback loops."""

import json
from datetime import datetime, timezone
from pathlib import Path

from . import config


def log_event(event: dict, path: Path | None = None) -> None:
    target = path or (config.LOG_DIR / "events.jsonl")
    target.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        **event,
    }
    with open(target, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
