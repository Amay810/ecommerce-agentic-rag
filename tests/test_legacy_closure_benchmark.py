from __future__ import annotations

import sqlite3

import pytest

from ecommerce_rag.domain import AgentAction
from ecommerce_rag.harness import UserSimulatorProtocolError
from ecommerce_rag.legacy_closure_benchmark import (
    FROZEN_TASK_SHA256,
    TypedScenarioUser,
    _redact,
    build_m1_tasks,
    clone_database,
    prepare_database,
    progress_gate,
)


def test_frozen_m1_manifest_shape_and_hash():
    tasks = build_m1_tasks()
    assert FROZEN_TASK_SHA256 == "e4346e3f99261d203f9fea57aeec48d58e5f769d9a1e856e43b9cf0b74a6c8e3"
    assert len(tasks) == len({task.task_id for task in tasks}) == 60
    assert sum(task.split == "dev" for task in tasks) == 40
    assert sum(task.split == "locked" for task in tasks) == 20


def test_database_is_pristine_and_clones_are_independent(tmp_path):
    tasks = build_m1_tasks()
    pristine = prepare_database(tasks, tmp_path / "pristine.sqlite")
    first = clone_database(pristine, tmp_path / "first.sqlite")
    second = clone_database(pristine, tmp_path / "second.sqlite")
    with sqlite3.connect(first) as conn:
        conn.execute("UPDATE orders SET return_status='requested' WHERE order_id=?", (tasks[0].order_id,))
        conn.commit()
    with sqlite3.connect(second) as conn:
        assert conn.execute("SELECT return_status FROM orders WHERE order_id=?",
                            (tasks[0].order_id,)).fetchone()[0] is None
    with pytest.raises(FileExistsError):
        prepare_database(tasks, pristine)
    with pytest.raises(FileExistsError):
        clone_database(pristine, first)


def test_typed_user_does_not_consume_confirmation_as_verification():
    user = TypedScenarioUser(["确认"])
    action = AgentAction.answer("请输入验证码", requires_user_response=True)
    with pytest.raises(UserSimulatorProtocolError):
        user.respond(action, "verification_code")
    assert user.responses == ["确认"]
    assert user.respond(action, "confirmation") == "确认"


def test_progress_gate_enforces_layer_responsibility():
    baseline = {"success_rate": .5, "illegal_state_change_count": 1,
                "inappropriate_handoff_count": 0, "p95_latency_ms": 100,
                "protocol_error_count": 0}
    progress = {**baseline, "p95_latency_ms": 110}
    records = [{"progress_spans": [{"step": 0}]} for _ in range(40)]
    assert progress_gate(baseline, progress, records)["passed"]
    progress["success_rate"] = .475
    failed = progress_gate(baseline, progress, records)
    assert not failed["passed"]
    assert failed["decision"] == "revert_progress_exposure"


def test_record_redaction_removes_credentials_without_destroying_order_ids():
    payload = _redact({
        "verification_code": "123456",
        "prompt": "user supplied 123456 for O000001",
        "arguments": {"order_id": "O000001"},
    })
    assert "123456" not in str(payload)
    assert payload["prompt"] == "user supplied [REDACTED] for O000001"
    assert payload["arguments"]["order_id"] == "O000001"
