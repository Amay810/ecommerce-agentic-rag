from __future__ import annotations

from ecommerce_rag.process_audit import (
    TaskAuditContract,
    audit_simulation,
    is_explicit_confirmation,
)


def _call(name, arguments=None):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": name, "function": {"name": name, "arguments": arguments or {}}}
        ],
    }


def test_confirmation_gate_is_conservative():
    assert is_explicit_confirmation("Yes, please do it.")
    assert is_explicit_confirmation("确认，请执行")
    assert not is_explicit_confirmation("I am thinking about it")


def test_compliant_write_path_passes():
    simulation = {
        "id": "S1",
        "messages": [
            {"role": "user", "content": "cancel my order"},
            _call("find_user_id_by_email", {"email": "a@example.com"}),
            {"role": "tool", "content": '"U1"', "error": False},
            _call("get_order_details", {"order_id": "O1"}),
            {"role": "tool", "content": '{"status":"pending"}', "error": False},
            {"role": "assistant", "content": "Confirm cancellation of O1?"},
            {"role": "user", "content": "Yes, cancel it."},
            _call("cancel_pending_order", {"order_id": "O1"}),
            {"role": "tool", "content": '{"status":"cancelled"}', "error": False},
        ],
    }
    result = audit_simulation(simulation)
    assert result.process_compliant
    assert not result.violations


def test_write_without_read_or_confirmation_is_rejected():
    simulation = {
        "id": "S2",
        "messages": [
            {"role": "user", "content": "maybe cancel it"},
            _call("find_user_id_by_email", {"email": "a@example.com"}),
            {"role": "tool", "content": '"U1"', "error": False},
            _call("cancel_pending_order", {"order_id": "O1"}),
        ],
    }
    result = audit_simulation(simulation)
    assert not result.process_compliant
    assert "write_without_successful_target_read" in result.violations
    assert "write_without_explicit_confirmation" in result.violations


def test_failed_auth_does_not_authorize_private_read():
    simulation = {
        "id": "S3",
        "messages": [
            _call("find_user_id_by_email", {"email": "a@example.com"}),
            {"role": "tool", "content": "not found", "error": True},
            _call("get_order_details", {"order_id": "O1"}),
            {"role": "tool", "content": '{"status":"pending"}', "error": False},
        ],
    }
    result = audit_simulation(simulation)
    assert "private_read_before_successful_authentication" in result.violations


def test_noop_cannot_pass_a_no_write_task_contract():
    contract = TaskAuditContract(
        task_id="T1",
        structure_id="S15_cancel_unconfirmed_no_write",
        expected_termination="refuse_no_write",
        confirmation_requirement="refused_or_absent",
        expected_actions=(("get_order_details", {"order_id": "O1"}),),
    )
    simulation = {
        "id": "S4",
        "messages": [
            {"role": "user", "content": "cancel it"},
            {"role": "assistant", "content": "I can help."},
        ],
    }
    result = audit_simulation(simulation, contract)
    assert "missing_successful_expected_action:get_order_details" in result.violations
    assert "required_refusal_or_no_change_not_stated" in result.violations
