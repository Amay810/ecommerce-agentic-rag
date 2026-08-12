"""Dynamic action contracts derived from TaskProgress.

This execution layer does **not** ask the model to retry.
Illegal actions are remapped once to the preferred legal action (when materializable)
or fail closed.  That keeps LLM call counts unchanged aside from normal format retries.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .domain import AgentAction
from .legacy_closure import TaskProgress


ASK_USER_TEMPLATES: dict[str, str] = {
    "verification_code": "请提供用于身份验证的六位验证码。",
    "order_id": "请提供订单号。",
    "return_reason": "请提供退货原因。",
    "confirmation": "订单符合条件，是否确认提交退货？",
}

KNOWN_ACTION_KEYS = frozenset({
    "handoff", "final_answer",
    "get_order", "check_return_eligibility", "create_return_request",
    "cancel_pending_order", "modify_pending_order_address",
    "modify_pending_order_items", "modify_pending_order_payment",
    "modify_user_address", "return_delivered_order_items",
    "exchange_delivered_order_items",
    "ask_user:verification_code", "ask_user:order_id",
    "ask_user:return_reason", "ask_user:confirmation",
})


@dataclass(frozen=True)
class ActionContract:
    state: str
    allowed_actions: tuple[str, ...]
    forbidden_actions: tuple[str, ...]
    terminal_allowed: bool
    preferred_action: str | None
    pending: tuple[str, ...] = ()
    blocked_by: str | None = None
    requested_input_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ConstraintResult:
    action: AgentAction
    accepted: bool
    remapped: bool
    fail_closed: bool
    reason: str | None = None
    original_action_key: str | None = None
    preferred_action: str | None = None
    contract: dict[str, Any] = field(default_factory=dict)
    llm_calls_added: int = 0

    def to_span(self, *, step: int) -> dict[str, Any]:
        return {
            "step": step,
            "accepted": self.accepted,
            "remapped": self.remapped,
            "fail_closed": self.fail_closed,
            "reason": self.reason,
            "original_action_key": self.original_action_key,
            "preferred_action": self.preferred_action,
            "llm_calls_added": self.llm_calls_added,
            "contract": copy_contract(self.contract),
            "resulting_action": {
                "action_type": self.action.action_type,
                "tool_name": self.action.tool_name,
                "requires_user_response": self.action.requires_user_response,
                "content": self.action.content,
                "arguments": dict(self.action.arguments),
            },
        }


def copy_contract(contract: dict[str, Any]) -> dict[str, Any]:
    return {key: (list(value) if isinstance(value, (list, tuple)) else value)
            for key, value in contract.items()}


def contract_from_progress(progress: TaskProgress) -> ActionContract | None:
    """Return a contract for return_resolution only; other workflows stay unconstrained."""
    if progress.workflow != "return_resolution":
        return None
    allowed = tuple(progress.allowed_next_actions)
    terminal_allowed = "final_answer" in allowed
    preferred = allowed[0] if len(allowed) == 1 else None
    forbidden = tuple(sorted(
        key for key in KNOWN_ACTION_KEYS
        if key not in allowed and key != "ask_user:unknown"
    ))
    if not terminal_allowed and "final_answer" not in forbidden:
        forbidden = tuple(sorted(set(forbidden) | {"final_answer"}))
    return ActionContract(
        state=progress.guard_state,
        allowed_actions=allowed,
        forbidden_actions=forbidden,
        terminal_allowed=terminal_allowed,
        preferred_action=preferred,
        pending=tuple(progress.pending),
        blocked_by=progress.blocked_by,
        requested_input_type=progress.requested_input_type,
    )


def action_key(action: AgentAction, requested_input_type: str | None) -> str:
    if action.action_type == "tool_call":
        return action.tool_name or "unknown_tool"
    if action.action_type == "handoff":
        return "handoff"
    if action.requires_user_response:
        return f"ask_user:{requested_input_type or 'unknown'}"
    return "final_answer"


def materialize_preferred(preferred: str, *, blocked_by: str | None = None) -> AgentAction | None:
    if preferred.startswith("ask_user:"):
        kind = preferred.split(":", 1)[1]
        content = ASK_USER_TEMPLATES.get(kind)
        if not content:
            return None
        return AgentAction.answer(content, requires_user_response=True)
    if preferred == "final_answer":
        return AgentAction.answer("当前步骤已处理完毕。")
    if preferred == "handoff":
        return AgentAction.handoff(blocked_by or "progress_required_handoff")
    # Tool preferred actions need live arguments; v1 does not invent them.
    return None


def apply_action_constraint(
    action: AgentAction,
    progress: TaskProgress | None,
    *,
    requested_input_type: str | None,
) -> ConstraintResult:
    if progress is None:
        return ConstraintResult(action, True, False, False, contract={})
    contract = contract_from_progress(progress)
    if contract is None:
        return ConstraintResult(action, True, False, False, contract={})
    key = action_key(action, requested_input_type)
    payload = contract.to_dict()
    if key in contract.allowed_actions:
        return ConstraintResult(
            action, True, False, False,
            original_action_key=key,
            preferred_action=contract.preferred_action,
            contract=payload,
        )
    preferred = contract.preferred_action
    remapped = materialize_preferred(preferred, blocked_by=contract.blocked_by) if preferred else None
    if remapped is not None:
        return ConstraintResult(
            remapped, True, True, False,
            reason="action_not_allowed_remapped_to_preferred",
            original_action_key=key,
            preferred_action=preferred,
            contract=payload,
            llm_calls_added=0,
        )
    return ConstraintResult(
        AgentAction.answer("无法在当前任务进度允许的动作集合内继续，已停止。"),
        False, False, True,
        reason="action_not_allowed_fail_closed",
        original_action_key=key,
        preferred_action=preferred,
        contract=payload,
        llm_calls_added=0,
    )
