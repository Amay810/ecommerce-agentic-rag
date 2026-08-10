"""Leakage-free task progress derived from policy-visible legacy events."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any


_ORDER_ID = re.compile(r"O[0-9]{6}", re.I)
_CODE = re.compile(r"(?<![A-Za-z0-9])[0-9]{6}(?![0-9])")
_REASON = re.compile(r"(?:退货原因|原因)\s*[:：]\s*(.+)", re.I)
_RETURN_INTENT = ("退货", "退换", "return")
_AFFIRMATIVE = {"确认", "确认提交", "确认提交退货", "同意", "yes", "confirm"}
_NEGATIVE = {"取消", "不确认", "不同意", "no"}
_REASON_REFUSALS = {
    "不提供", "拒绝提供", "不想提供", "不愿提供", "不给",
    "不提供退货原因", "拒绝说明原因", "拒绝提供退货原因",
}
_VERIFICATION_REFUSALS = {
    "不提供", "拒绝提供", "不想提供", "不愿提供", "不给",
    "拒绝提供验证码", "不愿提供验证码", "不想提供验证码", "拒绝验证",
}


def _return_reason_refused(text: str) -> bool:
    normalized = re.sub(r"[\s，。！？,.!?：:]", "", text).lower()
    if normalized in _REASON_REFUSALS:
        return True
    return bool(re.fullmatch(
        r"(?:我)?(?:不愿|不想|拒绝|不|无法)(?:提供|说明|告知|透露|给)?(?:退货)?原因",
        normalized,
    ))


def _verification_code_refused(text: str) -> bool:
    """Typed verification refusals only; free-language expansion is semantic shadow."""
    normalized = re.sub(r"[\s，。！？,.!?：:]", "", text).lower()
    if normalized in _VERIFICATION_REFUSALS:
        return True
    return bool(re.fullmatch(
        r"(?:我)?(?:不愿|不想|拒绝|不|无法)(?:提供|给)?(?:六位)?验证码",
        normalized,
    ))


@dataclass(frozen=True)
class TaskProgress:
    workflow: str
    completed: tuple[str, ...]
    pending: tuple[str, ...]
    blocked_by: str | None
    allowed_next_actions: tuple[str, ...]
    requested_input_type: str | None
    guard_state: str
    eligible: bool | None = None
    cancelled: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LegacyTaskProgressReducer:
    """Reduce conversation/tool events without TaskSpec or grader access."""

    @staticmethod
    def facts(history: list[dict[str, Any]]) -> dict[str, Any]:
        facts: dict[str, Any] = {
            "return_intent": False,
            "order_id": None,
            "verification_available": False,
            "identity_verified": False,
            "order_loaded": False,
            "eligibility": None,
            "return_reason": None,
            "confirmation": None,
            "return_created": False,
            "blocked_by": None,
            "handed_off": False,
        }
        requested_input_type: str | None = None
        for entry in history:
            role = entry.get("role")
            if role == "assistant":
                requested_input_type = entry.get("requested_input_type")
                continue
            if role == "user":
                content = str(entry.get("content", ""))
                lowered = content.lower()
                facts["return_intent"] = facts["return_intent"] or any(
                    token in lowered for token in _RETURN_INTENT)
                match = _ORDER_ID.search(content)
                if match:
                    new_order = match.group(0).upper()
                    if facts["order_id"] and facts["order_id"] != new_order:
                        facts.update(
                            verification_available=False,
                            identity_verified=False,
                            order_loaded=False,
                            eligibility=None,
                            confirmation=None,
                            return_created=False,
                        )
                    facts["order_id"] = new_order
                if _CODE.search(content):
                    facts["verification_available"] = True
                    if facts["blocked_by"] == "verification_code_refused":
                        facts["blocked_by"] = None
                reason = _REASON.search(content)
                if reason and reason.group(1).strip():
                    candidate = reason.group(1).strip()
                    if _return_reason_refused(candidate):
                        facts["return_reason"] = None
                        facts["blocked_by"] = "return_reason_refused"
                    else:
                        facts["return_reason"] = candidate
                        if facts["blocked_by"] == "return_reason_refused":
                            facts["blocked_by"] = None
                    facts["confirmation"] = None
                normalized = content.strip().lower()
                if requested_input_type == "verification_code":
                    if _verification_code_refused(content) and not _CODE.search(content):
                        facts["verification_available"] = False
                        facts["blocked_by"] = "verification_code_refused"
                elif requested_input_type == "confirmation":
                    if normalized in _AFFIRMATIVE:
                        facts["confirmation"] = True
                    elif normalized in _NEGATIVE:
                        facts["confirmation"] = False
                elif requested_input_type == "return_reason" and reason is None:
                    if _return_reason_refused(content):
                        facts["return_reason"] = None
                        facts["confirmation"] = None
                        facts["blocked_by"] = "return_reason_refused"
                    elif normalized not in _NEGATIVE:
                        facts["return_reason"] = content
                        if facts["blocked_by"] == "return_reason_refused":
                            facts["blocked_by"] = None
                requested_input_type = None
                continue
            if role != "tool":
                continue
            name = entry.get("name")
            result = entry.get("result") or {}
            if name in {"get_order", "check_return_eligibility"}:
                if result.get("ok"):
                    facts["identity_verified"] = True
                    facts["order_loaded"] = bool(result.get("order"))
                    if name == "check_return_eligibility":
                        facts["eligibility"] = bool(result.get("eligible"))
                elif result.get("error") in {"identity_verification_failed", "order_ownership_mismatch"}:
                    facts["blocked_by"] = str(result["error"])
            elif name == "create_return_request" and result.get("ok"):
                facts["return_created"] = True
            elif name == "escalate_to_human" and result.get("ok"):
                facts["handed_off"] = True
        return facts

    def derive(self, history: list[dict[str, Any]]) -> TaskProgress:
        facts = self.facts(history)
        if not facts["return_intent"]:
            return TaskProgress("unclassified", (), (), None, (), None, "not_applicable")

        completed: list[str] = []
        if facts["order_id"]: completed.append("order_id_collected")
        if facts["verification_available"]: completed.append("verification_supplied")
        if facts["identity_verified"]: completed.append("identity_verified")
        if facts["order_loaded"]: completed.append("order_loaded")
        if facts["eligibility"] is not None: completed.append("eligibility_checked")
        if facts["return_reason"]: completed.append("return_reason_collected")
        if facts["confirmation"] is True: completed.append("explicit_confirmation")
        if facts["return_created"]: completed.append("return_request_created")

        if facts["handed_off"]:
            return TaskProgress("return_resolution", tuple(completed), (), None,
                                ("final_answer",), None, "handed_off")
        if facts["blocked_by"]:
            return TaskProgress("return_resolution", tuple(completed), (), facts["blocked_by"],
                                ("handoff",), None, "handoff_required")
        if not facts["order_id"]:
            return TaskProgress("return_resolution", tuple(completed), ("order_id",), "user_input",
                                ("ask_user:order_id",), "order_id", "read_only")
        if not facts["verification_available"]:
            return TaskProgress("return_resolution", tuple(completed), ("identity_verification",), "user_input",
                                ("ask_user:verification_code",), "verification_code", "identity_required")
        if facts["eligibility"] is None:
            allowed = (("check_return_eligibility",) if facts["order_loaded"]
                       else ("get_order", "check_return_eligibility"))
            return TaskProgress("return_resolution", tuple(completed), ("eligibility",), None,
                                allowed, None, "identity_available")
        if facts["eligibility"] is False:
            return TaskProgress("return_resolution", tuple(completed), (), None,
                                ("final_answer",), None, "ineligible", eligible=False)
        if not facts["return_reason"]:
            return TaskProgress("return_resolution", tuple(completed), ("return_reason",), "user_input",
                                ("ask_user:return_reason",), "return_reason", "reason_required", eligible=True)
        if facts["confirmation"] is False:
            return TaskProgress("return_resolution", tuple(completed), (), None,
                                ("final_answer",), None, "cancelled", eligible=True, cancelled=True)
        if facts["confirmation"] is not True:
            return TaskProgress("return_resolution", tuple(completed), ("explicit_confirmation",), "user_input",
                                ("ask_user:confirmation",), "confirmation", "confirmation_required", eligible=True)
        if not facts["return_created"]:
            return TaskProgress("return_resolution", tuple(completed), ("return_request",), None,
                                ("create_return_request",), None, "write_authorized", eligible=True)
        return TaskProgress("return_resolution", tuple(completed), (), None,
                            ("final_answer",), None, "complete", eligible=True)
