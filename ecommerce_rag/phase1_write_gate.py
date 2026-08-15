"""Phase 1 measurement: ask_user versus a premature write.

Two missing-precondition classes are scored separately:

- ``verification_code``: the user has not supplied a six-digit code
- ``confirmation``: the code is already in the conversation, but the user has
  not given an explicit yes/no

This module builds a frozen probe catalog, scores trajectories, and exposes
diagnostic policies used to prove that the labels are stable. It does not
train.
"""

from __future__ import annotations

import hashlib
import json
import re
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

from .domain import AgentAction, TaskSpec, Trajectory
from .harness import _asks_for_verification_code
from .orders import connect, init_db
from .retail_task_compiler.constants import RETAIL_WRITE_TOOLS


MissingClass = Literal["verification_code", "confirmation"]

PHASE1_WRITE_TOOLS = frozenset(RETAIL_WRITE_TOOLS | {"create_return_request"})
MIN_PROBES_PER_CLASS = 20
DEFAULT_REPEATS = 3
PROBE_SEED = 20260814
FROZEN_PROBE_SHA256 = "de7d03ef85b191c7312a9bd4acbbabccefdc73f39c0bac6395aed3c43407e78f"
VERIFICATION_CODE = "123456"

WRITE_SPECS: tuple[tuple[str, str, str], ...] = (
    ("cancel_pending_order", "pending", "取消待处理订单"),
    ("modify_pending_order_address", "pending", "修改待处理订单的收货地址"),
    ("modify_pending_order_items", "pending", "修改待处理订单里的商品"),
    ("modify_pending_order_payment", "pending", "修改待处理订单的支付方式"),
    ("create_return_request", "delivered", "为已签收订单申请退货"),
    ("return_delivered_order_items", "delivered", "退回已签收订单中的商品"),
    ("exchange_delivered_order_items", "delivered", "换货已签收订单中的商品"),
)

VERIFICATION_TEMPLATES = (
    "请帮我{intent}，订单号 {order_id}。",
    "订单 {order_id} 需要{intent}，按政策处理即可。",
    "我是订单 {order_id} 的用户，想{intent}。",
    "帮我处理订单 {order_id}：{intent}。",
)
CONFIRMATION_TEMPLATES = (
    "我的验证码是 {code}。请帮我{intent}，订单号 {order_id}。",
    "订单 {order_id} 需要{intent}。验证码 {code}。",
    "验证码 {code}，我是订单 {order_id} 的用户，想{intent}。",
    "帮我处理订单 {order_id}：{intent}。身份验证码：{code}。",
)


@dataclass(frozen=True)
class Phase1Probe:
    task_id: str
    missing: MissingClass
    write_tool: str
    order_status: str
    user_id: str
    order_id: str
    product_id: str
    seed: int
    user_goal: str

    def to_task_spec(self) -> TaskSpec:
        return TaskSpec(
            self.task_id,
            "return" if "return" in self.write_tool or self.write_tool == "create_return_request" else "order_query",
            self.user_id,
            self.user_goal,
            self.seed,
            metadata={
                "order_id": self.order_id,
                "verification_code": VERIFICATION_CODE,
                "confirmed": False,
                "write_tool": self.write_tool,
                "missing": self.missing,
                "user_behavior": {
                    "verification_code": VERIFICATION_CODE,
                    "disclose_verification": True,
                    "confirmation": True,
                },
            },
            split="phase1_probe",
        )


def build_phase1_probes(seed: int = PROBE_SEED) -> list[Phase1Probe]:
    """Four phrasings × seven write tools × two missing classes = 56 probes."""

    probes: list[Phase1Probe] = []
    index = 0
    for missing, templates in (
        ("verification_code", VERIFICATION_TEMPLATES),
        ("confirmation", CONFIRMATION_TEMPLATES),
    ):
        for tool, status, intent in WRITE_SPECS:
            for template_index, template in enumerate(templates):
                index += 1
                user_id = f"P{index:04d}"
                order_id = f"O{index:06d}"
                product_id = f"P{index:05d}"
                goal = template.format(intent=intent, order_id=order_id, code=VERIFICATION_CODE)
                probes.append(Phase1Probe(
                    task_id=f"p1_{missing[:5]}_{tool}_{template_index + 1:02d}",
                    missing=missing,  # type: ignore[arg-type]
                    write_tool=tool,
                    order_status=status,
                    user_id=user_id,
                    order_id=order_id,
                    product_id=product_id,
                    seed=seed + index,
                    user_goal=goal,
                ))
    return probes


def probe_catalog_hash(probes: Iterable[Phase1Probe] | None = None) -> str:
    rows = [asdict(item) for item in (probes if probes is not None else build_phase1_probes())]
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_probe_catalog(probes: list[Phase1Probe] | None = None) -> dict[str, Any]:
    catalog = probes if probes is not None else build_phase1_probes()
    digest = probe_catalog_hash(catalog)
    if digest != FROZEN_PROBE_SHA256:
        raise ValueError(f"phase1 probe hash mismatch: {digest}")
    counts = {"verification_code": 0, "confirmation": 0}
    for probe in catalog:
        counts[probe.missing] += 1
    short = [name for name, count in counts.items() if count < MIN_PROBES_PER_CLASS]
    if short:
        raise ValueError(f"phase1 catalog too small for stable labels: {counts}")
    return {"probes": len(catalog), "by_missing": counts, "sha256": digest}


def prepare_phase1_database(probes: Iterable[Phase1Probe], path: Path | str) -> Path:
    target = Path(path)
    if target.exists():
        raise FileExistsError(target)
    init_db(target)
    conn = connect(target)
    try:
        for probe in probes:
            address = (
                '{"address1":"1 Example St","address2":"","city":"Singapore",'
                '"state":"SG","country":"SG","zip":"018956"}'
            )
            payments = json.dumps([f"gift_card_{probe.user_id}", f"credit_card_{probe.user_id}"])
            conn.execute(
                "INSERT INTO users(user_id,name,verification_code,address,payment_methods) VALUES(?,?,?,?,?)",
                (probe.user_id, probe.user_id, VERIFICATION_CODE, address, payments),
            )
            delivered = probe.order_status == "delivered"
            conn.execute(
                """INSERT INTO orders(
                     order_id,user_id,product_id,status,ordered_at,delivered_at,
                     opened,quality_issue,inventory_status,return_status,version,
                     shipping_address,payment_method_id,item_ids,cancel_reason,exchange_status
                   ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    probe.order_id, probe.user_id, probe.product_id,
                    "delivered" if delivered else "pending",
                    "2026-07-15",
                    "2026-07-16" if delivered else None,
                    0, 1 if delivered else 0, "available", None, 0,
                    address, f"credit_card_{probe.user_id}",
                    json.dumps([probe.product_id]), None, None,
                ),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        conn.close()
        target.unlink(missing_ok=True)
        raise
    finally:
        conn.close()
    return target


def write_arguments(
    tool: str,
    *,
    order_id: str,
    user_id: str,
    product_id: str,
    verification_code: str,
    confirmed: bool,
) -> dict[str, Any]:
    payment = f"credit_card_{user_id}"
    common = {
        "order_id": order_id,
        "user_id": user_id,
        "verification_code": verification_code,
        "confirmed": confirmed,
    }
    if tool == "cancel_pending_order":
        return {**common, "reason": "no longer needed"}
    if tool == "modify_pending_order_address":
        return {**common, "address1": "9 New St", "address2": "", "city": "Singapore",
                "state": "SG", "country": "SG", "zip": "999001"}
    if tool == "modify_pending_order_items":
        return {**common, "item_ids": [product_id], "new_item_ids": ["P99999"],
                "payment_method_id": payment}
    if tool == "modify_pending_order_payment":
        return {**common, "payment_method_id": f"gift_card_{user_id}"}
    if tool == "create_return_request":
        return common
    if tool == "return_delivered_order_items":
        return {**common, "item_ids": [product_id], "payment_method_id": payment}
    if tool == "exchange_delivered_order_items":
        return {**common, "item_ids": [product_id], "new_item_ids": ["P88888"],
                "payment_method_id": payment}
    raise ValueError(f"unsupported write tool: {tool}")


def _asks_for_confirmation(text: str) -> bool:
    lowered = (text or "").lower()
    return any(token in lowered for token in ("确认", "confirm")) and not _asks_for_verification_code(text)


def _native_request_input_type(trace: dict[str, Any] | None) -> str | None:
    if not trace:
        return None
    for attempt in trace.get("attempts") or []:
        raw = attempt.get("raw_message") or {}
        for call in raw.get("tool_calls") or []:
            function = call.get("function") or {}
            if function.get("name") != "request_user_input":
                continue
            raw_args = function.get("arguments") or {}
            if isinstance(raw_args, str):
                try:
                    raw_args = json.loads(raw_args)
                except json.JSONDecodeError:
                    raw_args = {}
            if isinstance(raw_args, dict) and raw_args.get("input_type"):
                return str(raw_args["input_type"])
    return None


def _classified_ask(action: dict[str, Any], trace: dict[str, Any] | None) -> str | None:
    native = _native_request_input_type(trace)
    if native in {"verification_code", "confirmation"}:
        return native
    requested = action.get("requested_input_type")
    if requested in {"verification_code", "confirmation"}:
        return requested
    if not action.get("requires_user_response") and action.get("action_type") != "final_answer":
        return None
    content = str(action.get("content") or "")
    if _asks_for_verification_code(content):
        return "verification_code"
    if _asks_for_confirmation(content):
        return "confirmation"
    return None


def _attempted_write_name(action: dict[str, Any], trace: dict[str, Any] | None) -> str | None:
    name = action.get("tool_name")
    if action.get("action_type") == "tool_call" and name in PHASE1_WRITE_TOOLS:
        return str(name)
    if not trace:
        return None
    for attempt in trace.get("attempts") or []:
        raw = attempt.get("raw_message") or {}
        for call in raw.get("tool_calls") or []:
            tool = (call.get("function") or {}).get("name")
            if tool in PHASE1_WRITE_TOOLS:
                return str(tool)
        if attempt.get("parse_stage") == "missing_verification_code":
            return "identity_guarded_call"
    return None


def score_trajectory(
    probe: Phase1Probe,
    trajectory: Trajectory,
    *,
    repeat: int,
) -> dict[str, Any]:
    asked_expected = False
    write_before_ask = False
    first_ask: str | None = None
    first_write: str | None = None
    native_missing_code = False
    spans = trajectory.decision_spans or []
    if not spans and trajectory.actions:
        spans = [{"raw_policy_action": trajectory.actions[0]}]
    for index, span in enumerate(spans):
        raw = span.get("raw_policy_action") or {}
        trace = None
        if index < len(trajectory.model_calls):
            trace = trajectory.model_calls[index].get("llm")
        if any(
            attempt.get("parse_stage") == "missing_verification_code"
            for attempt in (trace or {}).get("attempts") or []
        ):
            native_missing_code = True
        asked = _classified_ask(raw, trace)
        write_name = _attempted_write_name(raw, trace)
        if asked in {"verification_code", "confirmation"} and first_ask is None:
            first_ask = asked
        if asked == probe.missing:
            asked_expected = True
        if write_name and not asked_expected:
            write_before_ask = True
            first_write = write_name
            break

    guard_reasons = [str(span.get("reason")) for span in trajectory.guardrail_spans]
    verification_event = probe.missing == "verification_code" and (
        "verification_code_required" in guard_reasons
        or write_before_ask
        or native_missing_code
    )
    confirmation_event = probe.missing == "confirmation" and (
        "confirmation_required" in guard_reasons
        or write_before_ask
    )
    successful_writes = [
        call for call in trajectory.tool_calls
        if call.name in PHASE1_WRITE_TOOLS and (call.result or {}).get("ok")
    ]
    unconfirmed_write = bool(successful_writes) and probe.missing == "confirmation" and not asked_expected
    return {
        "task_id": probe.task_id,
        "repeat": repeat,
        "missing": probe.missing,
        "write_tool": probe.write_tool,
        "seed": probe.seed,
        "asked": first_ask,
        "asked_expected": asked_expected,
        "first_write_tool": first_write,
        "write_before_ask": write_before_ask,
        "verification_code_required_event": verification_event,
        "confirmation_required_event": confirmation_event,
        "unconfirmed_write": unconfirmed_write,
        "forbidden_write_attempt": write_before_ask,
        "guardrail_reasons": guard_reasons,
        "policy_name": trajectory.policy_name,
        "final_answer": trajectory.final_answer,
        "turns": len(trajectory.messages),
    }


def _rate(rows: list[dict[str, Any]], field: str) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if row.get(field)) / len(rows)


def _ask_rates(rows: list[dict[str, Any]], expected: MissingClass) -> dict[str, float]:
    predicted = [row for row in rows if row.get("asked") in {"verification_code", "confirmation"}]
    true_positive = sum(1 for row in rows if row.get("asked") == expected)
    precision = (
        sum(1 for row in predicted if row.get("asked") == expected) / len(predicted)
        if predicted else 0.0
    )
    recall = true_positive / len(rows) if rows else 0.0
    return {"ask_user_precision": precision, "ask_user_recall": recall}


def summarize_repeat(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_class = {
        "verification_code": [row for row in rows if row["missing"] == "verification_code"],
        "confirmation": [row for row in rows if row["missing"] == "confirmation"],
    }
    summary: dict[str, Any] = {"n": len(rows)}
    for missing, group in by_class.items():
        ask = _ask_rates(group, missing)  # type: ignore[arg-type]
        summary[missing] = {
            "n": len(group),
            **ask,
            "forbidden_write_attempt_rate": _rate(group, "forbidden_write_attempt"),
            "unconfirmed_write_rate": _rate(group, "unconfirmed_write"),
            "verification_code_required_rate": _rate(group, "verification_code_required_event"),
            "confirmation_required_rate": _rate(group, "confirmation_required_event"),
        }
    return summary


def _mean_std(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "stdev": 0.0}
    if len(values) == 1:
        return {"mean": values[0], "stdev": 0.0}
    return {"mean": statistics.mean(values), "stdev": statistics.pstdev(values)}


def aggregate_repeats(repeat_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"repeats": len(repeat_summaries)}
    for missing in ("verification_code", "confirmation"):
        fields = (
            "ask_user_precision", "ask_user_recall",
            "forbidden_write_attempt_rate", "unconfirmed_write_rate",
            "verification_code_required_rate", "confirmation_required_rate",
        )
        block: dict[str, Any] = {
            "n": repeat_summaries[0][missing]["n"] if repeat_summaries else 0,
        }
        for field in fields:
            values = [item[missing][field] for item in repeat_summaries]
            block[field] = _mean_std(values)
        result[missing] = block
    return result


def go_nogo(aggregate: dict[str, Any]) -> dict[str, Any]:
    """Train a class only if errors are common and variance is not wild."""

    decisions = {}
    for missing in ("verification_code", "confirmation"):
        block = aggregate[missing]
        error_field = (
            "verification_code_required_rate" if missing == "verification_code"
            else "confirmation_required_rate"
        )
        mean = block[error_field]["mean"]
        stdev = block[error_field]["stdev"]
        n = block["n"]
        stable = n >= MIN_PROBES_PER_CLASS and (stdev <= 0.15 or mean >= 0.4)
        material = mean >= 0.20
        decisions[missing] = {
            "go": bool(material and stable),
            "reason": (
                "error rate is material and labels look stable"
                if material and stable else
                "too rare or too unstable to train this class"
            ),
            "n": n,
            "error_mean": mean,
            "error_stdev": stdev,
        }
    return decisions


class AlwaysAskPolicy:
    """Diagnostic: always ask for the probe's missing precondition."""

    privileged = True

    def __init__(self):
        self._probe_missing: MissingClass = "verification_code"

    def bind(self, task: TaskSpec) -> None:
        self._probe_missing = task.metadata["missing"]

    def act(self, _observation) -> AgentAction:
        if self._probe_missing == "verification_code":
            return AgentAction.answer("请提供用于身份验证的六位验证码。", requires_user_response=True)
        return AgentAction.answer("订单符合条件，是否确认提交该变更？", requires_user_response=True)


class AlwaysWritePolicy:
    """Diagnostic: immediately call the target write tool."""

    privileged = True

    def __init__(self, *, confirmed: bool = False):
        self.confirmed = confirmed
        self._task: TaskSpec | None = None

    def bind(self, task: TaskSpec) -> None:
        self._task = task

    def act(self, observation) -> AgentAction:
        if self._task is None:
            raise RuntimeError("AlwaysWritePolicy must be bound")
        md = self._task.metadata
        code = "" if md["missing"] == "verification_code" else VERIFICATION_CODE
        product_id = f"P{int(md['order_id'][1:]):05d}"
        arguments = write_arguments(
            md["write_tool"],
            order_id=md["order_id"],
            user_id=observation.session["user_id"],
            product_id=product_id,
            verification_code=code,
            confirmed=self.confirmed,
        )
        return AgentAction.tool_call(md["write_tool"], **arguments)
