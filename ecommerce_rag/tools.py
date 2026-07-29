"""Typed retail tools with verification and state-transition guardrails."""

from __future__ import annotations

import hashlib
import re
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .domain import ToolCall
from . import orders


READ_TOOLS = {"search_catalog", "get_product", "compare_products", "get_policy", "get_order", "check_return_eligibility"}
WRITE_TOOLS = {"create_return_request", "escalate_to_human"}

#: Tools that touch one customer's order and therefore must never run without a
#: verification code the user actually supplied. A policy that asks for the code
#: and calls the tool in the same turn would otherwise reach the identity check
#: with an empty string and burn the attempt.
IDENTITY_GUARDED_TOOLS = {"get_order", "check_return_eligibility", "create_return_request"}

#: ``\d`` matches Unicode decimal digits, so it accepts full-width "１２３４５６"
#: and Arabic-Indic forms. An identity guard must be literal ASCII.
_VERIFICATION_CODE = re.compile(r"[0-9]{6}")

POLICY_CATEGORIES = {
    "return": "退换货",
    "warranty": "售后保修",
    "shipping": "物流",
    "invoice": "发票",
    "refund": "退款",
}
POLICY_ALIASES = {
    **POLICY_CATEGORIES,
    "退换货": "退换货",
    "保修": "售后保修",
    "售后保修": "售后保修",
    "物流": "物流",
    "发票": "发票",
    "退款": "退款",
}


class RetailTools:
    def __init__(self, db_path: Path | str, retriever: Any | None = None, today: date = date(2026, 7, 20)):
        self.db_path = Path(db_path)
        self.retriever = retriever
        self.today = today
        self.calls: list[ToolCall] = []
        self.guardrails: list[dict[str, Any]] = []
        self._registry: dict[str, Callable[..., dict]] = {
            "search_catalog": self.search_catalog,
            "get_product": self.get_product,
            "compare_products": self.compare_products,
            "get_policy": self.get_policy,
            "get_order": self.get_order,
            "check_return_eligibility": self.check_return_eligibility,
            "create_return_request": self.create_return_request,
            "escalate_to_human": self.escalate_to_human,
        }

    def _identity_guard(self, name: str, arguments: dict[str, Any]) -> dict | None:
        """Refuse an order-scoped tool that arrives without a usable code.

        Enforced here rather than inside each tool so a ninth order-scoped tool
        cannot be added without the protection.
        """
        if name not in IDENTITY_GUARDED_TOOLS:
            return None
        code = arguments.get("verification_code")
        # No str(), no strip(): coercing here would let 123456, " 123456 " and
        # full-width digits through a guard that promises literal six ASCII digits,
        # and no guardrail span would be recorded for them.
        if isinstance(code, str) and _VERIFICATION_CODE.fullmatch(code):
            return None
        self.guardrails.append({"tool": name, "blocked": True, "reason": "verification_code_required",
                                "supplied": code})
        return {"ok": False, "changed": False, "error": "verification_code_required"}

    def call(self, name: str, **arguments: Any) -> dict:
        started = time.perf_counter()
        stamp = datetime.now(timezone.utc).isoformat()
        call_id = hashlib.sha1(f"{name}:{len(self.calls)}:{arguments}".encode()).hexdigest()[:12]
        error = None
        blocked = self._identity_guard(name, arguments)
        if blocked is not None:
            self.calls.append(ToolCall(name, arguments, call_id, blocked, stamp,
                                       (time.perf_counter() - started) * 1000, blocked["error"]))
            return blocked
        try:
            if name not in self._registry:
                raise ValueError(f"unknown tool: {name}")
            result = self._registry[name](**arguments)
        except Exception as exc:
            error, result = str(exc), {"ok": False, "error": str(exc)}
        self.calls.append(ToolCall(name, arguments, call_id, result, stamp, (time.perf_counter() - started) * 1000, error))
        return result

    def search_catalog(self, query: str, top_k: int = 5, category: str | None = None, max_price: float | None = None) -> dict:
        if self.retriever is None:
            return {"ok": False, "error": "retriever_not_configured", "items": []}
        chunks = self.retriever.search(query, top_k=max(top_k * 3, top_k), source_type="product", category=category)
        seen, items = set(), []
        for chunk in chunks:
            pid = chunk.get("product_id")
            if not pid or pid in seen or (max_price is not None and (chunk.get("price") or float("inf")) > max_price):
                continue
            seen.add(pid)
            items.append({k: chunk.get(k) for k in ("product_id", "title", "category", "price", "inventory", "doc_id", "score")})
            if len(items) == top_k:
                break
        return {"ok": True, "items": items}

    def _product_chunks(self, product_id: str) -> list[dict]:
        if self.retriever is None:
            return []
        return [c for c in self.retriever.chunks if c.get("product_id") == product_id]

    def get_product(self, product_id: str) -> dict:
        chunks = self._product_chunks(product_id)
        if not chunks:
            return {"ok": False, "error": "product_not_found"}
        first = chunks[0]
        return {"ok": True, "product": {k: first.get(k) for k in ("product_id", "title", "category", "price", "inventory", "doc_id")}, "evidence": [c.get("text", "") for c in chunks[:5]]}

    def compare_products(self, product_ids: list[str]) -> dict:
        products = [self.get_product(pid) for pid in product_ids]
        return {"ok": all(p["ok"] for p in products), "products": products}

    def get_policy(self, policy_type: str) -> dict:
        if self.retriever is None:
            return {"ok": False, "error": "retriever_not_configured"}
        category = POLICY_ALIASES.get(policy_type)
        if category is None:
            return {"ok": False, "error": "unknown_policy_type"}
        chunks = self.retriever.search(category, top_k=3, source_type="policy", category=category)
        return {"ok": bool(chunks), "policies": [{"doc_id": c["doc_id"], "title": c["title"], "text": c["text"]} for c in chunks]}

    def _verified_order(self, order_id: str, user_id: str, verification_code: str) -> tuple[dict | None, str | None]:
        conn = orders.connect(self.db_path)
        try:
            user = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
            order = conn.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,)).fetchone()
            if not user or user["verification_code"] != verification_code:
                return None, "identity_verification_failed"
            if not order or order["user_id"] != user_id:
                return None, "order_ownership_mismatch"
            return dict(order), None
        finally:
            conn.close()

    def get_order(self, order_id: str, user_id: str, verification_code: str) -> dict:
        order, error = self._verified_order(order_id, user_id, verification_code)
        return {"ok": error is None, "order": order, "error": error}

    def check_return_eligibility(self, order_id: str, user_id: str, verification_code: str) -> dict:
        order, error = self._verified_order(order_id, user_id, verification_code)
        if error:
            return {"ok": False, "eligible": False, "error": error}
        if order["status"] != "delivered":
            return {"ok": True, "eligible": False, "reason": "order_not_delivered", "order": order}
        days = (self.today - date.fromisoformat(order["delivered_at"])).days
        eligible = bool(order["quality_issue"] or (days <= 7 and not order["opened"]))
        reason = "quality_issue" if order["quality_issue"] else "seven_day_return" if eligible else "return_window_expired_or_opened"
        return {"ok": True, "eligible": eligible, "reason": reason, "days_since_delivery": days, "order": order}

    def create_return_request(self, order_id: str, user_id: str, verification_code: str, confirmed: bool) -> dict:
        eligibility = self.check_return_eligibility(order_id, user_id, verification_code)
        if not eligibility.get("ok") or not eligibility.get("eligible") or not confirmed:
            reason = eligibility.get("error") or eligibility.get("reason") or "confirmation_required"
            if eligibility.get("eligible") and not confirmed:
                reason = "confirmation_required"
            self.guardrails.append({"tool": "create_return_request", "blocked": True, "reason": reason, "order_id": order_id})
            return {"ok": False, "changed": False, "error": reason}
        conn = orders.connect(self.db_path)
        try:
            cur = conn.execute(
                "UPDATE orders SET return_status='requested', version=version+1 WHERE order_id=? AND return_status IS NULL",
                (order_id,),
            )
            conn.commit()
            changed = cur.rowcount == 1
            return {"ok": changed, "changed": changed, "order_id": order_id, "return_status": "requested" if changed else "already_requested"}
        finally:
            conn.close()

    def escalate_to_human(self, user_id: str, reason: str, order_id: str | None = None) -> dict:
        stamp = datetime.now(timezone.utc).isoformat()
        hid = "H" + hashlib.sha1(f"{user_id}:{order_id}:{reason}:{stamp}".encode()).hexdigest()[:10]
        conn = orders.connect(self.db_path)
        try:
            conn.execute("INSERT INTO handoffs VALUES(?,?,?,?,?)", (hid, user_id, order_id, reason, stamp))
            conn.commit()
        finally:
            conn.close()
        return {"ok": True, "handoff_id": hid}
