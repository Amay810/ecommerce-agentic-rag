"""Guardrails and action-space alignment for compiler write tools in RetailTools."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from ecommerce_rag.orders import connect, seed_database
from ecommerce_rag.retail_protocol import RETAIL_WRITE_TOOLS
from ecommerce_rag.tool_schema import IDENTITY_TOOLS, SCHEMA_BY_NAME, TOOL_SCHEMAS
from ecommerce_rag.tools import IDENTITY_GUARDED_TOOLS, WRITE_TOOLS, RetailTools


def _pending(db: Path):
    conn = connect(db)
    try:
        order = dict(conn.execute("SELECT * FROM orders WHERE status='pending' LIMIT 1").fetchone())
        code = conn.execute(
            "SELECT verification_code FROM users WHERE user_id=?", (order["user_id"],)
        ).fetchone()[0]
        return order, code
    finally:
        conn.close()


def _delivered_eligible(db: Path):
    conn = connect(db)
    try:
        order = dict(conn.execute(
            "SELECT * FROM orders WHERE status='delivered' AND quality_issue=1 LIMIT 1"
        ).fetchone())
        code = conn.execute(
            "SELECT verification_code FROM users WHERE user_id=?", (order["user_id"],)
        ).fetchone()[0]
        return order, code
    finally:
        conn.close()


def _user_row(db: Path, user_id: str):
    conn = connect(db)
    try:
        return dict(conn.execute("SELECT * FROM users WHERE user_id=?", (user_id,)).fetchone())
    finally:
        conn.close()


def test_compiler_write_tools_are_executable_on_retail_tools():
    tools = RetailTools(":memory:")
    missing = sorted(RETAIL_WRITE_TOOLS - tools.executable_tool_names())
    assert missing == [], missing
    assert RETAIL_WRITE_TOOLS <= WRITE_TOOLS
    for name in RETAIL_WRITE_TOOLS:
        assert name in SCHEMA_BY_NAME
        assert name in IDENTITY_TOOLS
        assert name in IDENTITY_GUARDED_TOOLS


def test_tool_schemas_cover_full_write_surface():
    schema_names = {schema["name"] for schema in TOOL_SCHEMAS}
    assert WRITE_TOOLS <= schema_names
    assert RETAIL_WRITE_TOOLS <= schema_names


def test_cancel_requires_confirmation_and_blocks_non_pending():
    with tempfile.TemporaryDirectory() as directory:
        db = Path(directory) / "retail.db"
        seed_database(db, users=20, orders=100)
        tools = RetailTools(db)
        pending, code = _pending(db)
        blocked = tools.call(
            "cancel_pending_order",
            order_id=pending["order_id"],
            user_id=pending["user_id"],
            verification_code=code,
            reason="no longer needed",
            confirmed=False,
        )
        assert blocked == {
            "ok": False,
            "changed": False,
            "error": "confirmation_required",
            "order_id": pending["order_id"],
        }
        ok = tools.call(
            "cancel_pending_order",
            order_id=pending["order_id"],
            user_id=pending["user_id"],
            verification_code=code,
            reason="no longer needed",
            confirmed=True,
        )
        assert ok["ok"] and ok["changed"] and ok["status"] == "cancelled"
        again = tools.call(
            "cancel_pending_order",
            order_id=pending["order_id"],
            user_id=pending["user_id"],
            verification_code=code,
            reason="no longer needed",
            confirmed=True,
        )
        assert again["ok"] and again["idempotent_replay"] and again["changed"] is False

        delivered, dcode = _delivered_eligible(db)
        refused = tools.call(
            "cancel_pending_order",
            order_id=delivered["order_id"],
            user_id=delivered["user_id"],
            verification_code=dcode,
            reason="ordered by mistake",
            confirmed=True,
        )
        assert refused["error"] == "order_not_pending"
        assert refused["changed"] is False


def test_modify_address_payment_items_guards():
    with tempfile.TemporaryDirectory() as directory:
        db = Path(directory) / "retail.db"
        seed_database(db, users=20, orders=100)
        tools = RetailTools(db)
        pending, code = _pending(db)
        user = _user_row(db, pending["user_id"])
        payment = json.loads(user["payment_methods"])[0]
        alt_payment = json.loads(user["payment_methods"])[1]
        # Seed defaults to credit_card_*; switch onto the other method so the write is real.
        target_payment = payment if pending.get("payment_method_id") != payment else alt_payment

        no_confirm = tools.call(
            "modify_pending_order_address",
            order_id=pending["order_id"],
            user_id=pending["user_id"],
            verification_code=code,
            address1="9 New St",
            address2="",
            city="Singapore",
            state="SG",
            country="SG",
            zip="999001",
            confirmed=False,
        )
        assert no_confirm["error"] == "confirmation_required"

        addr = tools.call(
            "modify_pending_order_address",
            order_id=pending["order_id"],
            user_id=pending["user_id"],
            verification_code=code,
            address1="9 New St",
            address2="",
            city="Singapore",
            state="SG",
            country="SG",
            zip="999001",
            confirmed=True,
        )
        assert addr["ok"] and addr["changed"]

        pay = tools.call(
            "modify_pending_order_payment",
            order_id=pending["order_id"],
            user_id=pending["user_id"],
            verification_code=code,
            payment_method_id=target_payment,
            confirmed=True,
        )
        assert pay["ok"] and pay["changed"] and pay["payment_method_id"] == target_payment

        items = tools.call(
            "modify_pending_order_items",
            order_id=pending["order_id"],
            user_id=pending["user_id"],
            verification_code=code,
            item_ids=[pending["product_id"]],
            new_item_ids=["P99999"],
            payment_method_id=target_payment,
            confirmed=True,
        )
        assert items["ok"] and items["changed"] and items["product_id"] == "P99999"

        delivered, dcode = _delivered_eligible(db)
        blocked = tools.call(
            "modify_pending_order_payment",
            order_id=delivered["order_id"],
            user_id=delivered["user_id"],
            verification_code=dcode,
            payment_method_id=json.loads(_user_row(db, delivered["user_id"])["payment_methods"])[0],
            confirmed=True,
        )
        assert blocked["error"] == "order_not_pending"


def test_return_and_exchange_delivered_guards():
    with tempfile.TemporaryDirectory() as directory:
        db = Path(directory) / "retail.db"
        seed_database(db, users=40, orders=200)
        tools = RetailTools(db)
        delivered, code = _delivered_eligible(db)
        payment = delivered["payment_method_id"] or json.loads(
            _user_row(db, delivered["user_id"])["payment_methods"]
        )[0]

        pending, pcode = _pending(db)
        wrong_state = tools.call(
            "exchange_delivered_order_items",
            order_id=pending["order_id"],
            user_id=pending["user_id"],
            verification_code=pcode,
            item_ids=[pending["product_id"]],
            new_item_ids=["P88888"],
            payment_method_id=json.loads(_user_row(db, pending["user_id"])["payment_methods"])[0],
            confirmed=True,
        )
        assert wrong_state["error"] == "order_not_delivered"

        returned = tools.call(
            "return_delivered_order_items",
            order_id=delivered["order_id"],
            user_id=delivered["user_id"],
            verification_code=code,
            item_ids=[delivered["product_id"]],
            payment_method_id=payment,
            confirmed=True,
        )
        assert returned["ok"] and returned["changed"] and returned["return_status"] == "requested"
        again = tools.call(
            "return_delivered_order_items",
            order_id=delivered["order_id"],
            user_id=delivered["user_id"],
            verification_code=code,
            item_ids=[delivered["product_id"]],
            payment_method_id=payment,
            confirmed=True,
        )
        assert again["ok"] and again["idempotent_replay"]

        # Fresh delivered order for exchange (not the one already returned).
        conn = connect(db)
        try:
            exchange_order = dict(conn.execute(
                "SELECT * FROM orders WHERE status='delivered' AND quality_issue=0 "
                "AND return_status IS NULL AND exchange_status IS NULL LIMIT 1"
            ).fetchone())
            exchange_code = conn.execute(
                "SELECT verification_code FROM users WHERE user_id=?",
                (exchange_order["user_id"],),
            ).fetchone()[0]
        finally:
            conn.close()
        pay = exchange_order["payment_method_id"] or json.loads(
            _user_row(db, exchange_order["user_id"])["payment_methods"]
        )[0]
        exchanged = tools.call(
            "exchange_delivered_order_items",
            order_id=exchange_order["order_id"],
            user_id=exchange_order["user_id"],
            verification_code=exchange_code,
            item_ids=[exchange_order["product_id"]],
            new_item_ids=["P77777"],
            payment_method_id=pay,
            confirmed=True,
        )
        assert exchanged["ok"] and exchanged["changed"] and exchanged["exchange_status"] == "exchanged"


def test_modify_user_address_requires_confirmation_and_is_idempotent():
    with tempfile.TemporaryDirectory() as directory:
        db = Path(directory) / "retail.db"
        seed_database(db, users=8, orders=20)
        tools = RetailTools(db)
        user = _user_row(db, "U0001")
        blocked = tools.call(
            "modify_user_address",
            user_id=user["user_id"],
            verification_code=user["verification_code"],
            address1="2 Other St",
            address2="Unit 1",
            city="Singapore",
            state="SG",
            country="SG",
            zip="018957",
            confirmed=False,
        )
        assert blocked["error"] == "confirmation_required"
        ok = tools.call(
            "modify_user_address",
            user_id=user["user_id"],
            verification_code=user["verification_code"],
            address1="2 Other St",
            address2="Unit 1",
            city="Singapore",
            state="SG",
            country="SG",
            zip="018957",
            confirmed=True,
        )
        assert ok["ok"] and ok["changed"]
        again = tools.call(
            "modify_user_address",
            user_id=user["user_id"],
            verification_code=user["verification_code"],
            address1="2 Other St",
            address2="Unit 1",
            city="Singapore",
            state="SG",
            country="SG",
            zip="018957",
            confirmed=True,
        )
        assert again["ok"] and again["idempotent_replay"]


def test_identity_guard_blocks_new_write_tools_without_code():
    with tempfile.TemporaryDirectory() as directory:
        db = Path(directory) / "retail.db"
        seed_database(db, users=8, orders=40)
        tools = RetailTools(db)
        pending, _ = _pending(db)
        blocked = tools.call(
            "cancel_pending_order",
            order_id=pending["order_id"],
            user_id=pending["user_id"],
            verification_code="bad",
            reason="no longer needed",
            confirmed=True,
        )
        assert blocked["error"] == "verification_code_required"
        assert tools.guardrails[-1]["tool"] == "cancel_pending_order"
