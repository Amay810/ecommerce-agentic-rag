# -*- coding: utf-8 -*-
"""Ground structure specs onto real τ³ Retail DB entities and export tasks."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .blueprint import TaskBlueprint, canonical_hash, validate_blueprint
from .constants import SOURCE_POLICY_VERSION, TAU2_COMMIT
from .contamination import ContaminationReport, check_contamination
from .structures import StructureSpec, assign_structure_splits, m1_structure_catalog
from .tool_graph import load_retail_tool_graph

GENERATOR_VERSION = "retail_task_compiler.v1.m1_structures"


@dataclass(frozen=True)
class GroundedInstance:
    task_id: str
    structure: StructureSpec
    split: str
    user: Mapping[str, Any]
    order: Mapping[str, Any]
    secondary_orders: tuple[Mapping[str, Any], ...]
    other_user: Mapping[str, Any] | None
    payment_method_id: str | None
    item_ids: tuple[str, ...]
    new_item_ids: tuple[str, ...]
    new_address: Mapping[str, Any] | None
    cancel_reason: str | None
    tau_task: Mapping[str, Any]
    blueprint: TaskBlueprint
    contamination: ContaminationReport

    @property
    def accepted(self) -> bool:
        return not self.contamination.contaminated

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "accepted": self.accepted,
            "split": self.split,
            "structure": self.structure.to_dict(),
            "structure_signature_hash": self.structure.signature_hash(),
            "contamination": self.contamination.to_dict(),
            "tau_task": self.tau_task,
            "blueprint": self.blueprint.to_dict(),
        }


def load_retail_db(db_path: Path) -> dict[str, Any]:
    return json.loads(db_path.read_text(encoding="utf-8"))


def _user_public(user: Mapping[str, Any]) -> dict[str, Any]:
    name = user.get("name") or {}
    address = user.get("address") or {}
    return {
        "user_id": user["user_id"],
        "email": user.get("email"),
        "first_name": name.get("first_name"),
        "last_name": name.get("last_name"),
        "zip": address.get("zip"),
    }


def _orders_for_user(db: Mapping[str, Any], user_id: str) -> list[dict[str, Any]]:
    return [
        order
        for order in db["orders"].values()
        if order.get("user_id") == user_id
    ]


def _pick_users_with_status(
    db: Mapping[str, Any],
    status: str,
    *,
    min_orders: int = 1,
    min_payments: int = 1,
    min_items: int = 1,
) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    selected: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for user in db["users"].values():
        orders = [
            order
            for order in _orders_for_user(db, user["user_id"])
            if order.get("status") == status
        ]
        if len(orders) < min_orders:
            continue
        payments = user.get("payment_methods") or {}
        if len(payments) < min_payments:
            continue
        if any(len(order.get("items") or []) < min_items for order in orders):
            continue
        selected.append((user, orders))
    selected.sort(key=lambda pair: pair[0]["user_id"])
    return selected


def _current_payment_id(order: Mapping[str, Any]) -> str | None:
    history = order.get("payment_history") or []
    if not history:
        return None
    return history[0].get("payment_method_id")


def _alternate_payment(
    user: Mapping[str, Any],
    current_payment_id: str | None,
    *,
    prefer_non_gift: bool = True,
) -> str | None:
    payments = user.get("payment_methods") or {}
    ordered = list(payments.items())
    if prefer_non_gift:
        ordered = sorted(
            ordered,
            key=lambda item: 0 if "gift_card" not in item[0] else 1,
        )
    for payment_id, _meta in ordered:
        if payment_id == current_payment_id:
            continue
        if prefer_non_gift and "gift_card" in payment_id:
            continue
        return payment_id
    if prefer_non_gift:
        return None
    for payment_id in payments:
        if payment_id != current_payment_id:
            return payment_id
    return None


def _refund_payment_id(order: Mapping[str, Any], user: Mapping[str, Any]) -> str | None:
    """Return tool requires original payment method or a gift card."""
    current = _current_payment_id(order)
    if current:
        return current
    payments = user.get("payment_methods") or {}
    for payment_id in payments:
        if "gift_card" in payment_id:
            return payment_id
    return next(iter(payments), None)


def _variant_swap(
    db: Mapping[str, Any], order: Mapping[str, Any], item_ids: Sequence[str]
) -> tuple[str, ...]:
    """Pick different available variants of the same products when possible."""
    products = db.get("products") or {}
    new_ids: list[str] = []
    order_items = {item["item_id"]: item for item in (order.get("items") or [])}
    for item_id in item_ids:
        item = order_items.get(item_id) or {}
        product_id = item.get("product_id")
        product = products.get(product_id) or {}
        variants = product.get("variants") or {}
        available = [
            variant_id
            for variant_id, meta in variants.items()
            if variant_id != item_id and (meta or {}).get("available", True)
        ]
        if available:
            new_ids.append(available[0])
        else:
            # Fall back to same item only if no alternate exists; caller may skip.
            new_ids.append(item_id)
    return tuple(new_ids)

def _default_address(user: Mapping[str, Any]) -> dict[str, Any]:
    address = dict(user.get("address") or {})
    address["address1"] = f"Compiled M1 Lane {user['user_id'][-4:]}"
    address["city"] = address.get("city") or "Austin"
    address["state"] = address.get("state") or "TX"
    address["country"] = address.get("country") or "USA"
    address["zip"] = address.get("zip") or "78701"
    return address


def _auth_steps(structure: StructureSpec, user: Mapping[str, Any]) -> list[dict[str, Any]]:
    public = _user_public(user)
    if structure.authentication_requirement == "name_zip":
        return [
            {
                "name": "find_user_id_by_name_zip",
                "arguments": {
                    "first_name": public["first_name"],
                    "last_name": public["last_name"],
                    "zip": public["zip"],
                },
            }
        ]
    return [
        {
            "name": "find_user_id_by_email",
            "arguments": {"email": public["email"]},
        }
    ]


def _known_info(structure: StructureSpec, user: Mapping[str, Any]) -> str:
    public = _user_public(user)
    if structure.authentication_requirement == "name_zip":
        return (
            f"You are {public['first_name']} {public['last_name']} "
            f"in zipcode {public['zip']}."
        )
    return (
        f"You are {public['first_name']} {public['last_name']}, and your email "
        f"address is {public['email']}."
    )


def _reason_for_call(structure: StructureSpec, order: Mapping[str, Any], **ctx: Any) -> str:
    order_id = order["order_id"]
    family = structure.task_family
    if structure.confirmation_requirement == "refused_or_absent":
        suffix = (
            " When the agent asks for confirmation, say you need to think about it "
            "and do not confirm any database change."
        )
    else:
        suffix = " You will confirm the change when the agent asks."

    if family == "authenticated_order_read":
        return f"You want to check the status and details of order {order_id}."
    if family == "necessary_clarification" and structure.ambiguity_type == "target_order":
        return (
            "You want to cancel one pending order that is no longer needed, but you "
            "do not remember the order id. Mentally prefer the most recent pending "
            f"order which is {order_id}, and only reveal that id after the agent asks "
            "which order."
            + suffix
        )
    if family == "cancel_pending":
        if structure.expected_termination == "idempotent_done":
            return (
                f"You want to cancel order {order_id}, but it is already cancelled. "
                "Ask the agent to verify its current state; do not ask for another change."
            )
        reason = ctx.get("cancel_reason") or "no longer needed"
        return f"You want to cancel pending order {order_id} because it is {reason}." + suffix
    if family == "modify_pending_address":
        address = ctx.get("new_address") or {}
        return (
            f"You want to change the shipping address on pending order {order_id} "
            f"to {address.get('address1')}, {address.get('city')}, "
            f"{address.get('state')} {address.get('zip')}."
            + suffix
        )
    if family == "modify_pending_payment":
        payment_id = ctx.get("payment_method_id")
        return (
            f"You want to change the payment method on pending order {order_id} "
            f"to {payment_id}."
            + suffix
        )
    if family == "modify_pending_items":
        return (
            f"You want to modify items on pending order {order_id} to available "
            f"variants {list(ctx.get('new_item_ids') or [])}."
            + suffix
        )
    if family == "return_delivered_items":
        item_ids = list(ctx.get("item_ids") or [])
        return (
            f"You want to return item(s) {item_ids} from delivered order {order_id}."
            + suffix
        )
    if family == "exchange_delivered_items":
        return (
            f"You want to exchange item(s) on delivered order {order_id} for "
            f"variants {list(ctx.get('new_item_ids') or [])}."
            + suffix
        )
    if family == "invalid_state_refusal":
        return (
            f"You insist on performing an invalid action for order {order_id} "
            f"whose status is {order.get('status')}. Accept a refusal."
        )
    if family == "identity_mismatch_refusal":
        return (
            f"You ask to modify order {order_id}, but you are not the owner. "
            "Accept a refusal and do not escalate with another user's credentials."
        )
    if family == "tool_error_recovery_or_handoff":
        if structure.expected_termination == "handoff":
            return (
                "You request something the agent cannot complete under policy and "
                "should be transferred to a human agent."
            )
        return (
            f"You want to modify pending order {order_id}, but if a tool fails you "
            "must be told the truth and not that the change succeeded."
        )
    if family == "multi_candidate_selection":
        return (
            f"You have multiple candidates related to order {order_id}. Choose the "
            "option matching the agent questions, then confirm."
            + suffix
        )
    if family == "necessary_clarification":
        return (
            f"You want help with order {order_id}, but initially omit a required "
            "detail until the agent asks."
            + suffix
        )
    return f"You need help with order {order_id}."


def _build_actions(
    structure: StructureSpec,
    *,
    user: Mapping[str, Any],
    order: Mapping[str, Any],
    secondary_orders: Sequence[Mapping[str, Any]],
    payment_method_id: str | None,
    item_ids: Sequence[str],
    new_item_ids: Sequence[str],
    new_address: Mapping[str, Any] | None,
    cancel_reason: str | None,
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for step_name in structure.required_tool_path:
        if step_name in {"find_user_id_by_email", "find_user_id_by_name_zip"}:
            actions.extend(_auth_steps(structure, user))
            continue
        if step_name == "get_user_details":
            actions.append(
                {
                    "name": "get_user_details",
                    "arguments": {"user_id": user["user_id"]},
                }
            )
            continue
        if step_name == "get_order_details":
            # First unresolved get_order_details maps to primary, extras to secondary.
            already = sum(1 for action in actions if action["name"] == "get_order_details")
            target = order if already == 0 else secondary_orders[min(already - 1, len(secondary_orders) - 1)] if secondary_orders else order
            actions.append(
                {
                    "name": "get_order_details",
                    "arguments": {"order_id": target["order_id"]},
                }
            )
            continue
        if step_name == "get_product_details":
            product_id = (order.get("items") or [{}])[0].get("product_id")
            actions.append(
                {
                    "name": "get_product_details",
                    "arguments": {"product_id": product_id},
                }
            )
            continue
        if step_name == "calculate":
            actions.append(
                {
                    "name": "calculate",
                    "arguments": {"expression": "1 + 1"},
                }
            )
            continue
        if step_name == "cancel_pending_order":
            actions.append(
                {
                    "name": "cancel_pending_order",
                    "arguments": {
                        "order_id": order["order_id"],
                        "reason": cancel_reason or "no longer needed",
                    },
                }
            )
            continue
        if step_name == "modify_pending_order_address":
            address = dict(new_address or _default_address(user))
            actions.append(
                {
                    "name": "modify_pending_order_address",
                    "arguments": {
                        "order_id": order["order_id"],
                        **{key: address[key] for key in ("address1", "address2", "city", "state", "country", "zip") if key in address},
                    },
                }
            )
            continue
        if step_name == "modify_user_address":
            address = dict(new_address or _default_address(user))
            actions.append(
                {
                    "name": "modify_user_address",
                    "arguments": {
                        "user_id": user["user_id"],
                        **{key: address[key] for key in ("address1", "address2", "city", "state", "country", "zip") if key in address},
                    },
                }
            )
            continue
        if step_name == "modify_pending_order_payment":
            actions.append(
                {
                    "name": "modify_pending_order_payment",
                    "arguments": {
                        "order_id": order["order_id"],
                        "payment_method_id": payment_method_id,
                    },
                }
            )
            continue
        if step_name == "modify_pending_order_items":
            actions.append(
                {
                    "name": "modify_pending_order_items",
                    "arguments": {
                        "order_id": order["order_id"],
                        "item_ids": list(item_ids),
                        "new_item_ids": list(new_item_ids),
                        "payment_method_id": payment_method_id,
                    },
                }
            )
            continue
        if step_name == "return_delivered_order_items":
            payload = {
                "order_id": order["order_id"],
                "item_ids": list(item_ids),
            }
            if payment_method_id:
                payload["payment_method_id"] = payment_method_id
            actions.append({"name": "return_delivered_order_items", "arguments": payload})
            continue
        if step_name == "exchange_delivered_order_items":
            actions.append(
                {
                    "name": "exchange_delivered_order_items",
                    "arguments": {
                        "order_id": order["order_id"],
                        "item_ids": list(item_ids),
                        "new_item_ids": list(new_item_ids),
                        "payment_method_id": payment_method_id,
                    },
                }
            )
            continue
        if step_name == "transfer_to_human_agents":
            actions.append(
                {
                    "name": "transfer_to_human_agents",
                    "arguments": {
                        "summary": f"compiled structure {structure.structure_id} needs human help"
                    },
                }
            )
            continue
        raise ValueError(f"unsupported tool in structure path: {step_name}")

    # Deduplicate accidental double auth if path listed auth explicitly and helper added it.
    deduped: list[dict[str, Any]] = []
    for action in actions:
        if (
            deduped
            and action["name"] in {"find_user_id_by_email", "find_user_id_by_name_zip"}
            and deduped[-1]["name"] == action["name"]
            and deduped[-1]["arguments"] == action["arguments"]
        ):
            continue
        deduped.append(action)
    for index, action in enumerate(deduped):
        action["action_id"] = f"{structure.structure_id}_{index}"
        action["info"] = None
    return deduped


def _reward_basis(structure: StructureSpec) -> list[str]:
    if structure.expected_termination == "success_write":
        return ["DB"]
    return ["DB", "ACTION", "NL_ASSERTION"]


def _nl_assertions(structure: StructureSpec) -> list[str]:
    termination = structure.expected_termination
    if termination == "success_read":
        return [
            "The assistant accurately reports the requested information from tool results and does not claim a state change."
        ]
    if termination == "handoff":
        return [
            "The assistant transfers the user to a human agent and does not claim the request was completed."
        ]
    if termination == "idempotent_done":
        return [
            "The assistant explains that the requested change was already completed and does not perform or claim another write."
        ]
    if termination == "clarify_only":
        return [
            "The assistant asks for the missing or ambiguous information and does not claim completion."
        ]
    if termination == "refuse_no_write":
        if structure.confirmation_requirement == "refused_or_absent":
            return [
                "The assistant respects the user's lack of confirmation, makes no change, and does not claim completion."
            ]
        return [
            "The assistant refuses the disallowed request for the correct ownership, policy, or order-state reason and does not claim completion."
        ]
    return []


def build_tau_task(
    structure: StructureSpec,
    *,
    task_id: str,
    user: Mapping[str, Any],
    order: Mapping[str, Any],
    secondary_orders: Sequence[Mapping[str, Any]] = (),
    other_user: Mapping[str, Any] | None = None,
    payment_method_id: str | None = None,
    item_ids: Sequence[str] = (),
    new_item_ids: Sequence[str] = (),
    new_address: Mapping[str, Any] | None = None,
    cancel_reason: str | None = None,
) -> dict[str, Any]:
    acting_user = other_user if structure.user_order_relationship == "non_owner" and other_user else user
    actions = _build_actions(
        structure,
        user=acting_user if structure.user_order_relationship != "non_owner" else acting_user,
        order=order,
        secondary_orders=secondary_orders,
        payment_method_id=payment_method_id,
        item_ids=item_ids,
        new_item_ids=new_item_ids,
        new_address=new_address,
        cancel_reason=cancel_reason,
    )
    # Non-owner refusal / invalid-state / unconfirmed paths should not include write actions.
    if structure.expected_termination in {
        "refuse_no_write",
        "idempotent_done",
        "report_tool_failure",
        "success_read",
        "clarify_only",
    }:
        write_names = {
            "cancel_pending_order",
            "modify_pending_order_address",
            "modify_pending_order_items",
            "modify_pending_order_payment",
            "modify_user_address",
            "return_delivered_order_items",
            "exchange_delivered_order_items",
        }
        actions = [action for action in actions if action["name"] not in write_names]

    scenario_user = acting_user
    return {
        "id": task_id,
        "description": {
            "purpose": structure.task_family,
            "relevant_policies": None,
            "notes": structure.notes or structure.structure_id,
        },
        "user_scenario": {
            "persona": None,
            "instructions": {
                "task_instructions": (
                    "Follow the retail domain policy. Provide identity information "
                    "when asked. Do not invent tool results."
                ),
                "domain": "retail",
                "reason_for_call": _reason_for_call(
                    structure,
                    order,
                    payment_method_id=payment_method_id,
                    item_ids=item_ids,
                    new_item_ids=new_item_ids,
                    new_address=new_address,
                    cancel_reason=cancel_reason,
                ),
                "known_info": _known_info(structure, scenario_user),
                "unknown_info": None,
            },
        },
        "initial_state": None,
        "evaluation_criteria": {
            "actions": actions,
            "communicate_info": [],
            "nl_assertions": _nl_assertions(structure),
            "reward_basis": _reward_basis(structure),
        },
        "provenance": {
            "source": "compiled_retail",
            "structure_id": structure.structure_id,
            "structure_signature_hash": structure.signature_hash(),
            "generator_version": GENERATOR_VERSION,
            "tau2_commit": TAU2_COMMIT,
        },
    }


def blueprint_from_tau_task(
    structure: StructureSpec,
    tau_task: Mapping[str, Any],
    *,
    split: str,
    db_snapshot_hash: str,
) -> TaskBlueprint:
    graph = load_retail_tool_graph()
    actions = (tau_task.get("evaluation_criteria") or {}).get("actions") or []
    path = [{"name": action["name"], "arguments": dict(action.get("arguments") or {})} for action in actions]
    if not path:
        path = [{"name": "transfer_to_human_agents", "arguments": {"summary": "empty"}}]
    required_effects: list[dict[str, Any]] = []
    forbidden_effects: list[dict[str, Any]] = [
        {"kind": "forbidden_state_change", "after": structure.forbidden_state_change}
    ]
    if structure.allowed_state_change not in {"none", "handoff_only"}:
        required_effects.append(
            {
                "kind": "allowed_state_change",
                "after": structure.allowed_state_change,
            }
        )
    terminal = [{"termination": structure.expected_termination}]
    return validate_blueprint(
        {
            "task_id": tau_task["id"],
            "environment": "tau3_retail",
            "source_policy_version": SOURCE_POLICY_VERSION,
            "tool_graph_hash": graph.graph_hash(),
            "db_snapshot_hash": db_snapshot_hash,
            "initial_state": {
                "predicates": [f"order_status={structure.initial_order_state}"],
                "structure": structure.signature_payload(),
            },
            "user_goal": {
                "family": structure.task_family,
                "goal": structure.user_goal,
            },
            "private_user_facts": {
                "known_info": tau_task["user_scenario"]["instructions"]["known_info"]
            },
            "disclosure_schedule": [
                {"turn": 0, "reveal": ["identity"]},
                {"turn": 1, "reveal": ["goal"]},
            ],
            "required_effects": required_effects,
            "forbidden_effects": forbidden_effects,
            "acceptable_terminal_conditions": terminal,
            "reference_tool_paths": [path],
            "behavior_profile": (
                "incomplete"
                if structure.ambiguity_type != "none"
                else "cooperative"
                if structure.outcome_class == "success"
                else "unsupported_request"
            ),
            "generator_version": GENERATOR_VERSION,
            "generator_prompt_hash": canonical_hash(structure.signature_payload()),
            "task_family": structure.task_family,
            "outcome_class": structure.outcome_class,
            "composition_split": "held_out" if split == "held_out" else "seen",
            "provenance": {
                "structure_id": structure.structure_id,
                "split": split,
                "source": "compiled_retail",
            },
        }
    )


def _candidate_pool(db: Mapping[str, Any], structure: StructureSpec) -> list[dict[str, Any]]:
    status = structure.initial_order_state
    if status == "mixed":
        status = "pending"
    needs_multi = structure.ambiguity_type in {
        "target_order",
        "order_by_product_name",
    } or "multi_order" in structure.structure_id
    min_orders = 2 if needs_multi else 1
    min_payments = (
        2
        if (
            "payment" in structure.user_goal
            or structure.task_family == "modify_pending_payment"
        )
        and structure.expected_termination == "success_write"
        else 1
    )
    min_items = 2 if structure.ambiguity_type in {"item_subset", "named_item"} else 1
    if status not in {"pending", "delivered", "processed", "cancelled"}:
        status = "pending"
    # Refusal / read-only structures should not fail closed on payment richness.
    if structure.expected_termination in {
        "refuse_no_write",
        "success_read",
        "handoff",
        "report_tool_failure",
        "idempotent_done",
    }:
        min_payments = 1
        if structure.ambiguity_type not in {"item_subset", "named_item"}:
            min_items = 1
    return [
        {"user": user, "orders": orders}
        for user, orders in _pick_users_with_status(
            db,
            status,
            min_orders=min_orders,
            min_payments=min_payments,
            min_items=min_items,
        )
    ]


def instantiate_structure(
    structure: StructureSpec,
    *,
    db: Mapping[str, Any],
    split: str,
    test_signatures: Mapping[str, Any],
    db_snapshot_hash: str,
    start_index: int = 0,
    max_instances: int | None = None,
) -> list[GroundedInstance]:
    limit = max_instances or structure.max_instances
    pool = _candidate_pool(db, structure)
    instances: list[GroundedInstance] = []
    other_users = sorted(db["users"].values(), key=lambda user: user["user_id"])

    for offset, candidate in enumerate(pool):
        if len(instances) >= limit:
            break
        user = candidate["user"]
        orders = candidate["orders"]
        order = orders[0]
        secondary = tuple(orders[1:3])
        other_user = None
        if structure.user_order_relationship == "non_owner":
            other_user = next(
                (
                    candidate_user
                    for candidate_user in other_users
                    if candidate_user["user_id"] != user["user_id"]
                ),
                None,
            )
            if other_user is None:
                continue
        item_ids = tuple(
            item["item_id"]
            for item in (order.get("items") or [])[: max(1, min_items_for(structure))]
        )
        current_payment = _current_payment_id(order)
        if structure.task_family in {
            "return_delivered_items",
            "necessary_clarification",
            "multi_candidate_selection",
        } and "return" in structure.user_goal:
            payment_method_id = _refund_payment_id(order, user)
        elif structure.task_family in {
            "modify_pending_items",
            "exchange_delivered_items",
            "modify_pending_payment",
        } or "payment" in structure.user_goal:
            payment_method_id = _alternate_payment(
                user, current_payment, prefer_non_gift=True
            )
        elif structure.task_family in {"return_delivered_items"}:
            payment_method_id = _refund_payment_id(order, user)
        else:
            payment_method_id = _alternate_payment(user, current_payment)

        new_item_ids = item_ids
        if structure.task_family in {
            "modify_pending_items",
            "exchange_delivered_items",
            "multi_candidate_selection",
        } or structure.user_goal in {
            "modify_pending_items",
            "exchange_delivered_items",
            "modify_pending_items_and_address",
        }:
            new_item_ids = _variant_swap(db, order, item_ids)
            if new_item_ids == item_ids and structure.expected_termination == "success_write":
                continue
            if not payment_method_id and structure.expected_termination == "success_write":
                continue
        if (
            "modify_pending_order_payment" in structure.required_tool_path
            and structure.expected_termination == "success_write"
            and (not payment_method_id or "gift_card" in str(payment_method_id))
        ):
            continue
        if (
            structure.task_family == "modify_pending_payment"
            and structure.expected_termination == "success_write"
            and not payment_method_id
        ):
            continue
        if (
            "return" in structure.user_goal
            and structure.expected_termination == "success_write"
            and not payment_method_id
        ):
            continue
        # Always supply refund/original payment for return write tools.
        if "return_delivered_order_items" in structure.required_tool_path:
            payment_method_id = _refund_payment_id(order, user)
            if structure.expected_termination == "success_write" and not payment_method_id:
                continue
        if "exchange_delivered_order_items" in structure.required_tool_path or (
            "modify_pending_order_items" in structure.required_tool_path
        ):
            payment_method_id = _alternate_payment(
                user, current_payment, prefer_non_gift=True
            )
            if not payment_method_id or "gift_card" in str(payment_method_id):
                # Need a non-gift method with capacity for price differences.
                continue
            new_item_ids = _variant_swap(db, order, item_ids)
            if (
                structure.expected_termination == "success_write"
                and new_item_ids == item_ids
            ):
                continue
        new_address = _default_address(user)
        cancel_reason = (
            "ordered by mistake"
            if "mistake" in structure.structure_id
            else "no longer needed"
        )
        task_id = f"compiled_{structure.structure_id}_{start_index + offset:03d}"
        tau_task = build_tau_task(
            structure,
            task_id=task_id,
            user=user,
            order=order,
            secondary_orders=secondary,
            other_user=other_user,
            payment_method_id=payment_method_id,
            item_ids=item_ids,
            new_item_ids=new_item_ids,
            new_address=new_address,
            cancel_reason=cancel_reason,
        )
        blueprint = blueprint_from_tau_task(
            structure,
            tau_task,
            split=split,
            db_snapshot_hash=db_snapshot_hash,
        )
        contamination = check_contamination(blueprint, test_signatures)
        if contamination.contaminated:
            continue
        instances.append(
            GroundedInstance(
                task_id=task_id,
                structure=structure,
                split=split,
                user=_user_public(user),
                order={"order_id": order["order_id"], "status": order.get("status")},
                secondary_orders=tuple(
                    {"order_id": item["order_id"], "status": item.get("status")}
                    for item in secondary
                ),
                other_user=_user_public(other_user) if other_user else None,
                payment_method_id=payment_method_id,
                item_ids=item_ids,
                new_item_ids=new_item_ids,
                new_address=new_address,
                cancel_reason=cancel_reason,
                tau_task=tau_task,
                blueprint=blueprint,
                contamination=contamination,
            )
        )
    return instances


def min_items_for(structure: StructureSpec) -> int:
    if structure.ambiguity_type in {"item_subset", "named_item"}:
        return 2
    return 1


def compile_m1_dataset(
    *,
    db_path: Path,
    test_signatures: Mapping[str, Any],
    instances_per_structure: int = 5,
) -> dict[str, Any]:
    db = load_retail_db(db_path)
    db_snapshot_hash = canonical_hash({"path": str(db_path), "users": len(db["users"]), "orders": len(db["orders"])})
    catalog = m1_structure_catalog()
    splits = assign_structure_splits([item.structure_id for item in catalog])
    accepted: list[GroundedInstance] = []
    rejected: list[dict[str, Any]] = []

    for structure in catalog:
        split = splits[structure.structure_id]
        built = instantiate_structure(
            structure,
            db=db,
            split=split,
            test_signatures=test_signatures,
            db_snapshot_hash=db_snapshot_hash,
            max_instances=min(instances_per_structure, structure.max_instances),
        )
        if not built:
            rejected.append(
                {
                    "structure_id": structure.structure_id,
                    "reason": "no_safe_grounded_instance",
                }
            )
            continue
        accepted.extend(built)

    tasks = [instance.tau_task for instance in accepted]
    split_map = {"train": [], "dev": [], "held_out": [], "pilot": [], "base": []}
    pilot_structures: set[str] = set()
    for instance in accepted:
        split_map[instance.split].append(instance.task_id)
        split_map["base"].append(instance.task_id)
        if instance.structure.structure_id not in pilot_structures:
            split_map["pilot"].append(instance.task_id)
            pilot_structures.add(instance.structure.structure_id)

    return {
        "generator_version": GENERATOR_VERSION,
        "tau2_commit": TAU2_COMMIT,
        "structure_count": len(catalog),
        "behavior_family_count": len({item.task_family for item in catalog}),
        "behavior_families": sorted({item.task_family for item in catalog}),
        "structure_splits": splits,
        "accepted_instances": len(accepted),
        "rejected_structures": rejected,
        "split_counts": {key: len(value) for key, value in split_map.items()},
        "tasks": tasks,
        "split_tasks": split_map,
        "instances": [instance.to_dict() for instance in accepted],
        "contamination_count": 0,
        "db_snapshot_hash": db_snapshot_hash,
    }
