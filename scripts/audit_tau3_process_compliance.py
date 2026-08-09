"""Apply the minimal process-quality rules used by the hint pilots."""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path
from typing import Any


WRITE_TOOLS = frozenset(
    {
        "cancel_pending_order",
        "modify_pending_order_address",
        "modify_pending_order_items",
        "modify_pending_order_payment",
        "modify_user_address",
        "return_delivered_order_items",
        "exchange_delivered_order_items",
    }
)
MONEY = re.compile(r"(?:\$|USD\s*)([0-9][0-9,]*(?:\.[0-9]+)?)", re.I)
AFFIRMATIVE = re.compile(
    r"\b(?:yes|confirm(?:ed)?|go ahead|proceed|please do|do it|that's right|that is right|correct)\b",
    re.I,
)
EXPLICIT_REASON = {
    "no longer needed": re.compile(
        r"\b(?:no longer need(?:ed)?|do not need|don't need)\b", re.I
    ),
    "ordered by mistake": re.compile(
        r"\b(?:ordered?\s+(?:it|them|this|the order)?\s*by mistake|order(?:ed)?[^.?!]{0,30}\bmistake)\b",
        re.I,
    ),
}
REFUND_COMPLETION = re.compile(
    r"(?:refund[^.?!]{0,80}(?:has been|was|is now)\s+(?:processed|credited|completed)|"
    r"(?:has been|was)\s+refunded|"
    r"(?:new|current)\s+balance|balance\s+(?:will\s+now|is\s+now))",
    re.I,
)


def _numbers(text: str) -> set[str]:
    values = set()
    for raw in MONEY.findall(text):
        try:
            values.add(f"{float(raw.replace(',', '')):.2f}")
        except ValueError:
            continue
    return values


def _tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    return list(message.get("tool_calls") or [])


def _expected_writes(task: dict[str, Any]) -> collections.Counter[str]:
    actions = ((task.get("evaluation_criteria") or {}).get("actions") or [])
    return collections.Counter(
        action.get("name") for action in actions if action.get("name") in WRITE_TOOLS
    )


def audit_simulation(simulation: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    messages = simulation.get("messages") or []
    expected_writes = _expected_writes(task)
    actual_writes: collections.Counter[str] = collections.Counter()
    prior_tool_text = ""
    last_user = ""
    prior_assistant = ""

    for index, message in enumerate(messages):
        role = message.get("role")
        content = str(message.get("content") or "")
        if role == "tool":
            prior_tool_text += "\n" + content
            if content.startswith("Error:"):
                issues.append(
                    {"type": "tool_error", "message_index": index, "detail": content}
                )
            continue
        if role == "user":
            last_user = content
            continue
        if role != "assistant":
            continue

        for amount in sorted(_numbers(content) - _numbers(prior_tool_text)):
            issues.append(
                {
                    "type": "ungrounded_money",
                    "message_index": index,
                    "detail": amount,
                }
            )
        if REFUND_COMPLETION.search(content):
            recent_tool = ""
            for prior in reversed(messages[:index]):
                if prior.get("role") == "tool":
                    recent_tool = str(prior.get("content") or "")
                    break
            if '"transaction_type": "refund"' not in recent_tool and '"balance"' not in recent_tool:
                issues.append(
                    {
                        "type": "unsupported_refund_or_balance_state",
                        "message_index": index,
                        "detail": content[:240],
                    }
                )

        calls = _tool_calls(message)
        for call in calls:
            name = call.get("name")
            if name not in WRITE_TOOLS:
                continue
            actual_writes[name] += 1
            if not AFFIRMATIVE.search(last_user):
                issues.append(
                    {
                        "type": "write_without_explicit_confirmation",
                        "message_index": index,
                        "detail": name,
                    }
                )
            if name == "cancel_pending_order":
                reason = str((call.get("arguments") or {}).get("reason") or "")
                reason_pattern = EXPLICIT_REASON.get(reason)
                user_confirmed_reason = bool(reason_pattern and reason_pattern.search(last_user))
                user_stated_other_reason = bool(
                    re.search(r"\b(?:reason|because|due to|since)\b", last_user, re.I)
                    and not user_confirmed_reason
                )
                summarized_then_confirmed = bool(
                    reason
                    and reason.lower() in prior_assistant.lower()
                    and AFFIRMATIVE.search(last_user)
                    and not user_stated_other_reason
                )
                if not (user_confirmed_reason or summarized_then_confirmed):
                    issues.append(
                        {
                            "type": "unconfirmed_enum_mapping",
                            "message_index": index,
                            "detail": reason,
                        }
                    )
        prior_assistant = content

    for name, count in actual_writes.items():
        if count > expected_writes[name]:
            issues.append(
                {
                    "type": "unexpected_or_extra_write",
                    "message_index": None,
                    "detail": f"{name}: actual={count}, expected={expected_writes[name]}",
                }
            )

    reward = (simulation.get("reward_info") or {}).get("reward")
    return {
        "task_id": str(simulation.get("task_id")),
        "trial": simulation.get("trial"),
        "reward": reward,
        "termination_reason": simulation.get("termination_reason"),
        "issue_types": sorted({issue["type"] for issue in issues}),
        "issues": issues,
        "automatic_process_pass": not issues,
        "strict_candidate": reward == 1 and not issues,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.results.read_text(encoding="utf-8"))
    tasks = {str(task.get("id")): task for task in payload.get("tasks") or []}
    audits = [
        audit_simulation(simulation, tasks[str(simulation.get("task_id"))])
        for simulation in payload.get("simulations") or []
    ]
    raw = args.results.read_text(encoding="utf-8")
    summary = {
        "simulations": len(audits),
        "reward_successes": sum(row["reward"] == 1 for row in audits),
        "automatic_process_passes": sum(row["automatic_process_pass"] for row in audits),
        "strict_candidates": sum(row["strict_candidate"] for row in audits),
        "infrastructure_errors": sum(
            row["termination_reason"] == "infrastructure_error" for row in audits
        ),
        "private_hint_markers_in_results": sum(
            raw.count(marker)
            for marker in ("<private_semantic_plan>", "<resolution_steps>", "[Step ")
        ),
    }
    print(json.dumps({"summary": summary, "simulations": audits}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
