"""Minimal CLI demo for the AgentCase data-flywheel MVP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ecommerce_rag.agent_case import AgentCase, progress_signature_from_progress
from ecommerce_rag.agent_case_memory import approve_case, build_memory_advice, write_candidate
from ecommerce_rag.agent_case_store import list_cases


def main() -> None:
    parser = argparse.ArgumentParser(description="AgentCase flywheel MVP helpers")
    parser.add_argument("--db", type=Path, required=True)
    sub = parser.add_subparsers(dest="cmd", required=True)

    seed = sub.add_parser("seed-train-identity", help="Insert one approved train identity case")
    seed.add_argument("--case-id", default="ac_train_identity_demo")

    sub.add_parser("list", help="List cases")
    query = sub.add_parser("query-identity", help="Query identity_required advice")
    query.add_argument("--approve-case-id")

    args = parser.parse_args()
    if args.cmd == "seed-train-identity":
        progress = {
            "workflow": "return_resolution",
            "pending": ["identity_verification"],
            "blocked_by": "user_input",
            "guard_state": "identity_required",
            "eligible": None,
            "cancelled": False,
            "allowed_next_actions": ["ask_user:verification_code"],
        }
        case = AgentCase(
            case_id=args.case_id,
            split="train",
            training_approved=False,
            user_goal="demo seed without credentials",
            progress_before=progress,
            allowed_actions=["ask_user:verification_code"],
            chosen_action={
                "action_type": "final_answer",
                "requires_user_response": True,
                "requested_input_type": "verification_code",
                "content": "please provide verification code",
            },
            terminal_state={"illegal_state_change": False},
            success=True,
            failure_owner="none",
            reusable_pattern="identity verification must complete first",
            avoid_pattern="handoff|final_answer",
            workflow="return_resolution",
            progress_signature=progress_signature_from_progress(progress),
        )
        stored, admission = write_candidate(case, db_path=args.db, approve=True)
        print(json.dumps({"stored": stored.case_id, "admission": admission}, ensure_ascii=False, indent=2))
        return
    if args.cmd == "list":
        rows = [case.to_dict() for case in list_cases(db_path=args.db)]
        print(json.dumps({"count": len(rows), "cases": rows}, ensure_ascii=False, indent=2))
        return
    if args.cmd == "query-identity":
        if args.approve_case_id:
            approve_case(args.approve_case_id, db_path=args.db)
        advice = build_memory_advice({
            "workflow": "return_resolution",
            "pending": ["identity_verification"],
            "blocked_by": "user_input",
            "guard_state": "identity_required",
            "allowed_next_actions": ["ask_user:verification_code"],
        }, db_path=args.db)
        print(json.dumps(advice.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
