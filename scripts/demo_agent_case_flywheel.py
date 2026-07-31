"""Minimal CLI demo for the AgentCase data-flywheel MVP.

Run from repo root:

    python -m scripts.demo_agent_case_flywheel --db logs/demo_cases.db seed-train-identity
    python -m scripts.demo_agent_case_flywheel --db logs/demo_cases.db query-identity
    python -m scripts.demo_agent_case_flywheel --db logs/demo_cases.db list
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ecommerce_rag.agent_case import AgentCase, progress_signature_from_progress, provenance_hash
from ecommerce_rag.agent_case_memory import approve_case, build_memory_advice, write_candidate
from ecommerce_rag.agent_case_store import list_cases


def main() -> None:
    parser = argparse.ArgumentParser(description="AgentCase flywheel MVP helpers")
    parser.add_argument("--db", type=Path, required=True)
    sub = parser.add_subparsers(dest="cmd", required=True)

    seed = sub.add_parser("seed-train-identity", help="Insert one approved train identity case")
    seed.add_argument("--case-id", default="ac_train_identity_demo")
    seed.add_argument("--approved-by", default="demo_operator")
    seed.add_argument(
        "--approval-reason",
        default="curated identity seed with paired replay for flywheel demo",
    )

    sub.add_parser("list", help="List cases")
    query = sub.add_parser("query-identity", help="Query identity_required advice")
    query.add_argument("--approve-case-id")
    query.add_argument("--approved-by", default="demo_operator")
    query.add_argument("--approval-reason", default="demo approval")

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
        action = {
            "action_type": "final_answer",
            "requires_user_response": True,
            "requested_input_type": "verification_code",
            "content": "please provide verification code",
        }
        source = {
            "attribution": "human_seed",
            "attribution_source": "demo_cli",
            "split": "train",
        }
        case = AgentCase(
            case_id=args.case_id,
            split="train",
            training_approved=False,
            user_goal="demo seed without credentials",
            progress_before=progress,
            allowed_actions=["ask_user:verification_code"],
            chosen_action=dict(action),
            executed_action=dict(action),
            raw_policy_action=dict(action),
            constrained_action=dict(action),
            constraint_remapped=False,
            policy_followed_advice=True,
            step=0,
            step_outcome="allowed:ask_user:verification_code",
            terminal_state={"illegal_state_change": False, "return_status": "requested"},
            terminal_outcome={"success": True, "illegal_state_change": False},
            success=True,
            failure_owner="none",
            causal_credit="seed",
            reusable_pattern="identity verification must complete first",
            avoid_pattern="handoff|final_answer",
            workflow="return_resolution",
            progress_signature=progress_signature_from_progress(progress),
            source=source,
            source_hash=provenance_hash(source, args.case_id, 0),
            paired_replay_result={"ok": True, "note": "demo_paired_replay"},
        )
        stored, admission = write_candidate(
            case,
            db_path=args.db,
            approve=True,
            approved_by=args.approved_by,
            approval_reason=args.approval_reason,
        )
        print(json.dumps({
            "stored": stored.case_id,
            "admission": admission,
            "created_at": stored.created_at,
            "source_hash": stored.source_hash,
        }, ensure_ascii=False, indent=2))
        return
    if args.cmd == "list":
        rows = [case.to_dict() for case in list_cases(db_path=args.db)]
        print(json.dumps({"count": len(rows), "cases": rows}, ensure_ascii=False, indent=2))
        return
    if args.cmd == "query-identity":
        if args.approve_case_id:
            approve_case(
                args.approve_case_id,
                db_path=args.db,
                approved_by=args.approved_by,
                approval_reason=args.approval_reason,
            )
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
