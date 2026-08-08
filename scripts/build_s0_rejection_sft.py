"""S0: build SFT data from environment-verified rollouts on the tau3 train split.

Position in the plan
--------------------
S0 sits *before* P1 (Retail Task Compiler). It needs no new components: the 74
official train tasks are already blueprints with verifiable terminal states, so
sampling the base policy repeatedly and keeping only the rollouts the official
grader scores 1.0 yields policy-compliant SFT data at roughly zero engineering cost.

Its job is to check the training path: can verified trajectories be exported and
learned without breaking tool use? A flat S0 curve does not veto the compiler.
The known ceiling -- 74 tasks is thin, and self-generated successes carry no signal
on tasks the base model never solves -- is exactly the argument for P1.

This script deliberately does not run rollouts. Generation and grading stay with the
official runner so that tool observations and rewards are unambiguously the
environment's:

    tau2 run --domain retail --task-set-name retail --task-split train \\
        --agent-llm <base> --user-llm <fixed-sim> --num-trials 8 \\
        --save-to logs/s0_train_rollouts.json

    python scripts/build_s0_rejection_sft.py \\
        --results logs/s0_train_rollouts.json \\
        --out data/s0_rejection_sft.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

DEFAULT_TAU2_ROOT = Path("E:/cv_codex/external/tau2-bench")

# Terminal states that mean the conversation ended the way the harness intended.
# Anything else (error cap, step cap) is discarded even at reward 1.0, because the
# trajectory shape would teach the model a truncated protocol.
ACCEPTED_TERMINATIONS = {"user_stop", "agent_stop"}


def reconstruct_system_prompt(tau2_root: Path, domain: str) -> str:
    """Rebuild the exact system prompt the agent was served.

    Results files carry the conversation but not the agent's system message, and the
    training prompt must match the served one (see scripts/check_template_parity.py).
    Importing tau2's own constants keeps the two in lockstep instead of duplicating
    the template text here.
    """
    sys.path.insert(0, str(tau2_root / "src"))
    from tau2.agent.llm_agent import AGENT_INSTRUCTION, SYSTEM_PROMPT  # noqa: PLC0415
    from tau2.registry import registry  # noqa: PLC0415

    env = registry.get_env_constructor(domain)()
    return SYSTEM_PROMPT.format(
        agent_instruction=AGENT_INSTRUCTION, domain_policy=env.get_policy()
    )


def load_tool_schemas(tau2_root: Path, domain: str) -> list[dict[str, Any]]:
    sys.path.insert(0, str(tau2_root / "src"))
    from tau2.environment.toolkit import get_tool_signatures  # noqa: PLC0415
    from tau2.registry import registry  # noqa: PLC0415

    env = registry.get_env_constructor(domain)()
    signatures = get_tool_signatures(env.tools)
    return [
        {"type": "function", "function": sig.model_dump(exclude_none=True)}
        for sig in signatures.values()
    ]


def tool_signature(messages: Iterable[dict]) -> str:
    """Structural fingerprint: the ordered sequence of tool names the agent called.

    Used for dedup. Two rollouts of the same task that take the same tool path teach
    the same lesson; keeping both just reweights the task.
    """
    names: list[str] = []
    for msg in messages:
        if msg.get("role") == "tool_call":
            payload = json.loads(msg.get("content") or "{}")
            if payload.get("name"):
                names.append(str(payload["name"]))
            continue
        if msg.get("role") != "assistant":
            continue
        for call in msg.get("tool_calls") or []:
            function = call.get("function") or call
            if function.get("name"):
                names.append(str(function["name"]))
    return "|".join(names)


def normalize_json_content(value: Any) -> str:
    """Return a JSON string as required for ms-swift tool_response content."""
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = value
    else:
        parsed = value
    return json.dumps(parsed, ensure_ascii=False)


def convert_messages(
    raw_messages: list[dict], allow_tool_errors: bool = False
) -> tuple[list[dict], str | None]:
    """Convert tau2 messages to OpenAI/ms-swift format.

    Returns (messages, rejection_reason).
    """
    out: list[dict] = []
    for msg in raw_messages:
        role = msg.get("role")

        if role == "user":
            if msg.get("tool_calls"):
                # Retail exposes no user tools; a user-requestor call means the
                # results file came from a different domain or a modified env.
                return [], "user_requestor_tool_call"
            out.append({"role": "user", "content": msg.get("content") or ""})

        elif role == "assistant":
            content = msg.get("content") or ""
            tool_calls = msg.get("tool_calls") or []
            if content:
                out.append({"role": "assistant", "content": content})
            if tool_calls:
                for tc in tool_calls:
                    function = tc.get("function") or tc
                    arguments = function.get("arguments") or {}
                    if isinstance(arguments, str):
                        arguments = json.loads(arguments)
                    out.append(
                        {
                            "role": "tool_call",
                            "content": json.dumps(
                                {
                                    "name": function["name"],
                                    "arguments": arguments,
                                },
                                ensure_ascii=False,
                            ),
                        }
                    )
            if not content and not tool_calls:
                out.append({"role": "assistant", "content": ""})

        elif role == "tool":
            if msg.get("error") and not allow_tool_errors:
                return [], "tool_error_in_trajectory"
            out.append(
                {
                    "role": "tool_response",
                    "content": normalize_json_content(msg.get("content")),
                }
            )

        elif role == "system":
            continue

        else:
            return [], f"unknown_role:{role}"

    return out, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True, help="Official runner output JSON")
    parser.add_argument("--out", type=Path, default=Path("data/s0_rejection_sft.jsonl"))
    parser.add_argument("--tau2-root", type=Path, default=DEFAULT_TAU2_ROOT)
    parser.add_argument("--domain", default="retail")
    parser.add_argument(
        "--max-per-task",
        type=int,
        default=2,
        help="Cap kept trajectories per task so easy tasks do not dominate the mix.",
    )
    parser.add_argument(
        "--allow-tool-errors",
        action="store_true",
        help="Keep trajectories containing failed tool calls (teaches recovery, adds noise).",
    )
    parser.add_argument(
        "--exclude-task-ids",
        nargs="*",
        default=(),
        help="Task ids reserved for S0 dev; their rollouts never enter training data.",
    )
    args = parser.parse_args()

    payload = json.loads(args.results.read_text(encoding="utf-8"))
    simulations = payload.get("simulations", [])
    if not simulations:
        raise SystemExit(f"No simulations found in {args.results}")

    system_prompt = reconstruct_system_prompt(args.tau2_root, args.domain)
    tools = load_tool_schemas(args.tau2_root, args.domain)

    rejections = Counter()
    kept_by_task: dict[str, list[dict]] = defaultdict(list)
    seen_signatures: set[str] = set()
    excluded_task_ids = set(args.exclude_task_ids)

    for sim in simulations:
        task_id = sim.get("task_id")
        if task_id in excluded_task_ids:
            rejections["reserved_for_dev"] += 1
            continue
        reward_info = sim.get("reward_info") or {}
        reward = reward_info.get("reward")

        if reward != 1.0:
            rejections["reward_not_1"] += 1
            continue

        # reward is a product over reward_basis, so a 1.0 already implies the DB check
        # passed. Assert it explicitly so a future change to reward_basis cannot let
        # non-terminal-verified trajectories through silently.
        db_check = reward_info.get("db_check")
        if db_check is not None and not db_check.get("db_match", True):
            rejections["db_check_failed"] += 1
            continue

        if sim.get("termination_reason") not in ACCEPTED_TERMINATIONS:
            rejections[f"termination:{sim.get('termination_reason')}"] += 1
            continue

        messages, reason = convert_messages(
            sim.get("messages") or [], allow_tool_errors=args.allow_tool_errors
        )
        if reason:
            rejections[reason] += 1
            continue

        if not any(m.get("role") == "tool_call" for m in messages):
            rejections["no_tool_calls"] += 1
            continue

        signature = f"{task_id}::{tool_signature(messages)}"
        if signature in seen_signatures:
            rejections["duplicate_tool_path"] += 1
            continue
        seen_signatures.add(signature)

        if len(kept_by_task[task_id]) >= args.max_per_task:
            rejections["per_task_cap"] += 1
            continue

        record = {
            "messages": [{"role": "system", "content": system_prompt}] + messages,
            # ms-swift v4 agent format requires a JSON string here. tool_call and
            # tool_response roles above are converted by --agent_template hermes;
            # user/tool responses remain masked from loss by the template.
            "tools": json.dumps(tools, ensure_ascii=False),
            "provenance": {
                "stage": "S0_rejection_sampling",
                "domain": args.domain,
                "task_id": task_id,
                "simulation_id": sim.get("id"),
                "trial": sim.get("trial"),
                "seed": sim.get("seed"),
                "reward": reward,
                "termination_reason": sim.get("termination_reason"),
                "source_results": str(args.results),
                "tool_path": tool_signature(messages),
                "tool_path_hash": hashlib.sha256(
                    tool_signature(messages).encode("utf-8")
                ).hexdigest()[:16],
            },
        }
        kept_by_task[task_id].append(record)

    records = [r for recs in kept_by_task.values() for r in recs]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest = {
        "source_results": str(args.results),
        "domain": args.domain,
        "total_simulations": len(simulations),
        "kept": len(records),
        "tasks_covered": len(kept_by_task),
        "max_per_task": args.max_per_task,
        "excluded_task_ids": sorted(excluded_task_ids),
        "rejections": dict(rejections.most_common()),
        "unique_tool_paths": len(seen_signatures),
    }
    manifest_path = args.out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(manifest, indent=2))
    print(f"\nwrote {len(records)} records -> {args.out}")
    print(f"manifest -> {manifest_path}")

    # Coverage against the train split matters more than raw count: tasks the base
    # policy never solves contribute nothing, and that blind spot is what P1 exists
    # to fill. Surface it here rather than discovering it after training.
    print(f"\ntrain tasks with at least one verified success: {len(kept_by_task)}")


if __name__ == "__main__":
    main()
