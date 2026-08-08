"""P0 gate: training-time and serving-time prompt rendering must be byte-identical.

Why this exists
---------------
tau2-bench passes tool schemas through the OpenAI-compatible `tools` parameter
(`src/tau2/utils/llm_utils.py:409-415`), so the served prompt string is produced by
vLLM applying the model's own chat template. ms-swift renders the same conversation
with its own template implementation at training time.

If the two renderings differ -- tool-schema JSON key order, whitespace around the
system block, `<think>` handling, tool-response role naming -- SFT optimises a
distribution the agent is never evaluated under. The usual symptom is "dev improves,
tau3 test drops", which is expensive to diagnose after the fact.

This script renders one representative multi-turn tool-calling conversation both
ways and fails loudly on any difference. It also reports which spans ms-swift
assigns loss to, which is the machine-checkable form of the plan's §5 masking rule
(assistant text + tool_call trained; system / user / tool_response masked).

Usage
-----
    python scripts/check_template_parity.py --model Qwen/Qwen3-4B-Instruct-2507
    python scripts/check_template_parity.py --dump-only     # no ms-swift needed
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_TAU2_ROOT = Path("E:/cv_codex/external/tau2-bench")

# Mirrors tau2-bench src/tau2/agent/llm_agent.py SYSTEM_PROMPT. Kept as a literal so
# a drift in the upstream template shows up as a parity failure rather than silently
# tracking whatever the installed tau2 happens to render.
TAU2_SYSTEM_PROMPT = "<instructions>\n{agent_instruction}\n</instructions>\n<policy>\n{domain_policy}\n</policy>"


def load_retail_tools(tau2_root: Path) -> list[dict[str, Any]]:
    """Return OpenAI-format tool schemas for the retail domain.

    Prefers live introspection of the pinned tau2 tree so the schemas match what
    litellm actually sends. Falls back to a two-tool stub when tau2 is not importable,
    which is enough to catch structural template drift but not schema-ordering drift.
    """
    try:
        sys.path.insert(0, str(tau2_root / "src"))
        from tau2.environment.toolkit import get_tool_signatures  # noqa: PLC0415
        from tau2.registry import registry  # noqa: PLC0415

        env = registry.get_env_constructor("retail")()
        signatures = get_tool_signatures(env.tools)
        return [
            {"type": "function", "function": sig.model_dump(exclude_none=True)}
            for sig in signatures.values()
        ]
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] could not import tau2 retail tools ({exc}); using stub schemas")
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_order_details",
                    "description": "Get the status and details of an order.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "order_id": {
                                "type": "string",
                                "description": "The order id, such as '#W0000000'.",
                            }
                        },
                        "required": ["order_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "cancel_pending_order",
                    "description": "Cancel a pending order.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "order_id": {"type": "string", "description": "The order id."},
                            "reason": {
                                "type": "string",
                                "enum": ["no longer needed", "ordered by mistake"],
                                "description": "The reason for cancellation.",
                            },
                        },
                        "required": ["order_id", "reason"],
                    },
                },
            },
        ]


def build_fixture_messages() -> list[dict[str, Any]]:
    """One conversation exercising every role the training format must cover."""
    system = TAU2_SYSTEM_PROMPT.format(
        agent_instruction="You are a customer service agent.",
        domain_policy="# Retail agent policy\n\nYou must authenticate the user first.",
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "Hi, I want to cancel my order."},
        {
            "role": "assistant",
            "content": "Sure. Could you give me the order id?",
        },
        {"role": "user", "content": "It is #W0000000."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {
                        "name": "get_order_details",
                        "arguments": json.dumps({"order_id": "#W0000000"}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_0",
            "content": json.dumps({"order_id": "#W0000000", "status": "pending"}),
        },
        {
            "role": "assistant",
            "content": "That order is still pending, so I can cancel it. Confirm?",
        },
    ]


def render_serving(model: str, messages: list[dict], tools: list[dict]) -> str:
    """Serving-side rendering: exactly what vLLM does with the OpenAI `tools` field."""
    from transformers import AutoTokenizer  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    return tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=False,
    )


def render_training(model: str, messages: list[dict], tools: list[dict]) -> tuple[str, list[tuple[str, bool]]]:
    """Training-side rendering via ms-swift, plus per-span loss coverage.

    Returns the decoded prompt and a list of (text, is_trained) spans derived from
    the label mask, so the §5 masking rule can be checked rather than assumed.

    NOTE: the ms-swift import surface must be re-verified against the pinned commit
    recorded in verified_ecommerce_agent_learning_v2_sources.json before this gate is
    treated as authoritative.
    """
    from swift.llm import InferRequest, get_model_tokenizer, get_template  # noqa: PLC0415

    _, tokenizer = get_model_tokenizer(model, load_model=False)
    template = get_template(tokenizer.model_meta.template, tokenizer)
    template.set_mode("train")

    encoded = template.encode(InferRequest(messages=messages, tools=tools))
    input_ids = encoded["input_ids"]
    labels = encoded.get("labels")

    text = tokenizer.decode(input_ids)
    if labels is None:
        return text, []

    spans: list[tuple[str, bool]] = []
    current: list[int] = []
    current_trained = labels[0] != -100
    for token_id, label in zip(input_ids, labels):
        trained = label != -100
        if trained != current_trained:
            spans.append((tokenizer.decode(current), current_trained))
            current, current_trained = [], trained
        current.append(token_id)
    if current:
        spans.append((tokenizer.decode(current), current_trained))
    return text, spans


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tau2-root", type=Path, default=DEFAULT_TAU2_ROOT)
    parser.add_argument("--out-dir", type=Path, default=Path("docs/template_parity"))
    parser.add_argument(
        "--dump-only",
        action="store_true",
        help="Only render the serving side. Use before ms-swift is installed.",
    )
    args = parser.parse_args()

    tools = load_retail_tools(args.tau2_root)
    messages = build_fixture_messages()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    serving = render_serving(args.model, messages, tools)
    (args.out_dir / "serving.txt").write_text(serving, encoding="utf-8")
    print(f"serving render: {len(serving)} chars -> {args.out_dir / 'serving.txt'}")

    if args.dump_only:
        print("dump-only: skipping ms-swift comparison")
        return 0

    training, spans = render_training(args.model, messages, tools)
    (args.out_dir / "training.txt").write_text(training, encoding="utf-8")
    print(f"training render: {len(training)} chars -> {args.out_dir / 'training.txt'}")

    if spans:
        print("\nloss coverage (ms-swift label mask):")
        for text, trained in spans:
            marker = "TRAIN" if trained else "  --  "
            preview = text.replace("\n", "\\n")[:88]
            print(f"  [{marker}] {preview}")
        (args.out_dir / "loss_spans.json").write_text(
            json.dumps([{"trained": t, "text": s} for s, t in spans], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    if serving == training:
        print("\nPARITY OK: training and serving renderings are byte-identical")
        return 0

    diff = difflib.unified_diff(
        serving.splitlines(keepends=True),
        training.splitlines(keepends=True),
        fromfile="serving(vllm)",
        tofile="training(ms-swift)",
    )
    diff_text = "".join(diff)
    (args.out_dir / "parity.diff").write_text(diff_text, encoding="utf-8")
    print("\nPARITY FAILED. Diff:")
    print(diff_text)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
