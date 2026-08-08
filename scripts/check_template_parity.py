"""Check that training-time and serving-time prompts have identical token ids.

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
import subprocess
import sys
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_TAU2_ROOT = Path("E:/cv_codex/external/tau2-bench")

# Mirrors tau2-bench src/tau2/agent/llm_agent.py SYSTEM_PROMPT. Kept as a literal so
# a drift in the upstream template shows up as a parity failure rather than silently
# tracking whatever the installed tau2 happens to render.
TAU2_SYSTEM_PROMPT = "<instructions>\n{agent_instruction}\n</instructions>\n<policy>\n{domain_policy}\n</policy>"


def load_retail_tools(
    tau2_root: Path, tau2_python: Path | None = None
) -> list[dict[str, Any]]:
    """Return OpenAI-format tool schemas for the retail domain.

    Prefers live introspection of the pinned tau2 tree so the schemas match what
    litellm actually sends. Falls back to a two-tool stub when tau2 is not importable,
    which is enough to catch structural template drift but not schema-ordering drift.
    """
    if tau2_python is not None:
        code = (
            "import json,sys; "
            "sys.path.insert(0, sys.argv[1] + '/src'); "
            "from tau2.environment.toolkit import get_tool_signatures; "
            "from tau2.registry import registry; "
            "env=registry.get_env_constructor('retail')(); "
            "s=get_tool_signatures(env.tools); "
            "print(json.dumps([{'type':'function','function':x.model_dump(exclude_none=True)} "
            "for x in s.values()]))"
        )
        completed = subprocess.run(
            [str(tau2_python), "-c", code, str(tau2_root)],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return json.loads(completed.stdout)

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


def render_serving(model: str, messages: list[dict], tools: list[dict]) -> tuple[str, list[int]]:
    """Serving-side rendering: exactly what vLLM does with the OpenAI `tools` field."""
    from transformers import AutoTokenizer  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    text = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=False,
    )
    token_ids = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=True,
        add_generation_prompt=False,
    )
    return text, list(token_ids)


def to_swift_agent_messages(messages: list[dict]) -> list[dict]:
    """Convert OpenAI serving messages to ms-swift v4 agent dataset roles."""
    converted: list[dict] = []
    for message in messages:
        role = message["role"]
        content = message.get("content") or ""
        if role == "assistant" and message.get("tool_calls"):
            if content:
                converted.append({"role": "assistant", "content": content})
            for call in message["tool_calls"]:
                function = call["function"]
                arguments = function.get("arguments") or {}
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                converted.append(
                    {
                        "role": "tool_call",
                        "content": json.dumps(
                            {"name": function["name"], "arguments": arguments},
                            ensure_ascii=False,
                        ),
                    }
                )
        elif role == "tool":
            converted.append({"role": "tool_response", "content": content})
        else:
            converted.append({"role": role, "content": content})
    return converted


def render_training(
    model: str, messages: list[dict], tools: list[dict]
) -> tuple[str, list[int], list[tuple[str, bool]]]:
    """Training-side rendering via ms-swift, plus per-span loss coverage.

    Returns the decoded prompt and a list of (text, is_trained) spans derived from
    the label mask, so the §5 masking rule can be checked rather than assumed.

    The imports and call shape below follow ms-swift v4.2.2 at the commit pinned in
    verified_ecommerce_agent_learning_v2_sources.json.
    """
    from swift import get_processor, get_template  # noqa: PLC0415

    processor = get_processor(model)
    template = get_template(processor, agent_template="hermes")
    template.set_mode("train")

    encoded = template.encode(
        {
            "messages": to_swift_agent_messages(messages),
            "tools": json.dumps(tools, ensure_ascii=False),
        }
    )
    input_ids = list(encoded["input_ids"])
    labels = encoded.get("labels")

    text = processor.decode(input_ids)
    if labels is None:
        return text, input_ids, []

    spans: list[tuple[str, bool]] = []
    current: list[int] = []
    current_trained = labels[0] != -100
    for token_id, label in zip(input_ids, labels):
        trained = label != -100
        if trained != current_trained:
            spans.append((processor.decode(current), current_trained))
            current, current_trained = [], trained
        current.append(token_id)
    if current:
        spans.append((processor.decode(current), current_trained))
    return text, input_ids, spans


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tau2-root", type=Path, default=DEFAULT_TAU2_ROOT)
    parser.add_argument(
        "--tau2-python",
        type=Path,
        help="Use the pinned tau2 environment to read the real Retail tool schemas.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("docs/template_parity"))
    parser.add_argument(
        "--dump-only",
        action="store_true",
        help="Only render the serving side. Use before ms-swift is installed.",
    )
    args = parser.parse_args()

    tools = load_retail_tools(args.tau2_root, args.tau2_python)
    messages = build_fixture_messages()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    serving, serving_ids = render_serving(args.model, messages, tools)
    (args.out_dir / "serving.txt").write_text(serving, encoding="utf-8")
    print(f"serving render: {len(serving)} chars -> {args.out_dir / 'serving.txt'}")

    if args.dump_only:
        print("dump-only: skipping ms-swift comparison")
        return 0

    training, training_ids, spans = render_training(args.model, messages, tools)
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

    if serving_ids == training_ids:
        print("\nPARITY OK: training and serving token ids are identical")
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
