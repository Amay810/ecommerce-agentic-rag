from __future__ import annotations

import copy
import json

import pytest

from ecommerce_rag.agent_runtime import (
    AgentRuntime,
    RuntimeConfig,
    RuntimeDecisionError,
    RuntimeGeneration,
)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_order_details",
            "description": "Read an order.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    }
]


def test_runtime_preserves_audit_history_and_records_compaction():
    large = {"order_id": "O1", "status": "pending", "unused": "x" * 1000}
    history = [
        {"role": "user", "content": "check O1"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_order_details",
                        "arguments": '{"order_id":"O1"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(large)},
    ]
    original = copy.deepcopy(history)
    captured = {}

    def generate(messages, tools):
        captured["messages"] = messages
        captured["tools"] = tools
        return RuntimeGeneration(content="The order is pending.", prompt_tokens=10)

    runtime = AgentRuntime(RuntimeConfig(compact_context=True))
    turn = runtime.run_turn(
        history=history,
        domain_policy="Use the retail tools.",
        tools=TOOLS,
        generate=generate,
    )
    assert history == original
    assert captured["messages"][0]["role"] == "system"
    assert len(captured["messages"][-1]["content"]) < len(json.dumps(large))
    assert turn.trace["context_compaction"]["reduction_ratio"] > 0
    assert turn.generation.content == "The order is pending."


def test_runtime_retries_invalid_parallel_call_then_succeeds():
    calls = 0

    def generate(_messages, _tools):
        nonlocal calls
        calls += 1
        if calls == 1:
            tool_call = {
                "function": {"name": "get_order_details", "arguments": "{}"}
            }
            return RuntimeGeneration(tool_calls=[tool_call, tool_call])
        return RuntimeGeneration(content="Please provide the order id.")

    turn = AgentRuntime(RuntimeConfig(max_generation_retries=1)).run_turn(
        history=[{"role": "user", "content": "check my order"}],
        domain_policy="Use tools.",
        tools=TOOLS,
        generate=generate,
    )
    assert calls == 2
    assert turn.trace["attempts"][0]["stage"] == "parallel_tool_calls"
    assert turn.trace["attempts"][1]["ok"] is True


def test_runtime_rejects_unoffered_tool():
    with pytest.raises(RuntimeDecisionError, match="not offered"):
        AgentRuntime(RuntimeConfig(max_generation_retries=0)).run_turn(
            history=[{"role": "user", "content": "do it"}],
            domain_policy="Use tools.",
            tools=TOOLS,
            generate=lambda _m, _t: RuntimeGeneration(
                tool_calls=[
                    {"function": {"name": "delete_everything", "arguments": "{}"}}
                ]
            ),
        )
