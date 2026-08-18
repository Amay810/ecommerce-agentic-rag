"""τ³ Agent Adapter backed by the project's unified AgentRuntime.

This module is imported by the project-owned τ³ launcher. It registers a custom
agent without changing the pinned τ³ checkout, Retail tools, database, user
simulator, orchestrator, or official evaluator.
"""

from __future__ import annotations

import json
import os
from typing import Any

from .agent_runtime import AgentRuntime, RuntimeConfig, RuntimeGeneration
from .context_compaction import context_compaction_enabled


def _require_tau2():
    try:
        from tau2.agent.base.llm_config import LLMConfigMixin
        from tau2.agent.base_agent import HalfDuplexAgent, is_valid_agent_history_message
        from tau2.data_model.message import (
            AssistantMessage,
            MultiToolMessage,
            SystemMessage,
            ToolCall,
            ToolMessage,
            UserMessage,
        )
        from tau2.registry import registry
        from tau2.utils.llm_utils import generate
    except ImportError as exc:  # pragma: no cover - exercised in τ³ environment
        raise RuntimeError(
            "tau2 v1.0.1 must be importable to use EcommerceTau3Agent"
        ) from exc
    return {
        "LLMConfigMixin": LLMConfigMixin,
        "HalfDuplexAgent": HalfDuplexAgent,
        "is_valid_agent_history_message": is_valid_agent_history_message,
        "AssistantMessage": AssistantMessage,
        "MultiToolMessage": MultiToolMessage,
        "SystemMessage": SystemMessage,
        "ToolCall": ToolCall,
        "ToolMessage": ToolMessage,
        "UserMessage": UserMessage,
        "registry": registry,
        "generate": generate,
    }


def _message_to_wire(message: Any) -> dict[str, Any]:
    role = message.role
    if role == "assistant":
        calls = []
        for call in message.tool_calls or []:
            calls.append(
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments, ensure_ascii=False),
                    },
                }
            )
        return {"role": role, "content": message.content or "", "tool_calls": calls}
    if role == "tool":
        return {
            "role": role,
            "content": message.content or "",
            "tool_call_id": message.id,
        }
    return {"role": role, "content": message.content or ""}


def _wire_to_tau_messages(messages: list[dict[str, Any]], tau: dict[str, Any]) -> list[Any]:
    converted = []
    for message in messages:
        role = message["role"]
        if role == "system":
            converted.append(tau["SystemMessage"](role=role, content=message["content"]))
        elif role == "user":
            converted.append(tau["UserMessage"](role=role, content=message["content"]))
        elif role == "assistant":
            calls = []
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                raw = function.get("arguments") or "{}"
                arguments = json.loads(raw) if isinstance(raw, str) else raw
                calls.append(
                    tau["ToolCall"](
                        id=call.get("id", ""),
                        name=function.get("name", ""),
                        arguments=arguments,
                        requestor="assistant",
                    )
                )
            converted.append(
                tau["AssistantMessage"](
                    role=role,
                    content=message.get("content") or None,
                    tool_calls=calls or None,
                )
            )
        elif role == "tool":
            converted.append(
                tau["ToolMessage"](
                    id=message.get("tool_call_id", ""),
                    role=role,
                    content=message.get("content") or "",
                    requestor="assistant",
                )
            )
        else:
            raise ValueError(f"unsupported provider role: {role}")
    return converted


def build_tau3_agent_class(tau: dict[str, Any] | None = None):
    """Build the adapter class after τ³ is available on sys.path."""

    tau = tau or _require_tau2()
    Base = tau["HalfDuplexAgent"]

    class EcommerceTau3Agent(Base):
        """Half-duplex τ³ agent using the project's frozen native runtime."""

        def __init__(self, tools, domain_policy, llm, llm_args=None):
            super().__init__(tools=tools, domain_policy=domain_policy)
            self.llm = llm
            self.llm_args = dict(llm_args or {})
            self.runtime = AgentRuntime(
                RuntimeConfig(
                    runtime_version=os.getenv("ERAG_RUNTIME_VERSION", "system-v1"),
                    prompt_version=os.getenv(
                        "ERAG_PROMPT_VERSION", "ecommerce-native-v1"
                    ),
                    compact_context=context_compaction_enabled(default=False),
                    max_generation_retries=int(
                        os.getenv("ERAG_MAX_GENERATION_RETRIES", "1")
                    ),
                )
            )

        def get_init_state(self, message_history=None):
            history = list(message_history or [])
            assert all(tau["is_valid_agent_history_message"](m) for m in history)
            return {"messages": history, "runtime_traces": []}

        def _generate(self, messages, _tool_schemas):
            response = tau["generate"](
                model=self.llm,
                tools=self.tools,
                messages=_wire_to_tau_messages(messages, tau),
                call_name="ecommerce_native_agent_response",
                tool_choice="auto",
                parallel_tool_calls=False,
                **self.llm_args,
            )
            calls = []
            for call in response.tool_calls or []:
                calls.append(
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.arguments, ensure_ascii=False),
                        },
                    }
                )
            return RuntimeGeneration(
                content=response.content or "",
                tool_calls=calls,
                finish_reason=(response.raw_data or {}).get("finish_reason"),
                prompt_tokens=(response.usage or {}).get("prompt_tokens"),
                completion_tokens=(response.usage or {}).get("completion_tokens"),
                cost=response.cost,
                raw_message=response.raw_data or {},
            )

        def generate_next_message(self, message, state):
            if isinstance(message, tau["MultiToolMessage"]):
                state["messages"].extend(message.tool_messages)
            else:
                state["messages"].append(message)
            history = [_message_to_wire(item) for item in state["messages"]]
            schemas = [tool.openai_schema for tool in self.tools]
            turn = self.runtime.run_turn(
                history=history,
                domain_policy=self.domain_policy,
                tools=schemas,
                generate=self._generate,
            )
            generation = turn.generation
            tool_calls = []
            for call in generation.tool_calls:
                function = call.get("function") or {}
                raw = function.get("arguments") or "{}"
                arguments = json.loads(raw) if isinstance(raw, str) else raw
                tool_calls.append(
                    tau["ToolCall"](
                        id=call.get("id", ""),
                        name=function.get("name", ""),
                        arguments=arguments,
                        requestor="assistant",
                    )
                )
            successful_attempt = turn.trace["attempts"][-1]
            response = tau["AssistantMessage"](
                role="assistant",
                content=generation.content or None,
                tool_calls=tool_calls or None,
                usage={
                    "prompt_tokens": generation.prompt_tokens or 0,
                    "completion_tokens": generation.completion_tokens or 0,
                },
                cost=generation.cost,
                raw_data={
                    "finish_reason": generation.finish_reason,
                    "ecommerce_runtime": turn.trace,
                },
                generation_time_seconds=successful_attempt.get("generation_seconds"),
            )
            state["messages"].append(response)
            state["runtime_traces"].append(turn.trace)
            return response, state

    EcommerceTau3Agent.__name__ = "EcommerceTau3Agent"
    return EcommerceTau3Agent


def create_ecommerce_tau3_agent(tools, domain_policy, **kwargs):
    """Registry factory compatible with the pinned τ³ v1.0.1 contract."""

    cls = build_tau3_agent_class()
    return cls(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
    )


def register_tau3_agent(name: str = "ecommerce_native") -> None:
    """Register the project Agent Adapter in the live τ³ registry."""

    tau = _require_tau2()
    registry = tau["registry"]
    if registry.get_agent_factory(name) is None:
        registry.register_agent_factory(create_ecommerce_tau3_agent, name)
