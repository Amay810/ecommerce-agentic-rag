"""Tokenizer-side tool channel aligned to G0-E Chat Completions + hermes.

This module does not change tau2's action space. Hermes extraction matches
vLLM's ``hermes_tool_parser`` / VERL ``HermesToolParser``: only
``<tool_call>{"name", "arguments"}</tool_call>`` becomes a tool call.
The gym action remains ``{"name", "arguments"}`` for unchanged
``parse_action_string``.
"""

from __future__ import annotations

import json
import re
from typing import Any

_HERMES_TOOL_CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_GYM_TURN_PREFIX = re.compile(r"^(user|tool|assistant): ", re.MULTILINE)


def hermes_function_calls(text: str) -> list[tuple[str, Any]]:
    """Return ``(name, arguments)`` pairs from a hermes-formatted completion.

    Completions that only contain OpenAI function-call shells are ignored,
    matching G0-E's vLLM hermes parser.
    """
    if "<tool_call>" not in text or "</tool_call>" not in text:
        return []
    calls: list[tuple[str, Any]] = []
    for match in _HERMES_TOOL_CALL.findall(text):
        try:
            payload = json.loads(match)
            calls.append((payload["name"], payload["arguments"]))
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    return calls


def gym_action_from_function_call(name: str, arguments: Any) -> str:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            pass
    return json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)


def gym_action_from_completion(text: str) -> str:
    """Map a raw assistant completion to a tau2 gym action string.

    First hermes tool call wins (same as Tau3AgentLoop taking ``tool_calls[0]``).
    If hermes extracts nothing, the raw text is the action and gym treats it
    as a user-facing message.
    """
    calls = hermes_function_calls(text)
    if not calls:
        return text
    name, arguments = calls[0]
    return gym_action_from_function_call(name, arguments)


def assistant_message_from_function_call(
    name: str, arguments: Any, *, call_id: str = "call_0"
) -> dict[str, Any]:
    if isinstance(arguments, str):
        arguments_str = arguments
    else:
        arguments_str = json.dumps(arguments, ensure_ascii=False)
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments_str},
            }
        ],
    }


def initial_agent_messages(system_prompt: str, observation: str) -> list[dict[str, str]]:
    """Build the first chat-template messages. Tools are not concatenated here."""
    opening = messages_from_gym_observation(observation)
    if not opening:
        opening = [{"role": "user", "content": observation}]
    return [{"role": "system", "content": system_prompt}, *opening]


def messages_from_gym_observation(observation: str) -> list[dict[str, str]]:
    """Split a gym observation into Qwen chat roles.

    AgentGymEnv formats turns as ``role: content``. Tool results must stay
    ``role=tool`` so the Qwen template emits ``<tool_response>`` instead of
    wrapping them as a user utterance.
    """
    text = observation or ""
    if not text.strip():
        return []
    matches = list(_GYM_TURN_PREFIX.finditer(text))
    if not matches:
        return [{"role": "user", "content": text}]
    messages: list[dict[str, str]] = []
    for index, match in enumerate(matches):
        role = match.group(1)
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        content = text[start:end]
        if content.endswith("\n"):
            content = content[:-1]
        if role == "assistant":
            messages.append({"role": "user", "content": f"assistant: {content}"})
        else:
            messages.append({"role": role, "content": content})
    return messages
