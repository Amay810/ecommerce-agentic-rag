"""CPU-only wire from raw completion to gym action.

These tests cover the missing middle of the G0-E vs GRPO tool channel:
hermes-formatted model output must become ``{"name","arguments"}``;
OpenAI function-call shells must not. No GPU, no VERL, no network.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from ecommerce_rag.grpo.tool_channel import (
    gym_action_from_completion,
    hermes_function_calls,
    initial_agent_messages,
    messages_from_gym_observation,
)


HERMES_FIND_USER = (
    '<tool_call>\n'
    '{"name": "find_user_id_by_name_zip", '
    '"arguments": {"first_name": "Fatima", "last_name": "Johnson", "zip": "78712"}}'
    '\n</tool_call>'
)

OPENAI_SHELL = json.dumps(
    {
        "type": "function",
        "function": {
            "name": "find_user_id_by_name_zip",
            "arguments": {
                "first_name": "Fatima",
                "last_name": "Johnson",
                "zip": "78712",
            },
        },
    }
)

OPENAI_PARAMETERS_SHELL = json.dumps(
    {
        "function": {
            "name": "find_user_id_by_name_zip",
            "parameters": {"first_name": "Fatima", "last_name": "Johnson", "zip": "78712"},
        }
    }
)


def test_hermes_completion_becomes_gym_tool_json():
    action = gym_action_from_completion(HERMES_FIND_USER)
    payload = json.loads(action)
    assert payload == {
        "name": "find_user_id_by_name_zip",
        "arguments": {
            "first_name": "Fatima",
            "last_name": "Johnson",
            "zip": "78712",
        },
    }
    assert hermes_function_calls(HERMES_FIND_USER)


def test_openai_function_shell_is_not_a_tool_call():
    assert hermes_function_calls(OPENAI_SHELL) == []
    assert gym_action_from_completion(OPENAI_SHELL) == OPENAI_SHELL
    assert hermes_function_calls(OPENAI_PARAMETERS_SHELL) == []
    assert gym_action_from_completion(OPENAI_PARAMETERS_SHELL) == OPENAI_PARAMETERS_SHELL


def test_plain_assistant_text_stays_user_facing_chat():
    text = "Could you please provide your email address?"
    assert gym_action_from_completion(text) == text
    assert hermes_function_calls(text) == []


def test_first_hermes_call_wins_when_two_are_present():
    text = (
        HERMES_FIND_USER
        + '\n<tool_call>{"name": "get_user_details", "arguments": {"user_id": "x"}}</tool_call>'
    )
    payload = json.loads(gym_action_from_completion(text))
    assert payload["name"] == "find_user_id_by_name_zip"


def test_initial_messages_do_not_dump_tools_into_system_prompt():
    messages = initial_agent_messages(
        "# Retail agent policy\n\nAs a retail agent...",
        "user: Hi, I need to modify a pending order.",
    )
    assert messages[0] == {
        "role": "system",
        "content": "# Retail agent policy\n\nAs a retail agent...",
    }
    assert "# Available tools" not in messages[0]["content"]
    assert messages[1] == {
        "role": "user",
        "content": "Hi, I need to modify a pending order.",
    }


def test_gym_tool_observation_is_role_tool_not_wrapped_user():
    observation = 'tool: {"user_id": "fatima_johnson_7581"}'
    assert messages_from_gym_observation(observation) == [
        {"role": "tool", "content": '{"user_id": "fatima_johnson_7581"}'},
    ]


def test_gym_user_observation_strips_role_prefix():
    observation = "user: I actually don’t remember which email I used."
    assert messages_from_gym_observation(observation) == [
        {"role": "user", "content": "I actually don’t remember which email I used."},
    ]


def test_gym_tool_then_user_observation_keeps_both_roles():
    observation = (
        'tool: {"user_id": "fatima_johnson_7581"}\n'
        "user: Please go ahead and swap the item."
    )
    assert messages_from_gym_observation(observation) == [
        {"role": "tool", "content": '{"user_id": "fatima_johnson_7581"}'},
        {"role": "user", "content": "Please go ahead and swap the item."},
    ]


def _load_parse_action_string():
    root = Path(os.environ.get("TAU_ROOT", "/home/may/ecommerce-agentic-rag-archive"))
    direct = root / "src" / "tau2"
    nested = root / "vendor" / "tau2-bench-fc0055dc" / "src" / "tau2"
    source = direct if direct.is_dir() else nested
    if not source.is_dir():
        pytest.skip("vendored tau2 snapshot is not available")
    sys.path.insert(0, str(source.parent))
    from tau2.utils.tools import parse_action_string

    return parse_action_string


def test_hermes_gym_action_is_parse_action_string_tool_call():
    parse_action_string = _load_parse_action_string()
    message = parse_action_string(gym_action_from_completion(HERMES_FIND_USER))
    assert message.tool_calls is not None
    assert message.tool_calls[0].name == "find_user_id_by_name_zip"
    assert message.content is None


def test_openai_shell_stays_chat_under_parse_action_string():
    parse_action_string = _load_parse_action_string()
    message = parse_action_string(gym_action_from_completion(OPENAI_SHELL))
    assert message.tool_calls is None
    assert message.content == OPENAI_SHELL
