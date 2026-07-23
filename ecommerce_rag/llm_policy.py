"""OpenAI-compatible and local next-action policy.

The policy receives only ``AgentObservation``. It never loads task files or gold
labels, which makes it safe to evaluate on the locked split.

Action protocol: **one JSON object per turn**, validated against the JSON Schema
in :mod:`ecommerce_rag.tool_schema`. The same schemas can later be handed to a
native tool-calling API or a constrained decoder without changing this contract.

Every generation is recorded in :attr:`LLMPolicy.last_trace` — raw output,
finish reason, token counts, and the exact stage at which parsing failed. The
first 360-trajectory run could not be diagnosed because none of that was kept.
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

from .domain import AgentAction, AgentObservation
from .tool_schema import ToolArgumentError, prompt_block, validate_arguments

SYSTEM_PROMPT = """You are a retail support next-action policy.

Return EXACTLY ONE JSON object and nothing else. No prose, no markdown fence, no explanation.

Schema:
{"action_type": "tool_call" | "final_answer" | "handoff",
 "tool_name": string or null,      // required when action_type is tool_call
 "arguments": object,              // tool arguments; {} for non-tool actions
 "content": string,                // message to the user; "" for tool_call
 "requires_user_response": boolean}

Available tools:
{tools}

Rules:
- Use only a listed tool, with exactly its declared arguments and types.
- Ask the user for a missing order id, six-digit verification code, or explicit
  return confirmation by returning final_answer with requires_user_response=true.
- Never invent tool results, identity data, or policy facts.
- After a successful read tool, answer from its result.
- On identity or ownership failure, hand off.
"""

#: Tool results can be large; keep the rendered history inside a budget so the
#: request does not crowd out the response window on a small local model.
MAX_HISTORY_CHARS = 6000
MAX_TOOL_RESULT_CHARS = 900


@dataclass
class Generation:
    """A single model call, with whatever the backend can tell us about it."""

    text: str
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    truncated: bool | None = None
    meta: dict[str, Any] = field(default_factory=dict)


class ActionParseError(ValueError):
    """Parse failure carrying the stage at which it happened."""

    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage


def _extract_json_object(raw: str) -> str:
    """Return the first balanced ``{...}`` block.

    A greedy ``\\{.*\\}`` spans from the first brace to the last one, so a model
    that emits two objects, or prose containing a brace, yields something that
    is not valid JSON and the real cause is hidden behind a decode error.
    """
    start = raw.find("{")
    if start < 0:
        raise ActionParseError("no_json_object", "no '{' in model output")
    depth, in_string, escaped = 0, False, False
    for index in range(start, len(raw)):
        char = raw[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return raw[start:index + 1]
    raise ActionParseError("unbalanced_json", "'{' is never closed — output was likely truncated")


def _render_history(history: list[dict[str, Any]]) -> str:
    lines = []
    for entry in history:
        role = entry.get("role", "?")
        if role == "tool":
            content = str(entry.get("content", ""))
            if len(content) > MAX_TOOL_RESULT_CHARS:
                content = content[:MAX_TOOL_RESULT_CHARS] + f"…[+{len(content) - MAX_TOOL_RESULT_CHARS} chars]"
            lines.append(f"[tool:{entry.get('name')}] {content}")
        else:
            lines.append(f"[{role}] {entry.get('content', '')}")
    rendered = "\n".join(lines)
    if len(rendered) > MAX_HISTORY_CHARS:  # keep the most recent turns
        rendered = "…[earlier turns omitted]\n" + rendered[-MAX_HISTORY_CHARS:]
    return rendered


class LLMPolicy:
    privileged = False

    def __init__(self, generate: Callable[[str, str], "str | Generation"], *, max_parse_retries: int = 1,
                 generator_meta: dict[str, Any] | None = None):
        self.generate = generate
        self.max_parse_retries = max_parse_retries
        self.generator_meta = generator_meta or {}
        self.retry_count = 0
        self.last_trace: dict[str, Any] = {}

    # ------------------------------------------------------------------ setup

    @classmethod
    def from_env(cls) -> "LLMPolicy":
        backend = os.getenv("ARAG_AGENT_BACKEND", "openai").lower()
        if backend == "local":
            model = os.getenv("ARAG_LOCAL_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
            generate, meta = cls._local_generator(model)
            return cls(generate, generator_meta=meta)
        base_url = os.getenv("ARAG_LLM_BASE_URL", "https://api.openai.com/v1")
        model = os.getenv("ARAG_LLM_MODEL", "gpt-4o-mini")
        return cls(cls._openai_generator(base_url, os.getenv("ARAG_LLM_API_KEY", ""), model),
                   generator_meta={"backend": "openai", "model": model, "base_url": base_url})

    @staticmethod
    def _openai_generator(base_url: str, api_key: str, model: str) -> Callable[[str, str], Generation]:
        if not api_key:
            raise RuntimeError("ARAG_LLM_API_KEY is required for the openai backend")
        endpoint = base_url.rstrip("/")
        if not endpoint.endswith("/chat/completions"):
            endpoint += "/chat/completions"

        def generate(system: str, user: str) -> Generation:
            body = json.dumps({"model": model, "temperature": 0, "messages": [
                {"role": "system", "content": system}, {"role": "user", "content": user}
            ]}).encode("utf-8")
            request = urllib.request.Request(
                endpoint, data=body,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = json.loads(response.read().decode("utf-8"))
            choice = payload["choices"][0]
            usage = payload.get("usage") or {}
            finish = choice.get("finish_reason")
            return Generation(
                text=choice["message"]["content"] or "",
                finish_reason=finish,
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                truncated=finish == "length",
            )
        return generate

    @staticmethod
    def _local_generator(model_name: str, max_new_tokens: int = 512):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", torch_dtype="auto",
            trust_remote_code=True, local_files_only=True,
        )
        model.eval()
        # Record which template path was taken; a mismatched chat template is one
        # of the few things that can silently break every single generation.
        thinking_supported = True
        try:
            tokenizer.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False,
                                          add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            thinking_supported = False

        meta = {"backend": "local", "model": model_name, "max_new_tokens": max_new_tokens,
                "enable_thinking_supported": thinking_supported,
                "chat_template_present": bool(getattr(tokenizer, "chat_template", None))}

        def generate(system: str, user: str) -> Generation:
            messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            kwargs = {"tokenize": False, "add_generation_prompt": True}
            if thinking_supported:
                kwargs["enable_thinking"] = False
            text = tokenizer.apply_chat_template(messages, **kwargs)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            generated = output[0][inputs.input_ids.shape[1]:]
            completion = int(generated.shape[0])
            return Generation(
                text=tokenizer.decode(generated, skip_special_tokens=True),
                finish_reason="length" if completion >= max_new_tokens else "stop",
                prompt_tokens=int(inputs.input_ids.shape[1]),
                completion_tokens=completion,
                truncated=completion >= max_new_tokens,
                meta={"rendered_prompt_chars": len(text)},
            )
        return generate, meta

    # ------------------------------------------------------------------ parse

    @staticmethod
    def _parse(raw: str, allowed_tools: set[str]) -> AgentAction:
        if not raw or not raw.strip():
            raise ActionParseError("empty_output", "model returned no text")
        block = _extract_json_object(raw)
        try:
            value = json.loads(block)
        except json.JSONDecodeError as exc:
            raise ActionParseError("json_decode_error", str(exc)) from exc
        if not isinstance(value, dict):
            raise ActionParseError("not_an_object", f"top level is {type(value).__name__}")

        action_type = value.get("action_type")
        if action_type not in {"tool_call", "final_answer", "handoff"}:
            raise ActionParseError("bad_action_type", f"unknown action_type: {action_type!r}")

        arguments = value.get("arguments") or {}
        if not isinstance(arguments, dict):
            raise ActionParseError("arguments_not_object", "arguments must be an object")

        tool_name = value.get("tool_name")
        if action_type == "tool_call":
            if not tool_name:
                raise ActionParseError("missing_tool_name", "tool_call without tool_name")
            if tool_name not in allowed_tools:
                raise ActionParseError("unknown_tool", f"tool not offered: {tool_name!r}")
            try:
                validate_arguments(tool_name, arguments)
            except ToolArgumentError as exc:
                raise ActionParseError("schema_violation", str(exc)) from exc

        return AgentAction(action_type, tool_name, arguments, str(value.get("content", "")),
                           bool(value.get("requires_user_response", False)))

    # -------------------------------------------------------------------- act

    def _build_prompt(self, observation: AgentObservation) -> str:
        return (
            f"CONVERSATION SO FAR (step {observation.step}):\n{_render_history(observation.history)}\n\n"
            f"CURRENT USER MESSAGE:\n{observation.current_message}\n\n"
            f"SESSION: {json.dumps(observation.session, ensure_ascii=False)}\n\n"
            "Return the next action as one JSON object."
        )

    def act(self, observation: AgentObservation) -> AgentAction:
        system = SYSTEM_PROMPT.replace("{tools}", prompt_block())
        user = self._build_prompt(observation)
        allowed = {schema["name"] for schema in observation.tool_schemas}
        attempts: list[dict[str, Any]] = []
        error = ""

        for attempt in range(self.max_parse_retries + 1):
            request = user if not error else (
                f"{user}\n\nYour previous output could not be parsed: {error}\n"
                "Return corrected JSON only.")
            result = self.generate(system, request)
            if isinstance(result, str):
                result = Generation(text=result)

            record: dict[str, Any] = {
                "attempt": attempt,
                "system_chars": len(system), "user_chars": len(request),
                "prompt_tokens": result.prompt_tokens, "completion_tokens": result.completion_tokens,
                "finish_reason": result.finish_reason, "truncated": result.truncated,
                "raw_output": result.text, "raw_output_chars": len(result.text or ""),
                **({"generation_meta": result.meta} if result.meta else {}),
            }
            try:
                action = self._parse(result.text, allowed)
            except ActionParseError as exc:
                record.update(parse_ok=False, parse_stage=exc.stage, parse_error=str(exc))
                attempts.append(record)
                error = f"[{exc.stage}] {exc}"
                if attempt < self.max_parse_retries:
                    self.retry_count += 1
                continue
            record.update(parse_ok=True, parse_stage=None, parse_error=None,
                          action_type=action.action_type, tool_name=action.tool_name)
            attempts.append(record)
            self.last_trace = {"resolution": "parsed", "attempts": attempts, "generator": self.generator_meta}
            return action

        self.last_trace = {"resolution": "fallback_handoff", "attempts": attempts,
                           "generator": self.generator_meta,
                           "final_stage": attempts[-1].get("parse_stage") if attempts else None}
        return AgentAction.handoff("model_action_parse_failure")
