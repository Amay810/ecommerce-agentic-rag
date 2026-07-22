"""OpenAI-compatible and local-Qwen next-action policy.

The policy receives only ``AgentObservation``. It never loads task files or gold
labels, which makes it safe to evaluate on the locked split.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import asdict
from typing import Any, Callable

from .domain import AgentAction, AgentObservation


SYSTEM_PROMPT = """You are a retail support next-action policy.
Return exactly one JSON object and no prose:
{"action_type":"tool_call|final_answer|handoff","tool_name":null,"arguments":{},"content":"","requires_user_response":false}
Use only a listed tool. Ask the user for missing order id, six-digit verification code,
or explicit return confirmation. Never invent tool results, identity data, or policy facts.
After a successful read tool, answer from its result. On identity/ownership failure, hand off.
"""


class LLMPolicy:
    privileged = False

    def __init__(self, generate: Callable[[str], str], *, max_parse_retries: int = 1):
        self.generate = generate
        self.max_parse_retries = max_parse_retries
        self.retry_count = 0

    @classmethod
    def from_env(cls) -> "LLMPolicy":
        backend = os.getenv("ARAG_AGENT_BACKEND", "openai").lower()
        if backend == "local":
            return cls(cls._local_generator(os.getenv("ARAG_LOCAL_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")))
        return cls(cls._openai_generator(
            os.getenv("ARAG_LLM_BASE_URL", "https://api.openai.com/v1"),
            os.getenv("ARAG_LLM_API_KEY", ""),
            os.getenv("ARAG_LLM_MODEL", "gpt-4o-mini"),
        ))

    @staticmethod
    def _openai_generator(base_url: str, api_key: str, model: str) -> Callable[[str], str]:
        if not api_key:
            raise RuntimeError("ARAG_LLM_API_KEY is required for the openai backend")
        endpoint = base_url.rstrip("/")
        if not endpoint.endswith("/chat/completions"):
            endpoint += "/chat/completions"

        def generate(prompt: str) -> str:
            body = json.dumps({"model": model, "temperature": 0, "messages": [
                {"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}
            ]}).encode("utf-8")
            request = urllib.request.Request(endpoint, data=body, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return payload["choices"][0]["message"]["content"]
        return generate

    @staticmethod
    def _local_generator(model_name: str) -> Callable[[str], str]:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", torch_dtype="auto",
            trust_remote_code=True, local_files_only=True,
        )
        model.eval()

        def generate(prompt: str) -> str:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
            # Qwen3 supports an explicit non-thinking mode, which is preferable
            # for a short machine-parseable next-action JSON. Older templates
            # safely ignore this through the fallback.
            try:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            output = model.generate(**inputs, max_new_tokens=256, do_sample=False)
            return tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return generate

    @staticmethod
    def _parse(raw: str, allowed_tools: set[str]) -> AgentAction:
        match = re.search(r"\{.*\}", raw, flags=re.S)
        if not match:
            raise ValueError("no JSON object in model output")
        value = json.loads(match.group(0))
        action_type = value.get("action_type")
        if action_type not in {"tool_call", "final_answer", "handoff"}:
            raise ValueError(f"unknown action_type: {action_type}")
        tool_name = value.get("tool_name")
        if action_type == "tool_call" and tool_name not in allowed_tools:
            raise ValueError(f"unknown tool: {tool_name}")
        arguments = value.get("arguments") or {}
        if not isinstance(arguments, dict):
            raise ValueError("arguments must be an object")
        return AgentAction(action_type, tool_name, arguments, str(value.get("content", "")), bool(value.get("requires_user_response", False)))

    def act(self, observation: AgentObservation) -> AgentAction:
        public = asdict(observation)
        prompt = "POLICY-VISIBLE OBSERVATION:\n" + json.dumps(public, ensure_ascii=False)
        allowed = {x["name"] for x in observation.tool_schemas}
        error = ""
        for attempt in range(self.max_parse_retries + 1):
            raw = self.generate(prompt + (f"\nPrevious output error: {error}. Return corrected JSON." if error else ""))
            try:
                return self._parse(raw, allowed)
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                error = str(exc)
                if attempt < self.max_parse_retries:
                    self.retry_count += 1
        return AgentAction.handoff("model_action_parse_failure")
