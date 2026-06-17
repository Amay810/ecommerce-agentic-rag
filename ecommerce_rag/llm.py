# -*- coding: utf-8 -*-
"""Minimal OpenAI-compatible chat client using only the Python stdlib."""

import json
import urllib.request

from . import config


class LLMError(RuntimeError):
    pass


def chat(
    messages: list[dict],
    temperature: float = 0.2,
    max_tokens: int = 800,
    model: str | None = None,
) -> str:
    if not config.LLM_API_KEY:
        raise LLMError("未设置 ERAG_LLM_API_KEY，无法调用大模型 API。")
    url = config.LLM_BASE_URL.rstrip("/") + "/chat/completions"
    payload = json.dumps(
        {
            "model": model or config.LLM_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Authorization": f"Bearer {config.LLM_API_KEY}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        raise LLMError(f"LLM 调用失败: {exc}") from exc


def complete(system: str, user: str, **kwargs) -> str:
    return chat([{"role": "system", "content": system}, {"role": "user", "content": user}], **kwargs)
