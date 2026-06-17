# -*- coding: utf-8 -*-
"""Query rewriting and HyDE fallback."""

from . import llm

REWRITE_SYSTEM = (
    "你是电商检索 query 改写器。把用户问题改写成一句独立、清晰、适合检索商品或政策资料的中文查询。"
    "需要消解“它/这个/上面那款”等指代，但不要编造商品名。只输出改写后的查询。"
)

HYDE_SYSTEM = (
    "你是电商客服。针对用户问题写一段 1-2 句、像商品资料或政策说明中会出现的假设性答案，"
    "用于检索匹配，不要求一定准确。只输出这段文字。"
)


def rewrite_query(query: str, history: list[str] | None = None) -> str:
    ctx = ("\n对话历史：" + " / ".join(history[-4:])) if history else ""
    try:
        return llm.complete(REWRITE_SYSTEM, f"用户问题：{query}{ctx}", temperature=0.0, max_tokens=120) or query
    except llm.LLMError:
        return query


def hyde(query: str) -> str:
    try:
        return llm.complete(HYDE_SYSTEM, f"用户问题：{query}", temperature=0.3, max_tokens=120) or query
    except llm.LLMError:
        return query
