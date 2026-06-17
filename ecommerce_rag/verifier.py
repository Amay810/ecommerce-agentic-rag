# -*- coding: utf-8 -*-
"""Answer verification before returning a customer-facing response."""

import json
import re

from . import config, llm

SENT_SPLIT = re.compile(r"(?<=[。！？?\n])")
TAG = re.compile(r"\[资料\d+\]")
NUMERIC_FACT = re.compile(r"\d+(?:\.\d+)?\s*(?:元|ml|mAh|Pa|kg|g|小时|分钟|天|年|%)", re.I)


def split_sentences(text: str) -> list[str]:
    out = []
    for raw in SENT_SPLIT.split(text):
        sent = TAG.sub("", raw).strip(" \n\t-•")
        if len(sent) >= 6:
            out.append(sent)
    return out


class GroundingChecker:
    def __init__(self, embed_model):
        self.model = embed_model

    def check(self, answer: str, evidence_texts: list[str]) -> dict:
        sentences = split_sentences(answer)
        if not sentences or not evidence_texts:
            return {"ratio": 0.0, "per_sentence": [], "unsupported": sentences}
        sent_emb = self.model.encode(sentences, normalize_embeddings=True, convert_to_numpy=True)
        ev_emb = self.model.encode(evidence_texts, normalize_embeddings=True, convert_to_numpy=True)
        sims = sent_emb @ ev_emb.T
        rows, unsupported, grounded = [], [], 0
        for sent, row in zip(sentences, sims):
            score = float(row.max())
            ok = score >= config.GROUNDING_SENT_THRESHOLD
            grounded += int(ok)
            rows.append({"sentence": sent, "score": score, "grounded": ok})
            if not ok:
                unsupported.append(sent)
        return {"ratio": grounded / len(sentences), "per_sentence": rows, "unsupported": unsupported}


def citation_check(answer: str) -> dict:
    # 必须在去掉 [资料N] 标记之前判断引用，否则 TAG.search 永远命中不到。
    missing = []
    for raw in SENT_SPLIT.split(answer):
        sent = raw.strip()
        clean = TAG.sub("", sent).strip(" \n\t-•")
        if len(clean) < 6:
            continue
        if NUMERIC_FACT.search(clean) and not TAG.search(sent):
            missing.append(clean)
    return {"ok": not missing, "missing": missing}


def consistency_check(answer: str, context: str) -> dict:
    system = (
        "你是电商客服事实核查员。判断【回答】是否与【资料】一致。"
        "只输出 JSON：{\"verdict\":\"一致|矛盾|资料外\",\"problems\":[\"...\"]}。"
        "矛盾=回答和资料冲突；资料外=回答把资料未提及内容当作事实。"
    )
    user = f"【资料】\n{context}\n\n【回答】\n{answer}"
    try:
        raw = llm.complete(system, user, temperature=0.0, max_tokens=300)
    except llm.LLMError:
        return {"verdict": "跳过", "problems": [], "raw": ""}
    try:
        data = json.loads(raw)
        return {"verdict": data.get("verdict", "未知"), "problems": data.get("problems", []), "raw": raw}
    except json.JSONDecodeError:
        for verdict in ("矛盾", "资料外", "一致"):
            if verdict in raw:
                return {"verdict": verdict, "problems": [], "raw": raw}
    return {"verdict": "未知", "problems": [], "raw": raw}
