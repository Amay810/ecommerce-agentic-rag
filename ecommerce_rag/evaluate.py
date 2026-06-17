# -*- coding: utf-8 -*-
"""Evaluation harness for routing, retrieval, and optional RAGAS-style metrics."""

import argparse
import json
import re
from pathlib import Path

from . import config, llm


def doc_ranking(chunks: list[dict]) -> list[str]:
    """Return deduplicated parent-doc ranking from retrieved chunks."""
    seen, ranking = set(), []
    for chunk in chunks:
        doc_id = chunk.get("doc_id")
        if doc_id and doc_id not in seen:
            seen.add(doc_id)
            ranking.append(doc_id)
    return ranking


def recall_at_k(ranking: list[str], gold_doc_ids: list[str], k: int) -> float | None:
    if not gold_doc_ids:
        return None
    gold = set(gold_doc_ids)
    hits = len(gold.intersection(ranking[:k]))
    return hits / len(gold)


def reciprocal_rank(ranking: list[str], gold_doc_ids: list[str]) -> float | None:
    if not gold_doc_ids:
        return None
    gold = set(gold_doc_ids)
    for idx, doc_id in enumerate(ranking, 1):
        if doc_id in gold:
            return 1.0 / idx
    return 0.0


def _score_0_1(text: str) -> float:
    match = re.search(r"(0?\.\d+|[01](?:\.0+)?)", text)
    try:
        return max(0.0, min(1.0, float(match.group(1)))) if match else 0.0
    except ValueError:
        return 0.0


def faithfulness(answer: str, context: str) -> float:
    out = llm.complete(
        "你是评测器。判断【回答】中事实性内容有多少比例能被【资料】支持。只输出0到1的小数。",
        f"【资料】\n{context}\n\n【回答】\n{answer}",
        temperature=0.0,
        max_tokens=10,
    )
    return _score_0_1(out)


def answer_relevancy(question: str, answer: str) -> float:
    out = llm.complete(
        "你是评测器。判断【回答】对【问题】的相关性和切题程度。只输出0到1的小数。",
        f"【问题】{question}\n【回答】{answer}",
        temperature=0.0,
        max_tokens=10,
    )
    return _score_0_1(out)


def context_precision(question: str, contexts: list[str]) -> float:
    if not contexts:
        return 0.0
    hits = 0
    for context in contexts:
        out = llm.complete(
            "判断这段资料是否与问题相关。只输出 1（相关）或 0（不相关）。",
            f"【问题】{question}\n【资料】{context[:500]}",
            temperature=0.0,
            max_tokens=5,
        )
        hits += 1 if "1" in out else 0
    return hits / len(contexts)


def evaluate(testset_path: Path, agent, with_llm_metrics: bool = False) -> dict:
    rows = [json.loads(line) for line in open(testset_path, encoding="utf-8") if line.strip()]
    details = []
    route_ok = action_ok = retrieval_ok = 0
    metrics = {"faithfulness": [], "answer_relevancy": [], "context_precision": []}
    retrieval_metrics = {"recall@1": [], "recall@3": [], "recall@5": [], "mrr": []}

    for row in rows:
        question = row["question"]
        result = agent.run(question)
        expected_intent = row.get("expected_intent")
        expected_action = row.get("expected_action")
        expected_action_not = row.get("expected_action_not")
        is_route_ok = expected_intent is None or result.get("intent") == expected_intent
        is_action_ok = True
        if expected_action:
            is_action_ok = result.get("action") == expected_action
        if expected_action_not:
            is_action_ok = result.get("action") != expected_action_not
        is_retrieval_ok = result.get("action") in ("direct", "handoff") or bool(result.get("chunks"))
        ranking = doc_ranking(result.get("chunks", []))
        gold_doc_ids = row.get("gold_doc_ids", [])
        r1 = recall_at_k(ranking, gold_doc_ids, 1)
        r3 = recall_at_k(ranking, gold_doc_ids, 3)
        r5 = recall_at_k(ranking, gold_doc_ids, 5)
        rr = reciprocal_rank(ranking, gold_doc_ids)

        route_ok += int(is_route_ok)
        action_ok += int(is_action_ok)
        retrieval_ok += int(is_retrieval_ok)
        for key, value in (("recall@1", r1), ("recall@3", r3), ("recall@5", r5), ("mrr", rr)):
            if value is not None:
                retrieval_metrics[key].append(value)
        item = {
            "question": question,
            "intent": result.get("intent"),
            "action": result.get("action"),
            "route_ok": is_route_ok,
            "action_ok": is_action_ok,
            "retrieval_ok": is_retrieval_ok,
            "gold_doc_ids": gold_doc_ids,
            "retrieved_doc_ids": ranking,
            "recall@1": r1,
            "recall@3": r3,
            "recall@5": r5,
            "rr": rr,
        }

        if with_llm_metrics and result.get("answer") and result.get("chunks"):
            contexts = [c["text"] for c in result["chunks"]]
            context_block = "\n".join(contexts)
            scores = {
                "faithfulness": faithfulness(result["answer"], context_block),
                "answer_relevancy": answer_relevancy(question, result["answer"]),
                "context_precision": context_precision(question, contexts),
            }
            for key, value in scores.items():
                metrics[key].append(value)
            item.update(scores)
        details.append(item)

    n = len(rows) or 1
    summary = {
        "n": len(rows),
        "route_accuracy": round(route_ok / n, 3),
        "action_accuracy": round(action_ok / n, 3),
        "retrieval_coverage": round(retrieval_ok / n, 3),
        **{k: round(sum(v) / len(v), 3) if v else 0.0 for k, v in retrieval_metrics.items()},
    }
    if with_llm_metrics:
        summary.update({k: round(sum(v) / len(v), 3) if v else 0.0 for k, v in metrics.items()})
    return {"summary": summary, "details": details}


def main() -> None:
    from .agent import CustomerSupportAgent
    from .hybrid_retriever import HybridRetriever

    parser = argparse.ArgumentParser()
    parser.add_argument("--testset", default=str(config.DATA_DIR / "eval_questions.jsonl"))
    parser.add_argument("--with-llm-metrics", action="store_true")
    args = parser.parse_args()

    agent = CustomerSupportAgent(HybridRetriever(), enable_logging=False)
    report = evaluate(Path(args.testset), agent, with_llm_metrics=args.with_llm_metrics and bool(config.LLM_API_KEY))
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(json.dumps(report["details"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
