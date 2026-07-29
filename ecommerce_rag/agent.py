# -*- coding: utf-8 -*-
"""Agentic customer-support controller.

The agent routes the request, retrieves the right knowledge source, retries weak
retrieval with HyDE, decides whether to answer or hand off, and records a trace
for observability.
"""

import re

from . import catalog, case_memory, compound_query, config, freshness, intent, llm, price_filter, query_transform, support_case, telemetry, verifier
from .hybrid_retriever import reciprocal_rank_fusion


def assess_sufficiency(top_dense_sim: float) -> bool:
    # 用 dense 余弦相似度的绝对值判断证据是否可靠，而非 RRF 名次分。
    return top_dense_sim >= config.RETRIEVAL_MIN_DENSE_SIM


def final_decision(grounding_ratio: float, verdict: str, citation_ok: bool) -> dict:
    if verdict == "矛盾":
        return {"action": "handoff", "reason": "事实核查发现回答与资料矛盾。"}
    cautions = []
    if grounding_ratio < config.GROUNDING_MIN_RATIO:
        cautions.append(f"仅 {grounding_ratio:.0%} 的回答句子能在资料中找到支撑。")
    if verdict == "资料外":
        cautions.append("部分内容超出资料范围。")
    if not citation_ok:
        cautions.append("部分数字/参数缺少资料引用。")
    if cautions:
        return {"action": "caution", "reason": " ".join(cautions)}
    return {"action": "ok", "reason": "回答通过引用、grounding 和事实一致性检查。"}


class CustomerSupportAgent:
    """Legacy retrieval-oriented controller retained for regression coverage.

    Agent v2 orchestration lives in :mod:`ecommerce_rag.harness` with
    :class:`ecommerce_rag.llm_policy.LLMPolicy` and ``RetailTools``.
    """

    def __init__(self, retriever, enable_logging: bool = True):
        self.retriever = retriever
        self.grounder = verifier.GroundingChecker(retriever.model)
        self.enable_logging = enable_logging

    def _direct(self, query: str) -> str:
        try:
            return llm.complete("你是友好的电商客服助手。", query, temperature=0.5, max_tokens=120)
        except llm.LLMError:
            return "你好，我是智能客服。你可以问我商品参数、对比推荐、售后政策或物流发票问题。"

    def _handoff(self, query: str, reason: str, trace: list[str],
                 chunks: list[dict] | None = None, memory_hint: dict | None = None) -> dict:
        result = {
            "query": query,
            "display": f"{config.HANDOFF_MESSAGE}\n\n原因：{reason}",
            "trace": trace,
            "chunks": chunks or [],
            "action": "handoff",
            "intent": "handoff",
            "memory_hint": memory_hint,
        }
        self._log(result)
        return result

    def _generate(self, query: str, context: str, task: str) -> str:
        user = f"【任务类型】\n{task}\n\n【资料】\n{context}\n\n【用户问题】\n{query}\n\n请依据资料回答，并用 [资料N] 标注依据。"
        return llm.complete(config.SYSTEM_PROMPT, user, temperature=0.2, max_tokens=700)

    def _entity_title_match(self, entity: str) -> str | None:
        """Find a product doc_id whose title represents the given entity.

        Chinese entities (≥2 CJK chars): return the FIRST product in corpus order
        whose title contains any 2-char bigram of the entity. First-match gives
        JSONL-insertion-order priority — the "canonical" product (e.g. P006 保温旅行杯
        at ID=6) is found before more niche variants (e.g. P015 儿童吸管保温杯 at ID=15)
        even when the niche variant has a higher bigram score.

        English/mixed entities: score-based token overlap (Jaccard on lower-cased words)
        and return the highest-scoring product. Token overlap is used here because
        first-match would be unstable — common words like "Pro" might appear in many titles
        before reaching the intended product.

        Returns None when the entity is too short (< 2 chars) or no title matches.
        """
        entity = entity.strip()
        if len(entity) < 2:
            return None

        n_cjk = sum(1 for c in entity if "一" <= c <= "鿿")
        use_chinese = n_cjk >= 2  # requires ≥2 CJK chars; single chars are over-broad

        if use_chinese:
            bigrams = [entity[i : i + 2] for i in range(len(entity) - 1)]
            if not bigrams:
                return None
            seen: set[str] = set()
            for c in self.retriever.chunks:
                if c.get("source_type") != "product":
                    continue
                did = c["doc_id"]
                if did in seen:
                    continue
                seen.add(did)
                if any(bg in c.get("title", "") for bg in bigrams):
                    return did  # first match in corpus order
            return None

        else:
            # English/mixed: score-based so that multi-token names like "Air Pro 2"
            # find their product more reliably than pure first-match.
            e_tokens = set(re.findall(r"[A-Za-z0-9]+", entity.lower()))
            if not e_tokens:
                return None
            best_did: str | None = None
            best_score: float = 0.0
            seen_docs: set[str] = set()
            for c in self.retriever.chunks:
                if c.get("source_type") != "product":
                    continue
                did = c["doc_id"]
                if did in seen_docs:
                    continue
                seen_docs.add(did)
                t_tokens = set(re.findall(r"[A-Za-z0-9]+", c.get("title", "").lower()))
                score = len(e_tokens & t_tokens) / len(e_tokens)
                if score > best_score:
                    best_score = score
                    best_did = did
            return best_did if best_score > 0.0 else None

    def _retrieve_compound(
        self, query: str, sub_queries: list[str], route: intent.Intent, trace: list[str]
    ) -> tuple[list[dict], float]:
        """Compound decomposition: search each sub-entity + full query, RRF at doc level.

        Guarantee step: pre-compute which entities need injection, reserve those slots from
        the RRF budget upfront so the final result never exceeds TOP_K. Each entity's
        title-matched product is guaranteed to appear within the TOP_K window so that
        recall@k evaluation and context window usage stay bounded and consistent.
        """
        best_by_doc: dict[str, dict] = {}
        doc_rankings: list[list[str]] = []

        for sq in sub_queries + [query]:
            sq_chunks = self.retriever.search(sq, source_type=route.source_type)
            order: list[str] = []
            seen: set[str] = set()
            for c in sq_chunks:
                did = c["doc_id"]
                if did not in seen:
                    seen.add(did)
                    order.append(did)
                prev = best_by_doc.get(did)
                if prev is None or c.get("dense_sim", 0.0) > prev.get("dense_sim", 0.0):
                    best_by_doc[did] = c
            doc_rankings.append(order)
            trace.append(f"子查询「{sq[:20]}」: top={order[:3]}")

        fused = reciprocal_rank_fusion(doc_rankings)
        ranked = sorted(fused.items(), key=lambda x: -x[1])
        # Exact title matches for explicitly named comparison entities are the
        # primary evidence and must precede generic fused candidates.
        entity_docs: list[tuple[str, str]] = []
        seen_entity_docs: set[str] = set()
        for entity in sub_queries:
            did = self._entity_title_match(entity)
            if did and did not in seen_entity_docs:
                entity_docs.append((entity, did))
                seen_entity_docs.add(did)

        result: list[dict] = []
        in_result: set[str] = set()
        for entity, did in entity_docs:
            if did in best_by_doc:
                chunk = dict(best_by_doc[did])
            else:
                chunk = next(
                    (dict(c) for c in self.retriever.chunks if c["doc_id"] == did), None
                )
            if chunk:
                chunk["score"] = fused.get(did, 1e-4)
                chunk.setdefault("dense_sim", 0.0)
                result.append(chunk)
                in_result.add(did)
                trace.append(f"保底注入：{did}（实体='{entity}'，标题匹配）")

        for did, rrf_score in ranked:
            if did in in_result:
                continue
            chunk = dict(best_by_doc[did])
            chunk["score"] = rrf_score
            result.append(chunk)
            in_result.add(did)
            if len(result) >= config.TOP_K:
                break

        top_sim = max((c.get("dense_sim", 0.0) for c in result), default=0.0)
        trace.append(f"复合查询合并：{len(result)} 候选（≤TOP_K），实体={sub_queries}")
        return result, top_sim

    def _retrieve(self, query: str, route: intent.Intent, history: list[str] | None, trace: list[str]) -> tuple[list[dict], float]:
        # Compound query decomposition: only for compare intent.
        if config.COMPOUND_DECOMP_ENABLED and route.name == "compare":
            is_compound, sub_queries = compound_query.detect(query)
            if is_compound:
                trace.append(f"复合查询检测：实体={sub_queries}，触发分解检索")
                return self._retrieve_compound(query, sub_queries, route, trace)

        chunks: list[dict] = []
        top_sim = 0.0
        rewritten = query_transform.rewrite_query(query, history)
        for rnd in range(1, config.MAX_RETRIEVAL_ROUNDS + 1):
            # 第二轮基于改写后的独立 query 做 HyDE，避免丢失第一轮补回的上下文。
            search_q = rewritten if rnd == 1 else query_transform.hyde(rewritten)
            trace.append(f"第 {rnd} 轮检索：{search_q[:80]}")
            chunks = self.retriever.search(search_q, source_type=route.source_type)
            top_sim = max((c.get("dense_sim", 0.0) for c in chunks), default=0.0)
            if assess_sufficiency(top_sim):
                trace.append(f"检索充分：top_dense_sim={top_sim:.3f}")
                break
            trace.append(f"检索偏弱：top_dense_sim={top_sim:.3f}")
        # Post-retrieval price filter: dense embedding cannot enforce numerical constraints.
        if config.PRICE_FILTER_ENABLED:
            budget = price_filter.parse_budget(query)
            if budget is not None:
                chunks, note = price_filter.apply(chunks, budget)
                if note:
                    trace.append(note)
                    top_sim = max((c.get("dense_sim", 0.0) for c in chunks), default=0.0)
        return chunks, top_sim

    def _lookup_memory(self, query: str, intent_name: str, trace: list[str]) -> dict | None:
        if not case_memory.CASE_MEMORY_ENABLED:
            return None
        try:
            hint = case_memory.search(query, intent=intent_name, model=self.grounder.model)
            t = case_memory.hint_to_trace(hint)
            if t:
                trace.append(t)
            return {"matched": hint.matched, "top_sim": hint.top_sim,
                    "avoid_patterns": hint.avoid_patterns,
                    "suggested_action": hint.suggested_action,
                    "note": hint.note}
        except Exception:
            return None

    def run(self, query: str, history: list[str] | None = None) -> dict:
        trace: list[str] = []
        route = intent.route_intent(query)
        trace.append(f"意图路由：{route.name}（{route.reason}）")

        memory_hint = self._lookup_memory(query, route.name, trace)

        if route.name in ("empty", "chitchat"):
            result = {"query": query, "display": self._direct(query), "trace": trace, "chunks": [], "action": "direct", "intent": route.name, "memory_hint": memory_hint}
            self._log(result)
            return result
        if route.name == "handoff":
            return self._handoff(query, route.reason, trace, memory_hint=memory_hint)

        chunks, top_score = self._retrieve(query, route, history, trace)
        if not assess_sufficiency(top_score):
            trace.append("证据不足，触发人工兜底。")
            return self._handoff(query, "检索资料不足，不能可靠回答。", trace, chunks, memory_hint=memory_hint)

        context = self.retriever.format_context(chunks)
        if route.name == "recommend":
            answer = catalog.recommendation_brief(chunks)
        elif route.name == "compare":
            answer = catalog.comparison_brief(chunks)
        else:
            try:
                answer = self._generate(query, context, route.name)
            except llm.LLMError:
                answer = "已找到相关资料，但当前未配置大模型 API，无法生成完整自然语言回答。请查看下方来源或配置 ERAG_LLM_API_KEY。"

        grounding = self.grounder.check(answer, [c["text"] for c in chunks])
        citations = verifier.citation_check(answer, len({c.get("doc_id") for c in chunks}))
        consistency = verifier.consistency_check(answer, context) if config.LLM_API_KEY else {"verdict": "跳过", "problems": [], "raw": ""}
        decision = final_decision(grounding["ratio"], consistency["verdict"], citations["ok"])
        trace.append(
            f"验证：grounding={grounding['ratio']:.0%}，citation={citations['ok']}，"
            f"consistency={consistency['verdict']} -> {decision['action']}"
        )

        freshness_verdict = self._apply_freshness(route.name, answer, chunks, decision, trace)

        if decision["action"] == "handoff":
            display = f"{config.HANDOFF_MESSAGE}\n\n原因：{decision['reason']}"
        elif decision["action"] == "caution":
            display = f"提示：{decision['reason']}\n\n{answer}"
        else:
            display = answer

        result = {
            "query": query,
            "display": display,
            "answer": answer,
            "trace": trace,
            "chunks": chunks,
            "grounding": grounding,
            "citations": citations,
            "consistency": consistency,
            "freshness": freshness_verdict,
            "memory_hint": memory_hint,
            "action": decision["action"],
            "intent": route.name,
        }
        self._log(result)
        return result

    def _apply_freshness(self, intent_name, answer, chunks, decision, trace) -> dict | None:
        # Freshness guardrail: if the answer asserts price/inventory/policy that we cannot
        # confirm is current, hedge — downgrade ok->caution and append an advisory note.
        if not config.FRESHNESS_GUARD_ENABLED:
            return None
        snapshot = support_case.build_snapshot(chunks)
        verdict = freshness.assess(snapshot, intent_name, answer)
        if verdict["triggered"] and freshness.should_downgrade(verdict["status"]):
            note = freshness.note(verdict["status"], verdict["claims"])
            if decision["action"] == "ok":
                decision["action"] = "caution"
                decision["reason"] = note
            elif decision["action"] == "caution":
                decision["reason"] = f"{decision['reason']} {note}"
            trace.append(f"新鲜度护栏：{verdict['status']}（{'/'.join(verdict['claims'])}）-> {decision['action']}")
        return verdict

    def _persist_case(self, result: dict) -> None:
        # Persist the auditable SupportCase; never let a store failure break the response.
        if not config.SUPPORT_STORE_ENABLED:
            return
        try:
            from . import store
            from .support_case import SupportCase

            store.insert_case(SupportCase.from_agent_result(result))
        except Exception:
            pass

    def _log(self, result: dict) -> None:
        if not self.enable_logging:
            return
        self._persist_case(result)
        telemetry.log_event(
            {
                "query": result.get("query"),
                "intent": result.get("intent"),
                "action": result.get("action"),
                "trace": result.get("trace", []),
                "chunks": [
                    {
                        "chunk_id": c.get("chunk_id"),
                        "title": c.get("title"),
                        "source_type": c.get("source_type"),
                        "score": c.get("score"),
                    }
                    for c in result.get("chunks", [])
                ],
            }
        )
