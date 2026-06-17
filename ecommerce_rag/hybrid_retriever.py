# -*- coding: utf-8 -*-
"""Hybrid retrieval: dense semantic search + BM25 lexical search + RRF fusion."""

import json
import re
from pathlib import Path

from . import config


def tokenize_zh(text: str) -> list[str]:
    try:
        import jieba

        return [t.lower() for t in jieba.lcut(text) if t.strip()]
    except Exception:
        words = re.findall(r"[a-zA-Z0-9]+", text)
        chars = re.findall(r"[\u4e00-\u9fff]", text)
        return [t.lower() for t in words + chars if t.strip()]


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = config.RRF_K) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return scores


class HybridRetriever:
    def __init__(self, index_dir: Path = config.INDEX_DIR, embed_model: str = config.EMBED_MODEL):
        import numpy as np
        from rank_bm25 import BM25Okapi
        from sentence_transformers import SentenceTransformer

        self.embeddings = np.load(index_dir / "embeddings.npy")
        with open(index_dir / "chunks.jsonl", encoding="utf-8") as f:
            self.chunks = [json.loads(line) for line in f if line.strip()]
        with open(index_dir / "parents.json", encoding="utf-8") as f:
            self.parents = json.load(f)
        self.by_id = {c["chunk_id"]: c for c in self.chunks}
        self.bm25 = BM25Okapi([tokenize_zh(c["text"]) for c in self.chunks])
        self.model = SentenceTransformer(embed_model)
        # cross-encoder 重排器仅在开启时加载，保持纯 hybrid 路径轻量
        self.reranker = None
        if config.USE_RERANKER:
            from sentence_transformers import CrossEncoder

            self.reranker = CrossEncoder(config.RERANKER_MODEL)

    def _dense_scores(self, query: str):
        import numpy as np

        q = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
        return self.embeddings @ q  # 已归一化，等于余弦相似度，与 self.chunks 对齐

    def _dense_ranking(self, query: str, k: int) -> list[str]:
        import numpy as np

        order = np.argsort(-self._dense_scores(query))[:k]
        return [self.chunks[int(i)]["chunk_id"] for i in order]

    def _bm25_ranking(self, query: str, k: int) -> list[str]:
        import numpy as np

        scores = self.bm25.get_scores(tokenize_zh(query))
        order = np.argsort(-scores)[:k]
        return [self.chunks[int(i)]["chunk_id"] for i in order]

    def search(
        self,
        query: str,
        top_k: int = config.TOP_K,
        source_type: str | None = None,
        category: str | None = None,
    ) -> list[dict]:
        import numpy as np

        sims = self._dense_scores(query)
        dense = [self.chunks[int(i)]["chunk_id"] for i in np.argsort(-sims)[:config.DENSE_K]]
        sim_by_id = {c["chunk_id"]: float(sims[i]) for i, c in enumerate(self.chunks)}
        bm25 = self._bm25_ranking(query, config.BM25_K)
        fused = reciprocal_rank_fusion([dense, bm25])

        # 开启重排时多取候选（RERANK_CANDIDATES），精排后再截断到 top_k
        limit = config.RERANK_CANDIDATES if self.reranker else top_k
        candidates = []
        for cid, score in sorted(fused.items(), key=lambda x: -x[1]):
            chunk = dict(self.by_id[cid])
            if source_type and chunk.get("source_type") != source_type:
                continue
            if category and category not in chunk.get("category", ""):
                continue
            chunk["score"] = score
            chunk["dense_sim"] = sim_by_id.get(cid, 0.0)
            chunk["dense_rank"] = dense.index(cid) + 1 if cid in dense else None
            chunk["bm25_rank"] = bm25.index(cid) + 1 if cid in bm25 else None
            candidates.append(chunk)
            if len(candidates) >= limit:
                break

        if self.reranker and candidates:
            # 商品级去重：每个 doc 只保留融合分最高的 chunk，避免同一商品的兄弟 chunk
            # 在父卡重排下因得分相同而霸占 top_k、挤掉其他商品（破坏多样性）。
            if config.RERANK_DEDUP:
                seen, deduped = set(), []
                for c in candidates:  # candidates 已按融合分降序
                    if c["doc_id"] in seen:
                        continue
                    seen.add(c["doc_id"])
                    deduped.append(c)
                candidates = deduped

            # 默认对完整父卡精排（保留上下文）；chunk 模式仅看召回的子片段
            def rerank_text(c: dict) -> str:
                if config.RERANK_ON == "chunk":
                    return c["text"]
                return self.parents.get(c["doc_id"], c["text"])

            scores = self.reranker.predict([(query, rerank_text(c)) for c in candidates])
            for chunk, rs in zip(candidates, scores):
                chunk["rerank_score"] = float(rs)
            candidates.sort(key=lambda c: -c["rerank_score"])

        return candidates[:top_k]

    def format_context(self, chunks: list[dict]) -> str:
        seen, blocks = set(), []
        for c in chunks:
            doc_id = c["doc_id"]
            if doc_id in seen:
                continue
            seen.add(doc_id)
            blocks.append(f"[资料{len(blocks) + 1}]\n{self.parents[doc_id]}")
        return "\n\n".join(blocks)
