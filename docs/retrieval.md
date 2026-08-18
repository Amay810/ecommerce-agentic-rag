# Retrieval as an Agent capability

Hybrid retrieval supplies product and policy facts when the Agent selects a knowledge action. It combines multilingual dense retrieval, BM25 and reciprocal-rank fusion, then applies structured constraints such as category and budget.

## Build a local index

The repository ships a small product and policy corpus. Build the files consumed
by `HybridRetriever` with:

```bash
python -m scripts.build_retrieval_index --output-dir ecommerce_rag/index
```

The command writes `embeddings.npy`, `chunks.jsonl`, and `parents.json`. The
embedding model is controlled by `ERAG_EMBED_MODEL`; use `--products` and
`--policies` to point at another corpus with the same JSONL source fields. Use
`--dry-run` to inspect the resulting chunk counts without loading an embedding
model.

## Results

| Protocol | Recall@1 | Recall@5 | nDCG@5 | P95 |
|---|---:|---:|---:|---:|
| 5k hybrid scale set | 0.9542 | 0.9917 | 0.9826 | 125.73 ms |
| Hybrid + constraints | 0.9542 | 0.9917 | 0.9826 | 108.37 ms |
| + BGE reranker | 0.9750 | 0.9958 | 0.9958 | 219.12 ms |

The scale-set and constrained latency numbers come from different recorded runs and are not interchangeable. BGE improves first-rank quality but roughly doubles P95, so it is optional.

The independent v3 difficult set reached Recall@5=0.6889; typo/alias recall was 0.25 and no-answer accuracy was 0. These limitations prevent the 5k scale result from being presented as general fuzzy-query performance.

Dataset construction and timing details are recorded in [the scale summary](retrieval_scale_summary.md). Full intermediate reports remain in the raw tag.
