# Honest Evaluation: Reranker A/B

This project treats retrieval changes as deployable only when they pass gold-document evaluation. The reranker experiment below was run offline on NSCC with the same 40-product corpus and 28-query gold set.

## Three-Way Result

| Configuration | recall@1 | recall@3 | recall@5 | MRR |
|---|---:|---:|---:|---:|
| Hybrid baseline | 0.907 | 0.981 | 0.981 | 0.981 |
| + rerank child chunks | 0.907 | 0.981 | 0.981 | 0.975 |
| + rerank parent cards, no product dedup | 0.907 | 0.907 | 0.907 | 0.963 |
| + rerank parent cards + product-level dedup | 0.907 | 0.981 | 0.981 | 0.981 |

Data sources: `rerank_final.log` and `nscc/run_rerank_final.pbs`.

## What Failed

1. Child-chunk reranking lost parent context.
   For the query "养猫家庭适合买哪款清洁产品？", the correct product `P005` was rank 1 in the hybrid baseline but dropped to rank 3 after child-chunk reranking. The relevant evidence about pet hair lived in a different product chunk, so the cross-encoder saw an incomplete view of the product.

2. Parent-card reranking without product-level dedup hurt diversity.
   Scoring parent cards for every candidate chunk caused sibling chunks from the same product to flood the ranked list. This reduced recall@3/5 from 0.981 to 0.907.

3. Parent-card reranking with product-level dedup fixed the diversity issue, but only matched the baseline.
   The final version recovered recall@3/5 and MRR to the hybrid baseline level, but did not improve them.

## Decision

The reranker is not enabled in the current demo. The first-stage hybrid retriever already has limited headroom (`recall@5 = 0.981`), and the reranker adds latency without a measured net gain. This is an evaluation-driven decision rather than a claim that rerankers are generally ineffective.

## Real Bottlenecks

- Price constraints: the query "预算600以内，通勤降噪耳机" can still rank an 899 yuan product (`P009`) above the correct product. This should be handled with metadata filtering or constraint-aware scoring, not a generic reranker.
- First-stage candidate recall: for the thermos-cup versus food-jar query, the correct product `P006` was not recalled. Reranking cannot fix candidates that never enter the pool.

## Boundary

This is a local demo with offline NSCC evaluation. It has not been deployed as a production service and should not be described as serving real users.

## Interview Takeaway

The point of the experiment is not "reranker improves RAG." The stronger lesson is:

> Rerankers are not a silver bullet. In parent-child RAG, reranking granularity must match the document structure, and deployment decisions should be driven by gold-set metrics plus case analysis.
