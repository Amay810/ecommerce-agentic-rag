# Retrieval scale and ablation

All rows use the same 250-query set generated from a category-stratified
40-product subset (20 Electronics + 20 Home & Kitchen), so the
gold products remain present as distractors are added. The 200 attribute questions
have programmatic gold IDs. The 50 hard cases are generated candidates and remain
explicitly marked `needs_human_review`; they must be reviewed before publication.

| Products | Chunks | Recall@1 | Recall@5 | MRR | nDCG@5 | P50 ms | P95 ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 40 | 365 | 0.963 | 0.985 | 0.989 | 0.980 | 23.9 | 41.8 |
| 1,000 | 8,770 | 0.958 | 0.973 | 0.982 | 0.972 | 111.1 | 175.6 |
| 5,000 | 43,953 | 0.950 | 0.965 | 0.975 | 0.965 | 634.3 | 1,063.6 |

At 5,000 products, deterministic price filtering raises Recall@5 from 0.965 to
0.979 and nDCG@5 from 0.965 to 0.976, while P95 stays near 1,060ms.
It is therefore enabled for constrained queries. The new 5k reranker run is left
to the supplied A100 job; prior 40-product evidence showed no net reranker gain,
so reranking remains disabled by default until the scale run proves otherwise.

Raw machine-readable reports are `retrieval_40.json`, `retrieval_1000.json`,
`retrieval_5000.json` and `retrieval_5000_constraints.json`.
