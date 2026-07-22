# Final failure analysis

## 1. Original 28-query top-rank regression

A new regression index reproduced the NSCC result locally, ruling out random
generation or scheduler variance.

| Version | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---:|---:|---:|---:|
| Previous baseline | 0.944 | 0.981 | 1.000 | 1.000 |
| Fresh index before fix | 0.889 | 0.981 | 1.000 | 0.963 |
| Local complete run after fix | 0.944 | 1.000 | 1.000 | 1.000 |

Root causes:

- “养猫家庭” had the correct product at dense rank 1, but a generic vacuum won
  the nearly tied BM25+dense RRF score. Increasing the natural-language dense
  tie-break from 0.02 to 0.03 restores the semantically specific product.
- “Air Pro 2 vs QuietMax H900” retrieved both named products, but the compound
  merge only guaranteed top-k inclusion. Explicit title-matched entities now
  precede generic fused distractors.
- Multi-gold comparisons naturally cap Recall@1 at 0.5 even when both products
  occupy ranks 1 and 2; this is a metric property, not a complete miss.

The fix is locally verified and requires one NSCC confirmation run.

## 2. Rule Policy wrong-tool failures

The nine failures are three deterministic policy tasks repeated three times:

- v2_locked_policy_13: logistics rules;
- v2_locked_policy_15: refund rules;
- v2_locked_policy_18: logistics rules.

The old order checked “物流” as an order signal and “退款” as a return signal
before checking “规则/规定”. The new order prioritises explicit policy language
when no personal order id is present. A new 30-query routing holdout passes
30/30. The old locked score remains in the report because it was inspected
during debugging.

## 3. Hard retrieval

The v2 locked set is below target at Recall@5=0.8029. The new v3 set is lower at
0.6889 and confirms the failure rather than hiding it.

| v3 kind | Recall@5 | Interpretation |
|---|---:|---|
| Multi-constraint | 1.000 | deterministic filtering works |
| Near-SKU multi-gold | 1.000 | one relevant sibling is consistently retrieved |
| Attribute without title | 0.663 | field/value lexical evidence is underweighted |
| Alias/typo | 0.250 | no fuzzy title or spelling-recovery channel |
| No-answer | 0.000 | retriever always returns nearest neighbours |

A single dense threshold is rejected: it improves no-answer accuracy but causes
large answerable recall loss. Future work should tune on dev only and combine
field-aware exact lookup, typo candidate generation, dense/BM25 agreement and
score margins. Any new tuned result needs a fresh holdout.

## 4. RL gate

The gate fails because no real LLMPolicy trajectories or human audits exist.
Rule and Oracle traces intentionally cannot pass it. This is the correct
fail-closed decision.

