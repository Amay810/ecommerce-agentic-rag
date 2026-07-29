# Terminal-grounding v2 blinded-audit closeout

## Decision

The preregistered terminal-grounding v2 experiment is closed with status
`negative_or_inconclusive`. It did not meet the positive-result gate and must not
be rerun, retuned, or reinterpreted as a quality improvement.

The frozen Qwen3-4B-Instruct-2507 run completed all 80 dev trajectories with
`max_new_tokens=1024`. Both shadow and terminal-grounded structural gates passed,
including action, handoff, tool-call, terminal-state, and smoke-to-dev
configuration immutability. Of the 80 trajectories, 61 were eligible and received
new terminal answers; 19 were passed through unchanged.

## Blinded review result

The frozen 40-task selection produced 80 shuffled answers with variant and pairing
hidden. A dedicated Codex review context saw only the blinded review file and
labelled factual correctness, completeness, and contradictions from the supplied
user goal, evidence, and answer. This is a Codex audit, not an external or
independent human annotation study.

| Metric | Base | Terminal-grounded | Paired change |
|---|---:|---:|---:|
| Fact pass | 34/40 (85.0%) | 34/40 (85.0%) | 0.0 pp |
| Answer complete | 32/40 (80.0%) | 33/40 (82.5%) | +2.5 pp |
| Contradiction present | 6/40 (15.0%) | 6/40 (15.0%) | 0.0 pp |

The fact-pass discordant pairs were one base-only pass and one grounded-only pass.
The seeded 10,000-sample paired-bootstrap 95% interval for the fact-pass change was
[-7.5 pp, +7.5 pp]. There were no `unclear` labels. One new contradiction appeared
in the grounded variant, while one base contradiction was removed, leaving the
aggregate count unchanged.

## Cost and coverage

All 61 eligible generations were non-empty and changed from the frozen draft.
Incremental generation averaged 1,148.66 prompt tokens, 136.44 completion tokens,
and 4,975.66 ms. P95 values were 2,651 prompt tokens, 357 completion tokens, and
12,963.77 ms.

Verifier output remained diagnostic-only and was excluded from selection,
labelling, and the primary decision.

## Preregistered checks

| Check | Result |
|---|---|
| Positive fact-pass difference | Fail |
| Bootstrap 95% lower bound above zero | Fail |
| Completeness drop no worse than 5 pp | Pass |
| Total contradictions do not increase | Pass |
| Trajectory immutability | Pass |

Because the two primary factual-improvement checks failed, the overall positive
gate failed. Per protocol, this closes terminal-grounding work without a v3,
prompt or token retuning, further verifier work, canonical-product expansion,
external benchmark, SFT, or DPO.

## Frozen evidence

- Aggregate: `docs/answer_postprocess_blind_audit_v1_aggregate.json`
- Blinded review package: `docs/answer_postprocess_blind_audit_v1/`
- Dev reports and gates: `docs/answer_postprocess_dev_v2_*`
- Frozen sidecars: `logs/answer_postprocess_dev_v2_*`
- Frozen base store SHA-256:
  `087f37c655e5fb234a072e6e5294c7b071fcd3f35d6e910bab11ba6b53d5173f`
