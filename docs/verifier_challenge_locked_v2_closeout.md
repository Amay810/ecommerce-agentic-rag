# Verifier challenge locked_v2 closeout

`verifier_challenge_locked_v2` is retained as a constructed verifier regression
fixture. It is not an independent human-labelled holdout: the review CSV exposes
the generated `gold_*` columns, and its `human_*` columns were prefilled from those
labels before confirmation. Confirming those rows would therefore remain vulnerable
to anchoring and would not create independent ground truth.

The artifact frozen at commit `9ffc62d` remains unchanged. Its 150 review statuses
remain `assistant_prefilled_pending_user_confirmation`; the one-time formal locked
evaluator is not run, no admission report is produced, and no `locked_v3` is created.

Answer-only grounding experiments treat verifier output as diagnostic shadow data:

- verifier output does not block smoke or dev runs;
- verifier output does not select, repair, reject, or hand off an answer;
- verifier output is excluded from the primary success decision;
- factual success is decided by the preregistered blinded paired human audit.

This protocol supersedes the verifier-admission dependency for answer-postprocessing
jobs while preserving the frozen challenge for deterministic regression tests.
