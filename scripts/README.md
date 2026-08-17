# Script index

The supported Agent entry point is `python -m ecommerce_rag.harness` for run, replay and compare.

The active scripts are limited to:

- τ³ Retail execution and frozen-judge launch;
- Phase 1 write-gate measurement (`scripts/run_phase1_write_gate.py`);
- template parity and context-compaction measurement;
- current source/split audits;
- reproducible catalogue and harness fixture generation;
- trajectory diagnostics.

Retired S0/hint, Memory, verifier, evidence-ablation, legacy correction, and
one-off smoke scripts are stored in the private research archive. New runtime
features belong in the Agent contracts rather than in post-processing scripts.
