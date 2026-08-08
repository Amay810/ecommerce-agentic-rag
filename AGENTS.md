# AGENTS.md

## Cursor Cloud specific instructions

Python project (Python 3.12 in this environment). Dependencies live in a virtualenv at
`.venv` (created by the startup update script from `requirements-dev.txt`). Run tools via
`.venv/bin/python` / `.venv/bin/pytest` (or activate with `source .venv/bin/activate`).

### Services / entry points

- **Supported runtime = the harness CLI**: `python -m ecommerce_rag.harness {run,replay,compare}`.
  This is the real product (Agent v2). See `README.md` "Quick start" / `docs/reproduction.md`
  for the exact commands.
- `--policy rule` and `--policy oracle` are **fully deterministic and need no network or API
  key** — use these for smoke runs and CI-style checks. The documented smoke command
  (`--policy rule ... --seed-db`) works out of the box.
- `--policy llm` / `evidence_verify*` need a real model: either an OpenAI-compatible endpoint
  (`ERAG_LLM_API_KEY` / `ERAG_LLM_BASE_URL` / `ERAG_LLM_MODEL`, see `.env.example`) or a local
  HF model (`ARAG_AGENT_BACKEND=local`, downloads Qwen3-4B). Not required for basic dev/testing.

### Tests / lint / build

- Tests: `.venv/bin/python -m pytest -q` (281 tests, ~2s, no network needed).
- No linter/formatter is configured (no ruff/flake8/pre-commit/CI in the repo).
- There is no build step; it is a Python package run in place.

### Non-obvious caveats

- **Building the retrieval index** (`python -m ecommerce_rag.data_loader`, or `harness ... --index`)
  downloads a sentence-transformers embedding model from HuggingFace, so it **requires network**.
  Index files are written to `ecommerce_rag/index/` (gitignored). The deterministic harness smoke
  does NOT need the index; only retrieval / LLM-policy flows do.
- **`ecommerce_rag/app.py` (legacy Streamlit demo) does not run out of the box** and is explicitly
  labeled legacy in the README. It references `config.RETRIEVAL_MIN_SCORE` (not defined in
  `config.py`) and uses package-relative imports that break `streamlit run ecommerce_rag/app.py`.
  Do not treat it as the entry point — use the harness instead.
