# Reproduction

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements-dev.txt
```

## Deterministic smoke

```powershell
python -m ecommerce_rag.harness run `
  --tasks ecommerce_rag/data/harness_smoke.jsonl `
  --db logs/demo_agent.db `
  --store logs/demo_trajectories.sqlite `
  --output logs/demo_report.json `
  --policy rule `
  --repeats 1 `
  --seed-db
```

## Real LLM policy

Copy `.env.example`, configure either the local or OpenAI-compatible backend, and run the same command with `--policy llm`. Model, decoding configuration and task split must be recorded with every report.

## Replay

```powershell
python -m ecommerce_rag.harness replay `
  --store logs/demo_trajectories.sqlite `
  --trajectory-id TRAJECTORY_ID
```

## Tests

```powershell
python -m pytest tests -q
```

NSCC job files reproduce the historical fixed experiments and are indexed in `nscc/README.md`. Raw v2 artifacts are intentionally excluded from the release tree and are addressable by path and SHA-256 through `docs/release_manifest_agent_v2.json`.
