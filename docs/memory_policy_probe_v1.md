# memory_policy_probe_v1

Frozen causal probe for Policy Memory. Formal `40/40` dev is untouched.

```text
Data Flywheel MVP v1.1
status: engineering_loop_complete
policy_gain: unproven   ← this protocol is the sole test of that claim
default: off
```

## Question

Under identical TaskProgress + Action Constraint, does Memory advice make the
**raw** Policy choose the preferred legal action more often?

## Freeze

| Item | Value |
|---|---|
| Protocol | `memory_policy_probe_v1` |
| Probe tasks | 24 (4 states × 6), SHA-256 in module |
| Train cases | 16 curated decision seeds (4 × 4), SHA-256 in module |
| Model | Qwen3-4B-Instruct-2507 |
| Constraint | on for both arms |
| Memory | off vs on (only difference) |
| Retrieval | SQL exact/layered only |
| Splits | no `dev` / `locked` |

## Arms

| Config | Memory | Action Constraint |
|---|---:|---:|
| `memory_off` | off | on |
| `memory_on` | on | on |

Off arm still computes offline preferred actions from the frozen Memory DB for
scoring; advice is **not** injected into the Policy observation.

## Metrics

```text
retrieval coverage
raw policy action in allowlist
raw policy matches memory preferred action
constraint remap count
terminal success / illegal state change
```

Core evidence is raw-Policy preferred match ↑ and constraint remap ↓ — not
terminal success alone (Constraint can rescue both arms).

## Preregistered verdicts

### Positive

- ≥4 off-arm raw Policy errors
- Memory repairs ≥ half of them
- no `off correct → on wrong` regressions
- constraint remap count decreases
- terminal success does not drop
- illegal state change = 0

### Neutral / underpowered

- off arm has <4 raw errors → `Memory effect not identifiable`
- at most one probe expansion; if still underpowered, Memory is unnecessary here

### Negative / inconclusive

- no raw-Policy gain, remap not down, or regressions
→ keep AgentCase Store; leave runtime Memory **default off**;
  do **not** “rescue” with semantic retrieval or LoRA

## Run

```powershell
python -m scripts.run_memory_policy_probe_v1 `
  --output-dir logs/memory_policy_probe_v1 `
  --expected-code-commit <HEAD> `
  --case-db logs/memory_policy_probe_v1/cases.db `
  --seed-train
```

## Next only if Positive

SQL coverage → cross-phrasing generalization → then semantic retrieval.
LoRA only after repeated real Policy errors across states.
