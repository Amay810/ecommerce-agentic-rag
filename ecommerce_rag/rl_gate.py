"""Fail-closed readiness gate for next-action SFT/DPO experiments."""

import argparse
import csv
import json
import sqlite3
from pathlib import Path


HIDDEN_KEYS = {"category", "gold_doc_ids", "allowed_tools", "forbidden_tools", "expected_state", "initial_state", "metadata"}

# A human verdict is only accepted as strict "true"/"false". Anything else —
# blank, "pending", "yes", a stray note — is not a judgement and must not be
# counted, otherwise a half-filled or mistyped sheet silently inflates the
# reviewed count and feeds garbage into the agreement rate.
VALID_VERDICTS = {"true", "false"}

# Both columns are required by docs/HUMAN_AUDIT_GUIDE.md, so a row is only
# adjudicated once the reviewer has ruled on task success *and* policy compliance.
VERDICT_COLUMNS = ("human_success", "human_policy_compliant")


def _verdict(row, column):
    """Return 'true'/'false' if the cell holds a strict boolean, else None."""
    value = (row.get(column) or "").strip().lower()
    return value if value in VALID_VERDICTS else None


def assess(tasks, store, audit=None, preference_pairs=None):
    specs = [json.loads(x) for x in Path(tasks).read_text(encoding="utf-8").splitlines() if x.strip()]
    conn = sqlite3.connect(store)
    try:
        db_rows = conn.execute("SELECT trajectory_json,grade_json FROM trajectories").fetchall()
    finally:
        conn.close()
    trajectories = [(json.loads(t), json.loads(g)) for t, g in db_rows]
    llm_rows = [(t, g) for t, g in trajectories if t.get("policy_name") == "LLMPolicy"]
    leakage_safe = bool(llm_rows) and all(not (set(obs) & HIDDEN_KEYS) for t, _ in llm_rows for obs in t.get("observations", []))
    deterministic_grades = bool(llm_rows) and all(isinstance(g.get("success"), bool) and isinstance(g.get("terminal_state_match"), bool) for _, g in llm_rows)
    success_rate = sum(g["success"] for _, g in llm_rows) / len(llm_rows) if llm_rows else None
    audit_rows, reviewed, malformed, agreement = 0, 0, 0, None
    if audit and Path(audit).exists():
        with open(audit, encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        audit_rows = len(rows)
        # A row counts as reviewed only when every required verdict column holds a
        # strict boolean. Counting blank cells as reviewed used to compare "" against
        # "true"/"false" and publish agreement=0.0, which reads as "humans disagreed
        # with the grader" when the truth is that nobody had adjudicated yet.
        judged, malformed = [], 0
        for row in rows:
            verdicts = [_verdict(row, column) for column in VERDICT_COLUMNS]
            if all(verdicts):
                judged.append(row)
            elif any((row.get(column) or "").strip() for column in VERDICT_COLUMNS):
                # partially filled or not a boolean — surface it instead of dropping
                # it silently, so a mistyped sheet is visible rather than looking untouched
                malformed += 1
        reviewed = len(judged)
        if reviewed:
            agreement = sum(_verdict(r, "human_success") == (r.get("grader_success") or "").strip().lower()
                            for r in judged) / reviewed
    pairs = 0
    if preference_pairs and Path(preference_pairs).exists():
        pairs = sum(1 for x in Path(preference_pairs).read_text(encoding="utf-8").splitlines() if x.strip())
    checks = {
        "hidden_tasks_at_least_120": len(specs) >= 120,
        "policy_input_isolated": leakage_safe,
        "deterministic_graders": deterministic_grades,
        "real_llm_trajectories_at_least_360": len(llm_rows) >= 360,
        "human_audit_at_least_40": reviewed >= 40,
        "human_reward_agreement_at_least_90pct": agreement is not None and agreement >= .9,
        "preference_pairs_at_least_200": pairs >= 200,
        "base_llm_success_below_95pct": success_rate is not None and success_rate < .95,
    }
    eligible = all(checks.values())
    return {"eligible": eligible, "checks": checks, "task_count": len(specs), "all_trajectory_count": len(trajectories),
            "real_llm_trajectory_count": len(llm_rows), "base_llm_success": success_rate, "preference_pairs": pairs,
            "human_audit_rows": audit_rows, "human_audit_reviewed": reviewed,
            "human_audit_malformed_rows": malformed,
            "human_audit_status": "not_started" if reviewed == 0 else ("partial" if reviewed < audit_rows else "complete"),
            "human_reward_agreement": agreement,
            "decision": "train next-action SFT/DPO" if eligible else "stop at RL-ready harness; do not claim Agent RL"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True); parser.add_argument("--store", required=True)
    parser.add_argument("--audit"); parser.add_argument("--preference-pairs"); parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = assess(args.tasks, args.store, args.audit, args.preference_pairs)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
