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


def _load_grade_overlay(path, expected_ids):
    """Load a complete trajectory-id keyed sidecar, returning errors fail-closed."""
    if not path:
        return None, []
    source = Path(path)
    if not source.exists():
        return {}, [f"grade sidecar does not exist: {source}"]
    grades, errors = {}, []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_number}: invalid JSON: {exc.msg}")
            continue
        trajectory_id, grade = row.get("trajectory_id"), row.get("grade")
        if not isinstance(trajectory_id, str) or not isinstance(grade, dict):
            errors.append(f"line {line_number}: trajectory_id and grade are required")
            continue
        if trajectory_id in grades:
            errors.append(f"duplicate trajectory_id: {trajectory_id}")
            continue
        grades[trajectory_id] = grade
    missing = sorted(expected_ids - set(grades))
    extra = sorted(set(grades) - expected_ids)
    if missing:
        errors.append(f"missing {len(missing)} trajectory ids")
    if extra:
        errors.append(f"unknown {len(extra)} trajectory ids")
    return grades, errors


def assess(tasks, store, audit=None, preference_pairs=None, grades=None):
    specs = [json.loads(x) for x in Path(tasks).read_text(encoding="utf-8").splitlines() if x.strip()]
    conn = sqlite3.connect(store)
    try:
        db_rows = conn.execute("SELECT trajectory_id,trajectory_json,grade_json FROM trajectories").fetchall()
    finally:
        conn.close()
    trajectories = [(trajectory_id, json.loads(t), json.loads(g)) for trajectory_id, t, g in db_rows]
    llm_rows = [(trajectory_id, t, g) for trajectory_id, t, g in trajectories if t.get("policy_name") == "LLMPolicy"]
    expected_ids = {trajectory_id for trajectory_id, _, _ in llm_rows}
    overlay, overlay_errors = _load_grade_overlay(grades, expected_ids)
    effective_rows = [(trajectory_id, t, overlay[trajectory_id] if overlay is not None and trajectory_id in overlay else g)
                      for trajectory_id, t, g in llm_rows]
    leakage_safe = bool(llm_rows) and all(not (set(obs) & HIDDEN_KEYS) for _, t, _ in llm_rows for obs in t.get("observations", []))
    deterministic_grades = bool(effective_rows) and not overlay_errors and all(
        isinstance(g.get("success"), bool) and isinstance(g.get("terminal_state_match"), bool)
        for _, _, g in effective_rows)
    success_rate = sum(g["success"] for _, _, g in effective_rows) / len(effective_rows) if effective_rows else None
    grade_by_id = {trajectory_id: grade for trajectory_id, _, grade in effective_rows}
    audit_rows, reviewed, malformed = 0, 0, 0
    success_agreement = policy_agreement = None
    audit_linkage_errors = []
    if audit and Path(audit).exists():
        # Spreadsheet exports commonly include a UTF-8 BOM. utf-8-sig accepts
        # both forms and prevents the first header becoming "\ufefftrajectory_id".
        with open(audit, encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        audit_rows = len(rows)
        # A row counts as reviewed only when every required verdict column holds a
        # strict boolean. Counting blank cells as reviewed used to compare "" against
        # "true"/"false" and publish agreement=0.0, which reads as "humans disagreed
        # with the grader" when the truth is that nobody had adjudicated yet.
        judged, malformed, seen_audit_ids = [], 0, set()
        for row in rows:
            trajectory_id = row.get("trajectory_id") or ""
            if trajectory_id in seen_audit_ids:
                audit_linkage_errors.append(f"duplicate audit trajectory_id: {trajectory_id}")
            seen_audit_ids.add(trajectory_id)
            if trajectory_id not in grade_by_id:
                audit_linkage_errors.append(f"unknown audit trajectory_id: {trajectory_id}")
            verdicts = [_verdict(row, column) for column in VERDICT_COLUMNS]
            if all(verdicts) and trajectory_id in grade_by_id:
                judged.append(row)
            elif any((row.get(column) or "").strip() for column in VERDICT_COLUMNS):
                # partially filled or not a boolean — surface it instead of dropping
                # it silently, so a mistyped sheet is visible rather than looking untouched
                malformed += 1
        reviewed = len(judged)
        if reviewed:
            success_agreement = sum(
                (_verdict(row, "human_success") == str(grade_by_id[row["trajectory_id"]].get("success")).lower())
                for row in judged
            ) / reviewed
            policy_agreement = sum(
                (_verdict(row, "human_policy_compliant") == str(grade_by_id[row["trajectory_id"]].get("policy_compliant")).lower())
                for row in judged
            ) / reviewed
    pairs = 0
    if preference_pairs and Path(preference_pairs).exists():
        pairs = sum(1 for x in Path(preference_pairs).read_text(encoding="utf-8").splitlines() if x.strip())
    checks = {
        "hidden_tasks_at_least_120": len(specs) >= 120,
        "policy_input_isolated": leakage_safe,
        "deterministic_graders": deterministic_grades,
        "grade_sidecar_complete": not overlay_errors,
        "real_llm_trajectories_at_least_360": len(llm_rows) >= 360,
        "human_audit_at_least_40": reviewed >= 40,
        "human_audit_linkage_valid": not audit_linkage_errors,
        "human_success_agreement_at_least_90pct": success_agreement is not None and success_agreement >= .9,
        "human_policy_agreement_at_least_90pct": policy_agreement is not None and policy_agreement >= .9,
        # Compatibility alias for existing consumers; this has always meant
        # agreement on the success/reward verdict, not policy compliance.
        "human_reward_agreement_at_least_90pct": success_agreement is not None and success_agreement >= .9,
        "preference_pairs_at_least_200": pairs >= 200,
        "base_llm_success_below_95pct": success_rate is not None and success_rate < .95,
    }
    eligible = all(checks.values())
    return {"eligible": eligible, "checks": checks, "task_count": len(specs), "all_trajectory_count": len(trajectories),
            "real_llm_trajectory_count": len(llm_rows), "base_llm_success": success_rate,
            "base_llm_operational_success": success_rate,
            "automatic_grade_scope": "operational; natural-language answer quality requires human review",
            "grade_sidecar": str(grades) if grades else None,
            "grade_sidecar_errors": overlay_errors,
            "preference_pairs": pairs,
            "human_audit_rows": audit_rows, "human_audit_reviewed": reviewed,
            "human_audit_malformed_rows": malformed,
            "human_audit_status": "not_started" if reviewed == 0 else ("partial" if reviewed < audit_rows else "complete"),
            "human_reward_agreement": success_agreement,
            "human_success_agreement": success_agreement,
            "human_policy_agreement": policy_agreement,
            "human_audit_linkage_errors": audit_linkage_errors,
            "decision": "train next-action SFT/DPO" if eligible else "gate failed; do not train or claim Agent RL"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True); parser.add_argument("--store", required=True)
    parser.add_argument("--audit"); parser.add_argument("--preference-pairs"); parser.add_argument("--grades"); parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = assess(args.tasks, args.store, args.audit, args.preference_pairs, args.grades)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
