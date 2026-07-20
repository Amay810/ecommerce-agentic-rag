"""Fail-closed readiness gate for next-action SFT/DPO experiments."""
import argparse, csv, json, sqlite3
from pathlib import Path

def assess(tasks, store, audit=None):
    task_count=sum(1 for x in Path(tasks).read_text(encoding="utf-8").splitlines() if x.strip())
    conn=sqlite3.connect(store)
    try: trajectories=conn.execute("SELECT COUNT(*) FROM trajectories").fetchone()[0]
    finally: conn.close()
    agreement=None; reviewed=0
    if audit and Path(audit).exists():
        with open(audit,encoding="utf-8") as f: rows=list(csv.DictReader(f))
        reviewed=len(rows); agreement=sum(r.get("human_success","").lower()==r.get("grader_success","").lower() for r in rows)/reviewed if reviewed else None
    checks={"tasks_at_least_60":task_count>=60,"deterministic_graders":True,"trajectories_at_least_300":trajectories>=300,"human_audit_at_least_30":reviewed>=30,"human_reward_agreement_at_least_90pct":agreement is not None and agreement>=.9}
    eligible=all(checks.values())
    return {"eligible":eligible,"checks":checks,"task_count":task_count,"trajectory_count":trajectories,"human_audit_rows":reviewed,"human_reward_agreement":agreement,"decision":"train next-action SFT/DPO" if eligible else "stop at RL-ready harness; do not claim Agent RL"}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--tasks",required=True); p.add_argument("--store",required=True); p.add_argument("--audit"); p.add_argument("--output",required=True); a=p.parse_args(); result=assess(a.tasks,a.store,a.audit); Path(a.output).write_text(json.dumps(result,indent=2),encoding="utf-8"); print(json.dumps(result,indent=2))
if __name__=="__main__": main()
