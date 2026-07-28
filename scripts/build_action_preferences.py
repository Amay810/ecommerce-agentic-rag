"""Pair successful and failed LLM trajectories for next-action DPO."""
import argparse,json,sqlite3
from collections import defaultdict
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument("--store",required=True);p.add_argument("--grades");p.add_argument("--output",required=True);a=p.parse_args();conn=sqlite3.connect(a.store)
    try:rows=[(trajectory_id,json.loads(t),json.loads(g)) for trajectory_id,t,g in conn.execute("SELECT trajectory_id,trajectory_json,grade_json FROM trajectories")]
    finally:conn.close()
    if a.grades:
        overlay={}
        for line in Path(a.grades).read_text(encoding="utf-8").splitlines():
            if not line.strip():continue
            row=json.loads(line);trajectory_id=row["trajectory_id"]
            if trajectory_id in overlay:raise ValueError(f"duplicate trajectory_id: {trajectory_id}")
            overlay[trajectory_id]=row["grade"]
        expected={trajectory_id for trajectory_id,_,_ in rows}
        if set(overlay)!=expected:raise ValueError("grade sidecar trajectory ids do not exactly match the store")
        rows=[(trajectory_id,t,overlay[trajectory_id]) for trajectory_id,t,_ in rows]
    groups=defaultdict(list)
    for _,t,g in rows:
        if t.get("policy_name")=="LLMPolicy":groups[t["task_id"]].append((t,g))
    pairs=[]
    for task_id,items in groups.items():
        good=sorted((x for x in items if x[1].get("success")),key=lambda x:-x[1]["reward"]);bad=sorted((x for x in items if not x[1].get("success")),key=lambda x:x[1]["reward"])
        for (chosen,cg),(rejected,rg) in zip(good,bad):
            pairs.append({"task_id":task_id,"prompt":chosen.get("observations",[]),"chosen":chosen.get("actions",[]),"rejected":rejected.get("actions",[]),"chosen_reward":cg["reward"],"rejected_reward":rg["reward"]})
    Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in pairs),encoding="utf-8");print(json.dumps({"pairs":len(pairs),"output":a.output}))
if __name__=="__main__":main()
