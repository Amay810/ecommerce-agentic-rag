"""Pair successful and failed LLM trajectories for next-action DPO."""
import argparse,json,sqlite3
from collections import defaultdict
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument("--store",required=True);p.add_argument("--output",required=True);a=p.parse_args();conn=sqlite3.connect(a.store)
    try:rows=[(json.loads(t),json.loads(g)) for t,g in conn.execute("SELECT trajectory_json,grade_json FROM trajectories")]
    finally:conn.close()
    groups=defaultdict(list)
    for t,g in rows:
        if t.get("policy_name")=="LLMPolicy":groups[t["task_id"]].append((t,g))
    pairs=[]
    for task_id,items in groups.items():
        good=sorted((x for x in items if x[1].get("success")),key=lambda x:-x[1]["reward"]);bad=sorted((x for x in items if not x[1].get("success")),key=lambda x:x[1]["reward"])
        for (chosen,cg),(rejected,rg) in zip(good,bad):
            pairs.append({"task_id":task_id,"prompt":chosen.get("observations",[]),"chosen":chosen.get("actions",[]),"rejected":rejected.get("actions",[]),"chosen_reward":cg["reward"],"rejected_reward":rg["reward"]})
    Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in pairs),encoding="utf-8");print(json.dumps({"pairs":len(pairs),"output":a.output}))
if __name__=="__main__":main()
