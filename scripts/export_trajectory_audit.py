"""Export a stratified, blind-friendly trajectory audit sheet."""
import argparse, csv, json, sqlite3
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument("--store",required=True); p.add_argument("--output",required=True); p.add_argument("--limit",type=int,default=40); a=p.parse_args()
    conn=sqlite3.connect(a.store)
    try: rows=[(json.loads(t),json.loads(g)) for t,g in conn.execute("SELECT trajectory_json,grade_json FROM trajectories")]
    finally: conn.close()
    rows=[x for x in rows if x[0].get("policy_name")=="LLMPolicy"]
    rows.sort(key=lambda x:(x[1].get("success",False),x[1].get("failure_type") or "",x[0]["task_id"],x[0]["seed"]))
    if rows:
        stride=max(1,len(rows)//a.limit); rows=rows[::stride][:a.limit]
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    with open(a.output,"w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=["trajectory_id","task_id","seed","grader_success","grader_reward","failure_type","tool_sequence","final_answer","human_success","human_policy_compliant","review_notes"]); w.writeheader()
        for t,g in rows:w.writerow({"trajectory_id":t["trajectory_id"],"task_id":t["task_id"],"seed":t["seed"],"grader_success":str(g["success"]).lower(),"grader_reward":g["reward"],"failure_type":g.get("failure_type"),"tool_sequence":" > ".join(x["name"] for x in t.get("tool_calls",[])),"final_answer":t.get("final_answer","")})
    print(json.dumps({"exported":len(rows),"output":a.output}))
if __name__=="__main__":main()
