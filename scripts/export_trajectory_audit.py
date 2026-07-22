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
    fields=["trajectory_id","task_id","split","seed","user_request","tool_calls_json","guardrails_json",
            "final_answer","terminal_state_match","state_diff_json","grader_success","grader_policy_compliant",
            "grader_reward","failure_type","human_success","human_policy_compliant","review_notes"]
    with open(a.output,"w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for t,g in rows:
            observations=t.get("observations",[])
            request=next((x.get("current_message","") for x in observations if x.get("current_message")),"")
            calls=[{"name":x.get("name"),"arguments":x.get("arguments"),"error":x.get("error")} for x in t.get("tool_calls",[])]
            w.writerow({"trajectory_id":t["trajectory_id"],"task_id":t["task_id"],"split":g.get("split"),"seed":t["seed"],
                        "user_request":request,"tool_calls_json":json.dumps(calls,ensure_ascii=False),
                        "guardrails_json":json.dumps(t.get("guardrail_spans",[]),ensure_ascii=False),
                        "final_answer":t.get("final_answer",""),"terminal_state_match":str(g.get("terminal_state_match",False)).lower(),
                        "state_diff_json":json.dumps(g.get("state_diff",{}),ensure_ascii=False),
                        "grader_success":str(g["success"]).lower(),"grader_policy_compliant":str(g.get("policy_compliant",False)).lower(),
                        "grader_reward":g["reward"],"failure_type":g.get("failure_type")})
    print(json.dumps({"exported":len(rows),"output":a.output}))
if __name__=="__main__":main()
