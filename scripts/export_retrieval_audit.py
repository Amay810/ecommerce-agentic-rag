"""Export retrieval candidates for a real human adjudication pass."""
import argparse,csv,json
from pathlib import Path

def stratified_sample(rows):
    """Return five review scopes required by the benchmark plan."""
    multi=[x for x in rows if x.get("kind")=="multi_constraint"]
    groups=[
        ("budget",multi[:10]),
        ("multi_constraint",multi[10:20]),
        ("alias_typo",[x for x in rows if x.get("kind")=="alias_typo"][:10]),
        ("near_sku",[x for x in rows if x.get("kind")=="near_sku_multi_gold"][:10]),
        ("no_answer",[x for x in rows if x.get("kind")=="no_answer"][:10]),
    ]
    selected=[]
    for scope,items in groups:
        selected.extend((scope,item) for item in items)
    if any(len(items)!=10 for _,items in groups):
        raise ValueError("benchmark does not contain 10 rows for every audit scope")
    return selected

def main():
    p=argparse.ArgumentParser();p.add_argument("--testset",required=True);p.add_argument("--output",required=True);p.add_argument("--limit",type=int,default=50);p.add_argument("--stratified",action="store_true");a=p.parse_args()
    rows=[json.loads(x) for x in Path(a.testset).read_text(encoding="utf-8").splitlines() if x.strip()]
    rows=[x for x in rows if x.get("review_status") in {"needs_human_review","curated_unverified"}]
    selected=stratified_sample(rows) if a.stratified else [(x.get("kind","unspecified"),x) for x in rows[:a.limit]]
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    with open(a.output,"w",newline="",encoding="utf-8-sig") as f:
        fields=["id","audit_scope","kind","question","proposed_gold_doc_ids","human_gold_doc_ids","is_answerable","reviewer","review_notes"]
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for scope,x in selected:w.writerow({"id":x["id"],"audit_scope":scope,"kind":x.get("kind"),"question":x["question"],"proposed_gold_doc_ids":"|".join(x.get("gold_doc_ids",[]))})
    print(json.dumps({"exported":len(selected),"output":a.output}))
if __name__=="__main__":main()
