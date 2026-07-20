"""Export retrieval candidates for a real human adjudication pass."""
import argparse,csv,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument("--testset",required=True);p.add_argument("--output",required=True);p.add_argument("--limit",type=int,default=50);a=p.parse_args()
    rows=[json.loads(x) for x in Path(a.testset).read_text(encoding="utf-8").splitlines() if x.strip()]
    rows=[x for x in rows if x.get("review_status") in {"needs_human_review","curated_unverified"}][:a.limit]
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    with open(a.output,"w",newline="",encoding="utf-8-sig") as f:
        fields=["id","kind","question","proposed_gold_doc_ids","human_gold_doc_ids","is_answerable","reviewer","review_notes"]
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for x in rows:w.writerow({"id":x["id"],"kind":x.get("kind"),"question":x["question"],"proposed_gold_doc_ids":"|".join(x.get("gold_doc_ids",[]))})
    print(json.dumps({"exported":len(rows),"output":a.output}))
if __name__=="__main__":main()
