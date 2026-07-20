"""Stream Amazon Reviews 2023 and materialise the reproducible 5k corpus.

This avoids downloading the full multi-million-row dataset. It first retains a
bounded review pool, then joins matching valid metadata records.
"""
import argparse, json, os
from collections import defaultdict
from pathlib import Path
from datasets import load_dataset
from scripts.prepare_amazon import clean, normalize

DATASET="McAuley-Lab/Amazon-Reviews-2023"
PINNED_REVISION="2b6d039ed471f2ba5fd2acb718bf33b0a7e5598e"
CATEGORIES=("Electronics","Home_and_Kitchen")

def stream(kind, category, revision):
    # Newer `datasets` releases no longer execute dataset scripts. The repository
    # also publishes raw JSONL files, which the generic streaming JSON loader reads.
    folder = "review_categories" if kind == "review" else "meta_categories"
    prefix = "" if kind == "review" else "meta_"
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    url = f"{endpoint}/datasets/{DATASET}/resolve/{revision}/raw/{folder}/{prefix}{category}.jsonl"
    return load_dataset("json", data_files=url, split="train", streaming=True)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--output",type=Path,default=Path("ecommerce_rag/data/amazon_products_5k.jsonl")); p.add_argument("--per-category",type=int,default=2500); p.add_argument("--review-scan",type=int,default=250000); p.add_argument("--revision",default=PINNED_REVISION); a=p.parse_args()
    final=[]; category_stats={}
    for category in CATEGORIES:
        selected=[]
        for row in stream("meta", category, a.revision):
            item=normalize(row)
            if item:
                item["amazon_category_config"]=category; selected.append(item)
                if len(selected)>=a.per_category: break
        if len(selected)<a.per_category: raise SystemExit(f"{category}: only {len(selected)} valid joined products")
        chosen={x["source_asin"]:x for x in selected}
        for n,row in enumerate(stream("review", category, a.revision),1):
            asin=clean(row.get("parent_asin") or row.get("asin")); body=clean(row.get("text") or row.get("title"))
            if asin in chosen and len(body)>=30 and len(chosen[asin]["reviews"])<5: chosen[asin]["reviews"].append(body)
            if n>=a.review_scan: break
        category_stats[category]={"products":len(selected),"reviews":sum(len(x["reviews"]) for x in selected)}; final.extend(selected)
    for i,item in enumerate(final,1): item["id"]=f"P{i:05d}"
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in final),encoding="utf-8")
    stats={"dataset":DATASET,"revision":a.revision,"products":len(final),"reviews":sum(len(x["reviews"]) for x in final),"categories":category_stats}; a.output.with_suffix(".stats.json").write_text(json.dumps(stats,indent=2),encoding="utf-8"); print(json.dumps(stats,indent=2))
if __name__=="__main__": main()
