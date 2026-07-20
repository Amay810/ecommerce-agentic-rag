"""Convert Amazon Reviews 2023 JSONL(.gz) exports to the project schema."""
from __future__ import annotations
import argparse, gzip, hashlib, json, re
from collections import defaultdict
from pathlib import Path

ALIASES = {"Electronics":["电子产品","数码","电子配件"], "Home & Kitchen":["家居","厨房","家用电器"],
           "Cell Phones":["手机","手机配件"], "Computers":["电脑","计算机","电脑配件"], "Headphones":["耳机","蓝牙耳机"]}

def rows(path):
    opener = gzip.open if Path(path).suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            try: yield json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError): continue

def clean(value):
    if isinstance(value, list): return " ".join(clean(x) for x in value)
    if isinstance(value, dict): return "; ".join(f"{k}: {clean(v)}" for k,v in value.items())
    return re.sub(r"\s+", " ", str(value or "")).strip()

def parse_price(value):
    m = re.search(r"\d+(?:\.\d+)?", clean(value).replace(",", ""))
    return float(m.group()) if m else None

def normalize(row):
    asin, title = clean(row.get("parent_asin") or row.get("asin")), clean(row.get("title"))
    desc, details = clean(row.get("description") or row.get("features")), row.get("details") or {}
    if not asin or not title or (not desc and not details): return None
    category = clean(row.get("categories") or row.get("main_category") or row.get("category"))
    aliases = sorted({a for key, vals in ALIASES.items() if key.lower() in category.lower() for a in vals})
    attrs = dict(details) if isinstance(details, dict) else {"details": clean(details)}
    if row.get("store"): attrs.setdefault("brand_or_store", clean(row["store"]))
    if aliases: attrs["zh_aliases"] = ", ".join(aliases)
    return {"source_asin":asin,"title":title,"category":category,"category_aliases_zh":aliases,
            "price":parse_price(row.get("price")),"inventory":"unknown","attributes":attrs,
            "description":desc,"reviews":[],"qa":[],"updated_at":"2026-07-20"}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--metadata",nargs="+",type=Path,required=True); p.add_argument("--reviews",nargs="*",type=Path,default=[])
    p.add_argument("--limit",type=int,default=5000); p.add_argument("--output",type=Path,required=True); p.add_argument("--stats",type=Path); a=p.parse_args()
    candidates={}
    for path in a.metadata:
        for raw in rows(path):
            item=normalize(raw)
            if item: candidates.setdefault(item["source_asin"],item)
    selected=sorted(candidates.values(),key=lambda x:hashlib.sha1(x["source_asin"].encode()).hexdigest())[:a.limit]
    if len(selected)<a.limit: raise SystemExit(f"Only {len(selected)} valid products; requested {a.limit}")
    chosen={x["source_asin"]:x for x in selected}; reviews=defaultdict(list)
    for path in a.reviews:
        for raw in rows(path):
            asin=clean(raw.get("parent_asin") or raw.get("asin")); body=clean(raw.get("text") or raw.get("review_text") or raw.get("title"))
            if asin in chosen and len(body)>=30:
                reviews[asin].append(((int(raw.get("helpful_vote") or raw.get("helpful_votes") or 0),min(len(body),2000)),body))
    for i,item in enumerate(selected,1): item["id"]=f"P{i:05d}"; item["reviews"]=[b for _,b in sorted(reviews[item["source_asin"]],reverse=True)[:5]]
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in selected),encoding="utf-8")
    stats={"products":len(selected),"with_price":sum(x["price"] is not None for x in selected),"with_reviews":sum(bool(x["reviews"]) for x in selected),"reviews":sum(len(x["reviews"]) for x in selected),"categories":len({x["category"] for x in selected})}
    (a.stats or a.output.with_suffix(".stats.json")).write_text(json.dumps(stats,indent=2),encoding="utf-8"); print(json.dumps(stats,indent=2))
if __name__=="__main__": main()
