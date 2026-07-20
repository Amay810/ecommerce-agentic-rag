"""Create 200 programmatic-gold queries and 50 hard-review candidates."""
import argparse, json, random
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument("--products",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--seed",type=int,default=20260720); a=p.parse_args()
    products=[json.loads(x) for x in a.products.read_text(encoding="utf-8").splitlines() if x.strip()]; rng=random.Random(a.seed)
    eligible=[x for x in products if x.get("attributes")]; rng.shuffle(eligible)
    out=[]
    priced=[x for x in eligible if x.get("price") is not None]
    candidates=[]
    for item in eligible:
        for key,value in sorted(item["attributes"].items(),key=lambda kv:(-len(str(kv[1])),kv[0]))[:5]: candidates.append((item,key,value))
    while len(candidates)<200: candidates.extend(candidates)
    for i,(item,key,value) in enumerate(candidates[:200]):
        alias=(item.get("category_aliases_zh") or [item.get("category","商品")])[0]
        out.append({"id":f"auto_{i+1:03d}","question":f"找一款{alias}，要求 {key} 是 {value}，商品名是 {item['title']}","gold_doc_ids":[f"product:{item['id']}"],"kind":"attribute","review_status":"programmatic_gold"})
    for i in range(50):
        x,y=eligible[(2*i)%len(eligible)],eligible[(2*i+1)%len(eligible)]; mode=i%5
        if mode==0: q,g=f"比较 {x['title']} 和 {y['title']}",[f"product:{x['id']}",f"product:{y['id']}"]
        elif mode==1:
            x=priced[i%len(priced)]; q,g=f"预算不超过 {x['price']}，想买 {x['title']}",[f"product:{x['id']}"]
        elif mode==2: q,g=x["title"].replace("o","0",1),[f"product:{x['id']}"]
        elif mode==3: q,g=f"{x['title']} 与同类近似型号有什么差异",[f"product:{x['id']}"]
        else: q,g=f"寻找不存在的量子悬浮版本 {i}",[]
        item={"id":f"hard_{i+1:03d}","question":q,"gold_doc_ids":g,"kind":["compound","budget","typo","near_sku","no_answer"][mode],"review_status":"needs_human_review"}
        if mode==1: item["constraints"]={"max_price":x["price"]}
        out.append(item)
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in out),encoding="utf-8"); print(json.dumps({"total":len(out),"needs_human_review":50}))
if __name__=="__main__": main()
