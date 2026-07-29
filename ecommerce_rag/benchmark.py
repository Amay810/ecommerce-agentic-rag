"""Retrieval metrics and scale benchmark CLI."""
from __future__ import annotations
import argparse, json, math, statistics, time
from pathlib import Path
from . import config
from .evaluate import doc_ranking, recall_at_k, reciprocal_rank

def ndcg_at_k(ranking, gold, k=5):
    if not gold: return None
    gains=[1.0 if x in set(gold) else 0.0 for x in ranking[:k]]
    dcg=sum(g/math.log2(i+2) for i,g in enumerate(gains)); ideal=sum(1/math.log2(i+2) for i in range(min(len(set(gold)),k)))
    return dcg/ideal if ideal else 0.0

def percentile(values, p):
    if not values: return 0.0
    values=sorted(values); return values[min(len(values)-1,round((len(values)-1)*p))]

def evaluate_retriever(retriever, rows, constraints=False, no_answer_threshold=None):
    scores={"recall@1":[],"recall@3":[],"recall@5":[],"mrr":[],"ndcg@5":[]}; lat=[]; details=[]; no_answer=[]; timing={}
    for row in rows:
        start=time.perf_counter(); chunks=retriever.search(row["question"],top_k=20 if constraints else 5,source_type=row.get("source_type"))
        for key,value in getattr(retriever,"last_timing",{}).items():
            if isinstance(value,(int,float)): timing.setdefault(key,[]).append(float(value))
        if constraints and row.get("constraints",{}).get("max_price") is not None:
            ceiling=float(row["constraints"]["max_price"]); chunks=[c for c in chunks if c.get("price") is not None and float(c["price"])<=ceiling]
        top_dense_sim=max((c.get("dense_sim",-1.0) for c in chunks),default=-1.0)
        predicted_abstain = not chunks or (no_answer_threshold is not None and top_dense_sim < no_answer_threshold)
        if predicted_abstain: chunks=[]
        ranking=doc_ranking(chunks)[:5]; lat.append((time.perf_counter()-start)*1000); gold=row.get("gold_doc_ids",[])
        if not gold:
            no_answer.append(predicted_abstain)
            vals={"recall@1":None,"recall@3":None,"recall@5":None,"mrr":None,"ndcg@5":None}
        elif row.get("relevance_mode")=="any":
            vals={f"recall@{k}":float(bool(set(ranking[:k]) & set(gold))) for k in (1,3,5)}
            vals.update({"mrr":reciprocal_rank(ranking,gold),"ndcg@5":ndcg_at_k(ranking,gold,5)})
        else:
            vals={"recall@1":recall_at_k(ranking,gold,1),"recall@3":recall_at_k(ranking,gold,3),"recall@5":recall_at_k(ranking,gold,5),"mrr":reciprocal_rank(ranking,gold),"ndcg@5":ndcg_at_k(ranking,gold,5)}
        for k,v in vals.items():
            if v is not None: scores[k].append(v)
        details.append({"id":row.get("id"),"kind":row.get("kind"),"split":row.get("split"),"ranking":ranking,"top_dense_sim":top_dense_sim,"predicted_abstain":predicted_abstain,**vals})
    summary={k:round(statistics.fmean(v),4) if v else None for k,v in scores.items()}; summary.update({"queries":len(rows),"answerable_queries":len(rows)-len(no_answer),"no_answer_queries":len(no_answer),"no_answer_accuracy":round(sum(no_answer)/len(no_answer),4) if no_answer else None,"false_positive_rate":round(1-sum(no_answer)/len(no_answer),4) if no_answer else None,"latency_p50_ms":round(percentile(lat,.5),2),"latency_p95_ms":round(percentile(lat,.95),2),"no_answer_threshold":no_answer_threshold,"latency_breakdown_p50_ms":{k:round(percentile(v,.5),2) for k,v in timing.items()},"dense_backend":getattr(retriever,"dense_backend","unknown")})
    return {"summary":summary,"details":details}

def main():
    from .hybrid_retriever import HybridRetriever
    p=argparse.ArgumentParser(); p.add_argument("--testset",type=Path,required=True); p.add_argument("--index",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--require-reviewed",action="store_true"); p.add_argument("--require-human-reviewed",action="store_true"); p.add_argument("--constraints",action="store_true"); p.add_argument("--reranker",action="store_true"); p.add_argument("--split",choices=("dev","locked")); p.add_argument("--no-answer-threshold",type=float); a=p.parse_args()
    rows=[json.loads(x) for x in a.testset.read_text(encoding="utf-8").splitlines() if x.strip()]
    if a.split: rows=[x for x in rows if x.get("split")==a.split]
    if a.require_reviewed and any(x.get("review_status")=="needs_human_review" for x in rows): raise SystemExit("Hard cases still require human review")
    if a.require_human_reviewed and any(x.get("review_status") not in {"programmatic_gold","human_verified"} for x in rows): raise SystemExit("Curated cases are not human verified")
    if a.reranker: config.USE_RERANKER=True
    start=time.perf_counter(); retriever=HybridRetriever(index_dir=a.index); load_ms=(time.perf_counter()-start)*1000
    report=evaluate_retriever(retriever,rows,a.constraints,a.no_answer_threshold); report["summary"]["index_load_ms"]=round(load_ms,2); report["summary"]["index_size_bytes"]=sum(x.stat().st_size for x in a.index.rglob("*") if x.is_file()); report["summary"]["configuration"]={"constraints":a.constraints,"reranker":a.reranker,"split":a.split}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps(report["summary"],ensure_ascii=False,indent=2))
if __name__=="__main__": main()
