"""Build a self-contained retrieval evidence package for human adjudication."""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()


def typo_norm(value: Any) -> str:
    return norm(value).translate(str.maketrans({"0": "o", "1": "i"}))


def parse_constraints(row: dict[str, str], spec: dict[str, Any]) -> dict[str, Any]:
    question = row["question"]
    result: dict[str, Any] = dict(spec.get("constraints") or {})
    result["query"] = question
    scope = row["audit_scope"]
    if scope in {"budget", "multi_constraint"}:
        match = re.search(r"预算不超过\s*([0-9.]+).*?想买\s*(.*?)\s*的\s*(.*?)，关键词\s*(.*)$", question)
        if match:
            result.update(max_price=float(match.group(1)), brand=match.group(2).strip(),
                          category_alias=match.group(3).strip(), keyword=match.group(4).strip())
    elif scope == "alias_typo":
        match = re.search(r"有个(.*?)好像叫\s*(.*?)，", question)
        if match:
            result.update(category_alias=match.group(1).strip(), raw_clue=match.group(2).strip(),
                          normalized_clue=typo_norm(match.group(2)))
    elif scope == "near_sku":
        match = re.search(r"找\s*(.*?)\s*的\s*(.*?)\s*相近型号，线索是\s*(.*)$", question)
        if match:
            result.update(brand=match.group(1).strip(), category_alias=match.group(2).strip(),
                          keyword=match.group(3).strip())
    elif scope == "no_answer":
        code = re.search(r"\bZX-\d+\b", question, flags=re.I)
        result.update(required_model_code=code.group(0) if code else None,
                      impossible_description=re.sub(r"\s*编号ZX-\d+\s*$", "", question).strip())
    return result


def product_facts(product: dict[str, Any] | None, doc_id: str) -> dict[str, Any]:
    if not product:
        return {"doc_id": doc_id, "missing": True}
    attrs = product.get("attributes") or {}
    return {
        "doc_id": doc_id,
        "title": product.get("title"),
        "price": product.get("price"),
        "brand": attrs.get("brand_or_store") or attrs.get("Brand") or attrs.get("Manufacturer"),
        "category": product.get("category"),
        "attributes": attrs,
        "description": product.get("description"),
        "source_asin": product.get("source_asin"),
    }


def category_pass(alias: str | None, category: str | None) -> bool | None:
    if not alias:
        return None
    alias_n, category_n = norm(alias), norm(category)
    mappings = {"数码": "electronics", "厨房": "home kitchen"}
    expected = mappings.get(alias, alias_n)
    return all(token in category_n for token in expected.split())


def candidate_checks(facts: dict[str, Any], constraints: dict[str, Any]) -> dict[str, Any]:
    haystack = " ".join(str(facts.get(key) or "") for key in ("title", "brand", "category"))
    haystack += " " + " ".join(str(x) for x in (facts.get("attributes") or {}).values())
    checks: dict[str, bool | None | float] = {}
    if constraints.get("max_price") is not None:
        price = facts.get("price")
        checks["budget"] = price is not None and float(price) <= float(constraints["max_price"])
    if constraints.get("brand"):
        checks["brand"] = norm(constraints["brand"]) in norm(haystack)
    if constraints.get("category_alias"):
        checks["category"] = category_pass(constraints["category_alias"], facts.get("category"))
    clue = constraints.get("keyword") or constraints.get("raw_clue")
    if clue:
        exact = norm(clue) in norm(haystack)
        fuzzy = SequenceMatcher(None, typo_norm(clue), typo_norm(facts.get("title"))).ratio()
        clue_tokens = [x for x in typo_norm(clue).split() if len(x) >= 3]
        token_hits = sum(token in typo_norm(haystack) for token in clue_tokens)
        token_ratio = token_hits / len(clue_tokens) if clue_tokens else 0.0
        checks.update(keyword_exact=exact, keyword_fuzzy=round(max(fuzzy, token_ratio), 3),
                      keyword_pass=exact or fuzzy >= 0.55 or token_ratio >= 0.5)
    if constraints.get("required_model_code"):
        checks["model_code"] = norm(constraints["required_model_code"]) in norm(haystack)
    return checks


def failed_checks(checks: dict[str, Any]) -> list[str]:
    return [key for key, value in checks.items() if value is False and key != "keyword_exact"]


def build_html(payload: dict[str, Any]) -> str:
    embedded = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Retrieval Human Adjudication</title>
<style>
body{{font:14px/1.45 system-ui,sans-serif;margin:0;background:#f4f6f8;color:#17202a}}header{{position:sticky;top:0;background:#17202a;color:white;padding:12px 20px;z-index:2}}main{{max-width:1400px;margin:auto;padding:18px}}article{{background:white;margin:0 0 18px;padding:18px;border-radius:10px;box-shadow:0 1px 4px #ccd}}table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{border:1px solid #d8dde3;padding:6px;vertical-align:top}}th{{background:#edf1f5}}.facts{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:8px}}.card{{border:1px solid #ccd;padding:8px;border-radius:6px}}.pass{{color:#087830}}.fail{{color:#b42318;font-weight:600}}.controls{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:8px;margin-top:12px}}input,select,textarea,button{{font:inherit;padding:7px}}textarea{{min-height:50px}}details{{margin:8px 0}}code{{white-space:pre-wrap}}button{{cursor:pointer}}.scope{{background:#dde8ff;padding:2px 7px;border-radius:10px}}.pending{{color:#9a6700}}</style></head>
<body><header><strong>50 条困难检索人工审核</strong>　<span id="progress"></span>　<button id="export">导出审核 CSV</button></header><main id="app"></main>
<script id="payload" type="application/json">{embedded}</script><script>
const data=JSON.parse(document.getElementById('payload').textContent); const key='erag-audit-'+data.metadata.version;
const saved=JSON.parse(localStorage.getItem(key)||'{{}}'); const app=document.getElementById('app');
function E(tag,text,cls){{const x=document.createElement(tag);if(text!==undefined)x.textContent=text;if(cls)x.className=cls;return x}}
function set(id,field,value){{saved[id]=saved[id]||{{}};saved[id][field]=value;localStorage.setItem(key,JSON.stringify(saved));progress()}}
function progress(){{const n=data.cases.filter(x=>saved[x.id]?.decision&&saved[x.id].decision!=='pending').length;document.getElementById('progress').textContent=`已裁决 ${{n}} / ${{data.cases.length}}`}}
for(const c of data.cases){{const a=E('article');const h=E('h2');h.append(c.id+' ',E('span',c.audit_scope,'scope'));a.append(h,E('h3',c.question));
 const d=E('details');d.open=true;d.append(E('summary','结构化约束'));d.append(E('code',JSON.stringify(c.constraints,null,2)));a.append(d);
 a.append(E('h3','建议 gold 商品'));const cards=E('div',undefined,'facts');for(const p of c.proposed_products){{const z=E('div',undefined,'card');z.append(E('strong',p.doc_id+' · '+(p.title||'MISSING')),E('div',`价格: ${{p.price}} | 品牌: ${{p.brand}}`),E('div',p.category||''));const q=E('details');q.append(E('summary','属性与描述'),E('code',JSON.stringify(p,null,2)));z.append(q);cards.append(z)}}a.append(cards);
 a.append(E('h3',`Hybrid Top ${{c.candidates.length}} 候选`));const t=E('table');const head=E('tr');['选择','排名','ID','标题/事实','检索分数','约束核验','失败原因'].forEach(x=>head.append(E('th',x)));t.append(head);
 for(const p of c.candidates){{const tr=E('tr');const cb=document.createElement('input');cb.type='checkbox';cb.value=p.doc_id;const defaults=(saved[c.id]?.gold_ids??c.proposed_gold_doc_ids).split('|');cb.checked=defaults.includes(p.doc_id);cb.onchange=()=>{{const ids=[...t.querySelectorAll('input[type=checkbox]:checked')].map(x=>x.value);set(c.id,'gold_ids',ids.join('|'));const g=document.getElementById('gold-'+c.id);if(g)g.value=ids.join('|')}};const td=E('td');td.append(cb);const fact=E('td');fact.append(E('div',`${{p.title}}\n$${{p.price}} | ${{p.brand}}\n${{p.category}}`));const fd=E('details');fd.append(E('summary','候选属性'),E('code',JSON.stringify(p.attributes,null,2)));fact.append(fd);tr.append(td,E('td',p.rank),E('td',p.doc_id),fact,E('td',`hybrid=${{p.score}}\ndense=${{p.dense_sim}}\nBM25 rank=${{p.bm25_rank}}`));const checks=E('td');for(const [k,v] of Object.entries(p.checks))checks.append(E('div',`${{k}}: ${{v}}`,v===false?'fail':v===true?'pass':''));tr.append(checks,E('td',p.failed_checks.join(', ')||'无程序性失败',p.failed_checks.length?'fail':'pass'));t.append(tr)}}a.append(t,E('p','自动解释：'+c.automatic_rationale,c.automatic_conclusion==='pending'?'pending':''));
 const controls=E('div',undefined,'controls');const reviewer=document.createElement('input');reviewer.placeholder='审核人';reviewer.value=saved[c.id]?.reviewer||'';reviewer.onchange=e=>set(c.id,'reviewer',e.target.value);const decision=document.createElement('select');['pending','confirm','modify','uncertain'].forEach(v=>{{const o=E('option',v);o.value=v;decision.append(o)}});decision.value=saved[c.id]?.decision||'pending';decision.onchange=e=>set(c.id,'decision',e.target.value);const answer=document.createElement('select');['','true','false'].forEach(v=>{{const o=E('option',v||'is_answerable');o.value=v;answer.append(o)}});answer.value=saved[c.id]?.is_answerable??'';answer.onchange=e=>set(c.id,'is_answerable',e.target.value);const gold=document.createElement('input');gold.id='gold-'+c.id;gold.placeholder='最终 gold IDs，用 | 分隔';gold.value=saved[c.id]?.gold_ids??c.proposed_gold_doc_ids;gold.onchange=e=>set(c.id,'gold_ids',e.target.value);const notes=document.createElement('textarea');notes.placeholder='审核备注';notes.value=saved[c.id]?.notes||'';notes.onchange=e=>set(c.id,'notes',e.target.value);controls.append(reviewer,decision,answer,gold,notes);a.append(controls);app.append(a)}}
function csv(v){{return '"'+String(v??'').replaceAll('"','""')+'"'}}document.getElementById('export').onclick=()=>{{const cols=['id','audit_scope','decision','human_gold_doc_ids','is_answerable','reviewer','review_notes'];const lines=[cols.join(',')];for(const c of data.cases){{const s=saved[c.id]||{{}};lines.push([c.id,c.audit_scope,s.decision||'pending',s.gold_ids??c.proposed_gold_doc_ids,s.is_answerable??'',s.reviewer||'',s.notes||''].map(csv).join(','))}}const b=new Blob(['\ufeff'+lines.join('\n')],{{type:'text/csv;charset=utf-8'}});const u=URL.createObjectURL(b);const x=document.createElement('a');x.href=u;x.download='retrieval_human_adjudication_completed.csv';x.click();URL.revokeObjectURL(u)}};progress();
</script></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--products", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--html-output", type=Path, required=True)
    args = parser.parse_args()

    from ecommerce_rag.hybrid_retriever import HybridRetriever

    with args.audit.open(encoding="utf-8-sig", newline="") as handle:
        audit_rows = list(csv.DictReader(handle))
    benchmark = {row["id"]: row for row in read_jsonl(args.benchmark)}
    products = {f"product:{row['id']}": row for row in read_jsonl(args.products)}
    retriever = HybridRetriever(index_dir=args.index)
    cases = []
    for row in audit_rows:
        spec = benchmark[row["id"]]
        constraints = parse_constraints(row, spec)
        proposed_ids = [x for x in row["proposed_gold_doc_ids"].split("|") if x]
        limit = 20 if row["audit_scope"] in {"near_sku", "no_answer"} else 10
        results = retriever.search(row["question"], top_k=limit, source_type="product")
        candidates = []
        for rank, result in enumerate(results, 1):
            facts = product_facts(products.get(result["doc_id"]), result["doc_id"])
            checks = candidate_checks(facts, constraints)
            candidates.append({**facts, "rank": rank, "score": round(float(result.get("score", 0)), 6),
                               "dense_sim": round(float(result.get("dense_sim", 0)), 4),
                               "dense_rank": result.get("dense_rank"), "bm25_rank": result.get("bm25_rank"),
                               "checks": checks, "failed_checks": failed_checks(checks)})
        passing = [x["doc_id"] for x in candidates if not x["failed_checks"]]
        proposed_ranks = {x["doc_id"]: x["rank"] for x in candidates if x["doc_id"] in proposed_ids}
        if row["audit_scope"] == "no_answer":
            rationale = ("Top 20 中没有候选通过全部硬约束；仍需人工确认。" if not passing else
                         f"Top 20 中存在未被 gold 覆盖的程序性可行候选：{', '.join(passing)}。")
        else:
            rationale = f"建议 gold 的检索排名：{proposed_ranks or '未进入展示候选'}；程序检查无失败的候选：{passing or '无'}。"
        cases.append({"id": row["id"], "audit_scope": row["audit_scope"], "kind": row["kind"],
                      "question": row["question"], "constraints": constraints,
                      "proposed_gold_doc_ids": "|".join(proposed_ids),
                      "proposed_products": [product_facts(products.get(x), x) for x in proposed_ids],
                      "candidates": candidates, "automatic_rationale": rationale,
                      "automatic_conclusion": "pending"})
    payload = {"metadata": {"version": "v2-evidence-1", "cases": len(cases),
                            "notice": "AI-assisted pending human adjudication"}, "cases": cases}
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.html_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    args.html_output.write_text(build_html(payload), encoding="utf-8")
    print(json.dumps({"cases": len(cases), "json": str(args.json_output), "html": str(args.html_output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
