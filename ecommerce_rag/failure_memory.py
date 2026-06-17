# -*- coding: utf-8 -*-
"""FailureMemory: mine the SupportCase store for KB gaps and avoid-patterns.

Reads needs_review cases from SQLite, applies rule-based pattern detectors, and emits
a structured report. No ML, no extra deps — all clustering is frequency-based + regex.

Detected pattern types
───────────────────────
  price_constraint_ignored   — query specifies a budget but top evidence exceeds it
  compound_query_recall_gap  — compare/recommend with 2 entities but <2 product docs retrieved
  stale_data_caution         — freshness guardrail fired with stale/unverified status
  zero_retrieval_handoff     — handoff with 0 evidence (first-stage total miss = KB gap)
  handoff_cluster[<intent>]  — high-frequency handoff by intent (FAQ/KB-fill opportunity)

KB gap types
────────────
  retrieval_blind_spot  — doc that appears in ok-cases but never in needs_review candidates
  compound_entity_miss  — compound query consistently retrieves only one entity
  stale_doc             — specific doc repeatedly causing freshness caution

Usage
─────
  python -m ecommerce_rag.failure_memory --report
  python -m ecommerce_rag.failure_memory --report --json
  python -m ecommerce_rag.failure_memory --report --db /path/to/support.db
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field

from . import store

# ── regex helpers ─────────────────────────────────────────────────────────────

# Budget ceiling in query: "预算600", "600以内", "不超过800", "500元以下"
_BUDGET_RE = re.compile(
    r"(?:预算|不超过|低于|小于)[^\d]*(\d+)"
    r"|(\d+)\s*(?:元|块|¥)?(?:以内|以下)"
)

# Compound query: two noun-like segments joined by 和/与/还是/or/vs
_COMPOUND_RE = re.compile(
    r"[一-鿿\w]{1,8}(?:和|与|还是|or\b|vs\.?\b)[一-鿿\w]{1,8}",
    re.IGNORECASE,
)


# ── data classes ──────────────────────────────────────────────────────────────

@dataclass
class PatternExample:
    case_id: str
    query: str
    intent: str
    action: str
    detail: str


@dataclass
class FailurePattern:
    pattern_type: str
    count: int
    suggestion: str
    examples: list[PatternExample] = field(default_factory=list)


@dataclass
class KBGap:
    gap_type: str           # retrieval_blind_spot | compound_entity_miss | stale_doc
    intent: str
    doc_id: str | None      # specific doc_id when identifiable
    description: str
    affected_queries: list[str] = field(default_factory=list)


# ── row parsing helpers ───────────────────────────────────────────────────────

def _parse_row(row) -> dict:
    d = dict(row)
    for f in ("evidence", "snapshot", "trace", "freshness"):
        col = f"{f}_json"
        if col in d:
            raw = d.pop(col)
            d[f] = json.loads(raw) if raw else ([] if f in ("evidence", "trace") else None)
    d["needs_review"] = bool(d.get("needs_review"))
    return d


def _product_doc_ids(case: dict) -> list[str]:
    """Distinct product doc_ids from evidence, in retrieval order."""
    seen: set[str] = set()
    result: list[str] = []
    for e in (case.get("evidence") or []):
        if e.get("source_type") == "product":
            did = e.get("doc_id")
            if did and did not in seen:
                seen.add(did)
                result.append(did)
    return result


def _extract_budget(query: str) -> int | None:
    m = _BUDGET_RE.search(query)
    if not m:
        return None
    val = m.group(1) or m.group(2)
    return int(val) if val else None


def _snapshot_prices(case: dict) -> dict[str, float | None]:
    return {
        p.get("doc_id"): p.get("price")
        for p in (case.get("snapshot") or {}).get("products") or []
        if p.get("doc_id")
    }


# ── pattern detectors ─────────────────────────────────────────────────────────

def _detect_price_constraint(cases: list[dict]) -> list[FailurePattern]:
    examples: list[PatternExample] = []
    for c in cases:
        budget = _extract_budget(c["query"])
        if budget is None:
            continue
        doc_ids = _product_doc_ids(c)
        if not doc_ids:
            continue
        prices = _snapshot_prices(c)
        top_price = prices.get(doc_ids[0])
        if top_price is not None and top_price > budget:
            examples.append(PatternExample(
                case_id=c["case_id"],
                query=c["query"],
                intent=c["intent"],
                action=c["action"],
                detail=f"预算={budget}元，但 top 证据 {doc_ids[0]}@{top_price:.0f}元 超预算",
            ))
    if not examples:
        return []
    return [FailurePattern(
        pattern_type="price_constraint_ignored",
        count=len(examples),
        suggestion=(
            "dense embedding 无法可靠学习价格约束，应在检索后加元数据过滤层：\n"
            "  1. 用正则从 query 解析预算上限\n"
            "  2. 对 price > budget 的候选施加惩罚或直接过滤\n"
            "  目标：Q3 '预算600以内' recall@1 从 0→1，且无约束查询不受影响"
        ),
        examples=examples,
    )]


def _detect_compound_recall_gap(cases: list[dict]) -> list[FailurePattern]:
    examples: list[PatternExample] = []
    for c in cases:
        if c["intent"] not in ("compare", "recommend"):
            continue
        if not _COMPOUND_RE.search(c["query"]):
            continue
        doc_ids = _product_doc_ids(c)
        if len(doc_ids) < 2:
            detail = (
                f"复合查询只召回 {len(doc_ids)} 个商品文档"
                + (f": {doc_ids[0]}" if doc_ids else "（零召回）")
            )
            examples.append(PatternExample(
                case_id=c["case_id"],
                query=c["query"],
                intent=c["intent"],
                action=c["action"],
                detail=detail,
            ))
    if not examples:
        return []
    return [FailurePattern(
        pattern_type="compound_query_recall_gap",
        count=len(examples),
        suggestion=(
            "单 query embedding 倾向于锁定其中一个实体，另一侧漏召。修复路线：\n"
            "  1. compare 意图下用正则/LLM 拆分复合 query\n"
            "  2. 对每个子 query 分别检索，父级 merge 后再做 RRF\n"
            "  目标：Q28 '保温杯和焖烧罐' P006 能稳定进入候选池"
        ),
        examples=examples,
    )]


def _detect_stale_data(cases: list[dict]) -> list[FailurePattern]:
    examples: list[PatternExample] = []
    for c in cases:
        fr = c.get("freshness") or {}
        if not fr.get("triggered"):
            continue
        if fr.get("status") not in ("stale", "unverified"):
            continue
        stale_docs = [
            f"{p.get('doc_id')}({p.get('default_updated_at','?')})"
            for p in (c.get("snapshot") or {}).get("products") or []
            if p.get("doc_id")
        ]
        examples.append(PatternExample(
            case_id=c["case_id"],
            query=c["query"],
            intent=c["intent"],
            action=c["action"],
            detail=f"freshness={fr.get('status')} claims={fr.get('claims',[])} docs={stale_docs}",
        ))
    if not examples:
        return []
    return [FailurePattern(
        pattern_type="stale_data_caution",
        count=len(examples),
        suggestion=(
            "回答断言价格/库存/政策，但快照数据超过新鲜度阈值，护栏已降级为 caution。\n"
            "  KB 修复：更新过期商品的 updated_at 及相关字段，或缩短 FRESHNESS_MAX_AGE_DAYS。"
        ),
        examples=examples,
    )]


def _detect_zero_retrieval(cases: list[dict]) -> list[FailurePattern]:
    examples: list[PatternExample] = []
    for c in cases:
        if c["action"] != "handoff":
            continue
        if _product_doc_ids(c) or (c.get("evidence") or []):
            continue
        examples.append(PatternExample(
            case_id=c["case_id"],
            query=c["query"],
            intent=c["intent"],
            action=c["action"],
            detail="handoff with 0 evidence（一阶检索零召回）",
        ))
    if not examples:
        return []
    return [FailurePattern(
        pattern_type="zero_retrieval_handoff",
        count=len(examples),
        suggestion=(
            "一阶完全未召回，可能原因：\n"
            "  - intent 本身路由到 handoff（正常）\n"
            "  - dense_sim < RETRIEVAL_MIN_DENSE_SIM（正确降级）\n"
            "  - 查询领域超出 KB 覆盖范围\n"
            "  建议：梳理高频零召回 query 类型，补充 KB 内容。"
        ),
        examples=examples,
    )]


def _detect_handoff_clusters(cases: list[dict]) -> list[FailurePattern]:
    clusters: dict[str, list[PatternExample]] = defaultdict(list)
    for c in cases:
        if c["action"] != "handoff":
            continue
        trace_tail = "; ".join((c.get("trace") or [])[-2:])
        clusters[c["intent"]].append(PatternExample(
            case_id=c["case_id"],
            query=c["query"],
            intent=c["intent"],
            action="handoff",
            detail=trace_tail,
        ))
    return [
        FailurePattern(
            pattern_type=f"handoff_cluster[{intent}]",
            count=len(exs),
            suggestion=f"意图 '{intent}' 高频转人工（{len(exs)} 条）。考虑补 KB 或增加 FAQ 快捷回答。",
            examples=exs,
        )
        for intent, exs in sorted(clusters.items(), key=lambda x: -len(x[1]))
    ]


# ── KB gap analysis ───────────────────────────────────────────────────────────

def _find_kb_gaps(review_cases: list[dict], ok_cases: list[dict]) -> list[KBGap]:
    gaps: list[KBGap] = []

    # Docs that appear in ok-evidence but never in review-evidence for the same intent
    ok_by_intent: dict[str, set[str]] = defaultdict(set)
    for c in ok_cases:
        for did in _product_doc_ids(c):
            ok_by_intent[c["intent"]].add(did)

    review_by_intent: dict[str, set[str]] = defaultdict(set)
    for c in review_cases:
        for did in _product_doc_ids(c):
            review_by_intent[c["intent"]].add(did)

    for intent, ok_docs in ok_by_intent.items():
        blind = ok_docs - review_by_intent.get(intent, set())
        for did in sorted(blind):
            gaps.append(KBGap(
                gap_type="retrieval_blind_spot",
                intent=intent,
                doc_id=did,
                description=(
                    f"{did} 在意图 '{intent}' 的成功案例中出现，"
                    "但从未作为失败案例的候选（可能难以召回）。"
                ),
            ))

    # Compound queries that consistently miss one entity
    compound_miss: dict[str, list[str]] = defaultdict(list)
    for c in review_cases:
        if c["intent"] in ("compare", "recommend") and _COMPOUND_RE.search(c["query"]):
            if len(_product_doc_ids(c)) < 2:
                compound_miss[c["intent"]].append(c["query"])
    for intent, qs in compound_miss.items():
        gaps.append(KBGap(
            gap_type="compound_entity_miss",
            intent=intent,
            doc_id=None,
            description=(
                f"复合查询在意图 '{intent}' 下有 {len(qs)} 次只召回一侧实体。"
                " 建议：query decomposition（拆分子查询分别检索，再合并）。"
            ),
            affected_queries=qs[:5],
        ))

    # Specific docs repeatedly triggering freshness caution
    stale_counts: Counter = Counter()
    stale_info: dict[str, dict] = {}
    for c in review_cases:
        fr = c.get("freshness") or {}
        if fr.get("status") in ("stale", "unverified"):
            for p in (c.get("snapshot") or {}).get("products") or []:
                did = p.get("doc_id")
                if did:
                    stale_counts[did] += 1
                    stale_info[did] = p
    for did, cnt in stale_counts.most_common(10):
        info = stale_info.get(did, {})
        gaps.append(KBGap(
            gap_type="stale_doc",
            intent="*",
            doc_id=did,
            description=(
                f"{did}（{info.get('title','?')}）数据过期，"
                f"触发新鲜度护栏 {cnt} 次。"
                f"最近更新：{info.get('default_updated_at','未知')}"
            ),
        ))

    return gaps


# ── public API ────────────────────────────────────────────────────────────────

def analyze(max_cases: int = 200, db_path=None) -> dict:
    """Analyze needs_review cases; return structured report dict."""
    review_rows = store.needs_review(max_cases, db_path)
    all_rows = store.recent(max_cases * 2, db_path)

    review_cases = [_parse_row(r) for r in review_rows]
    ok_cases = [_parse_row(r) for r in all_rows if not dict(r)["needs_review"]]

    patterns: list[FailurePattern] = []
    patterns.extend(_detect_price_constraint(review_cases))
    patterns.extend(_detect_compound_recall_gap(review_cases))
    patterns.extend(_detect_stale_data(review_cases))
    patterns.extend(_detect_zero_retrieval(review_cases))
    patterns.extend(_detect_handoff_clusters(review_cases))

    kb_gaps = _find_kb_gaps(review_cases, ok_cases)

    return {
        "total_needs_review": len(review_cases),
        "total_ok": len(ok_cases),
        "patterns": patterns,
        "kb_gaps": kb_gaps,
    }


def render_report(report: dict) -> str:
    """Render the analysis report as a human-readable string."""
    lines = [
        "=" * 60,
        "FailureMemory Report",
        "=" * 60,
        f"总案例  needs_review={report['total_needs_review']}  ok={report['total_ok']}",
        "",
    ]

    active = [p for p in report["patterns"] if p.count > 0]
    if active:
        lines.append("PATTERNS DETECTED")
        lines.append("─" * 40)
        for p in active:
            lines.append(f"\n[{p.pattern_type}]  count={p.count}")
            for ex in p.examples[:3]:
                lines.append(f"  Q: {ex.query}")
                lines.append(f"     intent={ex.intent}  action={ex.action}")
                lines.append(f"     {ex.detail}")
            if len(p.examples) > 3:
                lines.append(f"  … 还有 {len(p.examples) - 3} 条")
            lines.append(f"  → {p.suggestion}")
    else:
        lines.append("无 needs_review 案例或未检测到已知失败模式。")

    kb_gaps: list[KBGap] = report["kb_gaps"]
    if kb_gaps:
        lines.append("\nKB GAPS")
        lines.append("─" * 40)
        for g in kb_gaps:
            doc_s = f"  doc={g.doc_id}" if g.doc_id else ""
            lines.append(f"\n[{g.gap_type}] intent={g.intent}{doc_s}")
            lines.append(f"  {g.description}")
            for q in g.affected_queries[:3]:
                lines.append(f"  - {q}")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


def _report_to_json(report: dict) -> str:
    """Serialize report (dataclasses -> dicts) to JSON."""
    def _conv(obj):
        if hasattr(obj, "__dataclass_fields__"):
            return asdict(obj)
        raise TypeError(type(obj))
    return json.dumps(report, default=_conv, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="FailureMemory: mine SupportCase store for KB gaps.")
    parser.add_argument("--report", action="store_true", help="print the failure analysis report")
    parser.add_argument("--json", action="store_true", help="output as JSON instead of plain text")
    parser.add_argument("--db", metavar="PATH", help="path to support.db (default: config.SUPPORT_DB_PATH)")
    parser.add_argument("--max-cases", type=int, default=200, metavar="N",
                        help="max needs_review cases to analyze (default: 200)")
    args = parser.parse_args()

    if not args.report:
        parser.print_help()
        return

    from pathlib import Path
    db_path = Path(args.db) if args.db else None
    report = analyze(max_cases=args.max_cases, db_path=db_path)

    if args.json:
        print(_report_to_json(report))
    else:
        print(render_report(report))


if __name__ == "__main__":
    main()
