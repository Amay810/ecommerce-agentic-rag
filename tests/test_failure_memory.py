# -*- coding: utf-8 -*-
"""Stdlib-only tests for failure_memory.py (no embed model, no real DB needed).

Each test injects synthetic SupportCase rows into a temp SQLite DB and verifies
that the correct pattern / KB gap is detected.
"""

import unittest
import json
import sqlite3
import tempfile
from pathlib import Path

from ecommerce_rag import failure_memory


# ── helpers to build synthetic rows ──────────────────────────────────────────

def _base_row(**overrides) -> dict:
    """Minimal valid support_cases row; caller overrides what it cares about."""
    row = {
        "case_id": "sc_test_default",
        "ts": "2026-06-12T00:00:00+00:00",
        "query": "这款耳机音质怎么样",
        "intent": "product_qa",
        "action": "ok",
        "grounding_ratio": 0.8,
        "citation_ok": 1,
        "consistency_verdict": "支持",
        "confidence": 0.75,
        "answer": "音质不错",
        "needs_review": 0,
        "evidence_json": json.dumps([
            {"chunk_id": "c1", "doc_id": "product:P001", "source_type": "product",
             "title": "X1 耳机", "score": 0.9, "dense_sim": 0.75, "citation_index": 1}
        ], ensure_ascii=False),
        "snapshot_json": json.dumps({
            "products": [{"doc_id": "product:P001", "title": "X1 耳机",
                          "price": 299, "inventory": "现货",
                          "version": None, "default_updated_at": "2026-06-01"}],
            "policies": [],
        }, ensure_ascii=False),
        "trace_json": json.dumps(["意图路由：product_qa"], ensure_ascii=False),
        "freshness_json": json.dumps({"triggered": False, "status": "n/a",
                                      "claims": [], "reasons": []}, ensure_ascii=False),
    }
    row.update(overrides)
    return row


def _make_db(rows: list[dict]) -> Path:
    """Write rows into a temp SQLite file and return its path."""
    tmp = tempfile.mktemp(suffix=".db")
    conn = sqlite3.connect(tmp)
    # infer columns from first row
    cols = list(rows[0].keys())
    conn.execute(
        f"CREATE TABLE support_cases ({', '.join(c + ' TEXT' for c in cols)})"
    )
    conn.execute("CREATE INDEX idx_nr ON support_cases(needs_review)")
    for r in rows:
        placeholders = ", ".join("?" for _ in cols)
        conn.execute(
            f"INSERT INTO support_cases ({', '.join(cols)}) VALUES ({placeholders})",
            [r.get(c) for c in cols],
        )
    conn.commit()
    conn.close()
    return Path(tmp)


# ── tests ─────────────────────────────────────────────────────────────────────


class FailureMemoryTests(unittest.TestCase):
    def test_price_constraint_detected(self):
        """Query with budget 600 but top evidence is P009@899 → price_constraint_ignored."""
        rows = [
            _base_row(
                case_id="sc_price_1",
                query="预算600以内通勤降噪耳机",
                intent="recommend",
                action="caution",
                needs_review=1,
                evidence_json=json.dumps([
                    {"chunk_id": "c9", "doc_id": "product:P009", "source_type": "product",
                     "title": "ProNoise 降噪耳机", "score": 0.88, "dense_sim": 0.70,
                     "citation_index": 1}
                ], ensure_ascii=False),
                snapshot_json=json.dumps({
                    "products": [{"doc_id": "product:P009", "title": "ProNoise 降噪耳机",
                                  "price": 899, "inventory": "现货",
                                  "version": None, "default_updated_at": "2026-06-01"}],
                    "policies": [],
                }, ensure_ascii=False),
            ),
        ]
        db = _make_db(rows)
        report = failure_memory.analyze(db_path=db)
        types = [p.pattern_type for p in report["patterns"]]
        assert "price_constraint_ignored" in types, f"expected price_constraint_ignored in {types}"
        p = next(p for p in report["patterns"] if p.pattern_type == "price_constraint_ignored")
        assert p.count == 1
        assert "P009" in p.examples[0].detail

    def test_compound_recall_gap_detected(self):
        """Compare query '保温杯和焖烧罐' but only 1 product retrieved → compound_query_recall_gap."""
        rows = [
            _base_row(
                case_id="sc_compound_1",
                query="推荐保温杯和焖烧罐各一款",
                intent="compare",
                action="caution",
                needs_review=1,
                evidence_json=json.dumps([
                    {"chunk_id": "c14", "doc_id": "product:P014", "source_type": "product",
                     "title": "ThermoMug 保温杯", "score": 0.85, "dense_sim": 0.72,
                     "citation_index": 1}
                ], ensure_ascii=False),
                snapshot_json=json.dumps({
                    "products": [{"doc_id": "product:P014", "title": "ThermoMug 保温杯",
                                  "price": 189, "inventory": "现货",
                                  "version": None, "default_updated_at": "2026-06-05"}],
                    "policies": [],
                }, ensure_ascii=False),
            ),
        ]
        db = _make_db(rows)
        report = failure_memory.analyze(db_path=db)
        types = [p.pattern_type for p in report["patterns"]]
        assert "compound_query_recall_gap" in types, f"expected compound_query_recall_gap in {types}"
        p = next(p for p in report["patterns"] if p.pattern_type == "compound_query_recall_gap")
        assert p.count == 1

    def test_stale_data_detected(self):
        """Freshness triggered with stale status → stale_data_caution pattern."""
        rows = [
            _base_row(
                case_id="sc_stale_1",
                query="养猫家庭适合买哪款清洁产品",
                intent="recommend",
                action="caution",
                needs_review=1,
                freshness_json=json.dumps({
                    "triggered": True, "status": "stale",
                    "claims": ["price", "inventory"],
                    "reasons": ["P005 updated 2026-01-01, age=163d > max=30d"],
                }, ensure_ascii=False),
                snapshot_json=json.dumps({
                    "products": [{"doc_id": "product:P005", "title": "CleanBot 扫地机器人",
                                  "price": 1499, "inventory": "现货",
                                  "version": None, "default_updated_at": "2026-01-01"}],
                    "policies": [],
                }, ensure_ascii=False),
            ),
        ]
        db = _make_db(rows)
        report = failure_memory.analyze(db_path=db)
        types = [p.pattern_type for p in report["patterns"]]
        assert "stale_data_caution" in types, f"expected stale_data_caution in {types}"
        p = next(p for p in report["patterns"] if p.pattern_type == "stale_data_caution")
        assert p.count == 1
        assert "stale" in p.examples[0].detail

    def test_zero_retrieval_handoff_detected(self):
        """Handoff with 0 evidence → zero_retrieval_handoff pattern."""
        rows = [
            _base_row(
                case_id="sc_zero_1",
                query="我的订单退款什么时候到账",
                intent="handoff",
                action="handoff",
                needs_review=1,
                confidence=0.0,
                evidence_json=json.dumps([], ensure_ascii=False),
                snapshot_json=json.dumps({"products": [], "policies": []}, ensure_ascii=False),
            ),
        ]
        db = _make_db(rows)
        report = failure_memory.analyze(db_path=db)
        types = [p.pattern_type for p in report["patterns"]]
        assert "zero_retrieval_handoff" in types, f"expected zero_retrieval_handoff in {types}"
        p = next(p for p in report["patterns"] if p.pattern_type == "zero_retrieval_handoff")
        assert p.count == 1

    def test_handoff_cluster_detected(self):
        """Two handoff cases with same intent → handoff_cluster pattern."""
        rows = [
            _base_row(
                case_id="sc_ho_1", query="我要退货", intent="handoff",
                action="handoff", needs_review=1,
                evidence_json="[]", snapshot_json='{"products":[],"policies":[]}',
            ),
            _base_row(
                case_id="sc_ho_2", query="退款进度查询", intent="handoff",
                action="handoff", needs_review=1,
                evidence_json="[]", snapshot_json='{"products":[],"policies":[]}',
            ),
        ]
        db = _make_db(rows)
        report = failure_memory.analyze(db_path=db)
        types = [p.pattern_type for p in report["patterns"]]
        assert any("handoff_cluster" in t for t in types), f"expected handoff_cluster in {types}"
        ho = next(p for p in report["patterns"] if "handoff_cluster" in p.pattern_type)
        assert ho.count == 2

    def test_kb_gap_compound_entity_miss(self):
        """compound_entity_miss KB gap surfaces for compare intent."""
        rows = [
            _base_row(
                case_id="sc_gap_1", query="保温杯和焖烧罐哪个保温效果更好",
                intent="compare", action="caution", needs_review=1,
                evidence_json=json.dumps([
                    {"chunk_id": "c14", "doc_id": "product:P014", "source_type": "product",
                     "title": "ThermoMug", "score": 0.8, "dense_sim": 0.68, "citation_index": 1}
                ], ensure_ascii=False),
                snapshot_json='{"products":[{"doc_id":"product:P014","title":"ThermoMug",'
                              '"price":189,"inventory":"现货","version":null,"default_updated_at":"2026-06-05"}],'
                              '"policies":[]}',
            ),
        ]
        db = _make_db(rows)
        report = failure_memory.analyze(db_path=db)
        gap_types = [g.gap_type for g in report["kb_gaps"]]
        assert "compound_entity_miss" in gap_types, f"expected compound_entity_miss in {gap_types}"
        g = next(g for g in report["kb_gaps"] if g.gap_type == "compound_entity_miss")
        assert g.intent == "compare"
        assert any("保温杯" in q for q in g.affected_queries)

    def test_kb_gap_stale_doc(self):
        """stale_doc KB gap identifies the specific doc_id causing repeated freshness cautions."""
        rows = [
            _base_row(
                case_id=f"sc_stale_{i}", query="清洁机器人有货吗",
                intent="product_qa", action="caution", needs_review=1,
                freshness_json=json.dumps({"triggered": True, "status": "stale",
                                           "claims": ["inventory"], "reasons": ["age>30d"]},
                                          ensure_ascii=False),
                snapshot_json=json.dumps({
                    "products": [{"doc_id": "product:P005", "title": "CleanBot",
                                  "price": 1499, "inventory": "现货",
                                  "version": None, "default_updated_at": "2026-01-01"}],
                    "policies": [],
                }, ensure_ascii=False),
            )
            for i in range(3)  # same doc triggers 3 times
        ]
        db = _make_db(rows)
        report = failure_memory.analyze(db_path=db)
        gap_types = [g.gap_type for g in report["kb_gaps"]]
        assert "stale_doc" in gap_types, f"expected stale_doc in {gap_types}"
        g = next(g for g in report["kb_gaps"] if g.gap_type == "stale_doc")
        assert g.doc_id == "product:P005"
        assert "3 次" in g.description

    def test_render_report_no_crash(self):
        """render_report should produce a non-empty string without raising."""
        rows = [
            _base_row(
                case_id="sc_render_1", query="不超过500元的键盘",
                intent="recommend", action="caution", needs_review=1,
                evidence_json=json.dumps([
                    {"chunk_id": "c20", "doc_id": "product:P020", "source_type": "product",
                     "title": "MechKey Pro", "score": 0.82, "dense_sim": 0.69, "citation_index": 1}
                ], ensure_ascii=False),
                snapshot_json=json.dumps({
                    "products": [{"doc_id": "product:P020", "title": "MechKey Pro",
                                  "price": 699, "inventory": "现货",
                                  "version": None, "default_updated_at": "2026-06-01"}],
                    "policies": [],
                }, ensure_ascii=False),
            ),
        ]
        db = _make_db(rows)
        report = failure_memory.analyze(db_path=db)
        rendered = failure_memory.render_report(report)
        assert "FailureMemory" in rendered
        assert "price_constraint_ignored" in rendered


if __name__ == "__main__":
    unittest.main()
