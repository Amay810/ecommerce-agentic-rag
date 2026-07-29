# -*- coding: utf-8 -*-
"""Stdlib-only tests for SupportCase + store (no model / no sentence-transformers).

Run: python -m pytest tests/test_support_case.py   (or: python tests/test_support_case.py)
"""

import unittest
import tempfile
from pathlib import Path

from ecommerce_rag.support_case import SupportCase, make_case_id
from ecommerce_rag import store


def _answer_result() -> dict:
    # product_qa answered but flagged caution (low grounding) -> needs_review
    return {
        "query": "保温杯可以装碳酸饮料吗？",
        "intent": "product_qa",
        "action": "caution",
        "answer": "不建议装碳酸饮料 [资料1]。",
        "trace": ["意图路由：product_qa", "第 1 轮检索：保温杯 碳酸"],
        "grounding": {"ratio": 0.4, "unsupported": ["xxx"]},
        "citations": {"ok": True, "missing": []},
        "consistency": {"verdict": "一致", "problems": []},
        "chunks": [
            {"chunk_id": "product:P006:desc:0", "doc_id": "product:P006", "source_type": "product",
             "title": "保温旅行杯 KeepWarm 500ml", "category": "户外/水杯", "price": 79,
             "inventory": "现货", "score": 0.03, "dense_sim": 0.71},
            {"chunk_id": "product:P006:qa:0", "doc_id": "product:P006", "source_type": "product",
             "title": "保温旅行杯 KeepWarm 500ml", "category": "户外/水杯", "price": 79,
             "inventory": "现货", "score": 0.02, "dense_sim": 0.55},
            {"chunk_id": "policy:POL001:body:0", "doc_id": "policy:POL001", "source_type": "policy",
             "title": "七天无理由退货", "category": "退换货", "price": None,
             "inventory": "", "score": 0.01, "dense_sim": 0.33},
        ],
    }


def _ok_result() -> dict:
    return {
        "query": "机械键盘支持 Mac 吗？", "intent": "product_qa", "action": "ok",
        "answer": "支持 Mac [资料1]。", "trace": ["意图路由：product_qa"],
        "grounding": {"ratio": 0.9}, "citations": {"ok": True}, "consistency": {"verdict": "一致"},
        "chunks": [
            {"chunk_id": "product:P004:qa:0", "doc_id": "product:P004", "source_type": "product",
             "title": "机械键盘 K8 87键", "category": "数码配件/键盘", "price": 299,
             "inventory": "少量现货", "score": 0.03, "dense_sim": 0.8},
        ],
    }


def _handoff_result() -> dict:
    # private/order intent -> handoff, no grounding/citation/consistency keys
    return {
        "query": "我的订单退款什么时候到账？", "intent": "handoff", "action": "handoff",
        "trace": ["意图路由：handoff"], "chunks": [],
    }


class SupportCaseTests(unittest.TestCase):
    def test_from_agent_result_fields(self):
        case = SupportCase.from_agent_result(_answer_result())
        assert case.case_id.startswith("sc_")
        # stable id reproducible from same query+trace+ts
        assert case.case_id == make_case_id(case.query, case.trace, case.ts)
        assert case.intent == "product_qa" and case.action == "caution"
        # confidence == max dense_sim
        assert abs(case.confidence - 0.71) < 1e-9
        # citation_index: same doc_id shares index; new doc gets next index
        idx = {e["chunk_id"]: e["citation_index"] for e in case.evidence}
        assert idx["product:P006:desc:0"] == 1 and idx["product:P006:qa:0"] == 1
        assert idx["policy:POL001:body:0"] == 2
        # snapshot split + reserved version fields
        assert len(case.snapshot["products"]) == 1 and len(case.snapshot["policies"]) == 1
        prod = case.snapshot["products"][0]
        assert prod["doc_id"] == "product:P006" and prod["price"] == 79
        assert prod["version"] is None and prod["default_updated_at"] is None
        assert case.snapshot["policies"][0]["policy_type"] == "退换货"

    def test_needs_review_rules(self):
        # caution -> review; low grounding even if action ok -> review; clean ok -> not review; handoff -> review
        assert SupportCase.from_agent_result(_answer_result()).needs_review is True
        assert SupportCase.from_agent_result(_handoff_result()).needs_review is True
        assert SupportCase.from_agent_result(_ok_result()).needs_review is False
        # citation failure alone triggers review
        assert SupportCase.compute_needs_review("ok", 0.9, False, "一致") is True
        # contradiction verdict triggers review
        assert SupportCase.compute_needs_review("ok", 0.9, True, "矛盾") is True

    def test_store_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "support.db"
            for r in (_answer_result(), _ok_result(), _handoff_result()):
                store.insert_case(SupportCase.from_agent_result(r), db_path=db)
            assert store.count(db_path=db) == 3
            assert len(store.needs_review(db_path=db)) == 2  # caution + handoff
            rows = store.recent(5, db_path=db)
            assert len(rows) == 3
            # JSON columns rehydrate
            export = Path(d) / "cases.jsonl"
            n = store.export_jsonl(export, db_path=db)
            assert n == 3
            first = export.read_text(encoding="utf-8").splitlines()[0]
            assert '"snapshot"' in first and '"evidence"' in first

    def test_insert_is_idempotent(self):
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "support.db"
            case = SupportCase.from_agent_result(_answer_result())
            store.insert_case(case, db_path=db)
            store.insert_case(case, db_path=db)  # same case_id -> upsert, no duplicate
            assert store.count(db_path=db) == 1


if __name__ == "__main__":
    unittest.main()
