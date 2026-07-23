# -*- coding: utf-8 -*-
"""Stdlib-only tests for price_filter.py (no embed model, no DB needed)."""

import unittest
from ecommerce_rag import price_filter


# ── parse_budget ──────────────────────────────────────────────────────────────

def _product(doc_id, price, title="x"):
    return {"chunk_id": f"{doc_id}_desc", "doc_id": doc_id,
            "source_type": "product", "title": title, "price": price,
            "score": 0.8, "dense_sim": 0.7}

def _policy(doc_id="policy:POL001"):
    return {"chunk_id": f"{doc_id}_c", "doc_id": doc_id,
            "source_type": "policy", "title": "退货政策",
            "score": 0.5, "dense_sim": 0.4}


class PriceFilterTests(unittest.TestCase):
    def test_parse_budget_yu_suan(self):
        assert price_filter.parse_budget("预算600以内通勤降噪耳机") == 600

    def test_parse_budget_bu_chao_guo(self):
        assert price_filter.parse_budget("不超过800元的机械键盘") == 800

    def test_parse_budget_yi_nei(self):
        assert price_filter.parse_budget("500元以内的蓝牙耳机") == 500

    def test_parse_budget_yi_xia(self):
        assert price_filter.parse_budget("600块以下的键盘") == 600

    def test_parse_budget_di_yu(self):
        assert price_filter.parse_budget("低于1000元的扫地机器人") == 1000

    def test_parse_budget_none(self):
        assert price_filter.parse_budget("这款耳机音质怎么样") is None
        assert price_filter.parse_budget("推荐一款降噪耳机") is None
        assert price_filter.parse_budget("保温杯和焖烧罐有什么区别") is None


    # ── apply ─────────────────────────────────────────────────────────────────────

    def test_filter_removes_over_budget(self):
        chunks = [_product("product:P009", 899), _product("product:P001", 499)]
        result, note = price_filter.apply(chunks, 600)
        ids = [c["doc_id"] for c in result]
        assert "product:P009" not in ids
        assert "product:P001" in ids
        assert "P009" in note
        assert "600" in note

    def test_filter_nothing_removed_when_all_within(self):
        chunks = [_product("product:P001", 499), _product("product:P004", 379)]
        result, note = price_filter.apply(chunks, 600)
        assert len(result) == 2
        assert note == ""

    def test_filter_keeps_policies(self):
        chunks = [_product("product:P009", 899), _policy()]
        result, note = price_filter.apply(chunks, 600)
        ids = [c["doc_id"] for c in result]
        assert "policy:POL001" in ids  # policy always kept
        assert "product:P009" not in ids

    def test_filter_all_over_budget_fallback(self):
        """When all products exceed budget, return original order (graceful)."""
        chunks = [_product("product:P009", 899), _product("product:P010", 799)]
        result, note = price_filter.apply(chunks, 300)
        # fallback: original unchanged
        assert len(result) == 2
        assert result[0]["doc_id"] == "product:P009"  # order preserved
        assert "原排序" in note

    def test_filter_none_price_passes_through(self):
        """Products with price=None are not filtered (price unknown = keep)."""
        chunks = [_product("product:PXX", None)]
        result, note = price_filter.apply(chunks, 600)
        assert len(result) == 1
        assert note == ""

    def test_q3_scenario(self):
        """Reproduce Q3: P009@899 should be removed, P001@499 retained under budget=600."""
        chunks = [
            _product("product:P009", 899, "QuietMax H900"),   # over budget, currently ranks 1st
            _product("product:P001", 499, "Air Pro 2"),        # within budget, should now rank 1st
            _product("product:P007", 249, "RunBuds Clip"),     # within budget
        ]
        result, note = price_filter.apply(chunks, 600)
        ids = [c["doc_id"] for c in result]
        assert ids[0] == "product:P001"  # P001 now ranks first
        assert "product:P009" not in ids
        assert "product:P007" in ids
        assert "P009" in note


if __name__ == "__main__":
    unittest.main()
