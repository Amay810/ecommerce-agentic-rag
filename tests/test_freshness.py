# -*- coding: utf-8 -*-
"""Stdlib-only tests for the freshness guardrail (no model needed)."""

import unittest
from datetime import datetime, timezone

from ecommerce_rag import freshness
from ecommerce_rag.support_case import build_snapshot

NOW = datetime(2026, 6, 12, tzinfo=timezone.utc)


def _snap(updated_at):
    chunks = [{"doc_id": "product:P1", "source_type": "product", "title": "x",
               "price": 99, "inventory": "现货", "updated_at": updated_at}]
    return build_snapshot(chunks)


class FreshnessTests(unittest.TestCase):
    def test_detect_claims(self):
        assert freshness.detect_claims("价格 99 元，现货") == {"price", "inventory"}
        assert freshness.detect_claims("支持七天无理由退货") == {"policy"}
        assert freshness.detect_claims("这款耳机适合跑步") == set()  # no transactional claim

    def test_no_claim_not_triggered(self):
        v = freshness.assess(_snap("2026-06-10"), "product_qa", "这款耳机适合跑步", now=NOW)
        assert v["triggered"] is False and v["status"] == "n/a"

    def test_fresh_path(self):
        v = freshness.assess(_snap("2026-06-10"), "recommend", "价格 99 元", now=NOW, max_age_days=30)
        assert v["status"] == "fresh" and freshness.should_downgrade(v["status"]) is False

    def test_stale_path(self):
        v = freshness.assess(_snap("2026-01-01"), "recommend", "价格 99 元", now=NOW, max_age_days=30)
        assert v["status"] == "stale" and freshness.should_downgrade(v["status"]) is True
        assert v["reasons"]

    def test_unverified_path(self):
        v = freshness.assess(_snap(None), "recommend", "价格 99 元", now=NOW)
        assert v["status"] == "unverified" and freshness.should_downgrade(v["status"]) is True

    def test_snapshot_carries_updated_at(self):
        snap = build_snapshot([{"doc_id": "product:P6", "source_type": "product", "title": "杯",
                                "price": 79, "inventory": "现货", "updated_at": "2026-06-10"}])
        assert snap["products"][0]["default_updated_at"] == "2026-06-10"
        assert snap["products"][0]["version"] is None


if __name__ == "__main__":
    unittest.main()
