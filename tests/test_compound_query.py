# -*- coding: utf-8 -*-
"""unittest-compatible tests for compound_query.py (no model, no DB)."""

import unittest

from ecommerce_rag import compound_query


class TestCompoundQueryDetect(unittest.TestCase):

    # ── Chinese connectors ──────────────────────────────────────────────────

    def test_he_connector(self):
        ok, parts = compound_query.detect("保温杯和焖烧罐有什么区别？")
        self.assertTrue(ok)
        self.assertEqual(parts, ["保温杯", "焖烧罐"])

    def test_yu_connector(self):
        ok, parts = compound_query.detect("降噪耳机与普通耳机的区别")
        self.assertTrue(ok)
        self.assertEqual(parts[0], "降噪耳机")
        self.assertEqual(parts[1], "普通耳机")

    def test_duibi_connector(self):
        ok, parts = compound_query.detect("保温杯对比焖烧罐")
        self.assertTrue(ok)
        self.assertIn("保温杯", parts)
        self.assertIn("焖烧罐", parts)

    def test_haishi_connector(self):
        ok, parts = compound_query.detect("买扫地机器人还是拖地机器人")
        self.assertTrue(ok)
        self.assertEqual(len(parts), 2)

    # ── English / mixed product names (regression) ──────────────────────────

    def test_vs_connector_english(self):
        ok, parts = compound_query.detect("Air Pro 2 vs QuietMax H900哪个好")
        self.assertTrue(ok)
        self.assertEqual(len(parts), 2)

    def test_mixed_english_names_full_extraction(self):
        """RunBuds Clip and Air Pro 2 must not be truncated to 'uds Clip' / 'Air'."""
        ok, parts = compound_query.detect("RunBuds Clip 和 Air Pro 2 哪个更适合跑步？")
        self.assertTrue(ok)
        self.assertEqual(parts[0], "RunBuds Clip")
        self.assertEqual(parts[1], "Air Pro 2")

    # ── Negative cases ──────────────────────────────────────────────────────

    def test_no_connector_returns_false(self):
        ok, _ = compound_query.detect("保温杯可以装碳酸饮料吗")
        self.assertFalse(ok)

    def test_single_entity_returns_false(self):
        ok, _ = compound_query.detect("这款耳机音质怎么样")
        self.assertFalse(ok)

    def test_identical_entities_returns_false(self):
        ok, _ = compound_query.detect("保温杯和保温杯的区别")
        self.assertFalse(ok)

    # ── Edge cases ──────────────────────────────────────────────────────────

    def test_entity_a_capped_at_8_for_pure_cjk(self):
        ok, parts = compound_query.detect("这是一款非常好用的保温杯和焖烧罐的区别是什么")
        self.assertTrue(ok)
        self.assertLessEqual(len(parts[0]), 8)
        self.assertIn("焖烧罐", parts[1])

    def test_strip_question_suffix_from_b(self):
        ok, parts = compound_query.detect("保温杯和焖烧罐有什么区别")
        self.assertTrue(ok)
        self.assertEqual(parts[1], "焖烧罐")  # "有什么区别" must be stripped


if __name__ == "__main__":
    unittest.main()
