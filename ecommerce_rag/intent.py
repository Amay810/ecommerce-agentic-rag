# -*- coding: utf-8 -*-
"""Intent routing for customer-support workflows."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Intent:
    name: str
    needs_retrieval: bool
    source_type: str | None
    reason: str


CHITCHAT = ("你好", "您好", "在吗", "谢谢", "多谢", "再见", "拜拜", "hi", "hello")
POLICY_TERMS = ("退货", "换货", "保修", "维修", "发票", "物流", "运费", "配送", "发货", "预售", "售后", "七天", "质保")
RECOMMEND_TERMS = ("推荐", "适合", "预算", "买哪", "怎么选", "选哪", "送人", "场景")
COMPARE_TERMS = ("对比", "比较", "区别", "哪个更", "哪款更", "和", "相比")
PRIVATE_OR_ORDER_TERMS = ("订单", "手机号", "地址", "支付", "退款到账", "投诉", "差评", "赔偿", "人工")


def route_intent(query: str) -> Intent:
    q = query.strip().lower()
    if not q:
        return Intent("empty", False, None, "空问题")
    # 显式问候才算闲聊；不能仅凭长度，否则“退货/保修/发票”等短关键词会被误判。
    if any(q == x or (q.startswith(x) and len(q) <= len(x) + 2) for x in CHITCHAT):
        return Intent("chitchat", False, None, "闲聊或问候")
    # 关键词意图判断必须先于“过短”兜底，确保短的政策/订单词能正确路由。
    if any(term in q for term in PRIVATE_OR_ORDER_TERMS):
        return Intent("handoff", False, None, "涉及订单、隐私、支付或投诉，需要人工处理")
    if any(term in q for term in POLICY_TERMS):
        return Intent("policy", True, "policy", "售后/物流/发票等政策问题")
    if any(term in q for term in COMPARE_TERMS):
        return Intent("compare", True, "product", "商品对比问题")
    if any(term in q for term in RECOMMEND_TERMS):
        return Intent("recommend", True, "product", "导购推荐问题")
    if len(q) <= 3:
        return Intent("chitchat", False, None, "过短且无明确意图，按闲聊处理")
    return Intent("product_qa", True, "product", "商品参数、使用或评价问题")
