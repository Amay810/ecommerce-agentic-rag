# -*- coding: utf-8 -*-
"""Streamlit demo for the standalone e-commerce Agentic RAG system."""

import streamlit as st

from . import config


@st.cache_resource
def get_agent():
    from .agent import CustomerSupportAgent
    from .hybrid_retriever import HybridRetriever

    return CustomerSupportAgent(HybridRetriever())


BADGE = {
    "ok": "通过核查",
    "caution": "部分证据不足",
    "handoff": "建议转人工",
    "direct": "直接回复",
}


def main() -> None:
    st.set_page_config(page_title="电商客服 Agentic RAG", page_icon="🛒", layout="wide")
    st.title("电商客服 Agentic RAG")
    st.caption("意图路由 · 商品/政策多源检索 · 推荐/对比 · 引用核查 · 人工兜底 · 日志闭环")

    with st.sidebar:
        st.subheader("配置")
        st.write(f"LLM: `{config.LLM_MODEL}`")
        st.write(f"Embed: `{config.EMBED_MODEL}`")
        st.write(f"Top-K: {config.TOP_K}")
        st.write(f"Retrieval threshold: {config.RETRIEVAL_MIN_SCORE}")
        if not config.LLM_API_KEY:
            st.warning("未设置 ERAG_LLM_API_KEY，商品问答会展示检索来源，但不调用大模型生成。")

    examples = [
        "扫地机器人适合养猫家庭吗？",
        "预算 600 以内，通勤降噪耳机推荐哪款？",
        "机械键盘支持 Mac 吗，和耳机比哪个更适合办公？",
        "保温杯可以退货吗？",
        "我的订单什么时候退款到账？",
    ]
    query = st.text_input("输入问题", placeholder=examples[0])
    cols = st.columns(len(examples))
    for col, example in zip(cols, examples):
        if col.button(example, use_container_width=True):
            query = example

    if not query:
        st.stop()

    agent = get_agent()
    with st.spinner("Agent 处理中..."):
        result = agent.run(query.strip())

    st.markdown(f"### {BADGE.get(result['action'], result['action'])} · `{result.get('intent')}`")
    st.write(result["display"])

    left, right = st.columns([1, 1])
    with left.expander("Agent 决策轨迹", expanded=True):
        for step in result.get("trace", []):
            st.markdown(f"- {step}")

    chunks = result.get("chunks", [])
    with right.expander("检索来源", expanded=bool(chunks)):
        if not chunks:
            st.caption("本次没有触发检索。")
        for i, c in enumerate(chunks, 1):
            st.markdown(
                f"**{i}. {c.get('title')}** · {c.get('source_type')} · RRF {c.get('score', 0):.3f} "
                f"· dense #{c.get('dense_rank', '-')} · BM25 #{c.get('bm25_rank', '-')}"
            )
            st.caption(c.get("text", "")[:260])

    grounding = result.get("grounding")
    if grounding:
        with st.expander(f"验证明细：grounding {grounding['ratio']:.0%}"):
            for row in grounding["per_sentence"]:
                mark = "OK" if row["grounded"] else "LOW"
                st.markdown(f"`{mark}` ({row['score']:.2f}) {row['sentence']}")


if __name__ == "__main__":
    main()
