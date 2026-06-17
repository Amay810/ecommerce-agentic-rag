# 电商客服可信 Agentic RAG

这是一个从 MedicalGPT 仓库中独立出来的电商客服 RAG 项目，目标不是只做“检索后生成”，而是模拟更接近真实客服系统的闭环：

- 意图路由：闲聊、商品问答、导购推荐、商品对比、售后政策、人工兜底。
- 多源知识库：商品结构化资料、用户评价、商品 QA、售后/物流/发票政策。
- 混合检索：dense embedding + BM25 + RRF，兼顾口语语义和型号/参数精确匹配。
- Agentic 控制：检索弱时 HyDE 重检，证据仍不足则转人工。
- 可信输出：引用标注、句级 grounding、事实一致性检查、数字参数引用检查。
- 落地闭环：JSONL 日志记录每次意图、检索命中、动作和 trace，便于后续补知识库和调阈值。

## 项目结构

```text
ecommerce-agentic-rag/
  ecommerce_rag/
    agent.py              # Agent 控制器
    intent.py             # 客服意图路由
    data_loader.py        # 商品/政策建索引
    hybrid_retriever.py   # BM25 + dense + RRF
    catalog.py            # 推荐/对比确定性摘要
    verifier.py           # grounding、引用、事实核查
    evaluate.py           # 回归评测与可选 RAGAS 风格指标
    app.py                # Streamlit demo
    data/
      sample_products.jsonl
      policies.jsonl
      eval_questions.jsonl
```

## 本地运行

```powershell
cd E:\cv_codex\ecommerce-agentic-rag
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt

# 构建索引
.\.venv\Scripts\python -m ecommerce_rag.data_loader

# 可选：配置 DeepSeek/Kimi/智谱/OpenAI-compatible API
$env:ERAG_LLM_API_KEY="你的key"
$env:ERAG_LLM_BASE_URL="https://api.deepseek.com"
$env:ERAG_LLM_MODEL="deepseek-chat"

# 启动 UI
.\.venv\Scripts\streamlit run ecommerce_rag/app.py
```

没有 LLM key 时，系统仍可运行意图路由、检索、推荐/对比摘要和回归评测；商品问答会提示需要配置 API 才能生成完整自然语言答案。

## 回归评测

```powershell
.\.venv\Scripts\python -m ecommerce_rag.evaluate

# 有 LLM key 时，可额外启用 faithfulness / relevancy / context precision
.\.venv\Scripts\python -m ecommerce_rag.evaluate --with-llm-metrics
```

## 面试故事线

可以讲成“从 RAG demo 到可落地客服系统”的演进：

1. 普通 RAG 容易在商品参数、库存、售后政策上乱答。
2. 所以先做意图路由，把商品问答、推荐、对比、政策和人工兜底拆开。
3. 商品资料使用父子文档，检索细粒度 chunk，生成时给完整商品卡。
4. 检索使用 dense + BM25 + RRF，既能处理口语问题，也能命中型号和参数。
5. 生成后用引用、grounding 和事实核查约束幻觉。
6. 所有 trace 写日志，后续用高频转人工问题反向补知识库。
