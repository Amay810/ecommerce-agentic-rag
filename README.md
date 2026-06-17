# 电商客服可信 Agentic RAG

这是一个从 MedicalGPT 仓库中独立出来的电商客服 RAG 项目，目标不是只做“检索后生成”，而是模拟更接近真实客服系统的闭环：

- 意图路由：闲聊、商品问答、导购推荐、商品对比、售后政策、人工兜底。
- 多源知识库：商品结构化资料、用户评价、商品 QA、售后/物流/发票政策。
- 混合检索：dense embedding + BM25 + RRF，兼顾口语语义和型号/参数精确匹配。
- Agentic 控制：检索弱时 HyDE 重检，证据仍不足则转人工。
- 可信输出：引用标注、句级 grounding、事实一致性检查、数字参数引用检查。
- 可审计闭环：每次回答写成 SupportCase，记录证据、快照、trace、验证结果与是否需要复盘。

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
    support_case.py       # 可审计 SupportCase
    store.py              # SQLite 落库与 CLI
    freshness.py          # 价格/库存/政策新鲜度护栏
    evaluate.py           # 回归评测与可选 RAGAS 风格指标
    app.py                # Streamlit demo
    data/
      sample_products.jsonl
      policies.jsonl
      eval_questions.jsonl
  docs/
    honest_evaluation.md
    slide_outline.md
  nscc/
    run_eval.pbs
    run_rerank_final.pbs
    run_smoke_supportcase.pbs
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

当前 10 题 baseline：

```text
route_accuracy:      1.000
action_accuracy:     1.000
retrieval_coverage:  1.000
recall@1:            0.944
recall@3:            1.000
recall@5:            1.000
MRR:                 1.000
```

`retrieval_coverage` 只表示触发检索后有结果；真正衡量“是否检索到正确父文档”的指标是 `recall@k` 和 `MRR`。评测集里的 `gold_doc_ids` 使用父文档 ID，例如 `product:P005`、`policy:POL001`。

## Honest Evaluation — Reranker A/B

完整记录见 [docs/honest_evaluation.md](docs/honest_evaluation.md)。结论很克制：在当前 40 商品 / 28 题 gold 评测上，reranker 没有带来净提升，因此当前 demo 不启用 reranker。

| 配置 | recall@1 | recall@3 | recall@5 | MRR |
|---|---:|---:|---:|---:|
| Hybrid baseline | 0.907 | 0.981 | 0.981 | 0.981 |
| +rerank 子chunk | 0.907 | 0.981 | 0.981 | 0.975 |
| +rerank 父卡(无去重) | 0.907 | 0.907 | 0.907 | 0.963 |
| +rerank 父卡+商品级去重 | 0.907 | 0.981 | 0.981 | 0.981 |

三类失败模式：

1. 子 chunk 重排丢上下文：养猫清洁产品问题中，正确商品 `P005` 从 rank 1 掉到 rank 3。
2. 父卡按 chunk 重排会让同商品兄弟 chunk 洪泛，破坏结果多样性，导致 recall@3/5 从 0.981 降到 0.907。
3. 父卡 + 商品级去重修复了多样性问题，但只追平 hybrid baseline，没有超过。

最终决策：不上线 reranker。一阶 hybrid 检索已经 `recall@5 = 0.981`，提升空间很小，而 reranker 会增加延迟。当前真正瓶颈是价格约束和上游候选召回：例如“预算600以内”需要 metadata filtering；保温杯 vs 焖烧罐问题中 `P006` 未进入候选池，reranker 无法修复未召回的文档。

边界：这是本地 demo + NSCC 离线评测结果，尚未作为生产服务上线。

## NSCC 安全运行

不要在 login node 上运行 `python -m ecommerce_rag.data_loader` 或 `python -m ecommerce_rag.evaluate`，它们会加载 embedding 模型，属于计算任务。请通过 PBS 提交到计算节点：

```bash
cd /scratch/users/ntu/s250045/ecommerce-agentic-rag
qsub nscc/run_eval.pbs
```

计算节点离线时，脚本会设置 `HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1`，要求 embedding 模型已经在 HuggingFace cache 中。

## 面试故事线

可以讲成“从 RAG demo 到可落地客服系统”的演进：

1. 普通 RAG 容易在商品参数、库存、售后政策上乱答。
2. 所以先做意图路由，把商品问答、推荐、对比、政策和人工兜底拆开。
3. 商品资料使用父子文档，检索细粒度 chunk，生成时给完整商品卡。
4. 检索使用 dense + BM25 + RRF，既能处理口语问题，也能命中型号和参数。
5. 生成后用引用、grounding 和事实核查约束幻觉。
6. 每次回答沉淀为 SupportCase，包含证据、快照、验证和 trace，可复盘、可审计。
7. 通过 Honest Evaluation 决定不上线无净收益的 reranker，把下一步优化集中到 metadata filtering 和一阶召回。
