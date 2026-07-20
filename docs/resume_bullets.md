# 简历与面试表述

## 简历两行版

- 基于 Amazon Reviews 2023 构建 5,000 商品、43,953 chunks 的中英跨语言
  Agentic RAG，完成 Dense+BM25+RRF、约束过滤、引用校验与三档规模评测；
  5k 下 Recall@5 0.965，价格约束后提升至 0.979。
- 构建含 1,000 用户/10,000 订单、8 类工具与写操作 Guardrail 的可重放
  Agent Harness；60 任务 × 3 次达到 policy compliance 1.000、pass³ 1.000，
  并以人工奖励一致率门槛阻止不成熟的 Agent RL 宣称。

## 必须主动说明的边界

- 250 题中的 200 题是程序化 gold；50 个困难候选仍需项目作者最终人工复核。
- Harness 当前高分来自透明规则策略，证明的是环境、工具和评分器可用，不是
  “LLM Agent 已达到 100%”。下一步才是替换 next-action policy 做 SFT/DPO。
- reranker 在旧小集没有净收益；5k A100 消融完成前默认关闭。
