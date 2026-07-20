# 简历与面试表述

## 简历三条

- 基于 Amazon Reviews 2023 构建 5,000 商品、43,953 个子块的中英跨语言检索系统，完成 Dense+BM25+RRF、约束过滤和父商品卡；通过倒排稀疏索引将本地 5k 检索 P95 从约 1.06s 降至 33.5ms。
- 构建包含 1,000 用户、10,000 订单、8 类真实工具和写操作 Guardrail 的可重放 Agent Harness，以数据库终态、政策合规、Tool-call F1、pass@1/pass³ 进行确定性评分。
- 发现并修复 Policy 直接读取 gold `TaskSpec` 的评测泄漏；在 60 条 locked 任务×3 次上，无泄漏 Rule Policy 的 task success/pass³ 为 0.950、政策合规与终态正确率均为 1.000，Oracle 上限独立报告。

## 必须主动说明的边界

- 300 条困难检索集不包含完整标题；locked Recall@5 为 0.803，未达到 0.85 目标，主要失败在属性查询和错别字查询。
- 120 条困难样本是 `curated_unverified`，没有真人复核前不称为人工标注。
- LLM Policy、reranker、A100 FAISS 与 Agent RL 仍为 pending；规则/Oracle 轨迹不能通过 RL 训练门槛。
