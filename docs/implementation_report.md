# 实现与评测报告

## 系统

- 5,000 个 Amazon 商品、5 份政策文档和 43,953 个检索子块；
- Dense + BM25 + RRF hybrid retrieval，可选 BGE reranker；
- 1,000 用户、10,000 订单和 8 类 typed tools；
- 身份验证、退货资格、用户确认和禁止写操作 guardrail；
- TaskSpec → AgentObservation → Trajectory → GradeResult 契约；
- run、replay、compare、数据库终态评分、人工审核和 fail-closed RL gate。

## 检索证据

5k scale set 的 Hybrid Recall@5=0.9917、nDCG@5=0.9826、P95=125.73ms。
BGE reranker 将 Recall@1 从 0.9542 提升至 0.9750，但 P95 从 108.37ms
增加至 219.12ms，因此不全局启用。

独立 v3 困难集 Recall@5=0.6889；其中 typo/alias=0.25，no-answer
accuracy=0。这些结果保留为当前检索局限。

## Agent 证据

Qwen3-4B 完成 120 个任务、360 条真实轨迹。原始自动操作分为 94.17%；补充精确
政策 gold 并将禁止工具“尝试”与“状态变更”分开后，v2 自动操作成功率为 84.17%，
合规率 95%，数据库终态准确率 100%，非法状态变更率 0%。

自动指标不判断全部自然语言质量。40 条系统抽样人工审核中，v2 success agreement
为 80.0%，policy agreement 为 77.5%。抽样不是随机样本，比例不外推到 360 条。
详细证据见 `docs/evaluation_closeout_v2.md`。

## RL 决策

当前 gate 为 `eligible=false`：两项人工一致性均低于 90%，并且同题三次确定性轨迹
没有形成成功/失败 preference pair（0 对）。因此不进行或宣称 SFT/DPO/PPO/GRPO。
