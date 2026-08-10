# Agent v2 implementation report

## Agent runtime

- Qwen3-4B `LLMPolicy` 在受控 action space 中选择检索、工具调用、用户追问、回答或转人工；
- 8 类 typed tools 连接 1,000 用户、10,000 订单、商品目录和政策语料；
- `TaskSpec → AgentObservation → ToolCall → Trajectory → GradeResult` 构成公共契约；
- Harness 支持 run、replay、compare、离线重评分和数据库终态检查。

## Transactional safety

订单与退货操作由身份验证、业务资格和用户显式确认保护。JSON Schema 在执行前检查工具名、参数类型、枚举和内部商品 ID；数据库终态评分独立于回答文本。

Qwen3-4B 在 120 个任务上形成 360 条真实轨迹。v2 自动操作成功率为 84.17%，策略合规率 95%，数据库终态准确率 100%，非法状态变更率 0%。5% 的轨迹仍尝试了禁止工具，因此不能把终态安全解释为模型决策完全合规。

## Fact retrieval

Agent 的知识行动覆盖 5,000 个 Amazon 商品、5 份政策文档和 43,953 个检索子块。5k hybrid scale set Recall@5=0.9917、nDCG@5=0.9826、P95=125.73ms。BGE reranker 将 Recall@1 从 0.9542 提升到 0.9750，但 constrained run 的 P95 从 108.37ms 增加到 219.12ms，因此不全局启用。

独立困难集 Recall@5=0.6889，typo/alias recall=0.25，no-answer accuracy=0；这些边界与 scale-set 结果同时报告。

## Evaluation boundary

自动指标不覆盖最终回答的全部自然语言质量。40 条系统抽样中的 v2 success agreement 为 80.0%，policy agreement 为 77.5%，不能外推为 360 条的人类成功率。RL gate 为 `eligible=false`，项目不宣称 SFT、DPO、PPO 或 GRPO。

terminal-grounding v2 的 40 对盲审 fact pass 差值为 0，正式状态为 `negative_or_inconclusive`。评分审计保留在 `docs/evaluation_closeout_v2.md`；已关闭实验的公开摘要见 `docs/history.md`，原始实现和产物迁至私有归档仓库。
