# 简历与面试表述

## 推荐简历三条

- 基于固定 revision 的 Amazon Reviews 2023 构建 5,000 商品、43,953 子块的中英跨语言检索系统，实现 Dense + BM25 + RRF、预算过滤和父商品卡；NSCC FAISS scale set Recall@5=0.992、P95=126ms，并完成 BGE reranker 效果—延迟消融。
- 构建包含 1,000 用户、10,000 订单和 8 类 typed tools 的可重放电商 Agent 环境，写操作由身份、资格和显式确认 guardrail 保护，并使用数据库终态而非字符串匹配评分。
- 在 Qwen3-4B 上完成 120 个任务、360 条真实 next-action 轨迹，建立轨迹回放、人工审核和 fail-closed RL gate；v2 自动操作成功率 84.17%、终态准确率 100%，人工一致性未达 90%，因此没有启动 DPO。

## 面试时主动说明

- 5k scale set 的 Recall@5=0.992 不等于模糊自然语言能力；新 v3 困难集 Recall@5=0.689，typo/alias 仍是明显短板。
- 360 条中的 84.17% 是自动化**操作指标**，不含对最终回答完整性、事实忠实度和推荐质量的全面判断。
- 40 条审核采用排序后固定步长抽样，不是随机样本；其中 success agreement=80.0%、policy agreement=77.5%，不能外推成人工评定的总体成功率。
- Guardrail 令非法状态变更为 0，不代表 Agent 决策完全合规：5% 的轨迹仍尝试了禁止工具。
- preference pairs 为 0，RL gate 未通过；项目没有宣称完成 DPO、PPO 或 GRPO。
- 三次重复使用确定性解码，不能把 pass³ 当作独立随机试验的可靠性估计。

## 不使用的表述

- 不把协议、测试或评分器缺陷的修正包装为项目工作量或技术亮点；
- 不把旧评分器的 94.17% 称为人工验证成功率；
- 不把 guardrail 阻断写操作称为 Agent 策略 100% 合规；
- 不把 0 条 preference pairs 表述为已具备 DPO 数据；
- 不把原始 21 条失败统一归因于 embedding 召回。
