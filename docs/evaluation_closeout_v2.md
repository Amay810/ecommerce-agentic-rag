# 360 条真实 LLM 轨迹评测收尾（v2）

## 结论

Qwen3-4B 已在同一套 120 个任务上完成 3 次确定性重复，共 360 条真实轨迹。
原始评分器给出的 339/360（94.17%）只代表旧口径下的**自动化操作成功率**，
不能解释为经过人工验证的语义任务成功率。

原始 SQLite 保持不变，SHA-256 为
`37e13a3a19c5780793c4f3e99a7b095eaabf9feae9e7c3f5a54693b668fa1408`。
新版评分写入独立 sidecar，不覆盖历史轨迹或历史 grade。

| v2 自动操作指标 | 全部 | dev | locked |
|---|---:|---:|---:|
| 轨迹数 | 360 | 180 | 180 |
| operational success | 84.17% | 83.33% | 85.00% |
| policy compliance | 95.00% | 95.00% | 95.00% |
| terminal-state accuracy | 100% | 100% | 100% |
| forbidden-tool attempt | 5.00% | 5.00% | 5.00% |
| illegal state change | 0% | 0% | 0% |

“禁止工具尝试率 5%”与“非法状态变更率 0%”必须同时报告：guardrail
保护了数据库终态，但 Agent 的决策并非因此自动合规。

## 原始 21 条失败的归因

原始自动评分中的 21 条失败来自 7 个唯一任务，每个任务以相同结果重复 3 次：

- 12 条、4 个政策任务：Agent 使用 `warranty` 等类型调用政策工具，而旧工具接口
  没有稳定的规范类型到语料类别映射；
- 9 条、3 个商品任务：Agent 将 `Ce-H22B12-S1`、`QGHXO`、
  `PSAMGLXTLC26` 等外部型号当成内部 `product_id`，没有先调用商品搜索。

这批失败没有证明 embedding 未召回 gold：9 条商品失败根本没有形成搜索候选集，
12 条政策失败首先是接口类别契约问题。因此本轮不据此升级 embedding、query
rewriting 或 reranker。

新版评分同时暴露了旧自动评分中的漏判，共得到 57 条操作失败：

| 原因 | 轨迹数 |
|---|---:|
| `forbidden-tool-attempt` | 18 |
| `required-tool-failed` | 12 |
| `retrieval-gold-missing` | 27 |

20 个政策任务现在均声明精确的 gold 文档；因此错误政策结果不再因“返回了任意政策”
而通过。

## 40 条人工审核

审核表不是随机样本。导出程序先按自动成功、失败类型、任务 ID、seed 排序，再按固定
步长抽取 40 条，因此下面的比例只描述这 40 条，**不得外推到全部 360 条**。

- 人工判定成功：20/40；
- 人工判定策略合规：25/40；
- 旧评分器 success agreement：57.5%；
- v2 评分器 success agreement：80.0%；
- v2 评分器 policy agreement：77.5%。

人工发现而自动操作评分无法可靠覆盖的问题包括：政策类别答非所问、推荐中混入错误
品类、证据与最终回答矛盾、无依据的商品或库存判断、遗漏用户真正的问题，以及编造
退货时限。自然语言答案质量继续由人工审核，不使用事后关键词规则把一致率拟合到 90%。

## RL 决策

fail-closed gate 当前为 `eligible=false`：

- 40 条人工审核已完成且全部关联到对应 trajectory；
- success agreement 80.0%，低于 90%；
- policy agreement 77.5%，低于 90%；
- 同一任务的三次确定性结果没有形成“成功 vs 失败”对，preference pairs 为 0。

因此当前不进行 SFT/DPO，也不宣称完成 Agent RL。下一次模型运行仅应覆盖上述 7 个
唯一失败任务，用于确认新的商品 ID 与政策类型契约；通过后再决定是否扩量。

## 证据文件

- 原始轨迹：`logs/harness_v2_llm_360.sqlite`
- v2 sidecar：`logs/harness_v2_llm_360_grades_v2.jsonl`
- v2 汇总：`docs/harness_v2_llm_360_regraded_v2.json`
- 人工审核：`docs/trajectory_audit_40.csv`
- RL gate：`docs/agent_rl_gate_regraded_v2.json`
- 偏好对：`logs/action_preferences.jsonl`（0 行）
