# 当前状态

状态日期：2026-07-28。

## 已完成

- Qwen3-4B 在 120 个任务上完成 360 条真实轨迹；原始 SQLite 已固化。
- 40 条系统抽样轨迹已完成人工 success 与 policy compliance 审核。
- 离线 v2 评分以 sidecar 保存，没有覆盖历史轨迹或历史 grade。
- RL gate 同时计算 success agreement 与 policy agreement，并对 sidecar 缺失、重复和 ID 不匹配 fail-closed。
- 商品详情工具只接受内部 `P[0-9]{5}`，政策工具使用五种规范类型并映射到精确语料类别。

## 当前结论

- v2 自动操作成功率 84.17%，终态准确率 100%，策略合规率 95%。
- 40 条审核中 success agreement=80.0%，policy agreement=77.5%。
- preference pairs=0，RL gate 为 false，因此不进入 SFT/DPO。

## 下一步

- 在 NSCC 仅重跑 7 个唯一失败任务，确认商品 ID 与政策类型契约。
- 若工具契约确认通过，再单独研究最终回答 grounding 与推荐质量评分；不通过事后关键词规则拟合人工标签。
- typo/alias 和 no-answer 属于独立的困难检索问题，使用 v3 retrieval benchmark 继续验证，不与这 21 条工具契约失败混为一谈。
