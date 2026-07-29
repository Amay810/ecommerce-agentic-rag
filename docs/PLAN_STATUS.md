# 当前状态

状态日期：2026-07-29。

## 已完成

- Qwen3-4B 在 120 个任务上完成 360 条真实轨迹；原始 SQLite 已固化。
- 40 条系统抽样轨迹已完成人工 success 与 policy compliance 审核。
- 离线 v2 评分以 sidecar 保存，没有覆盖历史轨迹或历史 grade。
- RL gate 同时计算 success agreement 与 policy agreement，并对 sidecar 缺失、重复和 ID 不匹配 fail-closed。
- 商品详情工具只接受内部 `P[0-9]{5}`，政策工具使用五种规范类型并映射到精确语料类别。
- terminal-grounding v2 已在 80 条冻结 dev 轨迹上完成；61 条 eligible 答案重生成，
  19 条按协议透传，两个结构与不变性 gate 均通过。
- 冻结的 40-task selection 已完成 80 条打乱答案的 Codex 盲审并按预注册规则聚合。

## 当前结论

- v2 自动操作成功率 84.17%，终态准确率 100%，策略合规率 95%。
- 40 条审核中 success agreement=80.0%，policy agreement=77.5%。
- preference pairs=0，RL gate 为 false，因此不进入 SFT/DPO。
- terminal-grounding 的 base 与 grounded fact pass 均为 34/40（85.0%），配对差值为
  0，bootstrap 95% 区间为 [-7.5pp, +7.5pp]。
- terminal-grounding positive gate 未通过，状态为 `negative_or_inconclusive`；不宣称
  质量提升。

## 后续边界

- terminal-grounding 实验正式关闭：不运行 v3，不调整 prompt/token，不继续 verifier、
  canonical-product、外部 benchmark、SFT 或 DPO。
- 项目下一项独立工作应回到统一 Agent baseline matrix；不得把该工作解释为
  terminal-grounding 的续跑或补救实验。
- 详细结果见 `docs/answer_postprocess_blind_audit_v1_closeout.md`。
