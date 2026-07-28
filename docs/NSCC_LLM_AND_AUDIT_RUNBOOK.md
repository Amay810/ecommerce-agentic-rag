# NSCC LLM 运行手册

## 当前状态

- 360 条真实 Qwen3-4B 轨迹已经完成并固化；
- 40 条人工审核已经完成；
- v2 离线评分和 RL gate 已生成；
- gate 未通过，preference pairs 为 0，不进行 DPO；
- 不重复提交完整 360 条作业。

## 七任务契约确认

下一次运行只确认原始 21 条失败对应的 7 个唯一任务：4 个政策任务和 3 个商品型号
任务。作业不会覆盖原始 SQLite，并会对 7 个场景逐项检查成功、合规、终态与动作解析。

```bash
cd /scratch/users/ntu/s250045/ecommerce-agentic-rag-git
git pull --ff-only
qsub nscc/run_llm_contract_confirmation.pbs
```

输出：

- `logs/llm_contract_confirmation_v2.sqlite`
- `docs/llm_contract_confirmation_v2_report.json`
- `docs/llm_contract_confirmation_v2_diagnosis.json`
- `docs/llm_contract_confirmation_v2_gate.json`
- `output_llm_contract_confirmation.log`

Gate 未通过时不扩大运行规模。Gate 通过只说明商品 ID 与政策类别工具契约得到模型端
确认，不改变现有人工审核结论，也不自动授权 DPO。

## 离线重评分与 gate

这两步不加载生成模型，可在登录节点或本地运行：

```bash
python -m scripts.regrade_trajectories \
  --tasks ecommerce_rag/data/harness_tasks_v2.jsonl \
  --store logs/harness_v2_llm_360.sqlite \
  --output-grades logs/harness_v2_llm_360_grades_v2.jsonl \
  --output-report docs/harness_v2_llm_360_regraded_v2.json

python -m ecommerce_rag.rl_gate \
  --tasks ecommerce_rag/data/harness_tasks_v2.jsonl \
  --store logs/harness_v2_llm_360.sqlite \
  --grades logs/harness_v2_llm_360_grades_v2.jsonl \
  --audit docs/trajectory_audit_40.csv \
  --preference-pairs logs/action_preferences.jsonl \
  --output docs/agent_rl_gate_regraded_v2.json
```

原始轨迹 SHA-256 必须保持：
`37e13a3a19c5780793c4f3e99a7b095eaabf9feae9e7c3f5a54693b668fa1408`。
