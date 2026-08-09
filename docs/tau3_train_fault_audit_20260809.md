# τ³ Retail train 轨迹故障审计

日期：2026-08-09

数据源：`tau3_s0_v1_train_rollout_qwen3_4b_k4/results.json`

范围：只使用官方 Retail **train 74×4**；没有用 test trace 设计训练数据。

## 结论

296 次 rollout 不是 296 个独立任务，而是 74 个任务各重复 4 次。任务级结果为：

| 四次成功数 | 任务数 |
|---:|---:|
| 4 | 16 |
| 3 | 9 |
| 2 | 10 |
| 1 | 18 |
| 0 | 21 |

因此 Base 至少成功一次的 train task 是 53/74，其中 37 个不稳定，21 个四次全失败。S0 filtered47 覆盖 40 个任务，缺失的 34 个任务由 **21 个零成功任务 + 13 个曾成功但被质量过滤/去重淘汰的任务**组成；不能把 34 全称为 Base 能力盲区。

167 条失败轨迹的“首个未满足官方 action check”集中在写操作和其前置读取：

| 首个未满足动作 | 失败轨迹 | 涉及任务 |
|---|---:|---:|
| `exchange_delivered_order_items` | 34 | 15 |
| `get_order_details` | 26 | 11 |
| `return_delivered_order_items` | 17 | 8 |
| `modify_pending_order_address` | 17 | 8 |
| `cancel_pending_order` | 17 | 8 |
| `modify_pending_order_items` | 15 | 7 |
| `get_product_details` | 9 | 4 |
| `calculate` | 9 | 4 |
| 仅 NL assertion / DB 无 action check | 11 | 7 |
| 身份查找、item 读取、handoff、用户地址修改 | 12 | 5 |

这支持把下一步放在“状态读取 → 写前确认 → 精确写入”的可靠性上，而不是继续优化表达。119/296 条轨迹出现至少一个工具错误；按“实际写工具数量/类型超出参考写动作”做的保守筛查得到 97 条候选，但其中可能包含任务允许的条件分支，必须逐条结合 policy 判断，不能直接宣称 97 条越权写入。

对写调用前最后一条用户消息做词面筛查得到 19 个“无 yes/confirm/proceed 等确认词”候选。人工抽看发现其中多数仍是明确祈使请求（如 “Just cancel …, please”），所以该数字也不能直接等同“缺少显式确认”。正式数据过滤必须判断 agent 是否在披露具体变更与后果后获得确认，不能用关键词代替协议语义。

## 21 个零成功任务

| task | 参考写动作 | 四次最常见首个故障 | 工具错误次数 | 多余/异类写候选 trials |
|---:|---|---|---:|---:|
| 14 | return ×2 | return | 0 | 0 |
| 19 | return | get order | 8 | 4 |
| 20 | modify items | modify items | 4 | 3 |
| 21 | modify items | get item / get order | 13 | 3 |
| 28 | return ×3 | calculate / return | 5 | 4 |
| 29 | exchange ×2 | exchange | 8 | 3 |
| 30 | return ×2 + cancel | get order / cancel | 0 | 0 |
| 37 | modify items | identity by email | 6 | 4 |
| 41 | address ×2 + items | identity by name/zip | 6 | 1 |
| 46 | return | get order | 0 | 0 |
| 54 | cancel ×2 + return | get order / writes | 12 | 0 |
| 57 | 无参考写动作 | DB mismatch | 0 | 4 |
| 59 | cancel + address | cancel | 3 | 1 |
| 76 | cancel ×2 | cancel | 1 | 2 |
| 85 | modify items | modify items | 4 | 4 |
| 92 | return ×2 | return | 1 | 1 |
| 98 | exchange ×2 + cancel | exchange | 11 | 2 |
| 103 | return ×2 + address + items | items / return | 3 | 2 |
| 104 | return ×3 + address + items | address / items | 4 | 3 |
| 105 | exchange | exchange | 3 | 1 |
| 109 | address + items（另含 user address） | address | 1 | 1 |

“工具错误次数”是四条轨迹中所有 `Error:` tool response 的总数；“候选 trials”只是筛查结果，不是违规判决。

## Hint-conditioned pilot 任务

首批选 8 个零成功 train task：

```text
14, 20, 29, 30, 46, 59, 85, 109
```

选择理由：覆盖多订单退货、复杂 pending items 修改、多次换货、退货+取消的条件链、错误订单号恢复、取消+地址修改、单一 items 修改、地址+items+用户地址的复合链；同时涵盖“前置读取失败”和“最终写动作失败”。这 8 个任务只用于验证 privileged-plan-conditioned self-distillation 能否产出新行为轨迹，不用于模型训练或能力主张。

生成时私下提供官方 `evaluation_criteria.actions` 作为计划提示，但不得把未向用户披露的参数当成用户已知事实。环境仍产生全部工具 observation；导出前剥离提示。只有满足以下条件的轨迹才保留：官方 reward=1、正常终止、无工具错误、身份与授权合法、写前获得语义上的显式确认、无额外/越权写入、提示内容未泄漏到训练记录。

pilot 完成后必须先报告：8 个独立任务、每任务采样上限、生成成功率、严格过滤保留率、结构去重损耗和最终有效任务/轨迹区间。若保留率不足，不进入 Task Compiler 扩量；只依据实际首个失败修正生成提示或停止该路径。

启动前数量漏斗：8 个独立任务 × 每任务 2 次 = 16 次 rollout。由于现有 Base 在完整 train 上的轨迹成功率为 43.58%，而 action-name plan 的实际增益未知，pilot 暂以生成成功率 40%–80%、严格质量保留率 50%–80%规划；每任务最多保留 1 条，预计最终约 2–6 条、覆盖 2–6 个独立任务，硬上限 8 条/8 任务。该区间只用于决定 pilot 成本，不冒充已有证据；实际结果将替换先验并用于后续 Task Compiler 漏斗。

执行使用 τ² 已有 `LLMGTAgent`，但关闭 `provide_function_args`，只提供 action 名序列，避免把订单号、item id、payment id 等未披露参数直接喂给模型。入口为 `scripts/run_tau3_hint_pilot_v1.ps1`；它只发起上述 16 次 rollout，不执行过滤、训练或评测扩量。

离线渲染已核对 8 个任务的提示：action 数分别为 6、10、6、13、7、5、1、3，未出现订单号、7–10 位实体 ID 或 payment method ID。需要注意，官方 `evaluation_criteria.actions` 是评分所需动作，不保证是完整策略：task 85 和 109 的 gold actions 只有目标写操作，没有身份验证或读取动作；其余 6 个任务同时包含认证与读取。因此这两题专门检验模型能否依据 Retail policy 自行补齐前置步骤，不能把 gold action list 当成完整 teacher trajectory。即使官方 reward=1，只要轨迹跳过认证、必要读取或显式确认，仍按严格质量规则拒收。
