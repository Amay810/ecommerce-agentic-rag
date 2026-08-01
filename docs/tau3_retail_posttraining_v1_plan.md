# τ³-bench Retail 后训练实验 v1（定版方案）

状态：决策已冻结，任务包 A 已完成，任务包 B 待实施。
定版日期：2026-08-01。
本文件用于向实施 agent 交接。**未经方案所有者同意，不得增删下方「冻结决策」与「不做清单」中的任何条目。**

---

## 1. 实验要回答的唯一问题

> 同一个 Qwen3-4B，在公开、可执行、非自建的电商客服环境中，经过同环境后训练后，**未受运行时约束保护的裸 agent 能力**是否提高？

这是一个可证伪的问题。v1 的全部工作只服务于它。

### 1.1 判据（三档，事先约定，不得事后调整）

| 观测结果 | 允许的结论 |
|---|---|
| τ³ retail test split 的 task success / pass¹ 上升 | **Agent 能力提升** |
| 仅内部 40 条 / 120 条上升，外部未动 | **过拟合**，不得声称能力提升 |
| 仅在开启 Action Constraint 时终态成功率上升 | **系统改善**，不是模型能力改善 |

### 1.2 v1 主指标

在 τ³ retail **test split（40 题）**、**Action Constraint 全程关闭**下测量：

- task success / pass¹
- 多次独立运行的一致性（pass^k，k 值见 §6 成本约束）
- 非法或错误写操作次数

已否决的指标：**"raw policy 合法动作率"**。原因：τ³ 多数状态下合法读操作不唯一，构造逐步 evaluator 等于重新掉回自建评测工程。

---

## 2. 冻结决策

| 决策 | 结论 | 理由 |
|---|---|---|
| 主评测 | τ³-bench Retail，**英文** | 公开、可执行、有官方 grader 与用户模拟器 |
| 语言轴 | **v1 全英文**；中文另开 v2，用独立 adapter | 中英混训 → 结果不可归因 |
| 实验臂 | **两臂**：Base / SFT，均 Constraint off | 见 §4.3 |
| 第三臂（SFT + Constraint） | **v1 砍掉** | 见 §4.3 |
| 主训练数据源 | τ³ retail **train split（74 题）** 的 teacher rollout | 同环境数据是 xLAM-2 增益的真实来源 |
| ABCD | **降为第二轮消融**，不进 v1 主线 | 其 30 个动作绑定 ABCD 自有政策，跨域迁移性存疑 |
| JDDC | **v1 删除**（非延后） | 需机构邮箱申请授权，周期不可控；且无 typed-tool action 标签 |
| Amazon ESCI | **移出本实验**，如需另立项 | 评的是检索器，action-only LoRA 物理上无法改变其数字 |
| ECom-Bench | 全流程**只在最后跑一次**，报 pass¹ 与绝对分，**不作 gate** | GPT-4o 在其上 pass³ 仅 10–20%，对 Qwen3-4B 分辨率不足 |
| DPO | **延后至 SFT 之后** | 偏好对需要 SFT 产生的干净失败分布 |
| 内部 120 / 退货 40 | **永久冻结为回归集**，不扩写、不进训练、不参与能力优化 | 只用于验证安全契约未回归 |
| SFT 目标格式 | **标准 tool-call JSON**，不用项目内五字段 envelope | 见 §4.1 |

---

## 3. 外部资源定版

| 资源 | 定版信息 |
|---|---|
| τ³-bench 仓库 | `https://github.com/sierra-research/tau2-bench`（**不要用旧的 `sierra-research/tau-bench`**） |
| Pin | tag `v1.0.1`，commit `fc0055dc4e0a316c3f83133267fbd6faaa770992` |
| 本地路径 | `E:\cv_codex\external\tau2-bench`（**刻意置于项目仓库之外**，避免嵌套仓库） |
| Python 要求 | `>=3.12,<3.14`；本机 3.13.7 满足 |
| 安装 extra | `.[gym]`（gym 接口供后续 RL 用；v1 不用 RL，但一次装好） |
| 模型接入 | 经 **LiteLLM**，需 OpenAI 兼容 HTTP 端点 |
| Retail split | train **74** / test **40** / base **114**，见 `data/tau2/domains/retail/split_tasks.json`（已核实） |

### 3.1 版本纪律（强制）

τ³ 官方明示：**< 1.0.1 与 >= 1.0.1 的 grading 不可比**。本实验横跨数周、要做三次测量（训练前 / SFT 后 / 最终冻结），期间**任何升级都会作废全部对比**。

要求：每一次产出的结果文件中必须内嵌 `tau2_commit`、`user_simulator_model`、`agent_model`、`pass_k` 四个字段。缺任一字段的结果视为无效。另记录 `nl_assertions_model`；该 judge 在 Base/SFT 间也必须完全一致。

### 3.2 用户模拟器是最大隐藏变量

τ³ 分数对"用哪个模型当模拟用户"高度敏感。**三次测量必须使用完全相同的 user simulator 模型与版本**，并写入结果文件。中途更换 = 结果作废。

---

## 4. 兼容性核对结论（已完成，实施 agent 直接采用，无需重做）

### 4.1 Action envelope 映射

项目内 `ecommerce_rag/domain.py` 的 `AgentAction`：`action_type`(tool_call | final_answer | handoff) + `tool_name` + `arguments` + `content` + `requires_user_response`。

| 字段 | τ³ 对应 | 备注 |
|---|---|---|
| `action_type=tool_call` + `tool_name` + `arguments` | assistant message 的 tool_calls | 直接对应 |
| `action_type=handoff` | 工具 `transfer_to_human_agents` | τ³ retail 原生工具，不需特判 |
| `action_type=final_answer` | 普通 assistant 文本消息 | **不再终结 episode** |
| `requires_user_response` | **无对应物** | τ³ 是真多轮，"问用户"与"结束"本就不是同一件事 |

**结论**：后两个字段在 τ³ 中语义塌缩。因此 **LoRA 训练目标必须用标准 tool-call JSON**，不能用五字段 envelope，否则是在教模型一个只有本项目 harness 认识的方言。

### 4.2 复用判定

| 组件 | 判定 | 依据 |
|---|---|---|
| User simulator | **不复用**，用 τ³ 官方的 | `harness.py:219` 的 `UserSimulator` 是脚本化、硬编码中文、逻辑绑死退货四分支（验证码/确认/退货原因/订单号） |
| Session loop | **由 τ³ 驱动**，本项目 policy 包成 τ³ 的 agent 插件 | 这样官方用户模拟器与 grader 白拿 |
| `max_steps` | **必须从 8 调高** | `harness.py:444` 默认 `max_steps=8`；τ³ retail 任务平均十余轮，8 步会让任务在中途被截断，表现为"模型不行"，实为自伤 |
| 数据库 | **完全归 τ³**；`ecommerce_rag/orders.py` 一行不参与 | — |
| Grader | **只用官方**；`GradeResult` 的 40+ 字段一个都不映射 | 那是内部回归集资产 |

### 4.3 raw / constraint 开关：已免费存在

`ecommerce_rag/action_constraint.py:86` 的 `contract_from_progress()` 首行即 `if progress.workflow != "return_resolution": return None`。τ³ 任务不属于该 workflow，故 `apply_action_constraint()` 自动退化为透传。

- **两臂（Base / SFT，Constraint off）开箱即用，无需写开关。**
- **第三臂在 τ³ 上不可实施**：约束层依赖 `legacy_closure.py` 的退货专用 progress reducer。为 τ³ 重写 reducer = 重新掉回评测工程，且会干扰后训练归因。故 v1 砍掉；"系统兜底是否有用"由内部 40 条回答，那本就是它该回答的问题。

---

## 5. 数据隔离规则（强制）

```text
teacher rollout 只在 train 74 上生成
最终评测只在 test 40 上进行
训练启动后，base 114 的成绩不再作为正式成绩引用
```

补充约束：

1. **不得**把同一条标准答案机械复制成数百条。扩量方式只能是：对 train 任务做**多次独立用户模拟与 teacher rollout**，再由环境回放过滤，只保留环境验证成功的轨迹。
2. 若后续引入 `Salesforce/APIGen-MT-5k`：该数据集由 τ-bench retail/airline 域合成，**必须先做任务级去重核对**，确认与 test 40 无重叠，并把隔离方式写入结果文档。否则最终数字不成立。
3. 人工审核只覆盖**准备进入训练集的少量案例**，不是人工评全部轨迹。

---

## 6. 阶段划分

### v1a（先做，验证链路）

```text
74 个官方 train task → teacher rollout → 环境回放过滤 → action-only SFT (LoRA)
```

目的：验证后训练链路本身能跑通。**暂不搭建完整 APIGen-MT 工厂。**

### v1b（仅在 v1a 数据量不足时启动）

参照 APIGen-MT 的真实做法——不是给现成 τ 题生成答案，而是：

```text
读取同环境 API / policy / DB
→ 合成新的任务蓝图及 ground-truth actions
→ 执行校验 + 政策校验 + 语义审核
→ 模拟用户—Agent 多轮交互
→ 只保留环境验证成功的轨迹
```

### 成本约束

τ³ 是多轮会话 + pass^k 重复运行，token 消耗比本项目现有 harness 高一个量级。**在全量跑之前，必须先由 smoke 量出单任务成本**，再据此确定 k 值与总预算。

---

## 7. 实施任务包

### 任务包 A：环境就绪 + 管道 smoke

前置状态（已完成部分）：

- 仓库已 clone 至 `E:\cv_codex\external\tau2-bench`，已 checkout 到 `v1.0.1` / `fc0055dc`。
- **`.venv` 创建与依赖安装被中断，状态未知。实施 agent 应先删除 `.venv` 后重建。**

步骤：

1. 重建 venv，`pip install -e ".[gym]"`，记录实际锁定的依赖版本。
2. 确认 retail 域可加载，任务数为 114，split 为 74/40。
3. 用**便宜 API 模型**（如 `deepseek-chat`，项目 `config.py` 默认即此）跑 **5 条 smoke**。
4. 本机 `ERAG_LLM_BASE_URL` / `ERAG_LLM_API_KEY` / `ERAG_LLM_MODEL` 及 `ARAG_*` 六个变量**当前均未设置**，需先配置（参考仓库内 `.env.example`）。

验收：τ³ 能启动；工具调用贯通；轨迹落盘；**产出单题 token 成本与耗时数字**。

**smoke 结果不计入 baseline，不与 Qwen 比较。**

### 任务包 B：NSCC 上的 Qwen3-4B Base baseline

已知冲突：项目现有跑法是 transformers 本地加载（`nscc/run_answer_postprocess_dev_v2.pbs`：`ARAG_AGENT_BACKEND=local`、`ARAG_LOCAL_MODEL=/scratch/.../Qwen3-4B-Instruct-2507`），且集群为 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` 离线态。**τ³ 需要 OpenAI 兼容端点，两者对不上。**

步骤：

1. 在有网环境预先备好 τ³ 及其依赖（litellm 等）的 wheel，供离线安装。
2. 写单个 PBS 作业：后台 `vllm serve` Qwen3-4B → 轮询端口就绪 → 前台 `tau2` 打 localhost。
3. 在 test 40 上跑 Base 臂，Constraint off。

验收：产出含 §3.1 四个必需字段的 Base 结果文件。

### 任务包 C：teacher rollout → SFT → 复测

1. 在 train 74 上采集 teacher rollout，环境回放过滤。
2. action-only LoRA，目标格式为标准 tool-call JSON（§4.1）。
3. 在 test 40 上跑 SFT 臂，Constraint off，与 Base 同 user simulator、同 commit、同 k。
4. 内部 120 / 40 跑一次回归，确认安全契约未破。
5. ECom-Bench 跑一次，报绝对分。

验收：按 §1.1 三档判据之一给出结论。

---

## 8. 不做清单

- 不做 Memory v2。
- 不同时接入七个 benchmark。
- 不为 τ³ 重写 progress reducer。
- 不扩写内部 40 条，不按 τ³ 风格给它加用户模拟器。
- 不把 ECom-Bench 或 ESCI 当作 v1 的 gate。
- 不在 v1 做 DPO / RL。
- 不把 benchmark 的正式 test 任务放进训练集。
- 不把 τ³ 当日常无限调试对象：全程只测三次（训练前 / 第一版 SFT 后 / 最终冻结版）。

---

## 9. 已知环境事实与遗留问题

| 项 | 事实 |
|---|---|
| 本机 Python | 3.13.7（满足 τ³ 要求） |
| 本机 git | 2.54.0.windows.1 |
| `uv` | **未安装**（τ³ 官方文档用 uv；可用 pip 替代） |
| 项目仓库 git | **已恢复**：2026-08-01 已由仓库所有者授权加入 `safe.directory`。 |
| 模型端点 | 六个 `ERAG_*` / `ARAG_*` 环境变量当前均未设置 |

### 9.1 任务包 A 实施记录（2026-08-01）

- 外部仓库已再次核验为 `v1.0.1` / `fc0055dc4e0a316c3f83133267fbd6faaa770992`。
- `.venv` 已删除后用 Python 3.13.7 重建，`.[gym]` 安装成功。
- Python 3.13 需额外安装 `audioop-lts==0.2.2`；否则 v1.0.1 CLI 会因标准库移除 `audioop` 而无法导入。未修改 benchmark 源码。
- `tau2 check-data` 已通过；Retail 数据再次核验为 114，split 为 train 74 / test 40。
- 新增 `scripts/run_tau3_retail_v1.py`：固定两臂与 split，拒绝 `max_steps < 20`，并将 `tau2_commit`、`user_simulator_model`、`agent_model`、`pass_k`、NL assertion judge 及 token/耗时统计写回结果文件；存在 infrastructure error 时结果自动标为无效并返回失败。
- v1.0.1 的 NL assertion grader 独立使用模型，不能由 `--user-llm` 控制；runner 因此增加显式、冻结的 `nl_assertions_model`，避免 Base/SFT 的 grader 漂移。
- 有效 5 条 train smoke 已完成：5 次正常终止、0 infrastructure error、0 DB mismatch；6 个 write action check 中 1 个未匹配。wall time 177.38 秒（35.48 秒/题）、累计上下文 token 415,557（83,111/题）。记录的 Agent/User API 成本合计约 `$0.00771`；grader 成本未计入这两个字段。
- smoke 的 5/5 reward **不计入 Base，不构成能力结论**。机器可读摘要见 `docs/tau3_retail_v1_smoke_result.json`；完整结果留在外部 benchmark 的 `data/simulations/tau3_retail_v1_smoke_valid/results.json`。

---

## 10. 最终项目定位（对外表述）

> v1 在英文公共电商客服环境（τ³-bench Retail）中证明后训练方法的有效性；v2 使用独立中文 adapter 验证向中文客服的迁移。

这条路线把三个问题拆开了：**先证明能不能训练变好 → 再验证中文迁移 → 最后才讨论更大的自生成数据飞轮。**
