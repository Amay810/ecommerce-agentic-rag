# 可验证电商客服 Agent 学习路线 v2

状态：研究与实施方案；取代“仅以 τ³ 74 条 teacher rollout 做一次 SFT”的主线，但不覆盖旧实验的历史记录。  
定版日期：2026-08-08。  
目标模型：`Qwen/Qwen3-4B-Instruct-2507`。  

## 0. 证据纪律

本文只使用三类表述：

- **仓库事实**：已在本地文件、数据或官方仓库中核实；
- **论文依据**：论文报告过该方法或现象，但不代表能在本项目复现；
- **待检验假设**：必须通过预注册实验决定，不提前写成结论。

任何由 LLM 生成的任务、轨迹、Skill 或评分都只是候选；没有通过真实工具执行、状态检查或独立人工核验，不进入正式训练或结论。

## 1. 研究问题与主张边界

唯一主问题：

> 在相同 Qwen3-4B、相同外部测试集和相同推理配置下，使用“可执行环境验证的数据”进行 Agent-turn SFT，并进一步进行多轮 GRPO，能否提高未受本项目 Action Constraint 保护的电商客服 Agent 能力？

次问题：

1. 失败定向的数据生成是否比只重复官方 train task 更有效？
2. 非合作用户是否揭示 cooperative simulator 隐藏的失败？
3. 经配对回放验证的 Skill 是否能在冻结模型上提供独立增益？

允许的三类贡献：

| 结果 | 允许的表述 |
|---|---|
| 外部测试上 Base → SFT 或 SFT → RL 配对提升 | 模型策略能力提升 |
| 只有 Skill on 提升，模型权重未变 | 推理时程序性知识改善 |
| 只有 Guardrail/Constraint on 提升 | 系统安全或可靠性改善 |

禁止把内部 40/120 回归修复、grader 调试、协议修补写成模型学习贡献。

## 2. 当前资产：能用什么，不能声称什么

### 2.1 本地已核实资产

| 资产 | 数量/范围 | 来源 | 本路线用途 |
|---|---:|---|---|
| typed tools | 8 | `ecommerce_rag/tool_schema.py` | 原生环境、格式与安全回归 |
| 原生事务工作流 | 1 个真实写流程：创建退货申请 | `ecommerce_rag/tools.py` | 安全、幂等、确认验证；不外推为完整客服业务 |
| 合成用户/订单 DB | 1,000 用户、10,000 订单，固定 seed | `ecommerce_rag/orders.py`、`agent_env_v2.db` | 可重置原生训练/诊断环境 |
| 商品语料 | 5,000 商品 | Amazon Reviews 2023 固定 revision | 研究环境的 RAG 事实；不直接进入模型权重训练 |
| 内部 Agent tasks | 120，7 类，dev/locked 各 60 | `harness_tasks_v2.jsonl` | 冻结安全与功能回归，不训练 |
| τ³ Retail | train 74 / test 40 | 本地 `v1.0.1`，commit `fc0055dc...` | 领域训练种子与外部主评测 |

当前 8 个工具覆盖商品检索、商品读取/比较、政策查询、订单读取、退货资格、创建退货申请、转人工。退款执行、换货、取消订单、修改地址/支付等并不存在于原生工具面，因此 v2 不得声称原生环境已覆盖这些能力。

### 2.2 扩大业务面的依据

τ³ Retail `v1.0.1` 在 MIT 许可证下公开了 Retail policy、DB 和工具实现。本地核实的公开工具包括：用户/订单/商品查询、取消 pending order、修改地址/商品/支付、退货、换货和转人工等。

采用方式：

- **不复制 test 40 的任务内容**；
- 通过 adapter 直接运行官方 Retail 环境；
- train 74 可用于 teacher rollout 和训练；
- 基于公开 policy、tool schema、DB 另行生成的任务必须保存独立 provenance，并与 test 40 去重；
- 原生环境和 τ³ 环境分别记分，不把两套数据库状态混为同一任务。

注意：τ² 论文描述了 compositional task generator，但本地 `v1.0.1` 只发现 Telecom 的 `create_tasks.py`，未发现可直接复用的 Retail generator。因此 Retail Task Compiler 是本项目需要实现的新组件，不能声称官方已有现成实现。

## 3. 数据来源总账

### 3.1 允许进入训练的数据

| 数据源 | 已核实事实 | 具体用途 | 限制与处理 |
|---|---|---|---|
| τ³ Retail train | 74 tasks；MIT；官方 policy/tool/DB/user simulator | 领域 seed、teacher rollout、SFT/RL 环境 | test 40 永不训练；固定 `v1.0.1/fc0055dc...` |
| 自建 verified retail trajectories | 从 τ³ MIT 环境或原生环境产生 | 主要领域训练数据 | 必须按 §4 生成、执行、拒收和去重 |
| APIGen-MT-5k | 5,000 条；Retail+Airline；CC-BY-NC-4.0；三层验证；数据卡含额外用途限制 | **可选**通用多轮 tool-use warm-up；默认只考虑 Airline 子集 | 使用前必须接受条款；不得用于商业用途；Retail 子集默认禁用，避免 τ 系测试污染 |
| ABCD train | 10K+ 人类客服对话、55 intents、30 actions；MIT | 提取 customer utterance 的语言现象与信息披露模式 | 不采用其 action label 训练本项目工具；不得把 ABCD policy 当成本项目 policy |

`APIGen-MT-5k` 不是默认必用数据。若许可证/使用条款与项目发布目标不一致，整个来源删除，不影响主路线。

### 3.2 只允许用于评测的数据

| 数据源 | 用途 | 不允许的用途 |
|---|---|---|
| τ³ Retail test 40 | 领域主评测 | prompt 调参、错误驱动生成、训练、Skill 构建 |
| BFCL Multi-Turn | 通用 function-calling 泛化 | 训练或选择领域数据模板 |
| ECom-Bench | 最终真实电商客服迁移评测；EMNLP 2025，Apache-2.0 | 日常调参或反复查看 test failure |
| 原生 locked 60 / 退货 40 | 安全和历史回归 | 训练、生成模板、模型能力主张 |

### 3.3 只作为环境事实或模拟器研究材料的数据

| 数据源 | 处理决定 | 原因 |
|---|---|---|
| Amazon Reviews 2023 5K slice | 只供 RAG 环境读取，不进 SFT/RL target，不随训练数据再分发 | 维护者明确未为数据分配许可证，仅认可研究用途 |
| RealUserSim | 默认不进训练；可在单独获准后用于模拟器 fidelity 研究 | ODC-BY，但数据卡含 OpenAI 辅助生成及用途限制 |
| JDDC/JDDC 2.0 | v2 不依赖 | 访问与授权不稳定，且不提供本项目可执行工具标签 |
| Amazon ESCI | 不进入本项目主实验 | 检索相关性数据，不回答多轮客服工具策略问题 |

## 4. 自建数据如何生成

### 4.1 Task Blueprint：先生成可执行规格，不先写对话

每个任务蓝图必须是机器可读对象，最少包含：

```json
{
  "task_id": "...",
  "environment": "tau3_retail|native_retail",
  "source_policy_version": "...",
  "tool_graph_hash": "...",
  "db_snapshot_hash": "...",
  "initial_state": {},
  "user_goal": {},
  "private_user_facts": {},
  "disclosure_schedule": [],
  "required_effects": [],
  "forbidden_effects": [],
  "acceptable_terminal_conditions": [],
  "reference_tool_paths": [],
  "behavior_profile": "cooperative|incomplete|impatient|digressive|unsupported_request|goal_shift",
  "generator_version": "...",
  "generator_prompt_hash": "..."
}
```

这里的 `reference_tool_paths` 是允许路径集合而非强制唯一答案。终态相同且未违反政策的替代路径可以通过。

依据：APIGen-MT 将 blueprint 与完整 trajectory 分离；Magnet 从 function-signature path 构造多轮数据；τ² 使用原子组件生成可验证组合任务。

### 4.2 工具依赖图

图节点不是自然语言意图，而是可执行动作及状态谓词，例如：

```text
get_user_details
  → pending_order_exists
  → modify_pending_order_address

get_order_details
  → delivered
  → item_in_order
  → user_confirms
  → return_delivered_order_items
```

每条边必须来自以下之一：

1. 工具 schema 的输入/输出字段；
2. 官方 policy 的明确前置条件；
3. 工具实现中的状态检查；
4. 人工审阅并写入版本化规则。

不得让 LLM 自行发明边。LLM 只可提出候选边，程序执行失败或无法在源码/policy 中定位依据即拒绝。

### 4.3 蓝图生成

1. 从干净 DB snapshot 选择真实存在的实体；
2. 枚举满足前置条件的合法工具路径；
3. 对同一初态构造合法目标、不可达目标和需要 handoff 的目标；
4. 添加信息披露计划和用户行为条件；
5. 在环境中执行 reference path，得到 expected effects；
6. 重置环境并执行第二次，确认可重复；
7. 生成 blueprint hash 后才进入自然语言阶段。

业务覆盖不使用“把模板换 1,000 个订单号”的计数。覆盖报告按下列轴记录：

- task family；
- tool-path length；
- read/write/handoff；
- initial-state predicate；
- required clarification；
- user behavior；
- success / impossible / unsafe；
- seen composition / held-out composition。

训练规模由 learning curve 决定：先生成能覆盖全部适用原子条件和二元组合的最小集合，此后按累计 verified token 数几何扩展；每次扩展只使用自建 dev 选择，连续扩展不再改善时停止。本文不预先编造“必须 600 或 1,000 条”。

### 4.4 自然语言与用户模拟

LLM 接收 blueprint，只生成：

- 用户初始表达；
- 后续信息披露的表面形式；
- 合作程度和语言风格；
- 不改变 latent goal 的改写。

行为类型依据 `Non-Collaborative User Simulators for Tool Agents`：不可用服务、跑题、不耐烦、信息不完整；本项目另将 goal shift 单列，因为电商客服中用户可能由查询转为修改/取消。

ABCD 只用于抽取诸如短句、反问、分段披露、修正前述信息等语言现象；不能复制其具体客户实体、公司规则或 action sequence。

模拟器不得看到 `required_effects`、`reference_tool_paths` 或 grader 结果，只能看到用户私有事实和 disclosure schedule。

### 4.5 轨迹采集

每个 blueprint 至少经过以下角色之一：

- teacher agent：生成高质量候选轨迹；
- current policy：生成 on-policy 成功与失败轨迹；
- scripted executor：仅用于验证 reference path，不进入语言训练。

训练轨迹保留 assistant 自己生成的 clarification、普通回复和 tool call；工具观察必须来自真实环境。禁止把 teacher 编造的 tool response 写入训练数据。

### 4.6 强制拒收条件

出现任一条件即拒收：

- blueprint 引用不存在的实体或工具；
- reference path 无法执行或重复执行终态不一致；
- required/forbidden effects 与真实 DB diff 不一致；
- assistant 调用了 schema 外工具或参数无法解析；
- 工具观察不是环境真实返回；
- 轨迹发生未授权写入、错用户/错订单操作；
- terminal success 仅由 LLM judge 支持，没有状态或规则证据；
- 与 τ³ test 40 在规范化目标、实体无关工具路径和关键约束上高度重合；
- provenance 字段不全；
- generator、teacher、environment 或 policy 版本不可复现。

LLM semantic reviewer 只能作为附加过滤器，不能覆盖确定性失败。

### 4.7 去重与 split

去重分三层：

1. 文本近重复：规范化文本 hash + embedding 相似度候选；
2. 结构近重复：`task_family + tool_path + state_predicates + required_effects`；
3. 外部污染：与 τ³ test 40 的结构签名比较。

自建数据先按结构签名分组，再分 train/dev/diagnostic-test；同一结构组不能跨 split。实体 ID、LLM paraphrase seed 和 user profile 也不能跨 split。自建 diagnostic-test 只证明生成器内泛化，不能替代外部 benchmark。

## 5. 训练数据格式

统一采用 ms-swift messages/tool schema，而不是本项目五字段 action envelope：

```json
{
  "messages": [
    {"role": "system", "content": "policy..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "需要先确认..."},
    {"role": "user", "content": "..."},
    {"role": "tool_call", "content": "{\"name\":\"get_order_details\",\"arguments\":{...}}"},
    {"role": "tool_response", "content": "{...}"},
    {"role": "assistant", "content": "..."}
  ],
  "tools": [...],
  "provenance": {...}
}
```

损失掩码：

- system、user、tool_response：不计算 loss；
- assistant clarification/final、tool_call：计算 loss；
- 不训练隐藏 blueprint、reference path、grader reasoning。

这不是纯“action-only”，因为客服能力也包含澄清与解释。另做 action-only ablation，回答自然语言监督是否必要。

## 6. 训练框架

### 6.1 统一框架选择

使用 [ModelScope ms-swift](https://github.com/modelscope/ms-swift)，实施时 pin `v4.2.2`（GitHub release 页面显示短 SHA `f279713`；P0 必须解析并记录完整 commit），原因：

- 官方支持 `Qwen3-4B-Instruct-2507`；
- 同一框架支持 LoRA SFT、vLLM、GRPO；
- 提供 agent template、`tool_call`/`tool_response` 格式；
- 官方文档明确 tool call 参与 loss、tool response 不参与；
- 提供 `MultiTurnScheduler` 和自定义异步 Reward 插件。

`Agent Lightning`保留为后续运行时—训练解耦参考，不在 v2 第一版同时引入，避免两套 tracing/store。`MUA-RL`官方示例要求多节点高端 GPU，不作为本项目默认实现。`Skill Self-Play`官方脚本假设 8 GPU，只借鉴 frontier curriculum，不直接复制整套训练。

所有 Python/CUDA/PyTorch/vLLM 依赖必须在 NSCC 预检后生成 lock；当前仓库没有足够证据写死 CUDA 和 GPU 数量。

### 6.1.1 当前不能伪造的四个待决事实

| 待决项 | 为什么现在不能填写 | 决定方法 |
|---|---|---|
| teacher model/version | 当前没有已确认可用且版本不可变的强模型端点 | 仅在 τ³ train 和自建 dev 上比较候选 teacher 的环境成功率、违规率与成本，选择后记录精确 model ID/date/hash；不看 test 40 |
| user simulator model/version | τ³ 分数对 simulator 很敏感，当前环境变量未给出 | 沿用一次 P0 smoke 通过的固定模型，并写入所有结果；中途更换则整组结果无效 |
| NL assertion judge | τ³ v1.0.1 单独调用 judge，不能从 agent 配置推断 | P0 明确设置并冻结模型 ID；Base/SFT/RL 完全一致 |
| NSCC GPU/CUDA/vLLM | 之前 SSH 预检未完成 | 恢复连接后只读查询硬件、driver、CUDA、Python；再选择 ms-swift 官方兼容组合并生成 lock |

teacher 选择不是“用最贵模型即可”。候选必须能输出标准 tool calls，并在 train-only calibration 上真实执行；没有一个候选明显强于学生时，暂停 teacher-distillation，改用环境成功的 on-policy rejection sampling。

### 6.2 Stage A：Agent-turn LoRA SFT

训练臂：

```text
B0  Base Qwen3-4B
S1  τ³ train teacher trajectories
S2  τ³ train + verified self-built trajectories
S3  S2 + optional APIGen Airline warm-up
```

S3 只有在数据许可 gate 与 BFCL 去重通过后才运行。

ms-swift 官方 Qwen3-4B 示例的 `LoRA rank=8, alpha=32, lr=1e-4`只作为 smoke 起点，不冒充本任务最优参数。正式超参数在自建 dev 上选择并冻结；τ³ test 不参与选择。`max_length`由训练轨迹 token 长度分布和显存预检确定，并报告删除/截断比例，不能静默截断工具链。

模型选择指标：自建 dev 的环境成功率优先；若相同，依次比较非法写入、schema error、平均轮数。不能用训练 loss 单独选 checkpoint。

### 6.3 Stage B：多轮 GRPO

前置条件：

1. S2 相对 B0 在自建 dev 有非零提升；
2. 环境可并发重置，每个 rollout 使用独立 DB copy；
3. Reward 全部由环境和规则计算；
4. 同 prompt 的多次 rollout 中存在不同 reward，否则 group-relative advantage 为零；
5. 先完成小规模 stability run，确认无 tool-call 格式崩溃和梯度异常。

使用 ms-swift `MultiTurnScheduler`：模型输出 tool call → scheduler 调用 τ³/native adapter → 返回真实 observation → 继续 rollout，直至终态或 max turns。

v2 主 Reward 不使用任意加权和：

```text
R = -1  若出现 schema 外动作、越权/非法状态变化或不可恢复的协议破坏
R =  1  若环境 goal/终态全部通过且无上述违规
R =  0  其他合法但未完成的轨迹
```

身份验证、显式确认、禁止写入等过程条件由 verifier 转成 hard validity gate。轮数、token 和 handoff 只作诊断指标，不在主 Reward 中人为调权。

第二轮消融才引入 CM2 风格 checklist reward，逐条记录 checklist 来源和 judge；不能把它和主结果混写。训练算法先用标准 GRPO；只有发现长度偏置或 group normalization 问题，才预注册 Dr.GRPO/GSPO 等替代，不能看 test 分数后换算法。

正式对比：

```text
B0: Base
S2: verified SFT
R1: S2 → GRPO
```

若算力允许，再加入相同 rollout 预算的 Base → GRPO，检验 SFT cold start 是否必要；它不是最低交付要求。

## 7. Skill 路线：独立实验，不与模型训练混归因

Skill 只从 train 轨迹构建。流程：

1. 按 first actionable fault、状态谓词、工具路径聚类成功/失败轨迹；
2. 只有跨实体重复的故障才提出 Skill；
3. 多条轨迹并行提取局部 lesson，再合并冲突，依据 Trace2Skill；
4. Skill 必须声明 applicability、preconditions、procedure、forbidden actions、recovery、evidence 和 verifier；
5. 在同一组 dev blueprint 上做 skill-off/on 配对 rollout；
6. 出现任何新增非法状态变化即拒绝；净修复为零或负数即拒绝；
7. 通过的 Skill 再在未参与构建的 diagnostic-test 上运行一次。

主指标是 paired repairs、regressions 和 net effect，不使用“Skill 被检索/被引用次数”代替效果。论文依据是 Trace2Skill 的跨轨迹汇总，以及 SkillGen/SkillOpt 类工作的 paired/held-out validation；材料同时显示未经验证的 Skill 可能负优化，因此不设置自动上线。

## 8. 评测协议

### 8.1 外部主评测：τ³ Retail test 40

固定项：

- benchmark `v1.0.1/fc0055dc...`；
- test 40；
- 相同 user simulator model/version；
- 相同 NL assertion judge；
- 相同 decoding 参数和 max steps；
- Constraint off；
- 每个模型使用相同 task × seed 配对运行。

指标：

- `pass@1`：单次任务成功率；
- `pass^k`：同一任务 k 次全部成功的比例，k 在成本 smoke 后冻结；
- write-action checks；
- 非法/错误状态变化；
- 平均轮数、tool calls、tokens、wall time；
- 按 task family 报告，但 40 题子组只作描述，不作强泛化结论。

统计方法：

- 对 Base/SFT/RL 使用同任务同 seed 配对；
- 二元 task success 使用 exact McNemar test；
- 差值置信区间使用以 task 为 cluster 的 paired bootstrap；
- 多次 rollout 不能当作独立任务扩大样本量；
- 同时报告修复数与退化数；
- 40 题若区间仍跨 0，结论为 inconclusive，不用内部数据补成 positive。

### 8.2 外部泛化

1. **BFCL Multi-Turn**：只测 Base、S2、R1，证明工具能力是否跨出电商环境；使用官方 executable/state-based evaluator。
2. **ECom-Bench**：最终冻结 checkpoint 后各跑一次；报告官方 action/search/output/time 指标和 pass 指标；不用于选模型。

### 8.3 原生安全回归

冻结内部 120 和退货 40，分别报告：

- raw policy operational success；
- policy compliance；
- forbidden-tool attempts；
- terminal-state accuracy；
- illegal state changes。

同时报告 Constraint off/on，但只有 off 进入模型能力结论；on 展示部署安全性。

### 8.4 非合作用户压力测试

从未见 blueprint 构建以下 paired variants：

- cooperative；
- incomplete disclosure；
- impatience；
- digression；
- unsupported request；
- goal shift。

同一 latent goal、初态和模型保持不变，只改变 user policy。报告相对 cooperative 的成功率下降、幻觉工具率、错误写入、恢复成功、handoff precision 和额外轮数。

该集合是本项目 stress test，不冒充真人效果。依据 `Mind the Sim2Real Gap` 的 451 人研究，LLM simulator 普遍过度合作且风格单一；因此最终报告必须明确“simulated-user robustness”，不能写“真实用户满意度”。

### 8.5 人工审核

人工只审核机器指标覆盖不到的内容：事实一致性、解释完整性、语气、是否误导用户、handoff 理由。样本在看模型标签前按 task family 和成功/失败分层抽取；审核员看不到模型身份；报告一致率和原始计数。人工审核不覆盖环境已能确定的 DB 终态。

## 9. 数据量与统计问题

### 9.1 为什么不再用固定“40 条证明能力”

40 个外部 test task 仍可用于可比 benchmark，但估计精度有限。项目必须报告置信区间，不把两题变化的 `5pp` 自动解释成稳定提升。

### 9.2 自建评测集如何确定规模

预先按所需置信区间计算，而不是拍脑袋。若估计单个主要能力的成功率，希望 95% 区间半宽不超过 `e`，保守取 `p=0.5`：

```text
n ≈ 1.96² × p(1-p) / e²
```

例如 `e=0.10` 时约需 97 个独立任务。这里的独立单位是 blueprint 结构组，不是同一任务的重复 rollout。最终 n 还要根据 paired discordance 和预算做 power analysis，并在生成前写入 manifest。

### 9.3 训练集如何确定规模

不承诺固定条数。采用 verified-token learning curve：

1. 先达到结构覆盖 gate；
2. 以几何增长的累计 token 预算训练 checkpoint；
3. 只在自建 dev 测增益；
4. 记录数据量—性能曲线；
5. 边际收益停止后不再机械扩写。

这样能区分“样本更多”和“覆盖更有效”，也避免把同模板换实体伪装成大数据。

## 10. 理论与论文映射

| 本项目设计 | 直接依据 | 借鉴边界 |
|---|---|---|
| blueprint → trajectory 两阶段 | APIGen-MT, arXiv:2504.03601 | 不复制其受限数据即可复用思想 |
| 工具依赖图生成路径 | Magnet, arXiv:2503.07826；FunReason-MT, arXiv:2510.24645 | 图边必须由 schema/policy/源码验证 |
| 可组合、状态可验证任务 | τ²-Bench, arXiv:2506.07982 | Retail generator 需本项目实现 |
| 环境反馈而非 LLM 自判 | Simia, arXiv:2511.01824；ASTRA, arXiv:2601.21558 | LLM 可模拟用户，不决定交易真值 |
| 从重复失败生成定向环境 | TRACE, arXiv:2604.05336 | 官方摘要报告 τ² +14.1，不采用 Consensus 文中的 +15.3 |
| SFT cold start + RL refinement | MUA-RL, arXiv:2508.18669；AReaL-SEA, arXiv:2601.22607 | 是否优于纯 SFT 仍由本项目实测 |
| checklist reward 消融 | CM2, arXiv:2602.12268 | 主实验先用确定性 Reward，避免 judge 混淆 |
| Skill 跨轨迹汇总 | Trace2Skill, arXiv:2603.25158 | 论文主要不是客服领域，需独立验证 |
| Skill paired validation | SkillGen, arXiv:2605.10999；SkillOpt, arXiv:2605.23904 | 不以使用率代替因果效果 |
| 非合作用户压力测试 | arXiv:2509.23124 | 训练泛化证据弱于评测证据 |
| simulator 不等于真人 | Mind the Sim2Real Gap, arXiv:2603.11245 | 只声称 simulated robustness |
| skill-routed frontier curriculum | Skill Self-Play, arXiv:2607.22529 | 其公开脚本默认 8 GPU，本项目后置 |

## 11. 实施任务包与验收

### P0：来源与环境冻结

产物：`data_source_manifest.json`、依赖 lock、benchmark commits、许可证快照。  
验收：每条数据能回答“来自哪里、允许做什么、是否接触 test”。

### P1：Retail Task Compiler v0

产物：tool graph、blueprint schema、generator、deterministic replay verifier、coverage report。  
验收：生成任务均可从干净 snapshot 重放两次；required/forbidden effects 与 DB diff 一致；test contamination 为 0。

### P2：Trajectory Factory

产物：用户模拟 adapter、teacher/current-policy rollout、turn-level validator、ms-swift exporter。  
验收：工具 observation 100% 来自环境；provenance 完整；拒收原因可统计。

### P3：SFT

产物：B0/S1/S2（可选 S3）、训练配置、token/coverage learning curve。  
验收：自建 dev 提升且安全不退化后，才允许第一次 τ³ test。

### P4：GRPO

产物：MultiTurnScheduler、hard Reward、group reward variance 报告、R1。  
验收：训练稳定；没有格式崩溃；完成 B0/S2/R1 配对外测。

### P5：Skill 与非合作用户

产物：版本化 Skill、off/on 配对结果、stress-test report。  
验收：Skill 无安全回归且 held-out 净效果为正；否则如实关闭。

## 12. 明确不做

- 不继续 SQL Memory advice 路线；
- 不把 40/120 失败改写成训练样本；
- 不允许 test failure 反向驱动 generator；
- 不将 LLM judge 作为交易终态唯一 Reward；
- 不把同模板换订单号计为任务多样性；
- 不同时引入 ms-swift、Agent Lightning、verl-tool 三套训练栈；
- 不在没有 GPU/依赖预检时承诺训练时长和显存；
- 不把模拟用户结果称为真实用户满意度；
- 不把 Skill、Guardrail 或 Memory 的增益记到模型权重上。

## 13. 官方来源

- τ²/τ³ benchmark and code: <https://github.com/sierra-research/tau2-bench>, <https://arxiv.org/abs/2506.07982>
- APIGen-MT paper/data: <https://arxiv.org/abs/2504.03601>, <https://huggingface.co/datasets/Salesforce/APIGen-MT-5k>
- ABCD: <https://github.com/asappresearch/abcd>, <https://arxiv.org/abs/2104.00783>
- BFCL: <https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard>
- ECom-Bench: <https://github.com/XiaoduoAILab/ECom-Bench>, <https://arxiv.org/abs/2507.05639>
- ms-swift: <https://github.com/modelscope/ms-swift>
- Magnet: <https://arxiv.org/abs/2503.07826>
- FunReason-MT: <https://arxiv.org/abs/2510.24645>, <https://github.com/inclusionAI/AgenticLearning>
- Simia: <https://arxiv.org/abs/2511.01824>, <https://github.com/microsoft/Simia-Agent-Training>
- TRACE: <https://arxiv.org/abs/2604.05336>
- CM2: <https://arxiv.org/abs/2602.12268>
- Trace2Skill: <https://arxiv.org/abs/2603.25158>, <https://github.com/Qwen-Applications/Trace2Skill>
- Skill Self-Play: <https://arxiv.org/abs/2607.22529>, <https://github.com/Qwen-Applications/skill-self-play>
- Non-Collaborative User Simulators: <https://arxiv.org/abs/2509.23124>
- Mind the Sim2Real Gap: <https://arxiv.org/abs/2603.11245>
- Amazon Reviews 2023 usage clarification: <https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/discussions/1>
