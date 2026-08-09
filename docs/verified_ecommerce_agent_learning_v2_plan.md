# 可验证电商客服 Agent 学习路线 v2

状态：研究与实施方案；取代“仅以 τ³ 74 条 teacher rollout 做一次 SFT”的主线，但不覆盖旧实验的历史记录。  
定版日期：2026-08-08（同日修订一次，见 §12.1；修订项均标注日期，涉及 §2.2、§3.2、§4.1、§6.1.1、§6.2、§6.3、§8.2、§9.1、§11）。  
2026-08-08 收紧一次（标注“收紧/新增”）：新增 §1.1 能力阶梯；收紧三处结论——不把三域 100 题合并成单一电商功效成功率并区分单比例 CI 与 McNemar 配对功效（§9.1、§9.2）；Retail test 40 声明为 test-aware informed 并要求独立 held-out 承担确认性（§3.2、§8.1、§11 P1）；S0 定位为训练链路检查而非新能力证明，P0.5 改为四分支决策（§6.2、§11 P0.5）。  
2026-08-09 执行交接更新：新增 §13，冻结 Windows/NSCC 分工、虚拟环境、路径、隧道、同步、Git 分支、S0 数据漏斗与当前完成状态；这些内容是后续 Agent 的执行事实，不得用旧对话中的临时 job、节点或建议覆盖。
2026-08-09 S0 服务更新：filtered47 四卡 LoRA 训练已完成并产出 `checkpoint-18`；冻结 vLLM 0.10.2 不支持 LoRA+DP，故 Base 服务保留四卡 DP=4，S0 LoRA 服务改为单卡 DP=1。该差异只改变吞吐，不改变能力评测条件，延迟不得跨臂比较。
2026-08-09 S0 诊断更新：Base 与 S0 Retail test 40×4 均已跑满且基础设施错误为 0；S0 为 86/160，Base 为 85/160，任务聚类 95% 区间跨 0，不能主张 S0 带来能力提升。完整审核见 §13.14。
2026-08-09 能力主线删减：S0 归档为过度规模化的 negative control；GRPO、Skill、Telecom/BFCL、完整回归矩阵和多臂消融退出当前主线。新增 train 296 故障审计与 8-task action-name-only hint pilot，先验证新行为数据获取，再决定是否实现 pending-order compiler。
目标模型：`Qwen/Qwen3-4B-Instruct-2507`。  

## 0. 证据纪律

本文只使用三类表述：

- **仓库事实**：已在本地文件、数据或官方仓库中核实；
- **论文依据**：论文报告过该方法或现象，但不代表能在本项目复现；
- **待检验假设**：必须通过预注册实验决定，不提前写成结论。

任何由 LLM 生成的任务、轨迹、Skill 或评分都只是候选；没有通过真实工具执行、状态检查或独立人工核验，不进入正式训练或结论。

## 1. 研究问题与主张边界

唯一主问题：

> 在相同 Qwen3-4B 和相同推理配置下，使用“覆盖 Base 未掌握行为、且经可执行环境验证的数据”进行 Agent-turn SFT，能否提高训练不可见 Retail held-out 上的电商客服能力，同时不增加错误写入？

次问题：

1. privileged-plan-conditioned self-distillation 能否在 Base 零成功任务上产生可严格保留的新行为轨迹？
2. pending-order 新任务能否补足官方 train 的工具与行为覆盖缺口？

允许的三类贡献：

| 结果 | 允许的表述 |
|---|---|
| 训练不可见 Retail held-out 上 Base → SFT 配对提升 | 模型策略能力提升 |
| 只有 Guardrail/Constraint on 提升 | 系统安全或可靠性改善 |

禁止把内部 40/120 回归修复、grader 调试、协议修补写成模型学习贡献。

### 1.1 能力主线与阶段阶梯（2026-08-08 新增）

项目目标不定义为“在 benchmark 上涨分”，而定义为一条能力主线：

> 让模型从只能回答商品问题和有限执行退货，逐步成长为能够理解用户目标、查询交易状态、遵守政策、完成多轮交易操作，并在异常与非合作对话中安全收尾的电商客服 Agent。

NSCC、template parity、LoRA 训练和 adapter 部署已经跑通，统一记为**已完成基础设施**，不再作为 M0/M1 能力阶段，也不再重复做完整规模链路实验。对应的能力阶梯从用户可感知的新行为开始（评测是验收门，不是阶段交付本身）：

| 阶段 | Agent 新增能力 | 可见交付 | 大致对应实施阶段 |
|---|---|---|---|
| M2 | 修改待处理订单 | 修改商品、支付方式；完整对话演示与 held-out 增益 | P1–P4 |
| M3 | 扩展交易业务 | 取消、退货、换货、地址修改，逐族增加 | P5 逐族扩展 |
| M4 | 处理真实对话困难 | 信息不全、拒绝、跑题、改变需求、工具故障 | cooperative 能力成立后再做 |
| M5 | 从失败中持续改善 | train bad case → 新任务/训练数据 → 独立回放验证 | 沿用 P1–P4 闭环 |
| M6 | 后训练优化 | SFT 仍有稳定盲区时再判断 Skill/RL | 当前冻结 |

**每个阶段必须同时交付以下六项，缺一不算阶段完成：**

```text
一个用户能感知的新能力
+ 一批来源明确的训练数据
+ 一个未参与训练的验证集
+ Base/新模型配对结果
+ 安全回归结果
+ 失败后是否继续的规则
```

这条主线把“增加客服能力”和“增加评测仪器”区分开：每一步先落一个可感知的新能力，再用配对与安全回归验收，而不是反复修评测。

### 1.2 训练数据规模漏斗硬规则（2026-08-09 新增）

所有训练数据 rollout 在正式启动前，必须先给出以下数量漏斗及合理区间：

```text
独立任务数
× 预期成功率
× 质量过滤保留率
× 去重保留率
× 每任务保留上限
= 最终有效训练数据规模区间
```

`pass_k`、rollout 总次数和同一任务的重复采样只表示采样预算，不表示独立任务多样性。不得再用 rollout 总次数冒充训练任务数或数据覆盖面。若正式结果显著偏离预估区间，必须先解释偏差来源，再决定扩采样、补独立任务或进入训练。

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

补充核对（2026-08-08，见 `docs/retail_task_compiler_portability_assessment.md`）：新组件的规模比初稿设想小。三项仓库事实——

1. `telecom/tasks/utils.py` 的 `compose_tasks` / `SelectionSet` / `BaseTask` / `ComposedTask` 不含任何 telecom 语义，可原样上移为共享组合引擎；
2. retail `Environment` 不构造 `user_tools` 且拒绝 solo mode，telecom 的 surroundings 相关逻辑既不可移植也不需要；
3. 114 条 retail 任务的 `reward_basis` 全部是 `[DB]` 或 `[DB, NL_ASSERTION]`，**无一使用 `ENV_ASSERTION` 或 `ACTION`**，而 telecom 的 `TaskManager` 硬编码 `ENV_ASSERTION`。

因此 retail 编译器不需要实现 telecom 中最复杂的 `is_fixed` 谓词与 `env_assertions` 构造，只需产出 `initialization_actions` 加一条参考 `actions` 轨迹，目标终态由官方在干净环境上重放该轨迹并做整库哈希自动导出。骨架估计约 6.5 人日（不含各 task family 的 selection sets）。

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
| τ³ Airline test 20 | 跨域 policy-following 泛化 | 任何训练；不得使用 airline train |
| τ³ Telecom test 40 | 跨域 policy-following 泛化 | 任何训练；不得使用 telecom train/full |
| BFCL Multi-Turn | 通用 function-calling 泛化 | 训练或选择领域数据模板 |
| ECom-Bench | 最终真实电商客服迁移评测；EMNLP 2025，Apache-2.0 | 日常调参或反复查看 test failure |
| 原生 locked 60 / 退货 40 | 安全和历史回归 | 训练、生成模板、模型能力主张 |

注（2026-08-08 收紧）：τ³ Retail test 40 的“仅评测”约束不变（不训练、不做错误驱动生成），但因 P1 竖片选择参考了其动作直方图，它已是 test-aware informed，不再是完全零污染的确认性测试。领域内的确认性角色改由编译器生成、训练不可见的 Retail held-out 承担，最外层由 ECom-Bench 单次验证，见 §8.1。

### 3.3 只作为环境事实或模拟器研究材料的数据

| 数据源 | 处理决定 | 原因 |
|---|---|---|
| Amazon Reviews 2023 5K slice | 只供 RAG 环境读取，不进 SFT/RL target，不随训练数据再分发 | 维护者明确未为数据分配许可证，仅认可研究用途 |
| RealUserSim | 默认不进训练；可在单独获准后用于模拟器 fidelity 研究 | ODC-BY，但数据卡含 OpenAI 辅助生成及用途限制 |
| JDDC/JDDC 2.0 | v2 不依赖 | 访问与授权不稳定，且不提供本项目可执行工具标签 |
| Amazon ESCI | 不进入本项目主实验 | 检索相关性数据，不回答多轮客服工具策略问题 |

## 4. 自建数据如何生成

### 4.1 Task Blueprint：先生成可执行规格，不先写对话

首个 pending-order 竖片直接使用 τ³ 官方任务对象和 `EnvironmentEvaluator`，不先建设通用 blueprint 框架。v0 最少字段为：

```json
{
  "task_id": "...",
  "source_policy_version": "...",
  "initialization_actions": [],
  "user_goal": {},
  "private_user_facts": {},
  "disclosure_schedule": [],
  "reference_actions": [],
  "behavior_profile": "normal|clarification|confirmation|illegal_state_refusal",
  "structure_signature": "...",
  "split": "train|dev|heldout",
  "generator_version": "...",
  "source": "tau3_retail_v1.0.1"
}
```

官方在干净环境重放 `reference_actions` 后生成目标 DB hash；终态相同且未违反政策的替代路径可以通过。v0 不维护第二套手写 `required_effects/forbidden_effects/reference_tool_paths` 真值。

依据：APIGen-MT 将 blueprint 与完整 trajectory 分离；Magnet 从 function-signature path 构造多轮数据；τ² 使用原子组件生成可验证组合任务。

历史方案中的 `required_effects` / `forbidden_effects` / `reference_tool_paths` 已从 v0 删除，而不是继续作为诊断必填字段：

- 官方 `RewardType.DB` 的语义是"在干净环境上重放参考 `actions` 得到目标库哈希，任何产生等价终态的路径都通过"。这已经提供了"允许路径集合"的语义，且比手工枚举更严格也更省事；
- `Environment.set_state` 在重放时跳过非 mutating 工具，因此读路径差异天然不影响终态比对；
- 避免维护第二套交易真值；
- 避免为首个竖片建设通用 effect/verifier 框架；
- 若未来回到 `native_retail`，再针对其缺少 DB hash 的实际问题设计最小判分字段，不提前实现。

### 4.2 Pending-order 硬编码前置条件表

v0 不实现通用 tool graph。只为 `modify_pending_order_items` 和 `modify_pending_order_payment` 写一张可读的前置条件表，例如：

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

每条规则必须来自以下之一：

1. 工具 schema 的输入/输出字段；
2. 官方 policy 的明确前置条件；
3. 工具实现中的状态检查；
4. 人工审阅并写入版本化规则。

不得让 LLM 自行发明规则。等扩展到多个业务族、确实出现重复维护问题后，再决定是否抽象为通用图。

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

正式生成前必须先完成 §1.2 数据漏斗。`5–10` 个任务只用于检查任务生成、官方双次回放和 hint 不泄漏，**不训练模型**。随后根据实测生成成功率、过程合规保留率和结构去重率，一次性确定 train/dev/held-out 规模；首个能力阶段不做几何扩量 learning curve。

### 4.4 自然语言与用户模拟

LLM 接收 blueprint，只生成：

- 用户初始表达；
- 后续信息披露的表面形式；
- 合作程度和语言风格；
- 不改变 latent goal 的改写。

行为类型依据 `Non-Collaborative User Simulators for Tool Agents`：不可用服务、跑题、不耐烦、信息不完整；本项目另将 goal shift 单列，因为电商客服中用户可能由查询转为修改/取消。

ABCD 只用于抽取诸如短句、反问、分段披露、修正前述信息等语言现象；不能复制其具体客户实体、公司规则或 action sequence。

模拟器不得看到参考 actions 或 grader 结果，只能看到用户私有事实和 disclosure schedule。

### 4.5 轨迹采集

当前只保留两种生成方式：current policy 的普通 rollout，以及向同一 policy 私下提供参考 action plan 的 privileged-plan-conditioned rollout。后者不是独立 teacher 实验臂，导出前必须剥离提示。scripted executor 只验证 reference path，不进入语言训练。

训练轨迹保留 assistant 自己生成的 clarification、普通回复和 tool call；工具观察必须来自真实环境。禁止把 teacher 编造的 tool response 写入训练数据。

### 4.6 强制拒收条件

出现任一条件即拒收：

- blueprint 引用不存在的实体或工具；
- reference path 无法执行或重复执行终态不一致；
- 官方参考 actions 无法在干净环境得到目标 DB 终态；
- assistant 调用了 schema 外工具或参数无法解析；
- 工具观察不是环境真实返回；
- 轨迹发生未授权写入、错用户/错订单操作；
- terminal success 仅由 LLM judge 支持，没有状态或规则证据；
- 与 τ³ test 40 在规范化目标、实体无关工具路径和关键约束上高度重合；
- provenance 字段不全；
- generator、teacher、environment 或 policy 版本不可复现。

LLM semantic reviewer 只能作为附加过滤器，不能覆盖确定性失败。

### 4.7 去重与 split

v0 去重只保留两层：

1. 结构近重复：`task_family + reference_actions + initial_state_predicates + behavior_profile`；
2. 外部污染：与 τ³ test 40 的结构签名比较。

首个竖片不实现 embedding 去重。只有数据规模扩大后出现真实文本近重复问题，才增加文本候选过滤。

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

这不是纯“action-only”，因为客服能力也包含澄清与解释。当前不安排 action-only ablation；只有实际出现语言能力或工具格式退化时，才把 loss mask 作为诊断变量。

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
| teacher model/version | 当前没有已确认可用且版本不可变的强模型端点 | 仅在 τ³ train 和自建 dev 上比较候选 teacher 的环境成功率、违规率与成本，选择后记录精确 model ID/date/hash；不看 test 40。**并须通过条款 gate，见下** |
| user simulator model/version | τ³ 分数对 simulator 很敏感，当前环境变量未给出 | 沿用一次 P0 smoke 通过的固定模型，并写入所有结果；中途更换则整组结果无效 |
| NL assertion judge | τ³ v1.0.1 单独调用 judge，不能从 agent 配置推断 | P0 明确设置并冻结模型 ID；Base/SFT/RL 完全一致。暴露面见下 |
| NSCC GPU/CUDA/vLLM | 之前 SSH 预检未完成 | 恢复连接后只读查询硬件、driver、CUDA、Python；再选择 ms-swift 官方兼容组合并生成 lock |

**teacher 条款 gate（新增）。** 本文对 APIGen 的 CC-BY-NC 和 Amazon 的无许可证都设了使用门槛，teacher 输出必须适用同一标准：多数闭源 API 供应商的服务条款限制将其输出用于训练模型。因此 teacher 候选必须先记录其服务条款对模型蒸馏的立场；条款不允许或不明确的候选一律排除。选择开放权重强模型可彻底绕开该问题，是默认倾向。

**NL judge 暴露面已量化（`docs/tau3_split_audit.json`）。** `NLAssertionsEvaluator` 在 `nl_assertions` 为空时短路返回 reward 1.0 且不调用 LLM。实测非空 `nl_assertions` 的任务数：

| 域 | test 任务数 | 实际触发 judge |
|---|---:|---:|
| Retail | 40 | 11 |
| Airline | 20 | 0 |
| Telecom | 40 | 0 |

即整个外部测试套件中只有 11 个任务会调用 judge。judge 仍需冻结（reward 是各分量乘积，judge 只会拉低这 11 题），但它既不是主要成本项也不是主要方差来源。这一结论允许把 P0 的注意力从 judge 转移到 user simulator。

teacher 选择不是“用最贵模型即可”。候选必须能输出标准 tool calls，并在 train-only calibration 上真实执行；没有一个候选明显强于学生时，暂停 teacher-distillation，改用环境成功的 on-policy rejection sampling。

### 6.2 Stage A：Agent-turn LoRA SFT

当前主动训练对比只保留两个模型：

```text
B0  Base Qwen3-4B（冻结、复用已有结果）
C1  含新行为监督的 verified SFT（每个能力阶段只训练一个主模型）
```

Teacher/hint-conditioned generation 只作为**数据获取方式**，不再单列 S1 实验臂；APIGen Airline warm-up、S3 和 action-only ablation 均移出主线。是否使用 teacher 由新行为数据的实际保留率决定，不能为了凑实验臂单独训练模型。

**S0 已降级为附录 negative control（2026-08-09）。** 它只训练 Base 自身成功轨迹，缺少零成功行为监督。合理规模本应是 `5–10 条数据 → LoRA 可训练 → adapter 可加载 → 5–10 题工具格式 smoke`，不需要完整 `74×4` 训练采样和 `40×4` 外测。实际执行得到 47 条高度同质记录，Base/S0 为 85/160 与 86/160，未证明能力提升。完整运行只保留为训练部署链路证据和规划失误记录，不再调参、补跑或作为后续能力阶段。

固定沿用已经跑通的 LoRA 配置作为首个能力模型起点，不做超参数 sweep 或多 checkpoint learning curve。只有含新行为数据的 C1 在 held-out 上明显退化，且已排除数据错误时，才允许一次有明确假设的训练参数调整。τ³ test 不参与选择。

### 6.3 Stage B：多轮 GRPO（冻结，非当前主线）

当前不实施 GRPO、Base→GRPO、CM2、Dr.GRPO 或 GSPO。Retail test 的 k=4 结果中确有 17/40 个任务同时出现成功和失败，说明“所有 group 必然全零”这一悲观判断没有数据支持；但这只证明可能存在组内方差，不构成启动 RL 的理由。

只有在 verified SFT 已在训练不可见 Retail held-out 上形成稳定、明确的 raw-policy 增益，并且新增任务数据仍不能解决的策略错误跨结构重复出现时，才另立新协议讨论 RL。届时重新核对环境重置、Reward、组内方差与训练预算；本方案不提前实现 scheduler、reward shaping、checklist、课程或算法消融。

## 7. Skill 路线（后置）

当前不构建 Skill 系统。先用可验证新行为数据证明模型权重能够获得客服能力；只有后续出现跨实体、跨任务重复且 SFT 难以吸收的程序性故障，才把 train 轨迹用于 Skill 候选，并以 off/on held-out 配对验证。Skill 检索或引用次数不作为效果证据。

## 8. 评测协议

### 8.1 外部主评测：τ³ Retail test 40

**Retail test 40 已是 test-aware informed，不再是完全零污染的确认性测试（2026-08-08 收紧）。** P1 首个竖片（pending-order 修改族）的选择参考了 `docs/tau3_split_audit.json` 中 test split 的动作直方图（见 §11 P1），这是合理的工程决策，但意味着 test 40 的结构分布已进入过设计视野。由此产生三条纪律：

- τ³ Retail test 40 **仍可作为公开、可比的 benchmark 报告**，Base/SFT/RL 的配对差值继续在其上测量；
- 但**不得再把它称为完全 untouched 的确认性测试**；确认性角色改由下面两项承担；
- **必须另外冻结一组编译器生成、但训练不可见的 Retail held-out**（`native`/`tau3_retail` 环境均可，provenance 独立、与 test 40 按 §4.7 去重、生成后即冻结不参与任何选择），作为领域内的确认集；
- **ECom-Bench 最终只跑一次**（见 §8.2、§11 P4/P5 后），承担更外部的电商泛化验证，不作 gate、不用于选模型。

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

1. **τ³ Airline test 20，k=1**：只在每个能力阶段的最终 checkpoint 跑一次，作为 retail LoRA 是否破坏通用 policy-following 的灾难性遗忘跳闸线；禁止使用其 train split，也不把它解释为电商能力提升。
2. **ECom-Bench**：仅在项目最终冻结 checkpoint 后跑一次，不用于选模型或迭代。
3. **Telecom 40 与 BFCL**：移出当前主线。只有最终论文问题确实需要更广泛跨域工具泛化时再恢复。

### 8.3 写操作安全验收

每个能力阶段只跑与新增写操作直接相关的 held-out 安全子集，报告终态、错误写入、越权写入和显式确认。旧内部 120 与退货 40 不再每轮全跑；只有修改共享运行时契约或最终收尾时才做回归。Constraint off 才进入模型能力结论，on 只表示部署保护。

### 8.4 非合作用户压力测试（后置）

不在首个能力竖片中构建六类压力测试。待 cooperative held-out 已证明能力提升后，再按明确的产品故障选择最少必要变体；模拟结果只称 simulated-user robustness。

### 8.5 人工审核（最终展示阶段）

只在最终候选 checkpoint 上抽样审核机器指标覆盖不到的表达质量，不作为每轮训练 gate，也不重复审核环境已确定的 DB 终态。

## 9. 数据量与统计问题

### 9.1 为什么不再用固定“40 条证明能力”

40 个外部 test task 仍可用于可比 benchmark，但估计精度有限。项目必须报告置信区间，不把两题变化的 `5pp` 自动解释成稳定提升。

不再把 Retail、Airline、Telecom 拼成“100 题”来制造样本充足的印象。跨域任务不能补足 Retail 独立结构；当前能力确认依赖训练不可见的 Retail held-out，Retail 40 只在能力阶段收尾跑一次，Airline 20 只作遗忘跳闸线。

### 9.2 自建评测集如何确定规模

预先按所需置信区间计算，而不是拍脑袋。若估计单个主要能力的成功率，希望 95% 区间半宽不超过 `e`，保守取 `p=0.5`：

```text
n ≈ 1.96² × p(1-p) / e²
```

例如 `e=0.10` 时约需 97 个独立任务。这里的独立单位是 blueprint 结构组，不是同一任务的重复 rollout。

该公式只估计单个成功率的区间宽度，不是配对比较的功效。最终任务量结合结构覆盖、数据漏斗、预算和预期效应确定；不把 97 当作能力门槛，也不为此新增 manifest 或独立 gate。

### 9.3 训练集如何确定规模

每次正式 rollout 前执行 §1.2 的数量漏斗，先区分独立任务与重复采样。当前阶段不做 token learning curve 或多档训练；用 hint pilot 实测保留率后一次性确定首个竖片规模，并固定一套 LoRA 配置。

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

已完成：τ³ 与 ms-swift/vLLM 版本、数据许可与 split、template parity、Base 服务和 LoRA 链路均已核实。P0 不再是活动任务；不新增 readiness gate、成本框架、manifest 或重复 parity。只有实际环境或模板版本发生变化时，才针对变化点做最小核对。

### P0.5：S0 negative control（已归档）

S0 已完成并证明训练、adapter 加载和工具格式链路可运行，但没有能力增益。完整规模原本没有必要：合理链路检查只需 5–10 条数据和 5–10 题 smoke。本阶段不再补跑、调参或派生实验，详见 §13.14。

### P1：train 故障证据与 hint-conditioned pilot

1. 只分析已落盘的 Retail train 74×4，不使用 test trace 驱动数据生成；按任务统计成功稳定性，并对失败轨迹记录首个 actionable fault、缺失显式确认候选、越权/多余写入候选。
2. 从 21 个 train 零成功任务中选 5–10 个结构不同的任务，使用官方 `evaluation_criteria.actions` 作为私有计划提示运行 agent；提示只在生成时可见，训练记录中必须剥离。
3. 严格保留条件：官方 reward=1、正常终止、无工具错误、身份/授权合法、写前有显式确认、无额外或越权写入、导出内容不含私有提示。
4. 该方法称为 **privileged-plan-conditioned self-distillation**，不是纯 on-policy Base 数据，也不是独立 teacher 实验臂。
5. pilot 只回答数据获取是否可行，实测 `生成成功率 × 严格过滤率 × 去重率`；不训练模型。

### P2：pending-order 最小 Task Compiler

只有 P1 pilot 能稳定产出新行为轨迹后才进入。首个竖片为 `modify_pending_order_items` 与 `modify_pending_order_payment`：官方 train 对后者零覆盖，因此不能靠增加现有 train 的 pass_k 产生该工具的训练样本；这不等于提前断言 Base 在未来新任务上的成功率为零。

v0 只保留：pending-order 前置条件表、最小 blueprint、官方双次回放、结构签名去重、与 test 隔离。行为类型只做四类：正常修改、必要澄清、显式确认、非法状态拒绝。先生成 5–10 个任务验证生成与官方回放，不拿它们训练；再用 P1 实测漏斗一次性决定 formal train/dev/held-out 的独立任务规模。

### P3：一次正式数据生成与一次 LoRA

formal 数据生成后冻结 split。只训练一个 verified SFT 模型，沿用已跑通的 LoRA 配置；不做 teacher 臂、S0 臂、S3、action-only ablation、配置 sweep 或 token learning curve。

### P4：能力验收

先做 pending-order held-out 配对与对应写操作安全子集；明确提升且安全不退化后，跑 Airline 20 k=1 遗忘跳闸线，再在阶段收尾跑一次 Retail 40。ECom-Bench 只留到项目最终 checkpoint。

### P5：后续能力扩展

首个竖片成功后，按 train 故障证据选择下一个真实客服业务族。GRPO、Skill、非合作用户压力测试与广泛跨域矩阵继续冻结，直到出现它们各自能解决且当前 SFT 主线不能解决的明确问题。

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

## 12.1 本方案的配套产物

| 产物 | 路径 | 状态 |
|---|---|---|
| split 审计脚本 | `scripts/audit_tau3_retail_split.py` | 已跑通 |
| split 审计结果 | `docs/tau3_split_audit.json` | 已生成 |
| Retail Compiler 移植评估 | `docs/retail_task_compiler_portability_assessment.md` | 已完成 |
| template 一致性脚本 | `scripts/check_template_parity.py` | 已在统一 Transformers 4.57.6 栈运行，`PARITY OK` |
| 评测成本模型 | `scripts/estimate_tau3_eval_cost.py` | 已跑通（先验值） |
| 成本模型结果 | `docs/tau3_eval_cost_model.json` | 已生成，待 smoke 实测替换 |
| S0 导出器 | `scripts/build_s0_rejection_sft.py` | 已运行：296 条 rollout 严格过滤为 47 条，覆盖 40 个任务 |

## 13. 执行环境、路径与跨 Agent 交接（2026-08-09）

本节是当前实施主线的执行事实。后续 Agent 接手时先核对本节和真实文件，再执行训练或评测；旧对话中的 job ID、计算节点、临时分支和建议均不得直接当成当前状态。

### 13.1 当前权威仓库、分支与状态语义

| 项目 | 当前权威值 |
|---|---|
| Windows 仓库 | `E:\cv_codex\ecommerce-agentic-rag-legacy-task-closure` |
| GitHub | `https://github.com/Amay810/ecommerce-agentic-rag.git` |
| 当前主线分支 | `feat/legacy-task-closure` |
| 本节写入前最新已推送提交 | `ca1c0ce`（`feat(s0): train audited filtered47 dataset`） |
| 主方案 | `docs/verified_ecommerce_agent_learning_v2_plan.md` |
| 旧 v1 方案 | `docs/tau3_retail_posttraining_v1_plan.md`，只保留历史协议语境，不覆盖本节现状 |

`cursor/tighten-tau3-plan-conclusions-5686` 等辅助工作树/分支不是 NSCC 执行源。分支对话给出的结论只有在以下链路走完后，才算进入主线：

```text
分支对话提出或发现
→ 主对话核对真实仓库/环境
→ 在 feat/legacy-task-closure 实现
→ 本地验证
→ commit
→ push
→ GitHub pull 或 N: 映射同步 NSCC
```

交接时必须使用以下五种状态词，不得混写：

```text
已建议：只有方案，文件可能不存在
已实现：本地文件已经修改
已验证：相应命令或产物已实际检查
已提交/已推送：Git 和 GitHub 已包含
已同步 NSCC：NSCC 路径已实际出现对应版本
```

截至 `ca1c0ce`，关键提交为：

| commit | 已进入主线的事实 |
|---|---|
| `9526386` | 将本地联网 rollout 与 NSCC 离线训练拆开 |
| `4cf93bb` | vLLM 服务加载 CUDA 12.8 toolkit |
| `2699bf6` | 单节点四卡；推理 DP=4/TP=1；训练 `NPROC_PER_NODE=4` |
| `4c37164` | 本地回环 vLLM 地址加入 `NO_PROXY`，修复 DeepSeek 调用后访问隧道端点的 502/连接问题 |
| `920588b` | PBS 提交队列改为 `normal`；同步 template parity 的 BatchEncoding 修复 |
| `ca1c0ce` | 提交已审核的 filtered47 数据、训练 PBS 和数据规模漏斗硬规则 |
| `2ad37cd` | 冻结 Windows/NSCC 环境、路径、隧道、同步与跨 Agent 交接事实 |

### 13.2 固定目录结构

Windows：

```text
E:\cv_codex\
├── ecommerce-agentic-rag-legacy-task-closure\   # 项目仓库
├── external\tau2-bench\                         # 外部 pinned τ²/τ³
│   └── .venv\Scripts\python.exe
└── .venv-agent-v2-release\Scripts\python.exe   # 本地主项目编排/检查环境
```

NSCC：

```text
/scratch/users/ntu/s250045/
├── ecommerce-agentic-rag-legacy-task-closure/   # 项目仓库
├── tau2-bench/                                   # 与项目仓库同级，不在仓库内部
│   └── .venv/bin/python
├── models/Qwen3-4B-Instruct-2507/
└── conda-envs/
    ├── ecommerce-vllm/
    └── ecommerce-swift/
```

固定外部版本：

```text
tau2-bench tag: v1.0.1
tau2-bench commit: fc0055dc4e0a316c3f83133267fbd6faaa770992
NSCC TAU2_ROOT: /scratch/users/ntu/s250045/tau2-bench
```

不得把 `tau2-bench/` 提交进项目仓库，也不得把 `TAU2_ROOT` 改成项目仓库内部路径。

### 13.3 虚拟环境及唯一职责

| 环境 | 已核实版本 | 唯一职责 |
|---|---|---|
| Windows `E:\cv_codex\.venv-agent-v2-release` | 当前本地主项目可用环境 | 调用项目 runner、检查/导出结果；不取代 τ² 自己的 `.venv` |
| Windows `E:\cv_codex\external\tau2-bench\.venv` | 来自 pinned `uv.lock` | 正式本地 τ²/τ³ Retail 编排、环境、grader |
| NSCC `ecommerce-vllm` | Python 3.12.13；torch 2.8.0+cu128；vLLM 0.10.2；Transformers 4.57.6 | Qwen Base 或 Qwen+LoRA 的 OpenAI-compatible vLLM 服务；提供 `uv` |
| NSCC `ecommerce-swift` | ms-swift 4.2.2；Transformers 4.57.6 | chat template、label mask、四卡 LoRA SFT |
| NSCC `tau2-bench/.venv` | τ² commit `fc0055dc...` | τ² Retail schema/环境；不向 `ecommerce-swift` 混装依赖 |
| NSCC `ecommerce-rag` | 旧环境 | 不再用于 vLLM、ms-swift 或 τ² |

`ecommerce-swift` 的 Transformers 曾为 5.8.1，已固定回 4.57.6，以匹配实际 vLLM tokenizer 栈。不得无实验协议变更再次升级到 Transformers 5。

### 13.4 Windows、登录节点与 GPU 计算节点分工

Windows 本地负责：

```text
τ²/τ³ 正式多轮编排
DeepSeek user simulator
DeepSeek NL assertion judge
通过隧道逐轮调用 Qwen
rollout 保存、auto-resume、reward 过滤和 SFT JSONL 导出
代码 commit/push
```

NSCC 登录节点负责：

```text
qsub / qstat / qdel
代码和数据接收
轻量文件检查与 template parity
查看服务输出中的 TAU3_VLLM_HOST
```

登录节点不负责 vLLM、LoRA 训练、大规模 rollout 或 CUDA JIT。

NSCC GPU 计算节点负责：

```text
Qwen Base vLLM
Qwen + LoRA vLLM
FlashInfer/CUDA JIT
单节点四卡 LoRA 训练
```

DeepSeek simulator 是冻结实验对象，不能为了在 NSCC 内部闭环而替换成本地 Qwen simulator。DeepSeek 必须留在联网的 Windows 环境；NSCC 不保存其 API key。

### 13.5 PBS 与四卡资源口径

用户入口队列只能写：

```bash
#PBS -q normal
```

`qstat` 运行时显示 `g1` 等名称是内部执行队列，不表示 PBS 可以写 `#PBS -q g1`。当前固定资源形式为：

```bash
#PBS -q normal
#PBS -P personal
#PBS -l select=1:ncpus=16:ngpus=4:mem=110gb
```

不得写 `select=4:ngpus=1`，因为这可能跨四个节点。推理使用：

```text
vLLM data_parallel_size = 4
vLLM tensor_parallel_size = 1
一个服务端口 = 8123
```

Qwen3-4B 单卡可以容纳；四卡的目的在于同时服务多个独立 rollout，而不是把 4B 模型切成 TP=4。训练使用 `NPROC_PER_NODE=4`，不能把 vLLM 的 DP 参数用于 ms-swift。

NSCC 可能把 `CUDA_VISIBLE_DEVICES` 暴露为四个 GPU UUID。`nscc/serve_tau3_agent_v1.pbs` 已逐个映射为数字 index 并保留全部四张卡，后续不得退回只支持单 UUID 的旧逻辑。

当前权威 PBS：

| 文件 | 用途 | 状态 |
|---|---|---|
| `nscc/serve_tau3_agent_v1.pbs` | 四卡 DP Base vLLM | 保持原样；Base 批量推理权威脚本 |
| `nscc/serve_tau3_s0_v1.pbs` | 单卡 S0 LoRA vLLM | 从 Base 脚本派生；适配冻结 vLLM 0.10.2 的 LoRA+DP 限制 |
| `nscc/run_tau3_s0_filtered47.pbs` | 只训练已审核的 47 条 S0 数据 | 四卡训练已成功完成，checkpoint 为 `output/tau3_s0_filtered47/checkpoint-18` |
| `nscc/run_tau3_s0_v1.pbs` | 旧 S0 作业 | 不得用于 filtered47；会重新过滤、改为 `max-per-task=1` 并另切 10 个 dev task |

filtered47 作业固定：

```text
input: data/s0_rejection_sft.jsonl
records: 47
NPROC_PER_NODE: 4
max_length: 32768
output: output/tau3_s0_filtered47
```

代表性 parity 轨迹约 18,661 tokens，旧 `max_length=16384` 会确定性截断，因此 filtered47 使用与服务侧一致的 32,768 上限。

冻结资源矩阵：

| 阶段 | 节点/GPU | 并行方式 | 说明 |
|---|---|---|---|
| S0 LoRA 训练 | 单节点四卡 | `NPROC_PER_NODE=4` | 已成功完成 |
| Base 批量推理 | 单节点四卡 | DP=4、TP=1 | 四个独立 Base 模型副本，提高 rollout 吞吐 |
| vLLM 0.10.2 S0 LoRA 服务 | 单节点单卡 | DP=1、TP=1 | 当前冻结版本限制；不是 LoRA 的一般性结论 |

四卡 S0 LoRA 服务的实际失败已定位为：

```text
vLLM 0.10.2
+ --enable-lora
+ --data-parallel-size 4
→ NotImplementedError: LoRA in DP mode is not supported yet
```

这不是 adapter、checkpoint、GPU UUID 映射或训练产物错误。不得通过改 adapter 或重训来“修复”该服务限制，也不得把“LoRA 必须单卡”外推到其他 vLLM 版本或其他服务框架。

提交 S0 单卡服务：

```bash
cd /scratch/users/ntu/s250045/ecommerce-agentic-rag-legacy-task-closure
qsub -v TAU3_ADAPTER=/scratch/users/ntu/s250045/ecommerce-agentic-rag-legacy-task-closure/output/tau3_s0_filtered47/checkpoint-18 \
  nscc/serve_tau3_s0_v1.pbs
```

### 13.6 Template parity 最终事实

唯一有效结论：

```text
Retail tools: 16
serving render: 70,179 chars
training render: 70,179 chars
training/serving token IDs: 完全一致
代表性 token IDs: 约 18,661
parity_exit: 0
结论: PARITY OK
```

label mask：

```text
system            不训练
user              不训练
assistant text    训练
tool_call         训练
tool_response     不训练
```

前两次运行无效：第一次把 `BatchEncoding` 直接 `list(...)`，只得到键名；第二次训练侧 Transformers 5.8.1 与服务侧 4.57.6 不一致。它们是脚本/环境失误，不是模板实验失败，更不是项目贡献。`scripts/check_template_parity.py` 已兼容 dict、tensor 和 batch 嵌套结构。

### 13.7 DeepSeek key 与本地 provider 配置

新 DeepSeek key 只以 Windows DPAPI 加密文本保存在：

```text
%LOCALAPPDATA%\ecommerce-agentic-rag\tau3_deepseek.key
```

一次性安全保存方式：

```powershell
$dir = "$env:LOCALAPPDATA\ecommerce-agentic-rag"
New-Item -ItemType Directory -Path $dir -Force | Out-Null
$secure = Read-Host "DeepSeek API key" -AsSecureString
$secure | ConvertFrom-SecureString | Set-Content "$dir\tau3_deepseek.key"
```

运行时读取后只注入当前进程，必须对文件内容 `.Trim()`，并在 finally 中清理 BSTR 和环境变量。不得在命令行、仓库、NSCC 或截图中出现明文 key。旧 key 若仍有效必须撤销。

本地 Qwen 端点配置：

```text
TAU3_AGENT_BASE_URL=http://127.0.0.1:8123/v1
TAU3_AGENT_API_KEY=local-vllm
agent model=hosted_vllm/Qwen3-4B-Instruct-2507
user model=deepseek/deepseek-chat
judge model=deepseek/deepseek-chat
NO_PROXY/no_proxy=127.0.0.1,localhost
```

`scripts/run_tau3_retail_v1.py` 已在 hosted-vLLM 地址为 localhost 时自动补 `NO_PROXY`。若直接调用 τ² CLI，也必须显式设置这两个变量，否则 DeepSeek 调用后的 HTTP 客户端可能把回环请求错误处理为 502。

### 13.8 两层隧道与服务检查

正式 rollout 链路：

```text
Windows τ² + DeepSeek
→ NSCC 工作流把登录节点暴露为 127.0.0.1:2222
→ Windows SSH 二次转发
→ 当前 GPU 计算节点:8123
→ 四卡 vLLM 单端点
```

第二层隧道示例：

```powershell
$node = "<从当前 qstat -f 或 TAU3_VLLM_HOST 读取的计算节点>"
ssh -p 2222 -N `
  -o ExitOnForwardFailure=yes `
  -o ServerAliveInterval=30 `
  -o ServerAliveCountMax=6 `
  -L 127.0.0.1:8123:${node}:8123 `
  s250045@127.0.0.1
```

PowerShell 变量值必须加引号。每个 PBS 作业的计算节点都会变化，历史节点如 `x1000c2s3b0n1`、`x1000c3s7b0n0` 只属于当时作业，不能写死。

可靠在线检查：

```powershell
Invoke-RestMethod http://127.0.0.1:8123/v1/models
```

若 `127.0.0.1:2222 connection refused`，是第一层 NSCC 工作流未启动，与 vLLM 无关。若 2222 正常但 8123 拒绝，检查第二层隧道、当前计算节点和 vLLM 服务。

PBS 输出可能暂存在 spool，旧 `logs/tau3_agent_v1.pbs.log` 不更新不等于当前作业失败；`qstat=R` 也不等于服务已就绪。必须以计算节点 `/v1/models` 或 Windows 隧道 `/v1/models` 返回为准。

### 13.9 正式 rollout 的本地执行纪律

多轮 rollout 期间隧道必须持续在线，因为每轮执行顺序是：

```text
DeepSeek 生成 user turn
→ NSCC Qwen 生成 agent text/tool call
→ Windows τ² 执行工具并更新环境
→ DeepSeek 根据新状态生成下一轮
→ 直到终止并判分
```

Base 四卡 DP 只增加同时进行的独立对话数；单条对话仍按轮次串行。S0 LoRA 单卡服务时，Windows τ³ 的 `--max-concurrency 4` 保持不变，四个并发请求会在单卡 vLLM 服务处排队；客户端并发数不要求等于 GPU 数。

当前 `scripts.run_tau3_retail_v1 --phase base` 对应官方 **test 40**，不是 train rollout。S0 训练数据必须直接使用 τ² CLI 的：

```text
--task-split-name train
--num-trials 4
--max-concurrency 4
--save-to tau3_s0_v1_train_rollout_qwen3_4b_k4
--auto-resume
```

不得误用 `--phase base` 生成训练数据。`--auto-resume` 已从 τ² 源码确认：保留非基础设施错误轨迹，删除 `infrastructure_error` 记录，只重跑失败的 task/trial。

第一次 S0 rollout 曾因隧道中断产生 130 条 `WinError 10061`；恢复隧道后同名 auto-resume 最终补齐。此过程只作为执行事实，不包装成工程贡献。

S0 诊断继续使用 Retail test 40×4、DeepSeek simulator/judge、冻结 temperature、`max_steps=200`、相同 task split、pass_k、reward 和 judge。Base 使用四卡 DP 服务，S0 使用单卡 LoRA 服务；两者的单条请求都在一张 A100 上完成，因此能力成功率可以配对比较，但服务吞吐和延迟不可作为模型能力差异。

### 13.10 S0 rollout、数据漏斗与主张边界

正式 Base rollout：

| 指标 | 最终值 |
|---|---:|
| 独立 Retail train tasks | 74 |
| 与 test 重叠 | 0 |
| pass_k | 4 |
| rollout 总次数 | 296 |
| reward 完整 | 296/296 |
| infrastructure error | 0 |
| 正常终止 | 296/296 |
| reward=1 成功轨迹 | 129 |
| Base train rollout 成功率 | 43.58% |
| 至少成功一次的任务 | 53/74 |
| 四次均未成功的任务 | 21/74 |

21 个零成功 task ID：

```text
14, 19, 20, 21, 28, 29, 30, 37, 41, 46, 54,
57, 59, 76, 85, 92, 98, 103, 104, 105, 109
```

严格过滤漏斗：

```text
296 rollouts（仅 74 个独立任务）
→ 129 条 reward=1
→ 排除 42 条含 tool error 的成功轨迹
→ 按 task + tool path 去重，排除 38 条重复路径
→ 每任务最多 2 条，再排除 2 条
→ 47 条训练记录
→ 覆盖 40 个独立任务
```

权威产物：

```text
原始 rollout:
E:\cv_codex\external\tau2-bench\data\simulations\
tau3_s0_v1_train_rollout_qwen3_4b_k4\results.json

训练 JSONL:
E:\cv_codex\ecommerce-agentic-rag-legacy-task-closure\data\s0_rejection_sft.jsonl

本地记录:
data/s0_rejection_sft.manifest.json
```

47 条只允许主张训练链路和局部信号检查；不能主张充分电商客服数据、广泛能力提升、解决 21 个零成功任务或稳定泛化。严格过滤后只覆盖 40/74 个任务，未覆盖部分是 Teacher/Task Compiler 的数据目标。

所有后续训练数据 rollout 必须执行 §1.2 的事前漏斗：

```text
独立任务数
× 预期成功率
× 质量过滤保留率
× 去重保留率
× 每任务上限
= 最终有效数据区间
```

`296` 是采样次数，不是 296 个独立训练任务。

### 13.11 Windows、GitHub 与 NSCC 同步方法

代码以 GitHub commit/pull 为主；当前权威分支是 `feat/legacy-task-closure`。NSCC 若切换到该分支，先核对工作树，再执行：

```bash
git status
git fetch origin
git switch feat/legacy-task-closure
git pull --ff-only origin feat/legacy-task-closure
```

若 Windows 仓库再次出现 dubious ownership，针对明确仓库设置：

```powershell
git config --global --add safe.directory E:/cv_codex/ecommerce-agentic-rag-legacy-task-closure
```

Windows `N:` 映射只指向 NSCC 项目仓库：

```text
N:\
= /scratch/users/ntu/s250045/ecommerce-agentic-rag-legacy-task-closure/
```

它不会显示同级 `/scratch/users/ntu/s250045/tau2-bench`，这是正常结构。若 `N:\data` 不存在，可直接创建并复制：

```powershell
New-Item -ItemType Directory -Path "N:\data" -Force
Copy-Item "$repo\data\s0_rejection_sft.jsonl" "N:\data\s0_rejection_sft.jsonl" -Force
Copy-Item "$repo\nscc\run_tau3_s0_filtered47.pbs" "N:\nscc\run_tau3_s0_filtered47.pbs" -Force
```

`net use N:` 返回系统错误 85 只表示盘符已经映射，不要重复映射。优先直接复制；只有实际发生大文件传输停滞时，才改用 GitHub 或 SCP，不预先增加 `.part`、SHA、转正流程等防御步骤。

截至 2026-08-09，以下两项已经在 NSCC 映射路径实际出现：

```text
data/s0_rejection_sft.jsonl                 4,601,362 bytes
nscc/run_tau3_s0_filtered47.pbs             2,105 bytes，Unix LF
```

manifest 只作本地/GitHub 数据记录，不参与训练，也未要求单独同步 NSCC。

### 13.12 分支对话已回写主线的内容

以下内容最初来自辅助/分支对话或 NSCC 手工执行交接，现已由主对话核对并写回本方案：

1. 项目目标是提升可执行电商客服能力，退货 40 条只作旧回归，不承担总体能力主张；
2. 外部 τ³ Retail 提供可重置环境、工具、DB 和确定性终态，S0 是训练链路检查，Task Compiler 才负责补新能力数据；
3. Airline 20 k=1 只作阶段末灾难性遗忘跳闸线；Telecom 40 与 BFCL 已退出当前能力主线；
4. Retail test 40 已被分析过，是 test-aware informed；确认性结论还需训练不可见 held-out；
5. Windows 承担 DeepSeek 与多轮编排，NSCC 只承担 Qwen 推理和 LoRA 训练；
6. PBS 入口队列是 `normal`，`g1` 只是可能显示的内部执行队列；
7. 四卡 Qwen3-4B 应使用 DP=4/TP=1，而非 TP=4；
8. template parity 在统一 Transformers 4.57.6 后正式通过，前两次失误不进入项目叙事；
9. 项目仓库与 `tau2-bench` 在 NSCC 上是同级目录；
10. S0 rollout 经 auto-resume 最终得到 296 条有效结果，严格过滤后只有 47 条/40 任务；
11. 任何训练前必须先估算数据漏斗，不再用 rollout 次数冒充任务多样性；
12. filtered47 必须使用新 PBS，不能使用会重新过滤/重切 dev 的旧 PBS。
13. 冻结 vLLM 0.10.2 的内置 LoRA 服务不支持 DP=4；Base 保持四卡 DP，S0 LoRA 服务使用单卡，但 Windows 评测并发仍可为 4。
14. S0 是一次过度规模化的链路检查，正式归档为 negative control；后续不再用完整训练采样和外测验证已跑通链路。
15. train 74×4 的任务级分布是 16 个稳定、37 个不稳定、21 个零成功；filtered47 缺失的 34 个任务不是 34 个零成功盲区，而是 21 个零成功加 13 个过滤/去重损耗。

后续其他 Agent 不需要重新争论上述事实；如果真实仓库、环境或实验协议发生变化，必须给出新证据、明确修改本节并产生新 commit。

### 13.13 当前下一步与完成边界

当前已经完成：

```text
template parity
→ 5 题 Base smoke（3/5，基础设施错误 0）
→ Retail train 74×4 Base rollout（129/296）
→ 严格过滤得到 47 条、覆盖 40 任务
→ filtered47 四卡训练 PBS 创建、验证、commit/push、同步 NSCC
→ S0 LoRA 四卡训练成功，产出 output/tau3_s0_filtered47/checkpoint-18
→ 四卡 LoRA 服务失败根因定位为 vLLM 0.10.2 不支持 LoRA+DP
→ 单卡 S0 LoRA 服务成功，Windows 并发 4 正常运行
→ Base/S0 Retail test 40×4 均完成，0 infrastructure errors
→ S0 归档为 negative control，能力主线与评测矩阵已删减
→ Retail train 296 故障审计完成，选出 8 个 hint-conditioned pilot 任务
→ 复用 τ² `LLMGTAgent` 的 action-name-only 模式，16 次 pilot launcher 已准备并完成语法/CLI 注册检查
→ 离线渲染确认 8 个提示无实体参数泄漏；识别出 task 85/109 的 gold actions 不含认证/读取，严格过滤不得把 gold list 当完整策略
```

当前尚未完成、不得提前写成完成态：

```text
8 个 train 盲区任务的 hint-conditioned 生成
严格过滤并测量真实数据漏斗
根据实测保留率决定是否进入 pending-order Task Compiler
```

下一执行顺序：

```text
只用官方 Retail train 的 8 个零成功任务
→ 私有 gold-action plan 辅助完整环境 rollout
→ 剥离私有提示并严格过滤
→ 报告独立任务数、生成成功率、质量保留率、去重损耗
→ 数据获取可行后才实现最小 pending-order Task Compiler
```

故障证据和 pilot 选择见 `docs/tau3_train_fault_audit_20260809.md`。S0 provenance 的说明性顶层字段可在归档时补齐，但不得因此阻塞能力主线，也不得重跑 S0。

本次需要永久记录但不包装成工程贡献的执行失误：提交四卡 LoRA 服务前，没有先核对冻结 vLLM 0.10.2 是否支持 `--enable-lora` 与 DP=4 的组合。以后涉及特定版本的并行组合时，应先查已安装版本的支持范围或用最小实际启动确认；这只是一项执行核对纪律，不扩展为新的 readiness gate、预检框架或项目交付物。

### 13.14 S0 Base/SFT 正式诊断结果（2026-08-09）

两臂运行完整性：

| 项目 | Base | S0 |
|---|---:|---:|
| Retail test 独立任务 | 40 | 40 |
| 每任务 trials | 4 | 4 |
| simulations | 160 | 160 |
| reward 完整 | 160/160 | 160/160 |
| infrastructure errors | 0 | 0 |
| 正常 `user_stop` | 160 | 160 |
| 成功 | 85 | 86 |
| 成功率 | 53.125% | 53.750% |

任务、task/trial 配对键、任务 payload、Retail policy、seed=300、agent/user temperature=0、DeepSeek user simulator、`max_steps=200` 均一致；两臂任务集合与官方 test 40 完全一致。服务资源差异仅为 Base 四卡 DP 与 S0 单卡 LoRA，单条请求都由一张 A100 完成，因此成功率可比较，延迟和吞吐不可比较。

配对结果：

```text
两臂都失败: 53
S0 修复:     22
S0 退化:     21
两臂都成功: 64
净差值:     +1/160 = +0.625pp
```

40 个独立任务聚合后：

```text
S0 改善任务: 11
S0 退化任务: 11
不变任务:    18
任务聚类 bootstrap 95% CI: [-8.75pp, +10.625pp]
```

辅助分量也没有形成正向证据：

| 分量 | Base | S0 | 配对变化 |
|---|---:|---:|---|
| DB match | 86/160 | 86/160 | 21 修复 / 21 退化 |
| NL assertion pass | 153/160 | 153/160 | 6 修复 / 6 退化 |
| failed write action checks | 86 | 90 | S0 多 4 个；只作诊断，不等同非法状态变更 |
| 至少成功一次的任务 | 31/40 | 30/40 | S0 少 1 个 |
| 四次全成功的任务 | 14/40 | 14/40 | 持平 |

正式能力判断：

> S0 adapter 已证明“47 条 verified rejection data → 四卡 LoRA 训练 → adapter 加载 → τ³ 多轮工具评测”的训练与部署链路可运行；但在本次 Retail test 40×4 上，没有证据表明 S0 提高了电商客服 Agent 能力。结果应记为 `negative_or_inconclusive`，不得把 +1/160 写成提升。

该结果符合 S0 的数据支持域：S0 只从 Base 已经成功的轨迹学习，严格过滤后仅 47 条、覆盖 40 个 train task，本来不能直接监督 21 个四次零成功任务。S0 无增益不否决 Task Compiler，反而确认下一步应补独立任务和能力盲区，而不是继续增加同任务 pass_k 或围绕 47 条调参。

产物：

```text
Base:
E:\cv_codex\external\tau2-bench\data\simulations\
tau3_s0_v1_base_test_qwen3_4b_k4\results.json

S0:
E:\cv_codex\external\tau2-bench\data\simulations\
tau3_s0_v1_sft_test_qwen3_4b_k4\results.json
```

provenance 审核：Base 已包含顶层 `tau2_commit/user_simulator_model/nl_assertions_model/agent_model/pass_k/protocol`。S0 运行文件尚未写入这些顶层注释字段，且 `info.git_commit` 记录的是启动 CLI 时的当前工作目录 commit；τ² 源码的 `get_commit_hash()` 本来就是对当前目录执行 `git rev-parse HEAD`，因此该值不是 τ² checkout 漂移证据。本地外部 checkout 已只读核实为 `v1.0.1/fc0055dc...`，两臂任务和冻结配置完全对齐。无需重跑，但正式 closeout 前必须把同一组冻结顶层字段补入 S0 产物。

本次稳定运行事实只记录以下四项：

1. vLLM 0.10.2 不支持内置 LoRA+DP；Base=四卡 DP、训练=四卡 DDP、S0 服务=单卡；
2. 绕过项目 wrapper 直接调用底层 CLI 时必须保留 `NO_PROXY/no_proxy=127.0.0.1,localhost`；此前 `BadGateway` 期间服务端没有收到正式 POST，因此不是并发或模型故障；
3. PBS 重启后必须按新 `exec_host` 重建 8123 隧道，旧隧道指向旧节点会产生 `WinError 10054`；
4. LiteLLM 的 `This model isn't mapped yet: Qwen3-4B-S0` 只表示无法计算自定义本地模型 API 成本，不影响推理、轨迹或 reward。

临时终端窗口、盘符可见性和某一次连接状态不作为项目/环境缺陷写入主叙事。排查 `BadGateway` 时先检查请求是否真正到达 vLLM，再判断模型、并发或服务故障。

## 14. 官方来源

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
