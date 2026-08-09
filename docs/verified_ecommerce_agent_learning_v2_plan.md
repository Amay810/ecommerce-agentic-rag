# 可验证电商客服 Agent 学习路线 v2

状态：研究与实施方案；取代“仅以 τ³ 74 条 teacher rollout 做一次 SFT”的主线，但不覆盖旧实验的历史记录。  
定版日期：2026-08-08（同日修订一次，见 §12.1；修订项均标注日期，涉及 §2.2、§3.2、§4.1、§6.1.1、§6.2、§6.3、§8.2、§9.1、§11）。  
2026-08-08 收紧一次（标注“收紧/新增”）：新增 §1.1 能力阶梯；收紧三处结论——不把三域 100 题合并成单一电商功效成功率并区分单比例 CI 与 McNemar 配对功效（§9.1、§9.2）；Retail test 40 声明为 test-aware informed 并要求独立 held-out 承担确认性（§3.2、§8.1、§11 P1）；S0 定位为训练链路检查而非新能力证明，P0.5 改为四分支决策（§6.2、§11 P0.5）。  
2026-08-09 执行交接更新：新增 §13，冻结 Windows/NSCC 分工、虚拟环境、路径、隧道、同步、Git 分支、S0 数据漏斗与当前完成状态；这些内容是后续 Agent 的执行事实，不得用旧对话中的临时 job、节点或建议覆盖。
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

### 1.1 能力主线与阶段阶梯（2026-08-08 新增）

项目目标不定义为“在 benchmark 上涨分”，而定义为一条能力主线：

> 让模型从只能回答商品问题和有限执行退货，逐步成长为能够理解用户目标、查询交易状态、遵守政策、完成多轮交易操作，并在异常与非合作对话中安全收尾的电商客服 Agent。

对应的能力阶梯（评测是每阶段的验收门，不是阶段交付本身）：

| 阶段 | Agent 新增能力 | 可见交付 | 大致对应实施阶段 |
|---|---|---|---|
| M0 | 可稳定运行与训练 | Base 在 NSCC 跑通；template parity 通过 | P0 / P0-a |
| M1 | 学会已有正确轨迹 | S0 checkpoint；证明训练链路有效 | P0.5（S0） |
| M2 | 修改待处理订单 | 修改商品、支付方式；完整对话演示与 held-out 增益 | P1 首个竖片 → P2/P3 |
| M3 | 扩展交易业务 | 取消、退货、换货、地址修改，逐族增加 | P1 扩量 → P3 |
| M4 | 处理真实对话困难 | 信息不全、拒绝、跑题、改变需求、工具故障 | P5 非合作用户 |
| M5 | 从失败中持续改善 | bad case 聚类 → 新任务/Skill/训练数据 → 独立回放验证 | P5 Skill 与失败飞轮 |
| M6 | 后训练优化 | SFT 有效后再验证 GRPO 是否提供额外收益 | P4（GRPO） |

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

**环境相关的字段降级（2026-08-08 修订）。** 在 `tau3_retail` 环境下，`required_effects` / `forbidden_effects` / `reference_tool_paths` 三个字段**不作为判分依据**，只作为诊断与覆盖率统计字段：

- 官方 `RewardType.DB` 的语义是"在干净环境上重放参考 `actions` 得到目标库哈希，任何产生等价终态的路径都通过"。这已经提供了"允许路径集合"的语义，且比手工枚举更严格也更省事；
- `Environment.set_state` 在重放时跳过非 mutating 工具，因此读路径差异天然不影响终态比对；
- 手写 effects 列表与 DB 哈希不一致时，以 DB 哈希为准，并把该蓝图记为拒收（生成器 bug）。

在 `native_retail` 环境下这三个字段仍然是判分依据，因为原生环境没有等价的整库哈希机制。两套环境分别记分的原则不变。

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

训练臂：

```text
B0  Base Qwen3-4B
S0  τ³ train on-policy rejection sampling（环境验证成功轨迹）
S1  τ³ train teacher trajectories
S2  S1 + verified self-built trajectories
S3  S2 + optional APIGen Airline warm-up
```

S3 只有在数据许可 gate 与 BFCL 去重通过后才运行。

**S0 是新增的一等实验臂，且排在 P1 之前（2026-08-08 修订）。** 初稿只把 on-policy rejection sampling 写成"没有候选 teacher 明显强于学生"时的退路。改为一等公民的理由：

- 它不需要任何新组件。74 条官方 train task 本身就是带可验证终态的蓝图，反复采样后用官方 grader 筛 `reward == 1.0` 即可，工程成本约为 P1 的百分之一；
- 它同时产出 P1 最需要的输入：train split 中基座策略从未成功过的任务集合，即覆盖盲区的直接证据。

**S0 是训练链路检查，不是新能力证明（2026-08-08 收紧）。** S0 从基座**已经成功**的轨迹中筛 `reward==1.0` 再训练，它能验证的是：数据格式是否正确、chat template 是否一致（P0-a）、LoRA 是否真正学到工具调用格式、环境验证轨迹能否进入训练、是否产生明显灾难性退化。但它**缺少基座从未成功过的行为**，因此**不能回答“能否获得新能力”**。

由此得到一个必须写死的判据：**S0 增益为 0 不能否决 Task Compiler。** 当成功轨迹只是重复了模型本来就会的东西时，零增益是预期结果，而不是编译器“建在沙上”的证据。S0 的天花板（74 条任务多样性有限、自采样对基座不会的任务零信号）本身就是 P1 的论证。S0 的 go/no-go 语义因此只针对训练链路，能力问题留给 P1，判据细则见 §11 P0.5。

实现：`scripts/build_s0_rejection_sft.py`。该脚本刻意不自行 rollout，生成与判分全部交给官方 runner，以保证工具观察和 Reward 无歧义地来自环境。

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

**方差塌缩的预注册对策（新增）。** 前置条件 4 只规定了检查，没有规定检查失败后怎么办。以 Qwen3-4B 的水平和 τ³ retail 的难度，多数 group 很可能全为 0（合法但未完成），group-relative advantage 归零，训练无梯度。这是多轮 agent RL 最常见的失败模式，必须提前定好对策，否则 P4 会在 P1 已经完工之后才卡死。

需要澄清一个概念区分：本文禁止的是**任意加权**，不是**任何 shaping**。由环境确定性计算的子目标进度不属于任意加权，与 CM2 那类 LLM checklist judge 有本质区别。据此预注册两条对策，按顺序启用：

1. **确定性子目标进度项。** 以蓝图 `required_effects` 中已满足的比例作为 `(0, 1)` 区间内的稠密项，仅在 group 内全部 rollout 的主 Reward 相同时生效。该值完全由环境状态计算，不涉及任何 judge。违规判定 `-1` 优先级不变，一旦触发即覆盖进度项。
2. **难度课程。** 用编译器的 `tool-path length` 轴从短路径任务起步建立方差，再逐步引入长路径任务。

两条都不改变主 Reward 的定义，只影响 advantage 的可计算性。若两条都无法产生方差，则如实关闭 GRPO 路线并只报告 SFT 结果。

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

1. **τ³ Airline test 20 + Telecom test 40**（新增，列为主要泛化证据）。同一个已 pin 的 MIT 仓库、同一套 runner 与判分实现、同一个 user simulator，零额外工程成本，零污染风险（训练只用 retail train）。回答的问题比 BFCL 更贴近本项目形态：在 retail 上做的可验证 SFT，是提升了跨域的 policy-following 与工具使用，还是只学会了 retail 的套路。

   两域的 `nl_assertions` 全部为空，判分 100% 确定性，无 judge 方差。禁止使用其 train split。

2. **BFCL Multi-Turn**：只测 Base、S2、R1，证明工具能力是否跨出客服环境；使用官方 executable/state-based evaluator。
3. **ECom-Bench**：最终冻结 checkpoint 后各跑一次；报告官方 action/search/output/time 指标和 pass 指标；不用于选模型。

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

**三域 100 题解决的是跨域泛化证据，不是 Retail 电商功效样本不足（2026-08-08 收紧）。** 实测三域 test split 合计：

```text
Retail 40 + Airline 20 + Telecom 40 = 100 个独立外部任务
```

100 在数量上越过 §9.2 的 97，且全部来自同一 pin 住的 MIT 仓库，无需任何额外构建。但这一百题混合了三个不同领域、不同工具面、不同难度的任务，不能合并成单一的“电商客服成功率”。正确的分层口径是：

- **Retail 40**：电商客服领域能力的主报告。样本仍然有限，结论必须带配对置信区间，且 Retail 40 已是 test-aware informed（见 §8.1），确认性由 §8.1 的独立 held-out 与 ECom-Bench 承担；
- **Airline 20 + Telecom 40**：跨域 policy-following 与工具使用的泛化证据。回答“在 retail 上做的可验证 SFT 是否迁移到未训练领域”，而不是补足 Retail 的电商样本量；
- **三域联合 100**：只在分层之后作为“通用多轮工具 Agent”的合并配对结果给出，且仅用于配对差值（同任务同 seed 下 Base 与 SFT 的差异），不报告跨域绝对成功率的平均，也不改名为电商客服成功率。

因此这 60 条跨域任务非常有价值，但它们解决的是跨域泛化，不是“Retail 样本不足”本身；后者仍需要 Retail 领域内的更多独立结构（自建 held-out）来收窄区间。

**样本量门槛与功效不是同一个问题（2026-08-08 收紧）。** §9.2 的 97 来自“估计单个成功率的 95% 置信区间半宽 ±10%”这一**单比例**公式；而本实验实际比较的是 **Base/SFT（及 SFT/RL）在同任务同 seed 下的配对差异**。配对实验的功效应依据 exact McNemar 的 **discordant pairs 数量**与**预期效应量**计算，而不是单比例 CI 的 n。100 越过 97 只说明“若要估计某一个绝对成功率，样本量勉强够”，**不等于**“已有足够功效检出 Base→SFT 的配对提升”。因此每次报告前仍须按 §9.2 末段做 paired power analysis 并写入 manifest。

### 9.2 自建评测集如何确定规模

预先按所需置信区间计算，而不是拍脑袋。若估计单个主要能力的成功率，希望 95% 区间半宽不超过 `e`，保守取 `p=0.5`：

```text
n ≈ 1.96² × p(1-p) / e²
```

例如 `e=0.10` 时约需 97 个独立任务。这里的独立单位是 blueprint 结构组，不是同一任务的重复 rollout。

必须强调该公式回答的问题：它估计的是**单个成功率的置信区间宽度**，而不是**配对比较的功效**。本项目的正式结论来自 Base/SFT/RL 的同任务同 seed 配对（§8.1 用 exact McNemar），其功效由 discordant pairs 的期望数量和预期效应量决定，可能显著大于或小于 97。因此 97 只是“估计单点成功率”的下限参考，不能当作“检出配对提升”的样本量结论。最终 n 须根据 paired discordance 与预算做 power analysis，并在生成前写入 manifest。

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

P0 新增四项验收（2026-08-08）：

**P0-a　chat template 一致性。** 训练侧渲染与服务侧渲染必须逐字节一致。τ³ 通过 litellm 的 `tools` 参数把工具 schema 交给 OpenAI 兼容端点，实际 prompt 字符串由 vLLM 套用模型自带 chat template 产生；ms-swift 训练时用自己的 template 实现渲染。两者不一致会得到"dev 涨、test 掉"，且极难事后定位。

工具：`scripts/check_template_parity.py`。它渲染同一条多轮工具调用会话，逐字节 diff，并输出 ms-swift 的 label 掩码分段，把 §5 的掩码规则从假设变成可机器校验的事实。ms-swift 的 import 接口须先对 pin 住的 commit 核实。

**P0-b　成本模型。** 产出 `docs/tau3_eval_cost_model.json`，覆盖全部实验臂 × 全部外部域 × pass^k。工具：`scripts/estimate_tau3_eval_cost.py`，任务数与 judge 暴露从 `docs/tau3_split_audit.json` 读取而非手抄。先验 token 参数须在 smoke 后用 `--from-smoke` 替换为实测值。

当前先验下的量级：test split、k=4、4 个实验臂，外部套件合计约 `99M` prompt token、`2.9M` output token，其中 judge 仅占每臂 `0.39M`。agent 侧跑本地 vLLM，真正的付费项是 user simulator，成本控制应针对它。

**P0-c　B0 外部基线提前到 P0。** 原 P3 验收写的是"自建 dev 提升后才允许第一次 τ³ test"，这条规则的本意（不让 test 参与迭代）正确，但把 Base 的 test 数字也一起推迟了。测一次 Base 不构成污染，且能提前验证整条评测链路、量出真实成本、确认任务是否在 4B 射程内。

纪律由"什么时候可以测"改为"谁能看到什么"：P0 跑一次 B0，评测脚本只输出聚合指标与置信区间；逐题 trace 写入明确标记为 quarantine 的目录，除最终报告阶段外不打开。

**P0-d　split 审计。** 产物 `docs/tau3_split_audit.json`，由 `scripts/audit_tau3_retail_split.py` 生成。已完成，结论已回写至 §2.2、§6.1.1、§8.2、§9.1。

### P0.5：S0 rejection sampling

产物：`data/s0_rejection_sft.jsonl` 及其 manifest、S0 checkpoint、S0 vs B0 的自建 dev 对比、train split 覆盖盲区清单。  
验收：产出轨迹 100% 来自官方 runner 且 `reward == 1.0`；拒收原因可统计；S0 相对 B0 在自建 dev 上的增益方向有明确结论（含配对置信区间）与安全回归结果。

**S0 只对训练链路做 go/no-go，不是 P1 的单一 go/no-go（2026-08-08 收紧）。** 按下表决策，不再用“正、零、负都算通过”“完全推不动便怀疑训练链路”这类过宽表述：

```text
S0 数据无效 / template 不一致（P0-a 未过）
→ 阻塞，先修训练链路，暂不进入 P1

S0 有效但增益为 0
→ 训练链路可能正常，说明现有成功轨迹缺少新信号
→ 不否决编译器，继续做 P1 的盲区数据

S0 明显退化或安全性下降
→ 暂停扩量，检查 masking、样本分布和训练参数

S0 有稳定正增益（配对区间不跨 0）
→ 证明 rejection SFT 有效，继续 P1 扩能力
```

即：只有“数据无效 / template 不一致”才阻塞并回修训练链路；“有效但零增益”是继续 P1 的信号而非否决理由；“退化或掉安全”触发扩量暂停与链路排查；“稳定正增益”确认 rejection SFT 有效。

### P1：Retail Task Compiler v0

产物：tool graph、blueprint schema、generator、deterministic replay verifier、coverage report。  
验收：生成任务均可从干净 snapshot 重放两次；required/forbidden effects 与 DB diff 一致；test contamination 为 0；**编译产出的任务能被官方 `EnvironmentEvaluator` 以 `reward_basis=[DB]` 打出满分**。

**竖片交付（2026-08-08 修订）。** P1 不做横切。先选 2 个 task family 把 compiler → trajectory factory → SFT 整条链路端到端打通，先产出约 20 条蓝图验证端到端链路，跑完 P2/P3 拿到第一个真实 dev 增益，再按结构覆盖回头补量。原顺序要求在没有任何模型反馈的情况下先完成最大的一块工程，风险不在正确性而在吞吐。**约 20 条只验证链路，不构成“该业务能力完成”的主张。**

首个竖片选 pending-order 修改族（`modify_pending_order_items` + `modify_pending_order_payment`）。选择理由（2026-08-08 收紧，按重要性排序）：

1. **官方 train 的工具覆盖缺口**：`docs/tau3_split_audit.json` 显示 `modify_pending_order_payment` 在 train 74 中出现 **0** 次——官方 train split 对这个写工具零覆盖，编译器在此处能提供 train 本身没有的信号；
2. **真实客服业务价值**：修改待处理订单的商品与支付方式是电商客服的高频真实写操作，属于能力阶梯 M2（见 §1.1）要交付的用户可感知能力；
3. **两个工具共享前置条件**：订单处于 pending、支付方式/商品归属校验相同，selection set 可复用，一次竖片覆盖两个写工具。

纪律说明：本竖片的选择参考了 test split 的动作直方图（`modify_pending_order_payment` 在 test 40 出现 1 次；`modify_pending_order_items` 的 train:test 为 22:17），这是 test-aware informed 的合理工程，但**不把“补 test 那一题”写成动机**；确认性由 §8.1 的独立 held-out 与 ECom-Bench 承担。

**竖片必须按行为结构覆盖，而不是只生成“正确调用某个工具”的样本。** 至少覆盖以下十类，每类都要有环境可验证的终态或拒绝判定：

- 正常修改商品；
- 正常更换支付方式；
- 订单不是 pending（不可修改）；
- 商品或支付方式不属于该用户；
- 用户信息不完整（需澄清）；
- 用户中途改变需求（goal shift）；
- 修改前需要显式确认；
- 无需修改的幂等情况；
- 工具故障与恢复；
- 应当拒绝或转人工。

移植细节见 `docs/retail_task_compiler_portability_assessment.md`。

### P2：Trajectory Factory

产物：用户模拟 adapter、teacher/current-policy rollout、turn-level validator、ms-swift exporter。  
验收：工具 observation 100% 来自环境；provenance 完整；拒收原因可统计。

### P3：SFT

产物：B0/S0/S1/S2（可选 S3）、训练配置、token/coverage learning curve。  
验收：自建 dev 提升且安全不退化后，才允许对 **SFT 臂**做 τ³ test。B0 的 test 数字已在 P0-c 取得，本阶段不重跑 B0，直接复用其冻结结果做配对。

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
| `nscc/serve_tau3_agent_v1.pbs` | 四卡 DP Base/LoRA vLLM | 已实现、验证过同类服务链路、已推送 |
| `nscc/run_tau3_s0_filtered47.pbs` | 只训练已审核的 47 条 S0 数据 | 已实现、Bash 语法验证、已推送并同步 NSCC；训练尚未声明完成 |
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

四卡只增加同时进行的独立对话数；单条对话仍按轮次串行。

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
3. Airline 20、Telecom 40 是跨域 policy-following 证据，不能与 Retail 40 合并成单一电商成功率；
4. Retail test 40 已被分析过，是 test-aware informed；确认性结论还需训练不可见 held-out；
5. Windows 承担 DeepSeek 与多轮编排，NSCC 只承担 Qwen 推理和 LoRA 训练；
6. PBS 入口队列是 `normal`，`g1` 只是可能显示的内部执行队列；
7. 四卡 Qwen3-4B 应使用 DP=4/TP=1，而非 TP=4；
8. template parity 在统一 Transformers 4.57.6 后正式通过，前两次失误不进入项目叙事；
9. 项目仓库与 `tau2-bench` 在 NSCC 上是同级目录；
10. S0 rollout 经 auto-resume 最终得到 296 条有效结果，严格过滤后只有 47 条/40 任务；
11. 任何训练前必须先估算数据漏斗，不再用 rollout 次数冒充任务多样性；
12. filtered47 必须使用新 PBS，不能使用会重新过滤/重切 dev 的旧 PBS。

后续其他 Agent 不需要重新争论上述事实；如果真实仓库、环境或实验协议发生变化，必须给出新证据、明确修改本节并产生新 commit。

### 13.13 当前下一步与完成边界

当前已经完成：

```text
template parity
→ 5 题 Base smoke（3/5，基础设施错误 0）
→ Retail train 74×4 Base rollout（129/296）
→ 严格过滤得到 47 条、覆盖 40 任务
→ filtered47 四卡训练 PBS 创建、验证、commit/push、同步 NSCC
```

当前尚未完成、不得提前写成完成态：

```text
S0 LoRA 实际训练
adapter checkpoint 产出
Qwen + S0 adapter 的 vLLM 服务
Base/S0 同口径配对评测
S0 是否产生局部增益或退化的结论
pending-order Task Compiler
```

下一执行顺序：

```bash
# 若 Base vLLM 作业仍在占用四卡，先释放真实 job id
qdel <current-base-service-job-id>

cd /scratch/users/ntu/s250045/ecommerce-agentic-rag-legacy-task-closure
qsub nscc/run_tau3_s0_filtered47.pbs
```

LoRA 训练期间不需要 Windows 开机、不需要 8123 隧道、不需要 DeepSeek。训练完成后才重新启动 `serve_tau3_agent_v1.pbs` 并通过 `TAU3_ADAPTER` 加载 checkpoint，再建立 Windows 隧道执行 Base/S0 配对评测。

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
