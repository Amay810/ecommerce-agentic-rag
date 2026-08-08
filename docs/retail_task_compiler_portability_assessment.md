# Retail Task Compiler：telecom generator 可移植性评估

状态：P0 调研产物，只读核对，未修改 `E:/cv_codex/external/tau2-bench`。
核对对象：`v1.0.1 / fc0055dc4e0a316c3f83133267fbd6faaa770992`。
评估日期：2026-08-08。

本文回答一个问题：`verified_ecommerce_agent_learning_v2_plan.md` §2.2 里"Retail Task Compiler 是本项目需要实现的新组件"这句话，具体要实现多少东西。

结论：**比原先设想的小得多。** telecom generator 的组合引擎可以零改动复用；而 retail 的判分机制（whole-DB hash）使得 telecom 里最复杂的那部分（`is_fixed` 谓词 + `env_assertions` 构造）在 retail 完全不需要。

## 1. 已核实的仓库事实

### 1.1 组合引擎与领域无关

`src/tau2/domains/telecom/tasks/utils.py` 虽然位于 telecom 包下，但其 import 只有：

```python
from tau2.data_model.message import ToolCall
from tau2.data_model.tasks import EnvAssertion, EnvFunctionCall
from tau2.environment.environment import Environment
```

其中 `BaseTask`、`SelectionSet`、`ComposedTask`、`compose_tasks` 没有任何 telecom 语义。`compose_tasks` 是对若干 `SelectionSet` 做 `product(tasks + [None])` 的纯组合，配一个可选 `task_validator` 过滤。

**处置：整体上移为共享模块，零逻辑改动。** 这就是 τ² 论文说的 compositional task generator 的核心。

### 1.2 TaskManager 通过三个注入回调耦合到 telecom

`src/tau2/domains/telecom/tasks/manager.py` 的 `TaskManager.__init__` 接收 `set_surrounding`、`is_fixed`、`get_env_assertions` 三个 callable。耦合点：

| 位置 | 耦合内容 | retail 是否适用 |
|---|---|---|
| `prepare_base_task` | 读 `env.user_tools.db.surroundings.{name,phone_number,is_abroad}` | **不适用**，见 1.3 |
| `set_surrounding` | 设置"用户在国内/国外"等环境背景 | 不适用 |
| `is_fixed` | 判定设备故障是否已修复 | 不适用，见 1.4 |
| `get_env_assertions` | 构造 `EnvAssertion` 列表 | 不需要，见 1.4 |
| `create_task` 中的 `fix_funcs` / `expected_failure` | 故障—修复语义 | 需替换为 retail 的目标—参考动作语义 |

### 1.3 retail 环境没有 user_tools

`src/tau2/domains/retail/environment.py:28-32` 构造 `Environment` 时只传 `domain_name`、`policy`、`tools`，未传 `user_tools`；且 `solo_mode=True` 直接 `raise ValueError`。

因此 `Environment.get_user_tools()` 会抛 `ValueError: User tools not available`（`environment.py:96-97`），`env.user_tools` 为 `None`。telecom 里所有基于 user env 的背景设置逻辑无法移植，也不需要移植。

### 1.4 retail 用 DB hash 判分，不用 env assertion

这是最重要的一条。`src/tau2/data_model/tasks.py:263-267` 定义的 `RewardType`，其中 `DB` 的语义是：

> predicted DB end state matches the target. Target is the result of replaying `EvaluationCriteria.actions` on a fresh env; any agent path producing an equivalent end state passes.

`Environment.check_db` / `get_db_hash`（`environment.py:263-282`）做整库哈希比对。

实测 114 条 retail 任务的 `reward_basis`：112 条是 `[DB, NL_ASSERTION]`，2 条是 `[DB]`。**没有一条使用 `ENV_ASSERTION` 或 `ACTION`。** 对照 telecom 的 `manager.py:104` 硬编码 `reward_eval_mode = ["ENV_ASSERTION"]`。

含义：

- retail 蓝图**不需要**手写 `env_assertions`，也**不需要** `is_fixed` 谓词；
- 目标状态由"在干净环境上重放 `actions`"自动导出，编译器只需产出 `initialization_actions` + 一条 `actions` 参考轨迹；
- "任意等价终态路径均通过"是环境自带属性，不需要我们枚举 `reference_tool_paths` 允许集合。

这一条同时修正了 plan §4.1 的一处过度设计：`required_effects` / `forbidden_effects` 在 `tau3_retail` 环境下是在重造 DB hash 已经提供的能力。建议保留该字段仅用于 `native_retail` 环境和诊断输出，不作为 τ³ 环境的判分依据。

### 1.5 确定性重放的现成范式

`manager.py:197-227` 的 `verify_task` 已经是 plan §4.6 要求的那套：重置环境 → `set_state(initialization_actions)` → 逐条执行参考动作 → 断言终态。可直接改写为 retail 版的 double-replay verifier。

需要注意 `Environment.set_state`（`environment.py:293-410`）的两个行为：

- 重放时**跳过非 mutating 工具**（`_is_mutating_tool`，`environment.py:388-389`），因此读路径差异不影响终态比对；
- 默认 `strict=True`，重放时工具返回内容与记录不一致会直接抛错。轨迹校验应保持 strict。

## 2. 移植清单

| # | 组件 | 处置 | 依据 | 估计 |
|---|---|---|---|---|
| 1 | `BaseTask` / `SelectionSet` / `ComposedTask` / `compose_tasks` | 原样复制为共享模块 | §1.1 | 0.5d |
| 2 | `RetailTaskManager` | 重写。构造 `initialization_actions` + `actions`，`reward_basis=[DB]` | §1.2 §1.4 | 2d |
| 3 | `prepare_base_task` 等价物 | 重写。从 `RetailDB` 选实体填充 `user_scenario` | §1.3 | 1d |
| 4 | `is_fixed` / `get_env_assertions` | **不实现** | §1.4 | 0d |
| 5 | retail selection sets（各 task family 的 init/goal 构造函数） | 新写，按 family 增量 | — | 每 family 0.5–1d |
| 6 | double-replay verifier | 改写 `verify_task` | §1.5 | 1d |
| 7 | 与 test 40 的结构去重 | 新写 | plan §4.7 | 1d |
| 8 | 覆盖率报告 | 新写 | plan §4.3 | 1d |

不含 selection sets 的骨架约 6.5 人日，而不是"重做一遍 Sierra 的工作"。

## 3. 竖片建议：先做哪两个 task family

`docs/tau3_split_audit.json` 里 retail 参考动作的名称直方图（全 114 条）：

| 工具 | 全集 | train 74 | test 40 |
|---|---:|---:|---:|
| `get_order_details` | 168 | 109 | 59 |
| `find_user_id_by_name_zip` | 61 | 37 | 24 |
| `get_user_details` | 57 | 36 | 21 |
| `get_product_details` | 54 | 40 | 14 |
| `return_delivered_order_items` | 41 | 32 | 9 |
| `modify_pending_order_items` | 39 | 22 | 17 |
| `exchange_delivered_order_items` | 35 | 25 | 10 |
| `cancel_pending_order` | 25 | 18 | 7 |
| `modify_pending_order_address` | 24 | 16 | 8 |
| `find_user_id_by_email` | 14 | 9 | 5 |
| `calculate` | 13 | 9 | 4 |
| `modify_user_address` | 11 | 7 | 4 |
| `transfer_to_human_agents` | 4 | 2 | 2 |
| `get_item_details` | 3 | 3 | 0 |
| `modify_pending_order_payment` | 1 | 0 | 1 |

两个直接可用的结论：

1. **`modify_pending_order_payment` 在 train 中出现 0 次，在 test 中出现 1 次。** 官方 train split 对这个写工具零覆盖。这是编译器价值最直接的证据，也是一条必须补的覆盖缺口。
2. **`modify_pending_order_items` 的 train/test 比例明显低于其他写工具**（22 : 17，其余写工具多在 2:1 以上）。test 相对偏重该 family。

因此竖片建议选 `modify_pending_order_items` + `modify_pending_order_payment` 这一组 pending-order 修改族，而不是我此前随口说的 cancel/return。理由是它同时覆盖了 train 的最大相对缺口和唯一的零覆盖工具，且共享同一套前置条件（订单处于 pending、支付方式归属校验），selection set 可复用。

## 4. 对 plan 文档的修订建议

1. §2.2 "Retail Task Compiler 是本项目需要实现的新组件" → 改为"基于官方 telecom `compose_tasks` 组合引擎移植，retail 专属部分新写；`is_fixed`/`env_assertions` 不适用"。
2. §4.1 blueprint 的 `required_effects` / `forbidden_effects` → 在 `tau3_retail` 环境下降级为诊断字段，判分以 `reward_basis=[DB]` 为准。
3. §4.1 `reference_tool_paths`"允许路径集合" → 改为单条参考轨迹即可，等价终态由 DB hash 自动放行。
4. P1 验收增加：编译产出的任务必须能被官方 `EnvironmentEvaluator` 以 `reward_basis=[DB]` 打出满分。
