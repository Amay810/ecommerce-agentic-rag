# 电商客服可信 Agentic RAG

这是一个可评测、可重放的电商客服 Agent 环境，而不是只有
`Embedding → Vector DB → LLM` 的教程 Demo。项目覆盖 5,000 商品检索、
10,000 订单状态、八类工具、写操作安全约束、轨迹回放、数据库终态评分和
fail-closed RL 门槛。

当前结论保持克制：正式规模检索与 Rule Policy Harness 已完成；困难检索、
人工审核和真实 LLMPolicy 尚未全部通过，因此不宣称完成 Agent RL。

## 系统结构

```mermaid
flowchart LR
    U[用户请求] --> P[Rule / LLM Policy]
    P --> R[Hybrid Retrieval]
    P --> T[Typed Tools]
    R --> D[5k 商品与政策]
    T --> O[1k 用户 / 10k 订单]
    T --> G[身份 + 政策 + 确认 Guardrails]
    R --> H[Trajectory]
    T --> H
    H --> E[终态 Grader / Replay / pass^3]
    E --> Q[Fail-closed RL Gate]
```

## 主要能力

- Hybrid Retrieval：多语言 dense embedding + 稀疏 BM25 + RRF；
- 商品父卡与描述、属性、评价、QA 子块；
- 预算约束过滤、复合实体分解、可选 BGE reranker；
- 八个工具：商品搜索、商品详情、比较、政策、订单、退货资格、创建退货、转人工；
- 1,000 用户和 10,000 订单的确定性模拟数据库；
- 修改型工具必须通过身份、政策资格和显式确认；
- TaskSpec、AgentObservation、ToolCall、Trajectory、GradeResult 公共契约；
- run、replay、compare、终态差异、失败分类和 pass@1/pass³；
- Oracle、Rule 和 LLM Policy 分开报告，禁止 gold TaskSpec 泄漏给普通策略。

## 目录

```text
ecommerce-agentic-rag/
├─ ecommerce_rag/          # Agent、检索、工具、订单环境、Harness、Grader
│  └─ data/                # 小型样例与可提交评测集；5k 原始语料不提交 Git
├─ scripts/                # 数据、holdout、审核、轨迹和偏好对生成
├─ nscc/                   # NSCC PBS、模型缓存和正式实验入口
├─ tests/                  # 路由、Guardrail、终态、引用与 fail-closed 测试
├─ docs/                   # 正式结果、报告、审核表与简历材料
└─ logs/                   # 本地轨迹数据库；默认不提交 Git
```

更细的保留、归档和可再生文件说明见
[目录与清理清单](docs/DIRECTORY_AND_CLEANUP.md)。

## 数据

- 来源：Amazon Reviews 2023，固定 revision
  `2b6d039ed471f2ba5fd2acb718bf33b0a7e5598e`；
- 2,500 Electronics + 2,500 Home and Kitchen；
- 5,000 商品、1,125 条有效评价、43,953 个检索子块；
- 商品事实保留英文，查询侧加入中文类别与属性别名；
- 5k 语料 SHA-256：
  `87e0f8a63a314d550dd6e68fa36ee8ae0cca492d037419215a5c9a702018c6a1`。

## 正式规模检索（NSCC FAISS）

同一 250-query scale set 在增加 distractor 商品时保持 gold 商品不变。

| 商品 | Recall@1 | Recall@5 | nDCG@5 | P50 | P95 | 索引大小 |
|---:|---:|---:|---:|---:|---:|---:|
| 40 | 0.9583 | 1.0000 | 0.9901 | 9.44 ms | 9.96 ms | 1.58 MB |
| 1,000 | 0.9625 | 0.9958 | 0.9888 | 20.52 ms | 28.93 ms | 37.30 MB |
| 5,000 | 0.9542 | 0.9917 | 0.9826 | 79.59 ms | 125.73 ms | 185.52 MB |

### Reranker 消融

| 5k 配置 | Recall@1 | Recall@5 | nDCG@5 | P95 |
|---|---:|---:|---:|---:|
| Hybrid + constraints | 0.9542 | 0.9917 | 0.9826 | 108.37 ms |
| + BGE reranker | 0.9750 | 0.9958 | 0.9958 | 219.12 ms |

Reranker 改善 top-rank 与 nDCG，但 Recall@5 仅增加 0.0041，P95 约翻倍，
因此不默认开启；适合在高价值比较或复杂约束请求上条件触发。

## 困难检索与负结果

scale set 用于规模与消融，不代表真实模糊查询上同样容易。去完整标题的 v2
locked set 得到：

| 配置 | Recall@5 | MRR | nDCG@5 | no-answer accuracy | P95 |
|---|---:|---:|---:|---:|---:|
| Hybrid + constraints | 0.8029 | 0.6246 | 0.6539 | 0.00 | 33.53 ms |
| + dev 阈值 0.65 | 0.4964 | 0.3926 | 0.4085 | 0.75 | 33.50 ms |

简单 dense 阈值虽然改善拒答，却误拒绝大量有答案问题，因此不启用。

在修复 28 题排序后生成的全新 v3 150-query holdout 得到 Recall@5=0.6889：
多约束和 near-SKU Recall@5 均为 1.0，但纯属性为 0.6625、错别字为 0.25，
无答案仍为 0。这是保留的负结果，说明困难检索仍是主要瓶颈。

## Agent Harness

120 个隐藏任务按 dev/locked 各 60 条。普通策略只接收 AgentObservation，
不能读取 category、gold 文档、允许工具或期望终态。

| Policy | Gold access | Task success | pass³ | Tool F1 | Compliance | Terminal state |
|---|---:|---:|---:|---:|---:|---:|
| Oracle upper bound | 是 | 1.000 | 1.000 | 0.944 | 1.000 | 1.000 |
| Rule Policy（NSCC） | 否 | 0.950 | 0.950 | 0.917 | 1.000 | 1.000 |
| LLM next-action | 否 | pending | pending | pending | pending | pending |

Rule Policy 的 9 次失败来自 3 个政策任务重复 3 次：物流/退款歧义词被错误地
优先解释为个人订单或退货。修复“明确政策语言优先”后，新的 routing v3
30-query holdout 达到 30/30；原 locked 结果仍保留，不把修复后的重复运行冒充
未见测试。

## 原 28 题回归

NSCC 复建索引结果为 Recall@1=0.889、Recall@5=1.000、MRR=0.963。
失败定位到自然语言 RRF tie-break 过弱和显式比较实体没有优先排序。通用修复后，
本地完整 28 题恢复到 Recall@1=0.944、Recall@3/5=1.000、MRR=1.000；
该修复仍需一次 NSCC confirmation run。

## RL 决策

当前 RL gate 为 `eligible=false`：真实 LLMPolicy 轨迹、人工审核和 preference
pairs 均未达到门槛。Oracle/Rule 轨迹不能代替真实模型轨迹，因此当前只声明
“RL-ready Harness”，不声明 Agent RL、DPO 或策略训练。

准备好的 [run_llm_policy_v2.pbs](nscc/run_llm_policy_v2.pbs) 将运行 dev 60×3
和 locked 60×3，共 360 条真实 LLMPolicy 轨迹，并导出 40 条人工审核表。

## 本地快速回归

```powershell
python -m unittest discover -s tests -v
python -m ecommerce_rag.data_loader
python -m ecommerce_rag.evaluate --index ecommerce_rag/index
```

不配置 API key 也能运行数据、检索、Rule Policy、工具安全和确定性评分。

## NSCC 入口

必须从真实仓库根目录提交：

```bash
cd /scratch/users/ntu/s250045/ecommerce-agentic-rag-git

# 规模实验
qsub nscc/build_5k_and_benchmark.pbs

# 仅测三档索引构建时间
qsub nscc/measure_index_builds.pbs

# 原 28 题确认
qsub nscc/run_regression_only.pbs

# 生成 50 条困难检索可审核证据面板
qsub nscc/build_retrieval_audit_evidence.pbs

# 360 条真实 LLMPolicy 轨迹；使用已存在的 Qwen3-4B-Instruct-2507
qsub nscc/run_llm_policy_v2.pbs
```

## 边界

- 不是生产部署，也不是临床或金融系统；
- 50 条证据面板目前是 AI-assisted pending human adjudication，用户完成
  confirm/modify/uncertain 裁决前不称人工标注；
- v2/v3 困难检索未达到 0.85 目标；
- 真实 LLMPolicy、人工 grader agreement 和 Agent RL 仍为 pending；
- 所有 easy/hard、Oracle/Rule/LLM 结果分开报告。

完整证据见 [implementation report](docs/implementation_report.md)、
[失败分析](docs/final_failure_analysis.md) 和
[人工审核指南](docs/HUMAN_AUDIT_GUIDE.md)。
