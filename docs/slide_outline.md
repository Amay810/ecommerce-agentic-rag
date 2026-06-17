# PPT 大纲 — 电商客服 Agentic RAG（面试版）
> 目标听众：LLM 算法/RAG 方向面试官
> 核心叙事：**科学方法论驱动的 RAG 工程**——量化评测 → 诚实负结果 → 定位真问题 → 修 → ablation 证明
> 页数：14 页，控制在 8-10 分钟讲完；每页一句 Takeaway
> 故事弧：定位(P1-3) → 技术深挖(P4-7) → 实验与进化(P8-11) → 系统成熟度(P12-14)
> 节奏：P1-P2=1分钟；P3过渡快讲；P4-P7=3分钟；P8-P11=4分钟（高潮）；P12-P14=2分钟

---

## 开篇：定位（3 页）

### P1 个人定位（20 秒）
**内容**
- 背景 + 求职方向（LLM 算法 / RAG）
- 关键词标签：Hybrid Retrieval · Retrieval Eval · Agentic RAG · Knowledge Grounding

**开场白（直接说这句）**：
> 我这个项目不是追求堆模块，而是把科学实验流程用在 RAG 工程里：先定义 gold-doc 指标，再通过负结果和 failure memory 找到真实瓶颈，最后用 targeted fix 做 ablation 证明。

**Takeaway**：我做的是可测、可量化的检索系统，不是调包 demo。

**面试钩子**（不用说出来，留给他们问）：
> "你说'可量化'，你怎么量化 RAG 的质量？" → 引到 P7 评测体系

---

### P2 项目定位（最重要一页）
**内容**
三栏对比表：

| | 普通 RAG demo | 传统规则客服 | **本系统** |
|---|---|---|---|
| 召回 | 无量化评测 | 精确匹配 | Recall@k / MRR |
| 回答 | 生成即发出 | 模板 | 生成→验证→置信度决策 |
| 失败处理 | 瞎编 | 转人工 | grounding 闸门→可溯源 handoff |
| 迭代能力 | 无 | 手工维护 | SupportCase 飞轮 |

定位句：
> "系统的产物不是一句回答，而是一条可复盘的 `SupportCase`：意图、证据、引用、置信度、验证结果、失败原因、最终动作——全部可审计。"

**Takeaway**：我把它定位成可验证、可审计的客服决策系统，而非 RAG demo。

---

### P3 真实痛点 × 架构回应（过渡页，快讲）
**内容**（两列 4 行）

| 客服痛点 | 系统级回应 |
|---|---|
| ①幻觉报错价格/参数 | Grounding + 数字引用检查 + 一致性验证 |
| ②答非所问 | 意图路由六分类（100% accuracy on 28 eval） |
| ③"600以内"买到 899 元产品 | 价格约束解析 + post-retrieval 过滤 |
| ④"A和B哪个好"遗漏一半候选 | Compound query 分解 + 双实体保底注入 |

**Takeaway**：架构不是堆模块，是对客服可信度痛点的逐条工程回应。

**面试钩子**：
> "③和④是你后来发现的问题还是预先设计的？" → 引到 P9 FailureMemory

---

## 技术深挖（4 页）

### P4 整体架构（流程图）
**内容**（一张端到端流程图）
```
用户问题
  → 意图路由（6类）
  → 混合检索（dense+BM25+RRF，父子文档）
    + 价格约束过滤 [NEW]
    + compound query 分解 [NEW]
  → 生成（LLM / catalog 模板）
  → 三层验证（grounding / 引用 / 一致性）
  → 置信决策（ok / caution / handoff）
  → SupportCase 落库
  → FailureMemory 分析 → CaseMemory 建索引
  → 记忆先验（advisory trace）
```

**Takeaway**：LLM 负责 propose，确定性工具负责 verify，记忆负责 accumulate。

---

### P5 混合检索（技术深挖 #1）
**内容**
- **Dense（语义）**：paraphrase-multilingual-MiniLM-L12-v2，余弦相似度闸门 0.35
- **BM25（词法）**：jieba 分词，捕捉型号/参数精确匹配（"H900"、"3000mAh"）
- **RRF 融合**：1/(k+rank) 加权，避免相似度绝对值不可比
- **父子文档**：检索细粒度子 chunk，喂 LLM 完整父卡（保留上下文）
- 40 商品 / 5 政策 / 205 chunks（NSCC 重建索引）；含同类混淆簇（7 款耳机、4 款键盘）

**Takeaway**：口语问题靠 dense，型号参数靠 BM25，RRF 兼顾两者。

**面试钩子**：
> "有了 dense embedding，还需要 BM25 做什么？" → BM25 对精确型号/数字的 IDF 权重远高于语义相似度；"H900"不在训练语料里，dense 向量退化为随机位置

---

### P6 生成后验证（技术深挖 #2）
**内容**（三层验证）

1. **句级 Grounding**：encode(回答句) ↔ encode(context 句)，余弦相似度 > 0.42 为有支撑，<50% 支撑率 → caution
2. **数字引用检查**：regex 检测价格/规格数字有无 `[资料N]` 标注
3. **LLM 一致性**：判断回答是否"矛盾/资料外/通过"（有 API key 时启用）
→ 三层 → `final_decision(grounding, consistency, citation)` → ok / caution / handoff

**Takeaway**：回答不是生成完就给用户，要先过可信度闸门。

---

### P7 评测体系（科学方法论基石）
**内容**
- **评测集**：28 题 gold 标注（expected_intent / expected_action / gold_doc_ids）
  - 刻意构造混淆：7 款降噪耳机、4 款机械键盘、5 款杯类
- **指标**：Recall@1 / Recall@3 / Recall@5 / MRR（基于 gold_doc_ids 的 IR 标准指标）
  - 指标为 query macro-average；所有 28 题均有 gold_doc_ids，全部参与 recall/MRR 计算
- **执行**：NSCC PBS CPU 节点，HF_HUB_OFFLINE=1，env flag 控制 ablation
- **基线**：recall@1=0.907 / recall@3=0.981 / recall@5=0.981 / MRR=0.981

**Takeaway**：先有量化指标，才有资格谈"改进了多少"。

**面试钩子**：
> "你为什么选 Recall@k 而不是 RAGAS / faithfulness？" → gold doc retrieval 是一阶问题；生成质量需要 LLM key，可以加，但先把检索搞对是前提
> "recall@1=0.907 怎么算的？" → 28 题，每题看 retrieved_doc_ids[0] 是否在 gold_doc_ids 里，命中数/28，macro-average

---

## 实验与进化（4 页，高潮区）

### P8 Honest Evaluation — Reranker 负结果
**内容**（三路对照表）

| 配置 | R@1 | R@3 | R@5 | MRR | 失败模式 |
|---|---|---|---|---|---|
| baseline hybrid | 0.907 | 0.981 | 0.981 | 0.981 | — |
| +reranker(chunk) | 0.907 | 0.981 | 0.981 | 0.975 | 子 chunk 丢上下文（猫毛信号在 review，不在 desc） |
| +reranker(parent, no dedup) | 0.907 | 0.907 | 0.907 | 0.963 | 同商品兄弟 chunk 刷屏，多样性崩塌 |
| +reranker(parent+dedup) | 0.907 | 0.981 | 0.981 | 0.981 | 追平基线，不提升 |

结论框：
> 一阶 hybrid recall@5 已达 0.981，无 headroom。reranker 两类失败：①子 chunk 丢上下文（MRR 0.981→0.975）；②父 chunk 无去重时同商品兄弟刷屏（R@3 0.981→0.907，多样性崩塌）。**评测驱动决策：不上线 reranker。**

**Takeaway**：敢讲负结果 = 有评测纪律；知道为什么不上 = 比知道怎么上更难。

**面试钩子**：
> "那重排器什么时候有用？" → 语料扩大到 500+ 商品、query-doc 分布更复杂时；当前 40 商品 corpus 不够"难"让 reranker 发力

---

### P9 FailureMemory — 定位真正瓶颈
**内容**
- 累积 SupportCase → `FailureMemory.analyze()` 自动挖掘模式，**直接产出修复优先级**：

| pattern | 根因诊断 | 产出修复目标 |
|---|---|---|
| `price_constraint_ignored` | dense embedding 无法捕捉数值约束 | Q3 → price_filter |
| `compound_query_recall_gap` | "A和B" 遗漏一半实体 | Q28 → compound_decomp |
| `stale_data_caution` | KB 文档 updated_at 超期 | freshness guardrail |

（其余 pattern：`zero_retrieval_handoff`、`handoff_cluster`；KB 缺口：`retrieval_blind_spot`、`stale_doc` — 见备注）

**Takeaway**：FailureMemory 不是日志，是修复优先级生成器；不靠直觉猜哪里坏。

**面试钩子**：
> "FailureMemory 和 CaseMemory 有什么区别？" → FM 是离线分析（batch pattern detection），CM 是在线 advisory（query-time semantic match）；一个诊断，一个预防

---

### P10 两个精准修复：Q3 + Q28
**内容**（分两栏）

**Q3 — 价格约束过滤**
- 根因：embedding "预算600以内" 仍把 P009@899元 排名第一（数值约束非语义问题）
- 修复：`price_filter.parse_budget()` 解析中文预算表达 → post-retrieval 过滤 price>budget
- 结果：Q3 recall@1: 0 → **1**

**Q28 — 复合查询分解**
- 根因："保温杯" 子查询返回 P014（因 P014 Q&A 含"和普通保温杯有什么不同"，语义污染）
- 修复：detect("A和B") → 分别检索各子实体 + 全量 query → doc级 RRF → **标题 bigram 保底注入**（绕过语义歧义）
- 英文名护栏："RunBuds Clip 和 Air Pro 2" 不截断（\s 不作 stop char；纯 CJK 才限 8 字符）
- 结果：Q28 recall@5: 0.5 → **1.0**（P006 进入候选池）

**关键定位句**：
> 这些不是拍脑袋规则，而是由 gold-doc evaluation 和 FailureMemory 定位出的确定性修复层——针对 embedding 的已测失效模式，可解释、可测试、可回滚。

**Takeaway**：知道 embedding 在哪里失效，才知道用什么规则补它。

**面试钩子**：
> "为什么不直接微调 embedding 模型来理解价格约束？" → 微调需要带 price 标注的对比训练对，成本高、泛化性差；rule-based filter 可解释、可测试、可回滚
> "标题 bigram 保底会不会误命中？" → 引出 guardrail 设计：中文实体要求≥2 CJK；英文走 token overlap；多命中按分数排序

---

### P11 Ablation Table（最终量化证明）
**内容**（最核心的一页）

| 配置 | Recall@1 | Recall@3 | Recall@5 | MRR | 说明 |
|---|---|---|---|---|---|
| baseline | 0.907 | 0.981 | 0.981 | 0.981 | 纯 hybrid |
| +price_filter | **0.944** | 0.981 | 0.981 | **1.000** | Δr@1=+0.037, ΔMRR=+0.019 |
| +compound_decomp | 0.907 | 0.981 | **1.000** | 0.981 | Δr@5=+0.019 |
| **+pf+cd (final retrieval)** | **0.944** | 0.981 | **1.000** | **1.000** | 两项可加，无回退 |
| +freshness_guard | 0.944 | 0.981 | 1.000 | 1.000 | 检索不变；guardrail 在生成层 |
| memory_prior | — | — | — | — | advisory trace，非检索项 |

关键结论（写进 slide）：
> price_filter fixes a **ranking error** (wrong #1); compound_decomp fixes a **coverage gap** (missing candidate entirely). Two failure modes, two targeted repairs, zero regression.

Reranker 单独表（Honest Eval）：三路均无提升，P8 已述。

**Takeaway**：每个模块的贡献独立可测，两项核心修复可加且无回退——这是工程可信度的来源。

**面试钩子**：
> "为什么 price_filter 提升的是 recall@1/MRR，compound_decomp 提升的是 recall@5？" → PF 修的是 top-1 排名错误（P009排第一），CD 修的是候选覆盖缺口（P006根本不在top-5）；两者是不同类型的召回失败

---

## 系统成熟度（3 页）

### P12 SupportCase + 记忆飞轮
**内容**
```
handoff / 低置信案例
    ↓ SupportCase 落库（SQLite，stable id = sc_{ts}_{hash10}）
    ↓ FailureMemory.analyze() → pattern + KB 缺口
    ↓ CaseMemory.build() → embed 历史 case → .npy（23 case，threshold 0.70）
    ↓ 新 query 命中（sim=0.701）→ 记忆先验 trace
    ↓ suggested_action=apply_price_filter → 工程修复优先级 → price_filter 上线并通过 ablation 验证
```
CaseMemory 实测 trace 示例：
> `记忆先验：sim=0.701 pattern=price_constraint_ignored suggested=apply_price_filter`

（CaseMemory 是 advisory trace，不自动执行修复；修复由工程实现+ablation 验证）

**Takeaway**：每次失败的客服请求都让下一次更聪明——这是 demo 和系统的分水岭。

**面试钩子**：
> "CaseMemory 是 cache 还是类似 episodic memory？" → 语义相似度检索（非精确命中）+ advisory（不覆盖检索结果）；更接近 episodic memory；threshold 0.70 控制误匹配
> "飞轮的终点是什么？" → KB 补全（KB gap → 补商品/政策文档）+ pattern 消失（fix 生效后 price_constraint_ignored 频率下降）

---

### P13 新鲜度护栏（业务 sense）
**内容**
- 每次生成回答前校验：answer 中的价格/库存/政策 → 追溯到 chunk 的 `updated_at` → 与 `FRESHNESS_MAX_AGE_DAYS=30` 对比
- 状态：`fresh`（正常）/ `unverified`（无时间戳）/ `stale`（超期）
- stale/unverified → 降级 ok→caution，追加 advisory note
- 演示：P005（updated_at=2026-01-01，故意留旧）触发 stale caution；P006（2026-06-10）正常通过
- **Ablation 验证**：+freshness_guard 检索指标不变（guardrail 在生成层，不影响召回）
- freshness 的成功指标不是 recall，而是**避免过期价格/库存被当作确定事实输出**

**Takeaway**：客服系统不能乱答价格和库存——可信比"答得像"更重要。

---

### P14 总结 + 边界声明
**内容**

**证明了三件事：**
1. **会做检索工程**：hybrid(dense+BM25+RRF) + 父子文档 + 复合分解，recall@1 从 0.907 → 0.944
2. **会做科学评测**：gold 标注 + recall@k/MRR + ablation table，每一步有数字
3. **会做"不加某模块"的判断**：reranker A/B → 负结果 → 定位真瓶颈 → 精准修复

**边界（诚实是加分项）：**
- 本地 demo + NSCC 离线评测，非生产系统
- 无 LLM key 时生成层降级为 catalog 模板（citation 检查失效）
- 语料 40 商品，测试集 28 题，corpus 规模受限
- reranker 实测无提升，未上线

**Takeaway**：我能把 RAG 从 demo 做成可验证、可审计、会自我迭代的系统。

---

## 数据/产物对照表（制 PPT 时直接查）

| 页 | 数据来源 |
|---|---|
| P7 baseline | `nscc/run_eval.pbs` + `ablation.log` |
| P8 reranker | `docs/honest_evaluation.md`（完整 4 列 3 行表） |
| P9 pattern | `nscc/smoke_failure_memory.py` NSCC 输出 |
| P10 Q3 fix | `ablation.log` TARGETED FIXES 段，Q3 recall@1 0→1 |
| P10 Q28 fix | `ablation.log` TARGETED FIXES 段，Q28 recall@5 0.5→1.0 |
| P11 全表 | `ablation.log` 完整输出 |
| P12 flywheel | `nscc/smoke_case_memory.py` NSCC 输出，sim=0.701 |
| P13 freshness | `ablation.log` +freshness 行，检索指标不变 |

**数字来源与可复现性说明**：
- 本地索引已重建为 205 chunks（40 商品 + 5 政策），与 NSCC 一致；`python -m ecommerce_rag.data_loader` 一键重建
- ablation 全表已在本地完全复现 NSCC 数字（baseline 0.907/0.981/0.981 → +pf+cd 0.944/1.000/1.000，逐格一致）
- Q28 P006 证据已确认：`+compound_decomp` 下 Q28 retrieved=`[P014, P002, P017, P006]` → recall@5=1.0；baseline 仅 `[P014]` → 0.5。保底注入真实生效，非空开关
- 复现命令：`$env:PYTHONPATH=repo; python nscc/run_ablation.py`（CPU 即可，~0.5s/condition）

**严禁写进 PPT 的内容（会被追问穿帮）：**
- "生产环境/上线/服务用户/训练过模型"
- 任何没有实测数字支撑的性能声称
- reranker 提升了性能（诚实评测结论是没提升）
- CaseMemory 自动执行了修复（它是 advisory trace，不是执行层）
