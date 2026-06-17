# 失败模式分类（Failure Taxonomy）

> 目的：把 baseline 条件下 28 道评测题的检索结果**逐题分类**，区分「真实可修的检索失败」与「指标假象」，并把每类失败映射到修复层和 FailureMemory pattern。
> 数据来源：`docs/eval_details_baseline.json`（baseline = price_filter / compound_decomp / freshness 全关；本地与 NSCC 数字一致）。
> 复现：`$env:ERAG_PRICE_FILTER=0; $env:ERAG_COMPOUND_DECOMP=0; ... ; python -m ecommerce_rag.evaluate`

---

## 一、总览：0.907 是怎么来的

28 题中 27 题有 gold（Q「退款到账」是 handoff，gold 为空，不计入 recall）。

baseline recall@1 = **24.5 / 27 = 0.907**，缺口来自 4 道题：

| 题目 | gold | baseline 检索结果 | r@1 | 性质 |
|---|---|---|---|---|
| 预算600以内降噪耳机 | P001 | **[P009, P001]** | 0.0 | ✅ 真实失败：数值约束 |
| 保温杯和焖烧罐区别 | P006, P014 | **[P014]** | 0.5 | ✅ 真实失败：复合实体漏召回 |
| RunBuds Clip 和 Air Pro 2 | P001, P007 | [P007, P001] | 0.5 | ⚠️ 指标假象：双 gold 都在 top-2 |
| Air Pro 2 和 QuietMax H900 | P001, P009 | [P001, P009] | 0.5 | ⚠️ 指标假象：双 gold 都在 top-2 |

**核心洞察**：4 个失分点里,只有 **2 个是真实可修的检索失败**(数值约束 + 复合实体漏召回)。另外 2 个是**双 gold 对比题的 recall@1 结构性上限**——两个 gold 不可能同时排第 1,即使检索完全正确,recall@1 也只能拿 0.5。换句话说,baseline 的 0.907 实际上**低估了**单 gold 题的 top-1 正确率。

这也解释了为什么修复后 recall@1 只能到 **0.944**：Q3 修好(0→1，+1/27=+0.037)，但 Q4/Q15/Q28 三道双 gold 题的半分是 recall@1 这个指标本身的天花板,不是检索能修的。

---

## 二、五类失败模式（+ 两类「非失败」）

### 真实失败（可修，已修）

**1. numeric_constraint_violation — 数值约束违反**
- 案例：Q「预算600以内」→ P009@899元 排在 P001@499元 之前
- 根因：dense embedding 把"预算600"当语义 token,不理解 899 > 600 的硬约束
- 修复：`price_filter`（post-retrieval 解析预算上限 + 过滤）
- FailureMemory pattern：`price_constraint_ignored`
- 效果：Q3 recall@1 0→1，全表 R@1 0.907→0.944，MRR 0.981→1.000

**2. compound_entity_miss — 复合实体漏召回**
- 案例：Q「保温杯和焖烧罐」→ 只召回 P014，P006 **完全不在 top-5**
- 根因：单 query 语义被 P014 的 Q&A（含"和普通保温杯有什么不同"）污染，把两侧实体压成一侧
- 修复：`compound_decomp`（拆子实体分别检索 + doc级 RRF + 标题 bigram 保底注入）
- FailureMemory pattern：`compound_query_recall_gap`
- 效果：Q28 retrieved 变为 [P014, P002, P017, **P006**]，recall@5 0.5→1.0

### 潜在失败（机制层，当前未致命）

**3. same_category_confusion — 同类混淆**
- 现象：rank-2 的填充项几乎总是与 gold **同品类**——Q3 [P009,P001] 都是降噪耳机；Q5 [P004,P018] 都是机械键盘；Q12 [P012,P009] 都是耳机
- 它是失败 1、2 的**底层机制**：正是因为同类商品语义高度相似,数值约束和复合实体才会被挤掉
- 当前未在 recall@5 致命,是因为混淆簇规模小(7 款耳机/4 款键盘),gold 仍能落进 top-5
- 风险预警：语料扩到 500+ 商品时,这是第一个会崩的地方（也是 reranker 真正该发力的场景）

### 非失败（正确行为，勿误判为 bug）

**4. trustworthy_caution_on_stale — 过期数据降级**
- 案例：Q「扫地机器人跨房间」「养猫清洁产品」→ 命中 P005（updated_at 故意留旧）→ action=caution
- 这不是检索失败：检索正确（recall@1=1.0），但 freshness guardrail 主动降级,避免把过期信息当确定事实
- 成功指标不是 recall,而是"过期价格/库存不被当确定事实输出"

**5. proper_handoff — 正确兜底**
- 案例：Q「我的订单退款什么时候到账」→ gold 为空 → action=handoff
- 这是系统设计的正确终点：个性化订单查询超出 KB 范围,转人工而非编造

---

## 三、给面试的一句话

> 我没有只报一个 recall@1=0.907,而是逐题拆解了这个数字:**真正可修的检索失败只有 2 个**（数值约束、复合实体漏召回），各对应一类 embedding 的已知盲区,分别用 price_filter 和 compound_decomp 修复;另外 2 个失分点是双 gold 对比题的指标结构性上限,检索本身是成功的。这个区分让我知道"修到 0.944 之后剩下的 gap 不该再投检索",而该投生成质量和混淆簇的鲁棒性。

---

## 四、附：分类计数

| 类别 | 数量 | 题目 |
|---|---|---|
| numeric_constraint_violation | 1 | 预算600降噪耳机 |
| compound_entity_miss | 1 | 保温杯和焖烧罐 |
| multi_gold_compare_artifact（指标假象） | 2 | RunBuds/Air、Air/QuietMax |
| same_category_confusion（机制层，rank-2 普遍） | — | 见 §2.3 |
| trustworthy_caution_on_stale | 2 | 扫地机器人、养猫清洁 |
| proper_handoff | 1 | 退款到账 |
| 完全正确（recall@1=1.0，单 gold） | 21 | 其余 |
