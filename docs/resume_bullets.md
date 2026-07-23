# 简历与面试表述

## 推荐简历三条

- 基于固定 revision 的 Amazon Reviews 2023 构建 5,000 商品、43,953 子块的中英跨语言检索系统，实现 Dense+稀疏 BM25+RRF、预算过滤和父商品卡；NSCC FAISS scale set 上 Recall@5=0.992、P95=126ms，并完成 BGE reranker 效果—延迟消融。
- 构建包含 1,000 用户、10,000 订单、8 类工具及身份/政策/确认 Guardrail 的可重放 Agent Harness；在无 gold 泄漏的 60 条 locked 任务×3 次上，Rule Policy task success/pass³=0.950，政策合规与数据库终态准确率均为 1.000。
- 修复早期 Policy 可读取完整 TaskSpec 的评测泄漏，建立 Oracle/Rule/LLM 分离、轨迹回放、失败分类和 fail-closed RL gate；困难检索 v2 Recall@5=0.803、简单拒答阈值会降至 0.496，因此保留为负结果并不宣称完成 Agent RL。

## 面试时必须主动说明

- 0.992 是带程序化 gold 的规模/消融集，不等于真实模糊查询能力；去标题困难集未达到 0.85。
- reranker 改善 top-rank 与 nDCG，但 P95 约翻倍，因此默认关闭、按场景触发。
- 0.950 是 Rule Policy baseline，不是真实 LLM Agent。更准确地说，它是**确定性规则系统的
  环境与评分器基线**：任务生成器与规则策略共用高度一致的关键词模板，dev/locked 只更换
  订单与商品实体，未改变语言分布，因此不能表述为“隐藏任务上的泛化成功率”。
- LLMPolicy 360 轨迹**已经跑完，但那一轮失败了**：360/360 触发
  `model_action_parse_failure`，全部退化为 `escalate_to_human`。汇总里的 0.1667 等于
  “永远转人工”的退化基线，不是 Qwen 的能力数字；`policy_compliance=1.000` 也是免费的
  （不调工具自然不违规）。当时 trace 未保存模型原始输出，无法定位是 prompt、chat
  template、输出格式还是 parser。该批次已标记为 invalid integration run 并保留，
  有效 LLM baseline 需重跑后才能给出。
- 简历第二条里的 pass³ 目前**没有统计意义**：Rule Policy 确定性、LLMPolicy
  `do_sample=False`，三次重复只改 seed 而无随机性来源，pass³ ≡ pass@1。现阶段只能说
  “确定性重复一致率”，引入用户模拟器措辞扰动后才是真正的多次运行可靠性。
- 困难检索审核面板会展示 Top 10/20 候选与逐项约束证据；人工完成
  confirm/modify/uncertain 裁决前，不称为人工标注基准。
- Agent RL gate 尚未通过，所以只称 RL-ready Harness，不称完成 DPO/PPO/GRPO。
