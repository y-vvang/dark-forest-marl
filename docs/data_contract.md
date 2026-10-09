# Data Contract — DarkForestGrid v1.2（机制阶梯 v1.0 → v1.1 → v1.2）

> **版本指针（2026-09-21 更新）**：本契约书写于 v1.x 机制阶梯期。机制阶梯已延伸至 **v2.3**
> （资源守恒 + 自由池 + 内生生成 + 无上限文明 + 定长刻结算 + Softsign 科技奖励 + 半倍科技护盾 +
> 打击三模式选目标；动作空间 10、obs 38 维、36 槽位）。v2.x 的数据路线、字段语义与产物清单以
> `experiment_report.md` 第三～五部分与 `README.md`「环境模型与奖励设计架构」板块为准；
> v2.x 各轮产物：`runs_v2*/` `results_v2*/` `sweep_v23_arms/`。下文 v1.x 内容保留原状作为阶梯早期记录。

## Task binding
- research question: 黑暗森林法则（生存公理 + 物质竞争 + 猜疑链 + 技术爆炸）在何种机制组合下会使"隐藏+先发打击"成为多智能体强化学习自博弈的主导生存策略？
- task type: multi-agent reinforcement learning (self-play, independent Q-learning with parameter sharing)；离线评估在冻结策略上跑 Monte-Carlo 回合
- target variable: 累计折扣回报（γ=0.995）；评估期派生指标（存活率、动作分布、打击率、首次探测后先发打击率）

## Dataset route
- mode: controlled（自建合成模拟环境，无任何外部真实数据集）
- source: 本仓库内 `experiment/env.py` 实现的 DarkForestGrid（12 个文明、二维环面星图、探测/广播/打击/技术爆炸机制）
- version ladder（同一代码库，config 门控）:
  - v1.0：无任何竞争通道（develop_cost=0, passive_cost=0, neighbor_suppression=0, tech_cap=10）→ pilot，数据在 `runs_v10_pilot/`
  - v1.1：+有限全局物质池（develop_cost=0.25, passive_cost=0.02, tech_cap=10）→ 数据在 `runs_v11/`
  - v1.2：tech_cap=1000 + neighbor_suppression=0.5（无生存衰减）→ 训练数据因运行事故损失，**统一代码库重测**于 `runs_v12_rerun/`（config_v12.yaml：decay_rate=0, enable_black_domain=false, strike_penalty=0.25）；事故前完成的 3 组冻结策略评估保留在 `results_v12/`
  - v1.3（head）：+ decay_rate=0.025 + strike_penalty=0（用户设计）+ BLACK_DOMAIN 动作（用户设计）→ 数据在 `runs/`
- repository / landing page: 本地运行目录，无外部托管
- accession / DOI / stable URL: 无（自建环境，以本契约+`experiment_design.md` §2/§2.1/§2.2 为稳定描述）
- version / snapshot / date accessed: DarkForestGrid v1.3，2026-09-05
- licence / access condition: 随代码包交付，无访问限制
- persistent identifier quality: none（自建模拟环境，如实标注）

## Files used in this run
- file or subset: `experiment/env.py`（环境=数据源）；`experiment/data.py`（按 config 实例化环境与种子，是唯一的环境构造入口）
- role: train / eval 全部回合由该环境按种子生成；无 metadata/labels 外部文件

## Split protocol
- split source: 不适用（RL 无静态数据集划分）
- split rule: 训练回合使用 env seed ∈ {42,43,44}（随策略种子联动）；评估回合使用 env seed ∈ {10001..10400}，与训练种子空间不重叠；基线（random / 启发式）评估使用同一评估种子集合
- leakage risks: 无外部数据泄漏风险；固有风险是"训练分布=评估分布"（同类环境），已在报告中作为模型相对性限制声明

## Pre-processing contract
- raw to model input: 观测向量 24 维，归一化常数写死于 `env.py` 文档字符串（tech/10，坐标/50，tech差/5，计数/11，时间/T）
- normalization / filtering: 位置差按环面最短路径计算后再归一化
- exclusions: 死亡文明不再产生转移样本；无效动作（对未知目标广播/打击）映射为兜底动作并在日志标注

## Reuse and publication notes
- can this run be marked measured: 可以——所有数字来自代码在指定硬件上的真实执行（measured, auto-executed by agent via Bash）；但环境本身是合成模型，报告必须同时声明"数字是实测的、结论是模型内的"
- source-data / FAIR follow-up needed: 环境规范+种子+config 足以复现，无外部数据追踪需求
- unresolved risks: 结论对奖励尺度、探测/击杀概率参数敏感；已通过 2 组消融部分覆盖，其余作为 limitation 声明
