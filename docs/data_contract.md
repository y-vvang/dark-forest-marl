# Data Contract — DarkForestGrid v1.2（机制阶梯 v1.0 → v1.1 → v1.2）

> **版本指针（2026-10-15 更新）**：本契约书写于 v1.x 机制阶梯期。机制阶梯已延伸至 **v2.5**
> （当前版本 **v2.5** = v2.4 全套——观测优先信息原则、BROADCAST_ALL/BROADCAST_OTHER 揭露、
> 禁科技继承、四奖励臂参数 `res_reward_scale`/`attack_reward`——加**信息完备化**：
> obs 新增自我状态 `under_attack`（被入侵，滑窗一刻）与 `self_public`（被揭露，持久），
> **动作空间 8、obs 38 维**（scal 7→9）；36 槽位、**出生数无上限**（死亡槽回收抹社交记忆，
> 资源预算为唯一刹车）。v2.5 obs 语义要点：他人科技/资源仅 observed 后可见不变；
> 新增自我状态是唯一的信息增量——受害者/被揭露者自知，无专用应对动作。
> v2.x 的数据路线、字段语义与产物清单以 `experiment_report.md` 第三～七部分、
> `reports/v24_arms_brief.md` 与 `README.md`「环境模型与奖励设计架构」板块为准；
> v2.4 各臂产物：`runs/v24_arms/` `results/v24_arms/`（summary.json + per-arm stage_b.json）。
> v2.4 观测语义要点：他人科技/资源仅在其被 `observed` 后进入 obs（`near_tech/near_res=-1` 占位）；
> v2.3 的"known 即可见科技"的打击排序动作已删除。下文 v1.x 内容保留原状作为阶梯早期记录。

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

## v2.5.1 / v3.0 增量契约（2026-10-17 增补，measured）
- **评估范式变更（v3.0 起）**：主评估=solo（`eval_v30_solo.py`）——成型模型冻结后单跑，36 槽同网络 greedy（eps=0），16ep×600tick（旗舰 24ep）；**训练期 metrics 不再充当评估数据**（v2.5 及以前的 StageB 混合池数据保留作历史对照）
- solo 指标口径：survival_fraction_of_lifetime = per-civ 存活 tick/(评估窗口−出生 tick)，仅计 born>0 槽；p_imm = 打击事件中"距该攻击者首次探测 ≤10 tick"占比；探测→打击延迟为均值（中位数未存）；strike/bd/invade/observe 按 100 存活 tick 归一
- league 口径（旗舰）：learner 12 槽 vs 单一对手 kind 24 槽，8ep×3 seeds，对手=启发式函数或四臂 seed42 冻结网络（eps=0.05）
- 训练契约（champion）：`train2_v30_champion.py --mode champion`，每 ep 重抽 36 槽=learner 12+对手池 24（random/df/pacifist/ckpt:a/b/c/d 均匀采样），transition 只收 learner 槽；run_name=`seed<s>_champion`
- 种子空间：v2.5.1 训练 seeds {42,43,44}（champion 同）；solo 评估 env seeds 从 900 起（seed0=900+ep），与训练空间不重叠
- 新增产物：`runs/v25m_arms/`、`runs/v30_champion/run/seed<s>_champion/`、`results/v25m_arms/{<arm>/seed<s>/{stageb_old,stageb_v2b}/stage_b.json,summary_multiseed.json,analysis_multiseed.json}`、`results/v30_solo/{<arm>/seed<s>/solo_eval.json,analysis_v30_solo.json}`、`results/v30_champion/seed<s>/{solo,league/<kind>}/solo_eval.json`、`figures/fig_14–18_{v25m,v30}_*`（pdf+png 双格式）
- 幂等判定契约：训练完成=metrics 尾行 episode≥1000（**checkpoint 存在不算完成**，其只是每 50ep 中途保存点）；评估完成=solo_eval.json 存在；流水线 flock 单实例
- 已知限制：league 为小样本（8ep×3 seeds）；champion solo 生存 0.406 低于 a/c 臂 solo——league 训练目标（异质对手泛化）与 solo 口径（同质宇宙最优）不同，解读时不可混用

## v3.2/v3.3 契约增补（2026-10-18）

- **v3.2 sweep 产物**（2026-10-18 归档至 v3）：`runs/v3/v32_sweep/seed42_sw_*/`（19 分支 metrics+checkpoint）+ `results/v3/v32_sweep/{stage_A1..A5,sweep_summary}.json`；分支字段 name/vs_arm/vs_heur/solo_surv/complete
- **v3.2 确认（离线早停）产物**：`runs/v3/v32_confirm/seed{42,43,44}_confirm_s*/`（metrics + `checkpoint_ep{250,500,750}.npz` + 终点 `checkpoint.npz`）；`results/v3/v32_confirm/seed<s>/{snap{250,500,750,1000}/league/<kind>/solo_eval.json, solo/solo_eval.json}`；汇总 `confirm_summary.json`（per_seed_peaks：seed/snap_ep/vs_arm/vs_heur/solo_surv）
- **快照契约（v3.3 起为训练器一等能力）**：`train3_champion.py --snap-every N`（默认 250，0=关），快照名 `checkpoint_ep<ep>.npz`；终点另存 `checkpoint.npz`；快照存在性参与幂等判定（v3.2 confirm 口径：tail_ep≥1000 且快照齐）
- **v3.3 容量轴产物**：`runs/v3/v33_capacity/seed42_cap_h{64x64,128x128,192x192}/` + `results/v3/v33_capacity/<tag>/{league/<kind>,solo}/solo_eval.json` + `capacity_summary.json`（baseline_h64=v3.2 ts500 分支引用 + capacity 数组 + delta_vs_arm_vs_h64；汇总键 ckpt:<arm>）
- **归档规则（2026-10-18 起）**：champion 训练线（rounds）产物统一在 `runs/v3/`、`results/v3/` 命名空间（v30_champion/v31_champion/v32_sweep/v32_confirm/v33_capacity）；代码引用已全量同步（run_v30/v32/v33_driver、train2/train3、analysis_v30_solo/v31、plot_v31、pipeline_v31/v33）；v30_solo（solo 评估产物）不在 rounds 范围，保留原位
- **v3.3 口径警示**：容量轴三宽度全部**从头训**（无 --init-from）——hidden 变宽使热启动 ckpt 的 Adam 动量缓冲与全层权重形状失效（128 宽加载 64 宽 Adam 状态首步广播崩溃实证）；与 v3.2 FINAL（热启动口径）的 0.4667/0.4132 **不可直接横比**，容量轴内部三点同口径横比有效
- **超参 override 铁律（v3.2 事故教训）**：所有 override 必须在消费对象构造之前写入；hidden 走 `cfg["model"]["hidden"]`（完整 config），lr/gamma 等走 `tc=cfg["train"]`；同 seed 确定性下"分数逐位相同=配置未生效"是收数必查项
