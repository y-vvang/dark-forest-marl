# Figure Contract — DarkForestGrid v1

Core conclusion: 在自建黑暗森林环境中，自博弈 DQN 收敛出"低打击、重发展、不广播"
的生存策略；"先发打击"只在特定参数下保留——策略形态由环境机制（技术爆炸/打击成本）
塑形，而非公理的必然推论。
Figure archetype: quantitative grid
Target output: figures/*.pdf (+ .png sibling), vector, fonts embedded (fonttype 42)
Final size: 9×3.6 in（双栏通栏）、6.5×4 in（单栏）

Panel map:
  fig_01_strategy_emergence:
    a: 主配置 3 种子动作频率随训练回合演化（SILENT/DEVELOP/STRIKE/BROADCAST，mean±seed range）
       —— 回答"训练把策略推到了哪里"
    b: P(首次探测后10 tick内先发打击) 与平均存活比例随回合演化
       —— 回答"黑暗森林签名（探测→打击）是否出现、何时出现"
  fig_02_policy_comparison:
    a: survival_mean 四策略对比（random / pacifist / DF-heuristic / DQN），3 policy-seed mean±std
    b: final_alive 同上 —— 回答"学到的策略是否优于手写规则与随机"
  fig_03_ablation:
    a: strike_per_100 与 p_strike_after_detect：main vs no_tech_explosion vs costly_strike
       —— 回答"技术爆炸/打击成本两个机制各自把策略推向何处"（hero panel 给 H2/H3）
    b: survival_mean 与 extinction_rate 同三配置 —— 消融对生存的代价
  fig_04_survival_curves:
    单panel：四策略平均存活文明数 vs tick（400 eval 回合均值）—— 灭绝动力学全景

Evidence hierarchy:
  hero evidence: fig_01a + fig_02a（涌现形态 + 相对优势）
  validation evidence: fig_03（机制归因）
  controls / robustness: fig_04（动力学过程）；3 seeds、400 eval 回合

Statistics needed: 3-seed mean±std（n=3 训练种子）或 per-seed range（训练曲线）；
评估统一 400 回合 × 3 policy-seed，env seeds 10001+ 与训练空间不重叠。
Source data needed: runs/seed{42,43,44}_{main,no_tech_explosion,costly_strike}/metrics.jsonl;
results/eval_*.json; results/results.json。

Reviewer risk:
  - 训练曲线只有 3 种子：用 seed range（非 std）呈现并明示 n=3
  - "涌现"易被读成"证明黑暗森林"：caption 固定措辞 statements-in-model
  - P(strike|detect) 的分母是"有首次探测的文明数"，caption 必须写明定义
  - 随机策略动作分布会拉高 BROADCAST 频率：对比时避免用单一频率下结论
