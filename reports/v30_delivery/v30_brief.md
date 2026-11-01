# v2.5.1 多种子 + v3.0 solo 范式/旗舰 champion 简报（2026-10-17，measured）

> 研究意图（用户 2026-10-16 明确化）：**"在建立的宇宙模型中，所有文明一起从零探索，将收敛出何种生存策略？"**
> 评估范式从本版起改为 **solo**：成型模型冻结后单跑，同一共识智能指挥全宇宙；训练期 metrics 不再充当评估数据。
> 附带目标：旗舰 champion（游戏应用价值，"各种策略都见过，最强大最能打"）。
> 完成判定：`logs/pipeline_v25m.log` 第二条 `PIPELINE_V25M_DONE`（22:59:30，四段 EXIT=0）+ `logs/pipeline_v30.log` `PIPELINE_V26_DONE`（05:36:47，五段 EXIT=0）。

---

## 1. v2.5.1 多种子扩展（4 臂 × 3 seeds × 1000ep + StageB 双套）

设计：v2.5 四奖励臂（a_classic / b_terminal / c_res / d_malintent）各补 seeds 43/44，训练 1000ep（v2.5 单种子 600ep 的延长兑现，回应 §8.5 "c/d 未达定态"披露）；StageB 150ep 新旧启发式双套 ×3 seeds；聚合 mean±std。

### 1.1 Stage B 生存（v2b 口径，3 seeds mean±std）

| 臂 | learner | random | df | **pacifist** |
|---|---|---|---|---|
| a_classic | 0.538 ± 0.035 | 0.350 ± 0.009 | 0.586 ± 0.009 | **0.640 ± 0.014** |
| b_terminal | 0.478 ± 0.028 | 0.333 ± 0.014 | 0.543 ± 0.004 | **0.604 ± 0.010** |
| c_res | 0.438 ± 0.011 | 0.326 ± 0.020 | 0.536 ± 0.015 | **0.602 ± 0.008** |
| d_malintent | 0.518 ± 0.030 | 0.363 ± 0.016 | 0.551 ± 0.018 | **0.617 ± 0.010** |

（old 启发式套同构：pacifist 0.508–0.588 全居首，见 `results/v25m_arms/analysis_multiseed.json`）

### 1.2 跨种子一致性核查（对 §8.4 假说的裁决）

| v2.5 假说（单种子 600ep） | 多种子裁决（1000ep×3） |
|---|---|
| pacifist 四臂居首 | **✓ 稳健升级为事实**：3/3 居首，std 仅 0.008–0.014，种子间几乎无重叠风险 |
| d 臂"疯狂揭露"信息战（后期 BOTHER 0.807、return +337） | **✓ 稳健复现**：late-train BROADCAST_OTHER 3 seeds = 0.768 / 0.756 / 0.712；**return 唯一转正臂复现**（+170 / +254 / +47） |
| ~~d 臂 BOTHER 0.807 为种子噪声（10-16 晨核查）~~ | **✗ 判断撤销——查错键名**：metrics 无 `BOTHER` 键（为 BROADCAST_OTHER 简写），按真实键核查后 3/3 复现 |
| c 臂掠夺流未收敛（ret −1278） | **✓ 3/3 保持负**：−682 / −818 / −570；INVADE 稳定 ~0.10/100 tick（solo 口径 2.62）——掠夺在 1000ep 仍不收敛，"贪婪反噬"跨种子成立 |
| learner 信息完备零利用 | ✓ 方向保持：learner 0.438–0.538 全面低于同池 pacifist |
| b 臂隐匿流 | ✓ late-train SILENT 0.345 + BLACK_DOMAIN 0.070（四臂最高黑域倾向） |

收敛曲线（种子均值+std 带）：`figures/fig_15_v25m_convergence_multiseed`；多种子生存对比：`fig_14_v25m_survival_multiseed`。

---

## 2. v3.0 solo 评估范式（冻结模型指挥全宇宙）

设计：`eval_v30_solo.py`，12 个成型 checkpoint（4 臂 × 3 seeds）各 16 episodes × 600 tick，全部 36 槽由同一网络 greedy 驱动；统计 per-civ 终身生存、8 动作画像、黑暗森林信号（首触 10 tick 内打击率 p_imm、探测→打击延迟、打击目标 tech）。

### 2.1 四臂 solo 画像（3 seeds mean±std）

| 臂 | 终身生存 | strike/100 | 黑域/100 | invade/100 | **p_imm** | SILENT | DEVELOP |
|---|---|---|---|---|---|---|---|
| a_classic | 0.499 ± 0.090 | 0.19 | 2.05 | 0.57 | 0.021 | 0.732 | 0.247 |
| b_terminal | **0.384** ± 0.014 | 0.04 | **9.24** | 1.39 | 0.000 | **0.884** | 0.082 |
| c_res | 0.500 ± 0.062 | 0.18 | 0.27 | **2.62** | 0.008 | 0.813 | 0.171 |
| d_malintent | 0.479 ± 0.137 | 0.08 | 0.00 | 2.12 | **0.395** | 0.781 | 0.022 |

### 2.2 核心发现（对研究问题的回答）

1. **先发打击不是从零自博弈的收敛解**。四臂 solo 绝对打击率 0.04–0.19/100 tick，除 d 臂外 p_imm ≤ 0.02；主流画像 = 沉默（0.73–0.88）+ 发展。**"黑暗森林式"行为在该环境公理体系下没有被从零探索收敛出来**——与 v2.3-fix 时代"森林真点燃但先发打击净害"的启发式世界结论一致，且现在是**同源策略世界**的内部事实。
2. **奖励塑形改变策略形态，不改"低打击"底色**：d 臂（恶意/揭露奖励）把首发倾向拉到 p_imm=0.395 并维持入侵骚扰（2.12/100），但绝对打击仍低；c 臂（资源奖励）推向掠夺（2.62/100）。
3. **极端隐匿反伤自身**：b 臂（仅终局奖励）收敛出最高黑域（9.24/100）+ 最高沉默（0.884），但 solo 生存最低（0.384±0.014，std 最小=高度一致的次优吸引子）——黑域挤占发展窗口。
4. **同质宇宙 vs 混合池**：同一批 learner 在 StageB 混合池 0.438–0.538，solo 0.384–0.500——全员同质时生存普遍更低（无pacifist"肉盾"参照、同质竞争直接对撞）。

图：`fig_16_v30_solo_profile`（8 动作画像 ×4 臂）、`fig_17_v30_solo_df_signals`（黑暗森林信号 1×4）。

---

## 3. 旗舰 champion（league 训练 + 评估）

设计：`train2_v30_champion.py`，3 seeds × 1000ep；每 episode 重抽 36 槽 = learner 12 槽（同网络共识，唯一学习方）+ 24 槽对手池均匀采样（random / df / pacifist / 四臂 seed42 冻结网络，对手 eps=0.05）；transition 只收 learner 槽。

### 3.1 训练画像（league 混战，measured）

| seed | late return | alive | 画像要点 |
|---|---|---|---|
| 42 | −667 | 0.206 | silent 0.18 / invade 0.11 / bd 0.07 |
| 43 | −495 | 0.287 | silent 0.09 / bd 0.14（黑域型） |
| 44 | −331 | 0.223 | silent 0.16 / invade 0.04 |

### 3.2 league 评估（champion 12 槽 vs 单一对手 24 槽，8ep × 3 seeds）

| 对手 | champion 生存 |
|---|---|
| random | **0.584** ± 0.048 |
| pacifist | 0.538 ± 0.194 |
| df | 0.484 ± 0.142 |
| a_classic ckpt | 0.356 ± 0.046 |
| c_res ckpt | 0.350 ± 0.061 |
| d_malintent ckpt | 0.360 ± 0.043 |
| **b_terminal ckpt** | **0.312** ± 0.032 |

### 3.3 诚实披露：**"最能打"未达成（负结果）**

champion 对全部四个同源臂模型全败（0.31–0.36），对启发式也仅 random 占优（0.584）、df 反而压制它（0.484）。归因假说（未验证）：①league 混战中学习信号被 7 类异质对手稀释，1000ep 不足以专门针对任何一类优化；②同源自博弈模型是"同类打法专家"，champion 的泛化打法吃亏；③12/36 槽配比下 champion 的训练生存压力结构（异质混战）与评估局（12 vs 24 同类）失配。champion solo 生存 0.406（低于 a/c 臂 0.499/0.500）同样印证：league 训练优化的是"对付异质对手"，不是"同质宇宙最优"。

**若要真旗舰**：升级为 curriculum（先混战后专攻）或对臂模型选择性过采样 + 延长训练。当前版本按"见过所有策略家族的泛化体"定位，不按"最强"定位。

图：`fig_18_v30_champion_league`。

---

## 4. 披露

- solo 生存口径 = per-civ 存活 tick / (评估窗口 − 出生 tick)，仅计 born>0 槽；16ep（旗舰 24ep）× 600 tick，与训练期 metrics 无关（冻结模型重跑）。
- p_imm = 打击事件中"距该攻击者首次探测 ≤10 tick"的占比；探测→打击延迟为均值（中位数需逐事件日志，未存）。
- champion league 为 8ep×3 seeds 小样本，±0.03–0.19 置信有限，按假说级对待。
- 05:20 存在一次流水线自动续跑（s44 训练夜间被回收后幂等重训 14.6min）——幂等断点机制按设计工作，产物完整。
- 复现：`bash experiment/pipeline_v25m.sh && bash experiment/pipeline_v30.sh`（幂等，完成态秒退）；单段 `python3 experiment/run_v30_driver.py --stage solo25m|champion_train|champion_eval`。
- 产物：`runs/v25m_arms/`、`runs/v3/v30_champion/`、`results/v25m_arms/analysis_multiseed.json`、`results/v30_solo/{<arm>/seed<s>,analysis_v30_solo.json}`、`results/v3/v30_champion/seed<s>/{solo,league/<kind>}`、`figures/fig_14–18_*`。
