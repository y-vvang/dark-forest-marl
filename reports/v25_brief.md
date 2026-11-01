# v2.5 信息完备化简报（2026-10-15 用户 spec → 2026-10-15 执行）

> 基线：v2.4（观测优先信息原则+同起跑线+四奖励臂，`reports/v24_arms_brief.md`）。
> v2.5 三条 spec：①单世界出生数上限取消（只以 tick 满为终局，自由池够即生成）；
> ②"被入侵"是典型环境状态（立即发现入侵者并知晓正被入侵，模型只按正常动作决策，无"反咬"类专用动作）；
> ③"自身被 BROADCAST_OTHER 广播"可作为环境状态被醒目发现。
> 主跑四臂 600ep+StageB150（旧启发式，批次内一致）+ **h2 追加**：四臂 StageB 用 v2.5 适配启发式重跑。

## 1. 环境改动（env2_v25.py，v2.4 链零改动）

1. **出生数无上限（死亡槽回收）**：36 槽位不变；出生优先未出生槽，耗尽后回收死亡槽给新生儿（抹除 known/observed/public/under_attack/in_domain/first_detect/last_action/death_cause，spawn_tech/res/peak_tech/spawn_tick 覆盖=本代语义）。ever-born uncapped，资源预算（150）为唯一刹车——36 个健康活体需 >150 资源，并发顶满实际不可达。终局条件不变（t_max 或热死）。
2. **"被入侵"第一类状态**：入侵通知既有 `known[j,i]=True`（立即发现入侵者），v2.5 加 obs 自我状态 `under_attack`（滑窗一刻：重选 obs 即见"正被入侵"，下一 tick 常规 obs 仍见"上 tick 被入侵"）+ `last_invader[j]`（最近入侵者身份，供画像/启发式）。动作空间不变。
3. **被揭露自知**：obs 加 `self_public`（=public 自我标志，持久——known 不可撤销、黑域不洗旧知识，被揭露者每 tick 醒目自知）。
4. obs 36→38（scal 7→9）；归一化分母固定 n_norm=36（防跨版本信号漂移，alive/known 计数可超 1 后 clip 2）。
5. Stage B 统计配套：出生 tick 时 alive_ticks/loot/kills/first_detect 归零（口径=当前占用者）；survival 分母 `t_max - spawn_tick` 自动为最后一代窗口。

## 2. 人工策略重写（stage_b_v25b.py，用户令"保证其仍适用"）

| 策略 | v2.4 行为 | v2b 适配 |
|---|---|---|
| df | 教科书森林：打击优先、observe 0.3、develop/silent；invade 最弱/揭露最强 | 保持画像 + **反咬**：under_attack 时下 tick 侵犯 last_invader（身份来自状态，非新动作）；tick 中被咬只能改无目标动作 → 保持 DEVELOP |
| pacifist | develop 0.5/silent | **躲避**：under_attack 或 self_public → 进黑域（tech≤gate_domain_tech=5 时黑域不可用 → 回退静默）；已入域则静默；永不攻击 |
| random | 均匀随机 | 不变（无信息对照） |
| learner | 冻结自博弈策略 | 不变（它看到的是同一 obs，能否利用新信息=实验问题本身） |

冒烟 `smoke_v25b.py`：df 反咬动作+目标=last_invader ✓；pacifist 被咬/被揭露进黑域、低科技回退静默 ✓；全链 episode ✓。

## 3. 结果（Stage B 混战池 150 回合；生存=当前代口径，v2.5 与 v2.4 不完全可比，见 §7）

**主跑 Stage B（v2.4 旧启发式，四臂批次内一致）**

| 臂 | learner | df | pacifist | random | learner 特征（last-50ep 训练画像） |
|---|---|---|---|---|---|
| a_classic | 0.562 | 0.569 | **0.603** | 0.283 | DEVELOP 0.76 + SILENT 0.12；invade 1.0/100 |
| b_terminal | **0.621** | 0.596 | 0.635 | 0.321 | SILENT 0.43 + DEVELOP 0.36 + BOTHER 0.10 |
| c_res | 0.487 | 0.524 | **0.549** | 0.301 | DEVELOP 0.73 + SILENT 0.08；invade 6.45/100（掠夺流保持） |
| d_malintent | 0.556 | 0.548 | **0.573** | 0.390 | **BOTHER 0.807**（信息战纯化到极端） |

**h2 追加 Stage B（v2b 启发式；shift=v2b−旧启发式）**

| 臂 | learner | df | pacifist | random |
|---|---|---|---|---|
| a_classic | 0.545 (−0.017) | 0.611 (+0.042) | **0.666** (+0.063) | 0.329 (+0.046) |
| b_terminal | 0.590 (−0.031) | 0.632 (+0.036) | **0.672** (+0.037) | 0.356 (+0.035) |
| c_res | 0.487 (0.000) | 0.563 (+0.039) | **0.607** (+0.058) | 0.365 (+0.064) |
| d_malintent | 0.556 (0.000) | 0.588 (+0.040) | **0.625** (+0.052) | 0.369 (−0.021) |

## 4. 训练收敛（600ep 全跑完，50ep 桶）

| 臂 | return last-50 | Δvs 前 50 | TD loss | invade/100 | 收敛判定 |
|---|---|---|---|---|---|
| a_classic | −198 | +79 | 0.078 | 1.00 | 收敛中，仍缓慢改善 |
| b_terminal | −269 | −26 | 0.086 | 1.11 | 基本平稳 |
| c_res | −1278 | **−538** | 0.402 | 6.45 | **未收敛且恶化**：掠夺流在 v2.5 更疯狂但总回报崩塌 |
| d_malintent | **+337** | +239 | 1.171 | 2.58 | **转正且仍在涨**：信息战流在 v2.5 大幅走强（loss 高系大 Q 值正常） |

## 5. 核心发现

1. **pacifist 四臂仍全居首（0.607–0.672，v2b 口径）**——信息完备化没有动摇"隐匿种田流"的版本答案地位；v2.4 结论在 v2.5 环境下稳健。
2. **知情启发式全面受益，learner 无感**：v2b 的 df（反咬）+3.6~4.2pp、pacifist（躲避）+3.7~6.3pp；learner −3.1~0pp——**观测给了新状态不等于策略会用**。learner 在 600ep 自博弈里对 under_attack/self_public 的利用几乎为零（其对入侵的"反应"仍以日常画像为主），这是"信息完备化 ≠ 策略完备化"的直接证据。
3. **v2.5 的"遗忘效应"重塑池生态**：死亡槽回收抹社交记忆 → 集体记忆总量下降。对比 v2.4（无回收、记忆终身积累）：random 0.05→0.28–0.39、df 0.46–0.52→0.52–0.63、learner 0.30–0.48→0.49–0.62 全线上浮。v2.4 的世界惩罚的是"被记住"，v2.5 的世界记忆会随文明死亡一起下葬。
4. **d 臂信息战流在 v2.5 戏剧性放大**：训练后期 BOTHER 占比 0.181（v2.4）→ **0.807**（v2.5），return 转正（+337，v2.4 为负）。机制假说：被揭露者自知后会躲（黑域），但揭露者不暴露、known 关系不可撤销——"借刀杀人"在世界更替更快（回收）+ 目标池更替下反而更高效；且主跑 StageB 里 d 臂 learner 生存 0.556 不再是最低（v2.4 为 0.296）——"恶意有代价"在 v2.5 环境下被削弱。样本单种子，不下强结论。
5. **c 臂掠夺流在 v2.5 未收敛**：invade 6.45/100（a 臂 1.0 的 6 倍）但 return −1278 持续恶化——回收机制带来更快的文明更替=更多可掠夺目标，贪婪加速反而崩塌更狠。
6. **learner 的 b 臂回升**：terminal-only 臂 learner 0.621 居首（旧启发式口径）——v2.4 里"无 shaping 更消极"（SILENT 0.82）的策略在 v2.5 里不再垫底；v2.5 SILENT 0.43+DEVELOP 0.36+BOTHER 0.10 画像更均衡。

## 6. 四臂动作画像（Stage B v2b，learner，占比>0.001）

- a_classic：SILENT 0.818, DEVELOP 0.144, INVADE 0.018, BROADCAST_ALL 0.012
- b_terminal：SILENT 0.895, DEVELOP 0.068, BROADCAST_OTHER 0.014, BROADCAST_ALL 0.011
- c_res：SILENT 0.726, DEVELOP 0.216, BROADCAST_OTHER 0.032, INVADE 0.014
- d_malintent：SILENT 0.691, **BROADCAST_OTHER 0.254**, DEVELOP 0.044

图：`figures/fig_12_v25_action_profiles.{pdf,png}`（四臂画像）、`figures/fig_13_v25_convergence.{pdf,png}`（收敛四面板）。

## 7. 局限与诚实披露

1. **跨版本生存数字口径不同**：v2.5 survival 分母=最后一代出生窗口（回收后 spawn_tick 覆盖），v2.4=全程。§5.3 的版本间对比是方向性证据（所有 kind 同口径内对比），不是严格同口径数字对比。
2. 单种子（seed 42），四臂同种子；不下统计显著结论。
3. df 反咬/pacifist 躲避是**披露规则**（启发式全知使用 last_invader/public 状态），不代表学习者可学到的上限；learner 对新状态零利用可能是探索不足（StageB eps=0.05）而非学不会。
4. 主跑 StageB=旧启发式、h2=v2b：两套池对手行为不同，对比时注意批次。
5. c/d 臂收敛未完成（c 恶化、d 仍在涨），600ep 可能不够这两臂定态。

## 8. 复现

```bash
# 主跑四臂（幂等；per-arm summary + merge）
python3 experiment/run_v25_arms.py
# v2b 启发式 StageB 四臂（幂等）
python3 experiment/run_v25_stageb_h2.py
# 分析+画图
python3 experiment/analysis_v25.py && python3 experiment/plot_v25.py
# 冒烟
python3 experiment/smoke_v25.py && python3 experiment/smoke_v25b.py
```

产物：`runs/v25_arms/`（config/日志/checkpoint）、`results/v25_arms/`（主跑 StageB+summary）、
`results/v25_arms_h2/`（v2b StageB+summary_h2+analysis.json）、`figures/fig_12/13_v25_*`。
