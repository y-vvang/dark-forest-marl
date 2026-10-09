# v2.4 四奖励臂简报（2026-10-09 用户 spec → 2026-10-09 执行）

> 隶属：Does the Dark Forest strategy emerge as a dominant survival policy under encoded axioms?
> 数据均为真实执行（measured）。沙箱：2× Xeon vCPU / 4GB / 无 GPU。驱动：`experiment/run_v24_arms.py`（幂等，2 并行×2 批，总耗时 ~27 min）。

## 1. 环境改动（env2_v24.py，v2.3-fix 权威链零改动）

| # | 改动 | 理由 |
|---|---|---|
| 1 | BROADCAST_SELF → **BROADCAST_ALL** | 原"SELF"读起来像私有动作，实际是向全宇宙广播自己的坐标 |
| 2 | **BROADCAST_OTHER 恢复**：揭露 known 目标坐标给全宇宙（`known[:,j]=True; public[j]=True`），**广播者不暴露** | 删除科技排序打击后唯一的远程"借刀杀人"通道；d 臂信息战工具 |
| 3 | **删 STRIKE_WEAK / STRIKE_STRONG** | 二者按 known（非 observed）的真实科技排序选目标=观测前作弊，违反观测优先原则。obs 编码本就把他人科技/资源锁在 `observed[i,j]` 之后，动作集现在遵守同一原则 |
| 4 | OBSERVE 改**显式选目标**（nearest 兜底，与 STRIKE 同协议） | 用户 spec |
| 5 | **禁科技继承**：端生文明科技原按当代存活者 P25–P75 校准（站在前人肩膀上），现一律 `tech_spawn_lo`=1.0 | 所有文明同一起跑线 |
| 6 | 动作 10→8、obs 38→36；stage_b 分层变量 spawn_tech→**peak_tech（earned tech）** | 无继承后 spawn 分层退化；分层分析改按"挣来的科技" |

配套：`data2_v24.py`（config 工厂+新参数）、`train2_v24.py`、`stage_b_v24.py`、`config_v24.yaml`；`model.py` 的 ReplayBuffer 有效掩码宽度参数化（默认 10，向后兼容 v2.3）。
冒烟：动作/obs 形状、300 tick 随机 rollout、揭露集成（目标下一刻全场 public+known、广播者不暴露）、(c) 臂奖励公式逐项吻合、无继承（spawn tech 全 1.0）——全部通过。

## 2. 奖励臂设计

| 臂 | override（config_v24 之上） | 设计意图 |
|---|---|---|
| a_classic | —（softsign 0.01，bd_entry 0.5） | v2.3-fix 原味基准 |
| b_terminal | softsign_reward_scale=0，bd_entry_penalty=0 | 唯一信号=死亡 -10 |
| c_res | res_reward_scale=0.0025 | 资源纳入奖励：DEVELOP 刻 +0.0025×res（5–20 → 0.0125–0.05/刻），与科技项量级相当、≪死亡 10 |
| d_malintent | c + attack_reward=0.5 | 每次成功打击击杀/入侵击杀/揭露他人 +0.5（=黑域入场罚）。**只奖成功**：被护盾挡下不给；BROADCAST_ALL 自曝不给——培养"聪明恶意"而非自杀倾向 |

## 3. 结果（Stage B：冻结自博弈策略投放到 learner/random/df/pacifist 混战池，150 回合）

| 臂 | learner | df | pacifist | random | learner 特征 |
|---|---|---|---|---|---|
| a_classic | 0.478 | 0.519 | **0.578** | 0.053 | 静默 0.62+种田 0.36；入侵 2.95/ep；loot 15.7 |
| b_terminal | 0.342 | 0.516 | **0.537** | 0.062 | 纯静默 0.82+种田 0.11（无 shaping 更消极） |
| c_res | 0.376 | 0.460 | **0.507** | 0.050 | **掠夺流**：入侵 11.08/ep、loot 59.5（其余臂 3–6 倍），生存不升反降 |
| d_malintent | 0.296 | 0.464 | **0.510** | 0.072 | **信息战流**：揭露他人 0.181（其余臂 ≤0.04）、入侵 4.17/ep、loot 22.8；生存最低 |

各臂 learner 动作画像（占比 >0.001 的项）：
- a_classic：SILENT 0.622, DEVELOP 0.363, BROADCAST_ALL 0.003, BROADCAST_OTHER 0.003, BLACK_DOMAIN 0.007
- b_terminal：SILENT 0.819, DEVELOP 0.108, BROADCAST_ALL 0.007, BROADCAST_OTHER 0.037, BLACK_DOMAIN 0.028
- c_res：SILENT 0.720, DEVELOP 0.258, BLACK_DOMAIN 0.012, INVADE 0.005
- d_malintent：SILENT 0.794, BROADCAST_OTHER 0.181, DEVELOP 0.019, BROADCAST_ALL 0.003, INVADE 0.002

earned-tech（peak）生存分层（四臂一致）：high 0.83–0.92 ≫ mid 0.57–0.63 ≫ low 0.13–0.18。
df 击杀压力：a 14.5/ep → c 9.6 → d 8.2（d 臂揭露是否扰乱了 df 的选择环境未深挖，不下结论）。

## 4. 核心发现

1. **池生态反转**：v2.3-fix 里 learner 0.627 居首；v2.4 里 **pacifist 四臂全居首**（0.51–0.58），learner 掉至第三，random 崩盘（0.51→0.05）。同起跑线+删科技排序打击后，"隐匿种田流"成为版本答案，学习者 600ep 自博弈习得的收割策略在混战中反而招仇恨（入侵引发反杀与揭露）。
2. **learner 全臂零打击**：即使 d 臂有攻击赏金，打击门槛（tech>7）+死亡惩罚威慑下仍不走打击路径——恶意表现为"揭露+入侵"。
3. **(c) 资源奖励→掠夺机器**：入侵频率 ×3.8、缴获 ×3.8，但生存率不升反降——贪婪有代价；资源奖励与入侵缴获形成双重激励。
4. **(d) 恶意文明如约养成**：揭露 0.181+中频入侵的"借刀杀人"画像；生存最低（0.296）——恶意有代价，可作为对抗实验对手池。
5. **挣来的优势依然真实**：同起跑线下 earned-tech 分层梯度陡峭（high≈low 的 5–6 倍），发展优势从"出生特权"变为"努力回报"。

## 5. 局限与诚实披露

- 单种子（42）单策略池布局（learner/random/df/pacifist 轮转 36 槽）；四臂间比较可信，跨版本比较（vs v2.3-fix 0.627）未控环境差异，是机制+参数双重变化的结果。
- df 仍为全知启发式（invade 选 known 中 tech 最低者，disclosed 基线）——其威胁强度在不同臂间一致，不影响臂内对照。
- summary.json 初版因 fork 并行写竞态丢臂，已从四个 stage_b.json 无损重建；驱动已修复（per-arm summary + 主进程合并）。
- Stage B 150 回合下 pacifist/df 差距（如 a 臂 0.578 vs 0.519）未做显著性检验，按描述性结果对待。

## 6. 复现

```bash
cd experiment
python3 run_v24_arms.py          # 幂等：checkpoint/stage_b.json 存在即跳过
# 单臂重跑：删除 runs/v24_arms/<arm>/run 与 results/v24_arms/<arm>/stage_b.json 后再执行
# 冒烟：python3 smoke_v24.py
```

产物：`runs/v24_arms/<arm>/{config.yaml, train.log, stageb.log, run/seed42_main/}`；
`results/v24_arms/<arm>/stage_b.json` + `results/v24_arms/summary.json`；日志 `logs/v24_arms.log`。
