# 黑暗森林法则的多智能体强化学习验证 · 实验总目录

> ⚠️ **致命 Bug 披露与全线重跑（2026-09-22）**：v2.0–v2.3 存在结算层回归——**打击击杀只在
> 事件流中记录（strike_events/kills），从未清除受害者 `alive` 标志**，被打击的文明实际存活。
> 自 v2 起所有含打击的结果（含 DF 打击率、死因构成、策略池生存率、"护盾耗散非单调"等）
> 均被污染，只可视为"打击空转"世界的结果。已修复（`env2.py`，镜像入侵路径语义；回归测试
> 10/10 通过：击杀置 `alive=False`、守恒残差 0、比值 ≥2 精确挡下）并经后台流水线重跑
> v2.3 全链：`runs/v23_fix/`、`results/v23_fix/`、`results/v23_fix/sweep.json`（11 臂含对照）、
> `logs/pipeline_v23_fix.log`（完成判定见日志 `PIPELINE_FIX_DONE`）。修复前产物保留在原目录
> （`runs_v23/` 等）仅作事故档案。**v3.2（2026-10-18）为最新交付版本**：链式贪心 5 轴 sweep+离线早停终审——**"最能打"结构性不可达（终审峰值均值 0.4132<0.45，sweep 单 seed 0.4667 系乐观偏差）**，与 v2.5.1/v3.0/v3.1 构成完整负结果叙事链（见版本表+`reports/v32_brief.md`）；**v3.3（10-18）champion 训练线独立大版本，容量轴收官：h192 0.4158 三指标最优但天花板 0.41–0.42 不动——容量无关定案**；**v3.1（10-17）** curriculum 专攻仅 +0.018 且 solo 反跌（诚实负结果）；**v2.5.1+v3.3（10-16/17）** 多种子+solo 评估范式（solo 答案：沉默+发展为主流，先发打击非收敛解）；**v2.5（2026-10-15）** 信息完备化+出生数无上限+人工策略 v2b（见版本表）；**v2.4（2026-10-09）** 观测优先信息原则+同起跑线+四奖励臂；
> v2.0–v2.3 历史结论以 **v2.3-fix** 行为准。v2.0–v2.2s 历史臂未重跑
> （工作量原因），如需可按同一修复后环境重测。

> Does the Dark Forest strategy emerge as a dominant survival policy under encoded axioms?
> 全部数据为真实执行（measured），无模拟填充。沙箱：2× Xeon vCPU / 4GB / 无GPU。

## 一句话结论

**黑暗森林不是公理的必然推论，而是世界参数的函数**：生存压力（资源稀缺/技术衰减）点火，攻击安全性（无反制）助燃；当验证成本可支付、反制可执行时，收敛的 RL 策略选择"带牙的和平"。

## 目录索引

| 内容 | 位置 |
|---|---|
| **总报告**（机制阶梯 v1.0–v1.4 + v2.0/v2.1 + 三问解答 + 猜疑链漏洞杂谈） | `experiment_report.md` |
| 实验设计文档（v1 系列） | `experiment_design.md` |
| 数据契约（字段字典） | `data_contract.md` |
| 最终汇总图（四面板） | `figures/fig_09_final_summary.png` |

## 版本演进（机制 → 结论）

| 版本 | 机制变化 | 关键结果 | 数据目录 |
|---|---|---|---|
| v1.0 | 基线（生存压力小） | 和平涌现，打击≈2% | `runs_v10_pilot/` |
| v1.1 | 有限物质池 | 仍和平 | `runs_v11/` |
| v1.2 | tech cap 1000 + 邻居压制 | 仍和平（STRIKE≈2%） | `runs_v12_rerun/` |
| v1.3 | **生存衰减**（存在主义压力） | **先发打击涌现 P(strike\|detect)=0.40** | `runs_v13/` |
| v1.4 | 零打击惩罚 + 黑域（不可逆/绝对免疫）+ 探测即暴露 + 终局分支 | **教科书式森林=自杀**（DF 0.06）；黑域=和平替代品 | `runs/` `runs_v14_ter/` |
| v2.0 | 资源经济+入侵（可反咬互毁）+观察+科技门槛+灭亡惩罚 | **参数反转**：无衰减+免费打击时 DF thriving（0.96） | `runs_v2/` `results_v2/` |
| v2.1 | **间隔生成文明**+批次科技校准+两段式协议 | DQN 收敛修复（0.59→0.81），"带牙的和平" | `runs_v21/` `results_v21/` |
| v2.2 | **公理化资源守恒**（自由池+内生生成）+取消科技上界（Softsign 增长）+科技奖励=Softsign(tech)（scale 1.0）+半倍科技护盾 | 森林点燃（饥饿占死亡 66–78%）；但奖励量级击穿值网络，学习者崩塌至 0.485 | `runs_v22/` `results_v22/` |
| v2.2s | **审计变体**：同形状 scale=0.01（恢复死亡惩罚回报级支配）+目标掩码修复 | 学习者 0.915=全项目最高；森林经济学不变；领跑者护盾优势 | `runs_v22s/` `results_v22s/` |
| v2.3 | **无上限文明**（生成器预滚+定长刻结算）+增长回归线性（奖励负责陡峭→趋平）+广播仅全体+打击三模式选目标 | ⚠️ **已作废（09-22 打击 bug 污染，仅作事故档案）**：学习者 0.665 居首、免死金牌 high 0.951、护盾非单调——全部是"打击空转"世界的伪影 | `runs_v23/` `results_v23/` `sweep_v23_arms/` |
| **v2.3-fix** | 同 v2.3 机制，仅修复打击结算（击杀清除 `alive`，09-22） | **v2.4 之前的权威基准**：学习者 0.627 居首且 600 回合全程零打击；DF 杀 15.7 人零缴获仍 0.418<random 0.514（森林真点燃、先发打击仍净害）；免死金牌破灭（high 0.582≈mid 0.557）；贫瘠臂 DF 反超 random；护盾恢复单调；打击致死 1,987 例实证闭环 | `runs/v23_fix/` `results/v23_fix/` `runs/v23_fix_sweep_arms/` |
| **v2.4**（10-09） | 观测优先信息原则 + 同起跑线：BROADCAST_SELF→**BROADCAST_ALL**（改名不换语义）；**BROADCAST_OTHER 恢复**（揭露目标坐标给全宇宙、广播者不暴露——删科技排序打击后唯一的远程"借刀杀人"通道）；**删 STRIKE_WEAK/STRONG**（按 known 而非 observed 的真实科技排序=观测前作弊）；OBSERVE 改显式选目标（nearest 兜底）；**禁科技继承**（端生科技不再按当代 P25–P75 校准，一律 1.0）；动作 10→8、obs 38→36；stage_b 分层改用 earned-tech（peak） | 四奖励臂（各 600ep+StageB150，同种子 42）：**池生态反转**——pacifist 0.51–0.58 四臂全居首、df 0.46–0.52 次之、learner 0.30–0.48 掉至第三、random 崩至 0.05–0.07；learner 全臂零打击、靠入侵收割（classic invade 2.9/ep loot 15.7；**c_res 资源奖励→掠夺流** invade 11.1/ep loot 59.5 但生存不升）；**d_malintent 攻击赏金→信息战流**（揭露他人占 0.181，invade 4.2/ep，生存最低 0.296——恶意有代价）；earned-tech 分层 high 0.83–0.92 ≫ low 0.13–0.18（挣来的发展优势同起跑线下依然真实） | `runs/v24_arms/` `results/v24_arms/`（详版 `reports/v24_arms_brief.md`） |
| **v2.5**（10-15） | **信息完备化 + 出生数无上限**（当前版本）：**出生数上限取消**——36 槽位不变+死亡槽回收（新生儿社交记忆清零，ever-born uncapped，资源预算 150 唯一刹车）；**"被入侵"第一类环境状态**——obs 加 `under_attack`（滑窗一刻，重选即见"正被入侵"）+ `last_invader` 身份字段，无"反咬"类专用动作；**"被揭露"自知**——obs 加 `self_public` 持久标志；obs 36→38、归一化分母固定 n_norm=36；Stage B 统计改当前代口径；**人工策略 v2b**：df 被咬反咬 last_invader、pacifist 被咬/被揭露进黑域（低科技回退静默）、random 不变 | 四奖励臂（各 600ep+StageB150×新旧启发式双套，同种子 42）：**pacifist 四臂仍全居首**（0.607–0.672 v2b 口径）——"隐匿种田流"版本答案稳健；**知情启发式全面受益**（df 反咬 +3.6~4.2pp、pacifist 躲避 +3.7~6.3pp）而 **learner 无感**（−3.1~0pp）——"信息完备 ≠ 策略完备"；**"遗忘效应"重塑生态**（vs v2.4 无回收记忆终身积累）：random 0.05→0.28–0.39、learner 0.30–0.48→0.49–0.62（b 臂 0.621 居首）全线上浮；**d 臂信息战放大**：训练后期 BOTHER 0.181→0.807、return 转正 +337 仍在涨、StageB 生存不再垫底——"恶意有代价"被削弱（单种子，假说级）；**c 臂掠夺流未收敛**（invade 6.45/100 但 return −1278 持续恶化） | `runs/v25_arms/` `results/v25_arms/`（主跑）、`results/v25_arms_h2/`（v2b 对照+analysis）、`figures/fig_12/13_v25_*`（详版 `reports/v25_brief.md`） |
| **v2.5.1**（10-16/17） | **多种子扩展**：4 臂×3 seeds（42/43/44）×1000ep=12 训练 + 24 StageB 双套 + 聚合（mean±std）+ fig_14/15；出生语义澄清为**完全取消限制**（探针证实与 v2.5 旧实现逐 tick 等价，600ep 数据有效，`results/v25_arms_h2/probe_cap.json`）；工程修复：幂等判定改 metrics 尾行（checkpoint 存在≠完成）、stageb/analysis 补 makedirs、流水线 flock 防重 | **pacifist 3/3 稳健居首**（v2b 0.602–0.640，std 0.008–0.014——"隐匿种田流"升级为跨种子事实）；**d 臂揭露流跨种子复现**（BROADCAST_OTHER 0.712–0.768、return 唯一转正 +47~+254；10-16 晨"种子噪声"判断系查错键名，已撤销）；**c 臂掠夺 3/3 负**（−570~−818，1000ep 仍不收敛）；learner 0.438–0.538 全面低于 pacifist | `runs/v25m_arms/` `results/v25m_arms/` `figures/fig_14/15_v25m_*`（详版 `reports/v30_brief.md` §1） |
| **v3.0**（10-16/17） | **评估范式重构 + 旗舰 champion**（研究意图明确化："所有文明一起从零探索，收敛出何种生存策略"）：①**solo 评估**——成型模型冻结后单跑，同一共识智能指挥全宇宙（36 槽同网络 greedy），测 per-civ 终身生存率+8 动作画像+黑暗森林信号（首触 10t 内打击率/探测→打击延迟/打击目标 tech 偏好），**禁止训练期 metrics 充当评估数据**；②**旗舰训练**（游戏应用）——league 制 3 seeds×1000ep，每 ep 重抽：learner 12 槽（同网络共识）+24 槽对手池均匀采样（random/df/pacifist+四臂冻结 ckpt），transition 只收 learner 槽；旗舰评估=solo+league 矩阵（vs 7 类对手生存对比） | **solo：先发打击不是从零自博弈的收敛解**——四臂 strike/100 仅 0.04–0.19、p_imm≤0.02（d 臂 0.395 例外），主流=沉默 0.73–0.88+发展；**b 臂极端隐匿反伤自身**（黑域 9.24/100 最高、生存 0.384 最低且 std 最小=一致次优吸引子）；c 臂推向掠夺（invade 2.62）但 solo 生存不差（0.500）。**champion：对启发式 random 0.584/pacifist 0.538/df 0.484，对四臂同源模型全败 0.312–0.360——"最能打"未达成（诚实负结果，见简报 §3.3 归因）**；champion solo 0.406 | `results/v30_solo/` `results/v3/v30_champion/` `figures/fig_16/17/18_v30_*`（详版 `reports/v30_brief.md` §2–3） |
| **v3.1**（10-17） | **curriculum 专攻旗舰**（回应 v3.0 champion vs 四臂全败）：热启动自 v3.0 champion s42 ckpt + 对手池臂过采样 80%（12 learner+臂槽 3×8）+ lr 减半 | **仅 +0.018**（vs 臂 0.363 vs v3.0 0.345）且 solo 反跌 0.406→0.323（专攻过拟合信号）——"最能打"未达成（诚实负结果；直接催生 v3.2 超参 sweep） | `runs/v3/v31_champion/` `results/v3/v31_champion/` `figures/fig_19/20_v31_*` |
| **v3.2**（10-18） | **链式贪心坐标搜索**：19 分支×500ep×5 轴（池×lr 9→γ 2→replay 2→eps 2→tsync+batch 4），warm-start 自 v3.0 champion；门控 vs_arm>0.45 触发确认轮；**工程事故与修复**：超参 override 写在构造后致 16/19 分支无效（同 seed 分数逐位相同=配置未生效铁证）→修复重扫；确认轮因长训退化（loss 0.42→1.03）改为**离线早停终审**（3 seeds×1000ep×每 250ep 快照→12 快照离线 8ep league eval→每 seed 峰值 ckpt→solo 哨兵） | **门控过（ts500 分支 0.4667、solo 0.615 双健康——tsync 2000→500 是唯一双赢轴）但 ES 终审否决**：12 快照无一破 0.45、峰值均值 0.4132=sweep 单 seed 乐观偏差证实；早停有效（s43 单调衰减、s44 终点跳水、跨种子最优点在 ep250 附近）但救不回——**"最能打"结构性不可达，非优化问题**（与 v2.5.1 pacifist 居首/v3.0 solo 沉默主流/v3.1 专攻反跌构成完整叙事链） | `runs/v3/v32_sweep/` `runs/v3/v32_confirm/` `results/v3/v32_sweep/` `results/v3/v32_confirm/`（confirm_summary.json 含 per_seed_peaks；详版 `reports/v32_brief.md`） |
| **v3.3**（10-18，已交付） | **champion 训练线独立大版本**：`train3_champion.py`（rounds/快照间隔/超参 override/`--hidden` 容量轴一等公民，override 强制在构造前）；首战**容量轴**：hidden [64,64] 基线（=v3.2 ts500 分支）vs [128,128]/[192,192]，**从头训口径**（hidden 翻倍使热启动 Adam 状态与全层形状失效——跨宽度热启动不兼容，2000ep Adam 广播崩溃实证），其余沿用 v3.2 FINAL（pool b/lr6e-4/γ0.995/replay150k/eps0.05/tsync500/batch128） | 已交付（seed42 三宽度×500ep+8ep league+solo，h64/h128/h192 vs_arm 0.4120/0.3871/0.4158——容量无关；结果见 `results/v3/v33_capacity/capacity_summary.json`） | `runs/v3/v33_capacity/` `results/v3/v33_capacity/` `experiment/train3_champion.py` `experiment/run_v33_capacity.py` |

## 一图看懂全系列

`figures/fig_09_final_summary.png`（v1.0→v2.1 四面板）+ `figures/fig_10_v22_conserved_forest.png`（v2.2 四面板：三臂对比/死因构成/护盾分辨率分箱/出生时点分箱）+ `figures/fig_11_reward_scale_audit.png`（奖励尺度审计四面板）+ `figures/fig_12_v25_action_profiles.png`（v2.5 四臂 Stage B learner 动作画像）+ `figures/fig_13_v25_convergence.png`（v2.5 四臂训练收敛四面板）

> **v2.5.1/v3.0 图契约（fig_14–18，均已产出 10-17）**：`fig_14_v25m_survival_multiseed`（多种子 Stage B 生存分组条+误差棒，新旧启发式双套，old 空心点=v2.5 单种子参照）✅；`fig_15_v25m_convergence_multiseed`（四臂种子均值收敛曲线+std 带）✅；`fig_16_v30_solo_profile`（solo 范式 8 动作画像×4 臂，冻结模型，3 seeds mean±sd）✅；`fig_17_v30_solo_df_signals`（黑暗森林信号 1×4：首触 10t 打击率/strike 与黑域每 100 tick/终身生存）✅；`fig_18_v30_champion_league`（旗舰 vs 7 类对手 league 生存矩阵）✅。

## 环境模型与奖励设计架构（DarkForestGrid v2.2s 基座；当前版本 v2.5 见版本表）

> **v2.3→v2.4 增量注记（2026-10-09）**：12 槽位 → 36 槽位（v2.3，无上限文明+定长刻结算）；
> 打击三模式（WEAK/STRONG 科技排序）→ 仅显式目标+nearest 兜底（v2.4，观测优先）；广播仅全体 →
> BROADCAST_ALL + BROADCAST_OTHER（揭露目标、广播者不暴露）；OBSERVE 最近者 → 显式选目标；
> 端生科技按当代 P25–P75 校准 → 一律 1.0（禁继承）；动作 10→8、obs 38→36；奖励新增
> `res_reward_scale` / `attack_reward`（四奖励臂）。下文其余机制（守恒池/护盾/黑域/门控）v2.4 沿用。
>
> **v2.4→v2.5 增量注记（2026-10-15）**：出生数无上限=死亡槽回收（新生儿社交记忆清零：known/
> observed/public/under_attack/in_domain/first_detect 全抹，spawn_* 与 peak_tech 本代覆盖），
> 资源预算为唯一刹车，终局条件不变；"被入侵"与"被揭露"进入 obs 自我状态块（scal 7→9：
> +`under_attack` 滑窗一刻、+`self_public` 持久），**动作空间不变**（信息完备化不添加专用应对
> 动作——模型只按正常动作决策）；归一化分母固定 n_norm=36；Stage B 生存口径改"当前占用者"
> （出生 tick 归零计数）；`last_invader` 为环境侧身份记录（供启发式/分析，非 obs 字段）。

### 世界与资源守恒（第二条公理）

- 环面网格 100×100；~~12 文明槽位~~（v2.3 起 36 槽位）；回合上限 250
- **总资源恒定**（预算 150）：自然流失、未掠夺遗体、护盾耗散全部回流**自由资源池**；掠夺为转移、不进池
- 逐 tick 不变量：`总量 − Σ存活资源 − 自由池 ≡ 0`（代码内置断言）
- **内生文明生成**：每诞生一个文明即预抽下一个文明的出生资源量（U[5,20]），自由池攒够即诞生并扣款；~~出生科技按当代存活文明 P25–P75 校准~~（v2.4 起禁继承，一律 1.0）

### 文明状态与动作池

| 动作 | 门槛 | 生效 | 机制 |
|---|---|---|---|
| SILENT | 无 | 即时 | 探测截面 ×0.25 |
| DEVELOP | 无 | 即时 | 科技增长（见下）+ 获得 Softsign 科技奖励 |
| BROADCAST_SELF / _OTHER | tech>1 或 资源>10 | 下轮 | 全体/定向暴露坐标 |
| OBSERVE | tech>5 | 即时+暴露下轮 | 精确情报；代价=下轮被随机文明发现 |
| BLACK_DOMAIN | tech>5 | 下轮（风窗期可被打击） | 不可逆绝对庇护：免疫探测/打击/入侵、增长 ×0.2、爆发 ×0.1、入场 −0.5 |
| STRIKE | tech>7 | 即时 | 打击最近已知者，p=clip(0.55+0.12Δtech, 0.15, 0.95)，零成本 |
| INVADE | 无 | 即时 | 保证击杀+掠夺半数资源；目标先发现入侵者并可反咬（互毁）；**不受护盾限制** |

- 探测：p=0.03√tech（静默 ×0.25，上限 0.35），半径 30，环面距离
- 科技：**无上界**；增长 = Softsign 变体 `Δtech = x/(1+0.2|x|)`（渐近 5/tick）；科技爆炸 p=0.01/tick ×1.6–2.2（复合，无封顶）；观测向量 log1p 尺度
- **半倍科技护盾（目标付账）**：目标科技 ≥ 2× 打击者 → 打击被挡，**从目标自身资源耗散 50% 入自由池**；打击者零成本零击杀。护盾挡打击、不挡入侵、不挡饥饿

### 奖励设计架构

| 信号 | 公式 | 量级 | 设计原则 |
|---|---|---|---|
| 科技奖励（DEVELOP） | `scale × tech/(1+0.2·tech)` | v2.2: ≤5/tick；v2.2s: ≤0.05/tick | 奖励随科技 Softsign 饱和：早科技回报陡峭、高科技趋平，永不与死亡惩罚同级 |
| 死亡惩罚 | −10（一次性，真终点） | 10 | 必须**同时**满足逐 tick 支配（单 tick 奖励 < 10）与回报级支配（发展流现值 `≈r̄/(1−γ)` 不淹没 10） |
| 黑域入场 | −0.5 | 0.5 | 轻推；真实成本来自增长 ×0.2 的机会成本 |
| 存活/静默 | 0 | — | 不塑造 |
| 掠夺/打击 | 0（资源转移非奖励） | — | 生存压力经资源通道传导，不经奖励函数 |

> 审计结论（见报告 4.2）：DQN 的有效奖励尺度由折扣流决定——scale=1.0 时发展流现值 ~700–1000、死亡惩罚失去回报级支配、值网络失稳；v2.2s（scale=0.01）恢复支配并取得全项目最高学习者存活。**逐点有界 ≠ 回报有界。**

### 学习架构（两段式协议）

- 参数共享 Double-DQN，MLP 64×64（纯 numpy）；γ=0.995，lr 6e-4，replay 150k，batch 128，target sync 2000，Huber 损失 + 全局范数裁剪
- **目标计算按 next-state 合法动作掩码取 argmax**（审计修复 F1）；死亡=真终点不 bootstrap，存活者时限截断自然 bootstrap
- Stage A：self-play 600 回合（ε 1.0→0.05）→ Stage B：冻结策略投 12 文明混合池（learner/random/DF 启发式/和平主义者 各 3）评测 150 回合
- 复现：`OMP_NUM_THREADS=1 python3 train2.py --seed 42 --mode main --out ../runs_v22s --config config_v22s.yaml && python3 stage_b.py --run-dir ../runs_v22s/seed42_main --episodes 150 --out ../results_v22s`

## 复现指南

```bash
cd experiment/
# v2.2 两段式（约 10 分钟，纯 CPU；守恒不变量内置断言）
OMP_NUM_THREADS=1 python3 train2.py --seed 42 --mode main --out ../runs_v22 --config config_v22.yaml
OMP_NUM_THREADS=1 python3 stage_b.py --run-dir ../runs_v22/seed42_main --episodes 150 --out ../results_v22
# v2.1（同结构：config_v21.yaml；取证 evidence_v21.py --episodes 60）
# v2.0（约 60-75 分钟/组）
python3 train2.py --seed 42 --mode mixed --out ../runs_v2 --config config_v2.yaml
# v1 系列
python3 train.py --config config_v13.yaml ...   # 机制阶梯，见 experiment_design.md
```

## 图表索引

| 图 | 内容 |
|---|---|
| `fig_01` | v1.3/v1.4 策略涌现时间线（P(strike\|detect) 轨迹） |
| `fig_02` | v1.4 策略对比（森林=自杀） |
| `fig_03` | v1 机制阶梯 |
| `fig_04` | v1.4 存活曲线 |
| `fig_05` | v2.0 策略对比（参数反转） |
| `fig_06` | v2.0 出生科技分层 |
| `fig_07` | v2.1 策略池 + 学习者动作画像 |
| `fig_08` | 取证：打击条件可得率 + 死因构成 |
| `fig_09` | 全系列汇总：阶梯 / 策略池 / 死因 / 科技特权 四面板 |
| `fig_10` | v2.2 三臂对比 / 死因构成 / 护盾分辨率×科技比分箱 / 出生时点分箱 |
| `fig_11` | 奖励尺度审计：loss 曲线 / 学习者动作画像对比 / 峰值科技分箱 / 世界间方差 |

## 诚实披露

- v2.0–v2.2 为单种子（沙箱工具轮数限制；v2.2 另有 3 个世界种子量化池间方差）；v1 系列为 3 种子
- v2.2 护盾耗散比例（50%）与 Softsign 系数（a=0.2）为用户指定首版参数，未做敏感性扫描
- v2.2 stage_b 与 evidence 两个运行器对 DF 启发式的实现细节略有差异（0.832 vs 0.60–0.67），跨版本比较统一 stage_b 口径
- v1.0–v1.3 数字为训练期指标（末 100 回合，ε=0.10）；v1.4+ 为冻结策略评估（400/100/150 eps）
- P(imm)（v2.0）为"事件比"语义，可大于 1；v2.1 已改用"打击事件/首次发现"口径并在图中标注
- 混入池 per-kind 明细字段在 v2.0 评估 schema 中缺失，v2.1 已补齐
