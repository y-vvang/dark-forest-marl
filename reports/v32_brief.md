# v3.2 简报：超参 sweep + 离线早停终审（2026-10-18）

**一句话结论**：五轴链式 sweep + 门控确认 + 离线早停终审的完整链路给出干净答案——**"最能打"的学习者在当前环境与方法栈内结构性不可达**；tsync 500（目标网络同步加频）是唯一同时改善 vs_arm 与 solo 的轴，但其 sweep 成绩 0.4667 被跨种子终审否决（峰值均值 0.4132 < 门控 0.45）。

## 1. Sweep 设计与执行

- **设计**：链式贪心坐标搜索，19 分支 × 500ep（seed42，热启动自 v3.0 champion s42）：A1 池×lr（9）→ A2 γ（2）→ A3 replay（2）→ A4 eps（2）→ A5 tsync+batch（4）；每轴最优传入下一轴。
- **工程事故（诚实记录）**：首版 16/19 分支无效——超参 override 块写在 `DQNAgent` 构造**之后**，构造期参数（lr/γ/replay/tsync/batch）全部未生效，仅运行时读取的 eps 生效；同 seed 确定性下分数逐位相同=配置未生效的铁证。修复（override 移至构造前）后重扫 16 分支，3 个池轴分支保留复用。此外踩掉：分支名双横线破坏 lr 解析（改从定义结构读回）、pkill -f 自匹配自杀、flock 不防"锁释放但旧 worker 存活"（双驱动并行写坏一个分支，已删重跑）。

## 2. Sweep 结果

- **池轴**（lr=3e-4 固定）：c 池（臂 100% 采样）稳定 0.425–0.431；b 池方差大（0.33–0.44）；d 池（b 加权）全面负贡献。
- **优化轴**：γ 0.995（base）、replay 150k（base）、eps 0.05（base）均为各自最优——三个轴的变体全部不如 base。
- **A5 破局**：tsync 500 分支 **vs_arm 0.4667 + solo 0.615 双健康**——唯一同时改善两端的配置，打破此前"专攻越狠 solo 越崩"的反向权衡（lr 轴上 b_lr6e-4 达 0.4374 但 solo 仅 0.182）。FINAL 配置：pool=b, lr=6e-4, γ=0.995, replay=150k, eps=0.05, **tsync=500**, batch=128。
- **链式缺陷（如实注记）**：driver 缺"base 留任"逻辑——某轴全部变体不如 base 时仍将较优变体传给下一轴，跨轴绝对分数不可横比（轴内结论不受影响）。

## 3. 确认轮与离线早停终审

- **门控触发**：vs_arm 0.4667 > 0.45 → 确认轮（3 seeds × 1000ep，热启动自各自 v3.0 champion ckpt）。
- **首轮固定长度确认暴露长训退化**：s42 训练 loss 0.42→1.03（ep≈300 后加速恶化，翻 2.4 倍），ret/alive 无趋势；confirm league 0.408 < sweep 0.467。
- **改用离线早停终审（方案 B，用户拍板）**：每 250ep 存快照 → 12 快照（3 seeds × ep250/500/750/1000）离线 8ep league eval（与 sweep 同口径）→ 每 seed 取 vs_arm 峰值快照 → 峰值补 solo 16ep 哨兵。
- **终审结果**：12 快照无一破 0.45；峰值均值 **0.4132**；s43 自 ep250 单调衰减（0.380→0.341）、s44 终点跳水（0.406→0.345）、s42 噪声主导——跨种子最优点在 ep250 附近，长训稳定更差。峰值 ckpt solo 均值 0.397 ≈ v3.0 champion 0.406（专攻无生存收益）。
- **判定**：sweep 单 seed 成绩 0.4667 系评估噪声+单种子乐观偏差；门控实质未过；**"最能打"结构性不可达，非优化问题**。

## 4. 完整叙事链（v2.5.1 → v3.2）

pacifist 跨种子稳健居首（v2.5.1）→ solo 自博弈收敛于沉默+发展、先发打击非收敛解（v3.0）→ 专攻 curriculum 仅 +0.018 且 solo 反跌（v3.1）→ 五轴调参+早停终审无一破线（v3.2）。**在这个无缴获打击的世界里，"最能打的学习者"与"最会活的策略"不可兼得——这是环境结构性质。**

## 5. 产物清单

- `runs/v3/v32_sweep/`（19 分支训练+日志）、`results/v3/v32_sweep/`（stage_A1–A5.json、sweep_summary.json）
- `runs/v3/v32_confirm/`（3 seeds×1000ep+每 250ep 快照）、`results/v3/v32_confirm/`（12 快照×7 kinds league eval + 3 solo + confirm_summary.json 含 per_seed_peaks）
- 代码：`experiment/run_v32_driver.py`（sweep+ES confirm 驱动）、`train2_v31_champion.py`（快照钩子版）
- 后续：v3.3 将 champion 训练线独立为大版本（`train3_champion.py`，rounds/快照/override/--hidden 一等公民），容量轴已收官（h192 0.4158 三指标最优，容量无关定案；见 README 版本表 v3.3 行）
