# DarkForestGrid v1.3 — 黑暗森林策略涌现实验（可运行代码包）

> **版本指针（2026-10-15 更新）**：本目录代码已演进至 **v2.5**（当前版本，`config_v25.yaml`）：
> v2.4 全套（观测优先信息原则、BROADCAST_ALL/BROADCAST_OTHER、无科技排序打击、OBSERVE 显式
> 选目标、禁科技继承、同起跑线）+ **信息完备化**（"被入侵"成为第一类环境状态——受害者立即
> 发现入侵者且知晓正被入侵，obs `under_attack` 滑窗标志；"自身被 BROADCAST_OTHER 广播"可醒目
> 发现——obs `self_public` 持久标志；不添加"反咬"等专用应对动作）+ **出生数无上限**
> （36 槽位不变，死亡槽回收给新生儿并抹除社交记忆，ever-born uncapped，资源预算为唯一刹车）；
> **动作空间 8、obs 38**（scal 7→9）。**v2.5 入口**：`train2_v25.py` + `stage_b_v25.py` +
> `run_v25_arms.py` + `smoke_v25.py`；v2.4 链（`*_v24.py`/`config_v24.yaml`）保留原状；
> v2.3-fix 链与 v1.x 记录同前。机制说明见上级 `README.md` 版本表与 `reports/v25_brief.md`。

把《三体》黑暗森林法则的两条公理、两个中间机制（猜疑链 / 技术爆炸）**逐条编码**成
12 文明的多智能体环境，用参数共享的 Double-DQN 自博弈训练，观察"生存策略"是否
自发涌现为"隐藏 + 先发打击"。本包经历 v1.0→v1.3 四版机制阶梯（每版的发现与修订
理由见上级目录 `experiment_design.md` §2–§2.3），最终版 v1.3 包含：

- **领土拥挤**：探测半径内每个存活邻居把你的增长速度除以 (1+0.5·邻居数)；
- **生存衰减**（公理2存在主义化）：每 tick tech ×0.975，低于 0.5 崩溃死亡——不扩张就死；
- **免费打击**（用户设计）：strike_penalty=0，对应原著"我们只是唤起宇宙的其他力量进行打击"；
- **黑域**（用户设计，第6动作）：不可被发现/致盲/发展×0.2/爆炸×0.1/禁广播禁打击，
  进入付 0.5 瞬时惩罚，域内选 SILENT 解封；旧知识仍允许记忆打击；衰减域内照常。

## 安装

```bash
pip install -r requirements.txt   # numpy, pyyaml, matplotlib（无 torch 依赖）
```

## 训练（单组）

```bash
python train.py --config config.yaml --seed 42 --ablation main --out ../runs
```

- `--ablation` 可选 `main | no_tech_explosion | costly_strike | no_black_domain`
- 输出：`../runs/seed42_main/{checkpoint.npz, metrics.jsonl, train_log.txt}`

## 评估

```bash
# 训练策略（dqn 需要指定训练目录）
python evaluate.py --policy dqn --ablation main --policy-seed 1001 \
    --run-dir ../runs/seed42_main --out ../results

# 基线（无需 checkpoint）
python evaluate.py --policy random       --ablation main --policy-seed 1001 --out ../results
python evaluate.py --policy df_heuristic --ablation main --policy-seed 1001 --out ../results
python evaluate.py --policy pacifist     --ablation main --policy-seed 1001 --out ../results

# 汇总为 results.json
python evaluate.py --aggregate --config config.yaml --out ../results
```

## 硬件与时长预期（实测）

| 环境 | 单组训练（500 回合） | 全套（12 组训练 + 21 组评估） |
|---|---|---|
| 本沙箱（2 vCPU Xeon / 4GB / 无 GPU） | ≈ 3–4.5 min | ≈ 45–60 min（2 路并行） |
| 普通笔记本（≥2 核） | 同量级 | ≈ 45–70 min |

**不需要 GPU，独立显卡对这个实验没有作用。** 瓶颈是环境步进与小矩阵乘法。

## 诚实声明（scope）

- 教学/探索级玩具模型（12 文明、瞬时打击、单槽位目标选择、黑域为"战术掩体"
  而非原著的"永久安全声明"——理由见 `experiment_design.md` §2.3）。
- 一切结论 statements-in-model，不构成对原著论证或现实星际社会学的证明。
- 版本阶梯数据全部保留：`../runs_v10_pilot/`（v1.0）、`../runs_v11/`（v1.1）、
  `../results_v12/`（v1.2 评估）与 `../runs/`（v1.3 主数据）。
