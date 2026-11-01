# 目录地图（2026-09-23 整理后；10-09 增补 v2.4；10-15 增补 v2.5；10-16 增补 v2.5.1/v3.0；10-17 完成态更新；2026-10-31 版本号重排+完成态更新）

> 权威结论 = **v2.3-fix**（真击杀世界，报告第六部分）；v2.4（10-09）= 观测优先+四奖励臂（`reports/v24_arms_brief.md`）；v2.5（10-15）= 信息完备化+出生数无上限（`reports/v25_brief.md`）；**v2.5.1 多种子 + v3.0 solo 范式/旗舰（10-16/17）；v3.1 curriculum（10-17）；**v3.2 sweep+ES 终审 + v3.3 容量轴（10-18）= 当前最新版本（champion 线收官，已交付）**（`reports/v30_brief.md`+`reports/v32_brief.md`，报告第九、十部分；solo 答案：沉默+发展为主流，黑暗森林式先发打击非收敛解）；v2.0–v2.3 旧产物 = 事故档案（打击结算 bug，只作追溯）。

```
2026-09-04_221544/
├── README.md                  # 总览 + 版本表 + Bug 披露
├── INDEX.md                   # 本文件
├── reports/                   # v24_arms_brief.md（v2.4）+ v25_brief.md（v2.5）
├── experiment/                # 代码根（env2.py 含 09-22 打击修复；tests/ 回归测试）
│   ├── *_v24.py / config_v24.yaml / run_v24_arms.py / smoke_v24.py   # ⭐ v2.4 模块（env2_v24 隔离，不动 v2.3 链）
│   ├── *_v25.py / config_v25.yaml / run_v25_arms.py / smoke_v25.py   # ⭐ v2.5 模块（信息完备化，不动 v2.4 链）
│   ├── run_v25_multiseed.py / analysis_v25m.py / plot_v25m.py        # ⭐ v2.5.1 多种子（train_done 幂等=metrics 尾行）
│   ├── pipeline_v25m.sh                                              #   v2.5.1 流水线（flock 锁）
│   └── eval_v30_solo.py / train2_v30_champion.py / run_v30_driver.py # ⭐ v3.0 solo 评估范式 + 旗舰 champion
│       analysis_v30_solo.py / plot_v30m.py / pipeline_v30.sh         #   v3.0 聚合/图/流水线（等 v2.5.1 DONE 接力）
├── runs/                      # 全部训练产物
│   ├── seed42_main/ …         #   v1.x 主线（train.py 引用锚点，勿改名）
│   ├── v10_pilot/ v11/ v12_partial/ v12_rerun/ v13/ v14_ter/   # v1.x 系列
│   ├── v23_fix/               #   ⭐ v2.3-fix 主训练（600ep checkpoint）
│   ├── v23_fix_sweep_arms/    #   ⭐ v2.3-fix sweep 11 短臂
│   ├── v23_fix_term_only/     #   v2.3-fix 仅终局奖励消融臂（10-02，未入报告）
│   ├── v24_arms/              #   ⭐ v2.4 四奖励臂 a_classic/b_terminal/c_res/d_malintent
│   ├── v25_arms/              #   ⭐ v2.5 四奖励臂（信息完备化重跑；v3.0 旗舰对手池来源）
│   ├── v25m_arms/             #   ⭐ v2.5.1 多种子 4 臂×3 seeds×1000ep
│   └── v30_champion/          #   ⭐ v3.0 旗舰 3 seeds×1000ep league（run/seed<s>_champion）
├── results/                   # 分析/评测产物
│   ├── eval_*.json results.json  # v1.x 评测（evaluate.py 锚点，勿改名）
│   ├── v11/ v12/              #   v1.x 系列
│   ├── v23_fix/               #   ⭐ stage_b.json / analysis.json / sweep.json（v2.3 权威）
│   ├── v23_fix_term_only/     #   v2.3 消融臂 stage_b.json
│   ├── v24_arms/              #   ⭐ 四臂 stage_b.json + summary.json（v2.4）
│   ├── v25_arms/              #   ⭐ v2.5 四臂 stage_b.json + summary.json（v2.5 权威）
│   ├── v25_arms_h2/           #   v2.5 v2b 对照 + analysis + probe_cap.json（出生上限探针）
│   ├── v25m_arms/             #   ⭐ v2.5.1 多种子 stageb_old/stageb_v2b + summary/analysis_multiseed.json
│   ├── v30_solo/              #   ⭐ v3.0 四臂 12 ckpt solo 评估 + analysis_v30_solo.json
│   └── v30_champion/          #   ⭐ v3.0 旗舰 solo + league/<kind> 评估
├── figures/                   # fig_01–13（v3.0 增 fig_14–18，见 README 图契约）
├── docs/                      # experiment_report.md / experiment_design.md / data_contract.md
├── scripts/                   # pipeline_v23_fix.sh（幂等）/ launch_final.sh
├── logs/                      # 各版本流水线日志（pipeline_v25m.log / pipeline_v30.log 为当前活跃）
└── archive_incident_20260922/ # 🗄️ 事故档案（见其 README）
```

## 常用命令

```bash
# 回归测试（打击修复行为锁定）
python3 experiment/tests/test_strike_fix.py
# 重跑 fix 流水线（幂等，已全部完成则秒退）
bash scripts/pipeline_v23_fix.sh
# v2.4 四奖励臂（幂等；单臂重跑见 reports/v24_arms_brief.md §6）
python3 experiment/run_v24_arms.py
# v2.5 四奖励臂（幂等；信息完备化重跑）
python3 experiment/run_v25_arms.py
# v2.5.1 多种子（幂等；判定=metrics 尾行 episode>=1000，勿以 checkpoint 存在为准）
python3 experiment/run_v25_multiseed.py --stage train && python3 experiment/run_v25_multiseed.py --stage stageb
# v3.0 solo 评估 / 旗舰训练 / 旗舰评估（幂等）
python3 experiment/run_v30_driver.py --stage solo25m|champion_train|champion_eval
# 全链自动（等 v2.5.1 DONE 后接力，flock 防重）
bash experiment/pipeline_v25m.sh && bash experiment/pipeline_v30.sh
# v2.4/v2.5 冒烟
python3 experiment/smoke_v24.py && python3 experiment/smoke_v25.py
```

代码路径引用已同步本次整理（analysis_v23/sweep_v23/train2/stage_b/evidence_v21/evaluate2/pipeline_v23_fix.sh/launch_final.sh，共 11 处引用 + sweep.json arm_dir 11 条）。

## v3.2 超参 sweep + 离线早停终审（2026-10-18 交付）

- 代码：`experiment/run_v32_driver.py`（链式贪心 5 轴+门控+ES confirm 驱动）、`train2_v31_champion.py`（快照钩子版）
- sweep：`runs/v3/v32_sweep/`（19 分支）+ `results/v3/v32_sweep/`（stage_A1–A5.json、sweep_summary.json）
- 终审：`runs/v3/v32_confirm/`（3 seeds×1000ep×每 250ep 快照）+ `results/v3/v32_confirm/`（12 快照 league eval、confirm_summary.json per_seed_peaks）
- 结论：门控 0.4667 系单 seed 乐观偏差，ES 终审峰值均值 0.4132<0.45——"最能打"结构性不可达；tsync=500 唯一双赢轴
- 详版：`reports/v32_brief.md`

## v3.3 champion 训练线独立大版本（2026-10-18，已交付——容量轴收官：容量无关）

- 代码：`experiment/train3_champion.py`（rounds/快照间隔/override/--hidden 一等公民）+ `run_v33_capacity.py` + `pipeline_v33.sh` + `config_v33.yaml`
- 容量轴首战：hidden [64,64] 基线 vs [128,128]/[192,192]，从头训口径（跨宽度热启动不兼容），其余=v3.2 FINAL
- 产物：`runs/v3/v33_capacity/` + `results/v3/v33_capacity/capacity_summary.json`
- 目录整理：`runs/logs_v1_archive/`（v1 时代散 log 归档）

## v3 归档（2026-10-18 20:43，champion rounds 命名空间）

- **归档范围 [v3.0, v3.3]**：`runs/v3/{v30_champion, v31_champion, v32_sweep, v32_confirm, v33_capacity}` + `results/v3/` 同构——champion 训练线（rounds）统一收编；`results/v30_solo/`（solo 评估产物）不在范围，保留原位
- **代码引用同步**：run_v30_driver / run_v32_driver / run_v33_capacity / train2_v31_champion / train3_champion / analysis_v30_solo / analysis_v31 / plot_v31 / pipeline_v31 全量改 `runs/v3/...`（连写+os.path.join 分段两种写法都扫）
- **回归验证**：run_v33_capacity.py 新路径幂等重跑 EXIT=0（capacity_summary 数字重建一致）；run_v32_driver.py 新路径全幂等跳过+CONFIRM-ES 复用 EXIT=0
- **容量轴终局（measured，seed42 从头训 500ep）**：h64 vs_arm 0.4120 / h128 0.3871 / **h192 0.4158**（vs_heur 0.7794、solo 0.5443 同向最优，h128 谷值）——vs_arm 天花板 0.41–0.42 与热启动 ES 终审峰值 0.4132 跨口径互证：**容量不是"最能打"瓶颈**，v3.2 负结果叙事闭合
- 文档：报告第十部分（docs/experiment_report.md §10，v3.1/v3.2/v3.3 全记）、README 版本表路径列、data_contract 归档规则、reports/v32_brief.md 路径
