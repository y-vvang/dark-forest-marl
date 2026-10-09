# 目录地图（2026-09-23 整理后；2026-10-09 增补 v2.4）

> 权威结论 = **v2.3-fix**（真击杀世界，报告第六部分）；**v2.4（10-09）= 当前最新版本**（观测优先信息原则+同起跑线+四奖励臂，`reports/v24_arms_brief.md`）；v2.0–v2.3 旧产物 = 事故档案（打击结算 bug，只作追溯）。

```
2026-09-04_221544/
├── README.md                  # 总览 + 版本表 + Bug 披露
├── INDEX.md                   # 本文件
├── reports/                   # v24_arms_brief.md（v2.4 四奖励臂简报）
├── experiment/                # 代码根（env2.py 含 09-22 打击修复；tests/ 回归测试）
│   └── *_v24.py / config_v24.yaml / run_v24_arms.py / smoke_v24.py   # ⭐ v2.4 模块（env2_v24 隔离，不动 v2.3 链）
├── runs/                      # 全部训练产物
│   ├── seed42_main/ …         #   v1.x 主线（train.py 引用锚点，勿改名）
│   ├── v10_pilot/ v11/ v12_partial/ v12_rerun/ v13/ v14_ter/   # v1.x 系列
│   ├── v23_fix/               #   ⭐ v2.3-fix 主训练（600ep checkpoint）
│   ├── v23_fix_sweep_arms/    #   ⭐ v2.3-fix sweep 11 短臂
│   ├── v23_fix_term_only/     #   v2.3-fix 仅终局奖励消融臂（10-02，未入报告）
│   └── v24_arms/              #   ⭐ v2.4 四奖励臂 a_classic/b_terminal/c_res/d_malintent
├── results/                   # 分析/评测产物
│   ├── eval_*.json results.json  # v1.x 评测（evaluate.py 锚点，勿改名）
│   ├── v11/ v12/              #   v1.x 系列
│   ├── v23_fix/               #   ⭐ stage_b.json / analysis.json / sweep.json（v2.3 权威）
│   ├── v23_fix_term_only/     #   v2.3 消融臂 stage_b.json
│   └── v24_arms/              #   ⭐ 四臂 stage_b.json + summary.json（v2.4 权威）
├── figures/                   # fig_01–11
├── docs/                      # experiment_report.md / experiment_design.md / data_contract.md
├── scripts/                   # pipeline_v23_fix.sh（幂等）/ launch_final.sh
├── logs/                      # fix 流水线日志 + v24_arms.log + term_only.log
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
# v2.4 冒烟
python3 experiment/smoke_v24.py
```

代码路径引用已同步本次整理（analysis_v23/sweep_v23/train2/stage_b/evidence_v21/evaluate2/pipeline_v23_fix.sh/launch_final.sh，共 11 处引用 + sweep.json arm_dir 11 条）。
