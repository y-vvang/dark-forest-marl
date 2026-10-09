# 🗄️ 事故档案：打击结算 Bug 污染产物（2026-09-22 判定）

**Bug**：env2.py 打击结算只记 strike_events/kills/death_cause，从不清除受害者 alive 标志（入侵路径有、打击路径漏）→ v2.0–v2.3 所有含打击结果描述的是"打击不致命"的空转世界。

**本目录内容**：runs_v2 / v21 / v22 / v22s / v23（含 sweep_v23_arms）、results_v2 / v21 / v22 / v22s / v23 及各 *_analysis.json / sweep.json、当期流水线脚本 pipeline_v23.sh 与日志。全部**只作追溯，不作为任何结论依据**。

**权威替代**：../runs/v23_fix/、../results/v23_fix/（报告第六部分）。v1.x 结果不受影响（bug 自 v2 起）。
