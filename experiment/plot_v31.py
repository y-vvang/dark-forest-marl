"""fig_19: v3.0 vs v3.1 league uplift grouped bars; fig_20: v3.1 training convergence."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN = os.path.abspath(os.path.join(".", ".."))
FIG = os.path.join(RUN, "figures")
KINDS = ["random", "df", "pacifist", "a_classic", "b_terminal", "c_res", "d_malintent"]
LBL = {"random": "random", "df": "DF heur.", "pacifist": "pacifist heur.",
       "a_classic": "ckpt a_classic", "b_terminal": "ckpt b_terminal",
       "c_res": "ckpt c_res", "d_malintent": "ckpt d_malintent"}
a = json.load(open(os.path.join(RUN, "results", "v3", "v31_champion", "analysis_v31.json")))
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})

# ---- fig_19 league uplift ----
fig, ax = plt.subplots(figsize=(8, 4.2))
xs = np.arange(len(KINDS))
v31 = [a["league_v31"][k] for k in KINDS]; v30 = [a["league_v30"][k] for k in KINDS]
ax.bar(xs - 0.2, [x["mean"] if x else 0 for x in v30], 0.38, yerr=[x["std"] if x else 0 for x in v30],
       label="v3.0 champion (uniform league)", color="#94a3b8", capsize=3)
ax.bar(xs + 0.2, [x["mean"] if x else 0 for x in v31], 0.38, yerr=[x["std"] if x else 0 for x in v31],
       label="v3.1 flagship (curriculum, 80% arm oversample)", color="#2563eb", capsize=3)
ax.axhline(0.5, ls="--", c="#dc2626", lw=1, alpha=0.7)
ax.text(len(KINDS) - 0.45, 0.505, "parity", color="#dc2626", fontsize=8)
ax.set_xticks(xs); ax.set_xticklabels([LBL[k] for k in KINDS], rotation=20, ha="right")
ax.set_ylabel("champion 12-slot survival (mean±std, 3 seeds)")
ax.set_title("fig_19  v3.1 vs v3.0 champion: survival vs 7 opponent kinds (12 vs 24, 8ep)")
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_19_v31_league_uplift.png"), dpi=150)
fig.savefig(os.path.join(FIG, "fig_19_v31_league_uplift.pdf")); plt.close(fig)

# ---- fig_20 v3.1 training convergence (3 seeds) ----
fig, ax = plt.subplots(figsize=(8, 4.2))
for s, c in zip((42, 43, 44), ("#2563eb", "#7c3aed", "#059669")):
    p = os.path.join(RUN, "runs", "v3", "v31_champion", "run", f"seed{s}_v31", "metrics.jsonl")
    if not os.path.exists(p): continue
    rec = [json.loads(l) for l in open(p)]
    ep = [r["episode"] for r in rec]; ret = [r["return"] for r in rec]
    k = 50; sm = [sum(ret[max(0, i-k):i+1])/len(ret[max(0, i-k):i+1]) for i in range(len(ret))]
    ax.plot(ep, sm, c=c, alpha=0.85, label=f"seed {s} (ret, MA-50)")
ax.set_xlabel("episode"); ax.set_ylabel("learner mean return (smoothed)")
ax.set_title("fig_20  v3.1 flagship curriculum training (warm-start from v3.0 champion)")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_20_v31_convergence.png"), dpi=150)
fig.savefig(os.path.join(FIG, "fig_20_v31_convergence.pdf")); plt.close(fig)
print("-> fig_19/fig_20")
