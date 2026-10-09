"""fig_03_mechanism_ladder — how each competition mechanism shifts the equilibrium.

Consistent protocol across ALL bars: training-time behaviour over the last 100
episodes (epsilon already annealed to 0.10), 3-seed mean ± range.
Ladder: v1.0 (no competition) / v1.1 (finite pool) / v1.2 (uncapped tech +
territorial crowding, unified rerun) / v1.3 (survival decay + free strikes +
black domain) and the three v1.3 ablations.
Panel a: strike rate. Panel b: survival fraction.
"""
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "..")

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9.5,
    "legend.fontsize": 8, "xtick.labelsize": 7.6, "ytick.labelsize": 8.5,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "axes.spines.top": False, "axes.spines.right": False,
})

GROUPS = [
    ("v1.0\nno competition",      f"{BASE}/runs_v10_pilot", "main",              "#94a3b8"),
    ("v1.1\nfinite pool",         f"{BASE}/runs_v11",       "main",              "#14b8a6"),
    ("v1.2\nuncapped+crowding",   f"{BASE}/runs_v12_rerun", "main",              "#0ea5e9"),
    ("v1.3\ntactical domain",     f"{BASE}/runs_v13",       "main",              "#6366f1"),
    ("v1.4\nabsolute domain",     f"{BASE}/runs",           "main",              "#2563eb"),
    ("v1.4\nno explosion",        f"{BASE}/runs",           "no_tech_explosion", "#8b5cf6"),
    ("v1.4\ncostly strike",       f"{BASE}/runs",           "costly_strike",     "#d97706"),
    ("v1.4\nno black domain",     f"{BASE}/runs",           "no_black_domain",   "#f43f5e"),
    ("v1.4\nterminal reward",     f"{BASE}/runs_v14_ter",   "main",              "#16a34a"),
]


def last100(base: str, abl: str) -> list[dict]:
    out = []
    for fp in sorted(glob.glob(os.path.join(base, f"seed*_{abl}", "metrics.jsonl"))):
        recs = [json.loads(l) for l in open(fp)][-100:]
        if len(recs) < 100:
            continue
        out.append({
            "strike": float(np.mean([r["strike_per_100"] for r in recs])),
            "surv": float(np.mean([r["mean_alive_frac"] for r in recs])),
            "silent": float(np.mean([r["action_freqs"]["SILENT"] for r in recs])),
        })
    return out


def bars(ax, key, ylabel, title):
    xs, ms, los, his, cs = [], [], [], [], []
    for i, (lbl, base, abl, c) in enumerate(GROUPS):
        cells = last100(base, abl)
        if not cells:
            continue
        v = [x[key] for x in cells]
        xs.append(i); ms.append(float(np.mean(v)))
        los.append(float(np.mean(v) - np.min(v))); his.append(float(np.max(v) - np.mean(v)))
        cs.append(c)
    b = ax.bar(xs, ms, yerr=[los, his], width=0.62, color=cs, capsize=3,
               error_kw={"lw": 1.0, "ecolor": "#334155"})
    for x, v in zip(xs, ms):
        ax.text(x, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8.3, color="#0f172a")
    ax.set_xticks(xs, [g[0] for g in GROUPS][:len(xs)])
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.7))
    bars(axes[0], "strike", "strikes / 100 agent-ticks", "(a) Strike rate")
    axes[0].axhline(0.0, color="#64748b", lw=0.8)
    bars(axes[1], "surv", "mean survival fraction", "(b) Survival")
    axes[1].set_ylim(0, 1.0)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"fig_03_mechanism_ladder.{ext}"), dpi=300)
    print("fig_03 saved")


if __name__ == "__main__":
    main()
