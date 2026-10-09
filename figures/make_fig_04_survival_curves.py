"""fig_04_survival_curves — mean number of civilizations alive vs tick, 4 policies.

Data: results/eval_{random,pacifist,df_heuristic,dqn}_main_s*.json ("alive_curve",
mean over 400 eval episodes; averaged again across 3 policy seeds).
"""
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9.5,
    "legend.fontsize": 8.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
})
ORDER = [("random", "Random", "#94a3b8", ":"),
         ("pacifist", "Pacifist", "#16a34a", "-."),
         ("df_heuristic", "DF-heuristic", "#dc2626", "--"),
         ("dqn", "DQN-trained", "#2563eb", "-")]


def main():
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for pol, lbl, c, ls in ORDER:
        fps = sorted(glob.glob(os.path.join(RES, f"eval_{pol}_main_s*.json")))
        assert fps, f"no eval files for {pol}"
        curves = np.array([json.load(open(fp))["alive_curve"] for fp in fps])
        m = curves.mean(0)
        s = curves.std(0)
        x = np.arange(len(m))
        ax.plot(x, m, color=c, lw=1.9, ls=ls, label=lbl)
        ax.fill_between(x, m - s, m + s, color=c, alpha=0.13, lw=0)
    ax.set_xlabel("tick")
    ax.set_ylabel("civilizations alive (mean, 12 spawned)")
    ax.set_ylim(0, 12.4)
    ax.set_xlim(0, 250)
    ax.set_title("Extinction dynamics by policy (main environment)")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"fig_04_survival_curves.{ext}"), dpi=300)
    print("fig_04 saved")


if __name__ == "__main__":
    main()
