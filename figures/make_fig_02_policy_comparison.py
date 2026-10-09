"""fig_02_policy_comparison — survival & final population across 4 policies (main env).

Data: results/eval_{random,pacifist,df_heuristic,dqn}_main_s*.json (3 policy seeds each).
Panel a: survival_mean. Panel b: final_alive. mean ± std across policy seeds (n=3).
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
    "legend.fontsize": 8, "xtick.labelsize": 9, "ytick.labelsize": 8.5,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "axes.spines.top": False, "axes.spines.right": False,
})
ORDER = [("random", "Random", "#94a3b8"), ("pacifist", "Pacifist", "#16a34a"),
         ("df_heuristic", "DF-heuristic", "#dc2626"), ("dqn", "DQN-trained", "#2563eb")]


def main():
    vals = {"survival_mean": [], "final_alive": []}
    for pol, _, _ in ORDER:
        fps = sorted(glob.glob(os.path.join(RES, f"eval_{pol}_main_s*.json")))
        assert fps, f"no eval files for {pol}"
        v1, v2 = [], []
        for fp in fps:
            r = json.load(open(fp))
            v1.append(r["metrics"]["survival_mean"])
            v2.append(r["metrics"]["final_alive"])
        vals["survival_mean"].append((np.mean(v1), np.std(v1)))
        vals["final_alive"].append((np.mean(v2), np.std(v2)))

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4))
    for ax, key, ylab, title in (
        (axes[0], "survival_mean", "mean survival fraction", "(a) Survival fraction"),
        (axes[1], "final_alive", "civilizations alive at T=250", "(b) Survivors at episode end"),
    ):
        xs = np.arange(len(ORDER))
        m = [vals[key][i][0] for i in range(len(ORDER))]
        s = [vals[key][i][1] for i in range(len(ORDER))]
        bars = ax.bar(xs, m, yerr=s, width=0.62, color=[c for _, _, c in ORDER],
                      capsize=3, error_kw={"lw": 1.0, "ecolor": "#334155"})
        for b, v in zip(bars, m):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8.5, color="#0f172a")
        ax.set_xticks(xs, [lbl for _, lbl, _ in ORDER])
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.set_ylim(0, max(1.05, max(m) + max(s) + 0.12))
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"fig_02_policy_comparison.{ext}"), dpi=300)
    print("fig_02 saved")


if __name__ == "__main__":
    main()
