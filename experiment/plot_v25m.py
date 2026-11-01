#!/usr/bin/env python3
"""v2.5 multi-seed figures: fig_14 survival (mean+-sd bars + seed dots, both
suites), fig_15 convergence (seed-mean with std band, 2x2). Contract sizes."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["pdf.fonttype"] = 42
HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.dirname(HERE)
FIG = os.path.join(RUN, "figures")
ARMS = ["a_classic", "b_terminal", "c_res", "d_malintent"]
LBL = {"a_classic": "(a) classic", "b_terminal": "(b) terminal-only",
       "c_res": "(c) +resource", "d_malintent": "(d) +attack/broadcast"}
COL = {"a_classic": "#4477aa", "b_terminal": "#ee6677", "c_res": "#228833", "d_malintent": "#ccbb44"}
KINDS = ["learner", "random", "df", "pacifist"]
KCOL = {"learner": "#4477aa", "random": "#aa3377", "df": "#ee6677", "pacifist": "#228833"}

def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig); print("->", name)

def main():
    an = json.load(open(os.path.join(RUN, "results", "v25m_arms", "analysis_multiseed.json")))
    # fig_14: survival by kind, arms x kinds, v2b suite (old as hollow markers)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    w = 0.19
    for ai, arm in enumerate(ARMS):
        v2b = an["survival"][arm].get("v2b", {})
        old = an["survival"][arm].get("old", {})
        for ki, k in enumerate(KINDS):
            if k not in v2b:
                continue
            x = ai + (ki - 1.5) * w
            ax.bar(x, v2b[k]["mean"], w * 0.92, color=KCOL[k], alpha=0.85,
                   yerr=v2b[k]["std"], capsize=2,
                   error_kw={"lw": 0.9}, label=k if ai == 0 else None)
            ax.plot([x], [old[k]["mean"]] if k in old else [], marker="o", ms=3.5,
                    mfc="none", mec=KCOL[k], mew=1.2)
    ax.set_xticks(range(len(ARMS)))
    ax.set_xticklabels([LBL[a] for a in ARMS], fontsize=9)
    ax.set_ylabel("Stage-B survival (1000ep, 3 seeds)")
    ax.set_title("Mixed-pool survival by kind — mean$\\pm$sd of 3 seeds (bars: v2b heuristics; open dots: old)", fontsize=9.5)
    ax.legend(fontsize=8, ncol=4)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig_14_v25m_survival_multiseed")

    # fig_15: convergence 2x2, seed-mean + std band
    fig, axs = plt.subplots(2, 2, figsize=(9, 3.6))
    panels = [("return", "return (100-ep mean)"), ("loss", "TD loss (100-ep mean)"),
              ("invade_per_100", "invades / 100 ticks"), ("alive_frac", "mean alive fraction")]
    for ax, (key, ylab) in zip(axs.flat, panels):
        for arm in ARMS:
            c = an["convergence_mean_curve"].get(arm)
            if not c:
                continue
            xs = [r["ep"] for r in c]
            mu = np.array([r[key]["mean"] for r in c])
            sd = np.array([r[key]["std"] for r in c])
            ax.plot(xs, mu, color=COL[arm], label=LBL[arm], lw=1.4)
            ax.fill_between(xs, mu - sd, mu + sd, color=COL[arm], alpha=0.15, lw=0)
        ax.set_xlabel("episode", fontsize=8); ax.set_ylabel(ylab, fontsize=8)
        ax.tick_params(labelsize=7); ax.spines[["top", "right"]].set_visible(False)
    axs[0, 0].legend(fontsize=7)
    fig.suptitle("Training convergence by reward arm (v2.5, 3 seeds, mean$\\pm$sd)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, "fig_15_v25m_convergence_multiseed")

if __name__ == "__main__":
    main()
