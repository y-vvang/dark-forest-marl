#!/usr/bin/env python3
"""v2.5 figures: fig_12 four-arm StageB learner action profiles (grouped bars),
fig_13 training convergence (2x2: return / loss / invade rate / alive frac).
Contract: vector pdf + png sibling, fonttype 42, 9x3.6 in."""
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
ACTS = ["SILENT", "DEVELOP", "BROADCAST_ALL", "BROADCAST_OTHER", "STRIKE", "BLACK_DOMAIN", "INVADE", "OBSERVE"]

def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("->", name)

def main():
    an = json.load(open(os.path.join(RUN, "results", "v25_arms_h2", "analysis.json")))
    # fig_12: grouped bars, x=action, groups=arms
    fig, ax = plt.subplots(figsize=(9, 3.6))
    w = 0.2
    for gi, tag in enumerate(ARMS):
        prof = an["profile_by_arm"].get(tag)
        if not prof:
            continue
        vals = [prof.get(a, 0.0) for a in ACTS]
        ax.bar(np.arange(len(ACTS)) + (gi - 1.5) * w, vals, w, label=LBL[tag], color=COL[tag])
    ax.set_xticks(np.arange(len(ACTS)))
    ax.set_xticklabels([a.replace("_", "\n") for a in ACTS], fontsize=8)
    ax.set_ylabel("Stage-B action frequency")
    ax.set_title("Learner action profiles by reward arm (v2.5 pool, updated heuristics)", fontsize=10)
    ax.legend(fontsize=8, ncol=4)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig_12_v25_action_profiles")

    # fig_13: convergence 2x2
    fig, axs = plt.subplots(2, 2, figsize=(9, 3.6))
    panels = [("return", "return (50-ep mean)"), ("loss", "TD loss (50-ep mean)"),
              ("invade_per_100", "invades / 100 ticks"), ("alive_frac", "mean alive fraction")]
    for ax, (key, ylab) in zip(axs.flat, panels):
        for tag in ARMS:
            b = an["convergence_by_arm"].get(tag)
            if not b:
                continue
            xs = [(r["ep_range"][0] + r["ep_range"][1]) / 2 for r in b]
            ax.plot(xs, [r[key] for r in b], color=COL[tag], label=LBL[tag], lw=1.4)
        ax.set_xlabel("episode", fontsize=8); ax.set_ylabel(ylab, fontsize=8)
        ax.tick_params(labelsize=7); ax.spines[["top", "right"]].set_visible(False)
    axs[0, 0].legend(fontsize=7)
    fig.suptitle("Training convergence by reward arm (v2.5)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, "fig_13_v25_convergence")

if __name__ == "__main__":
    main()
