#!/usr/bin/env python3
"""v3.0 figures: fig_16 solo action profile by arm; fig_17 dark-forest signals
by arm; fig_18 champion league matrix."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["pdf.fonttype"] = 42
RUN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(RUN, "figures")
ARMS = ["a_classic", "b_terminal", "c_res", "d_malintent"]
LBL = {"a_classic": "(a) classic", "b_terminal": "(b) terminal-only",
       "c_res": "(c) +resource", "d_malintent": "(d) +attack/broadcast"}
COL = {"a_classic": "#4477aa", "b_terminal": "#ee6677", "c_res": "#228833", "d_malintent": "#ccbb44"}
ACTS = ["SILENT", "DEVELOP", "BROADCAST_ALL", "BROADCAST_OTHER", "STRIKE", "BLACK_DOMAIN", "INVADE", "OBSERVE"]

def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig); print("->", name)

def main():
    an = json.load(open(os.path.join(RUN, "results", "v30_solo", "analysis_v30_solo.json")))
    # fig_16: solo action profile (grouped bars, 8 actions x 4 arms, mean over 3 seeds)
    fig, ax = plt.subplots(figsize=(9.5, 3.4))
    w = 0.2
    for ai, arm in enumerate(ARMS):
        if arm not in an["arms"]: continue
        prof = an["arms"][arm]["action_freqs"]
        for ai_, act in enumerate(ACTS):
            x = act if False else ai_ + (ai - 1.5) * w
            m, sd = prof[act]["mean"], prof[act]["std"]
            ax.bar(x, m, w * 0.9, color=COL[arm], alpha=0.9, label=LBL[arm] if ai_ == 0 else None,
                   yerr=sd, capsize=2, error_kw={"lw": 0.8})
    ax.set_xticks(range(len(ACTS))); ax.set_xticklabels(ACTS, fontsize=8, rotation=20)
    ax.set_ylabel("action frequency (solo, frozen)")
    ax.set_title("What the converged policy does when it commands the whole universe (v3.0 solo, 3 seeds mean$\\pm$sd)", fontsize=9.5)
    ax.legend(fontsize=8); ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig_16_v30_solo_profile")

    # fig_17: dark-forest signals (2x2: p_strike_imm, strike/100, bd/100, survival)
    sig = [("p_strike_within_10_of_detect", "P(strike within 10t of detect)"),
           ("strike_per_100_ticks", "strikes / 100 alive-ticks"),
           ("blackdomain_per_100_ticks", "black-domain / 100 alive-ticks"),
           ("survival_fraction_of_lifetime", "civ lifetime survival")]
    fig, axs = plt.subplots(1, 4, figsize=(11, 3.0))
    for ax, (k, lab) in zip(axs.flat, sig):
        for arm in ARMS:
            if arm not in an["arms"]: continue
            d = an["arms"][arm][k]
            ax.errorbar([arm[:2]], [d["mean"]], yerr=[d["std"]], fmt="o", color=COL[arm],
                        label=LBL[arm] if k == sig[0][0] else None, capsize=3, ms=6)
        ax.set_title(lab, fontsize=9); ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    axs[0].legend(fontsize=7)
    fig.suptitle("Dark-forest signature of the consensus policy — v3.0 solo eval (3 seeds)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, "fig_17_v30_solo_df_signals")

    # fig_18: champion league — learner survival vs opponent kind
    lg = an["champion"].get("league", {})
    if lg:
        kinds = list(lg.keys())
        fig, ax = plt.subplots(figsize=(7.5, 3.2))
        xs = np.arange(len(kinds))
        mu = [lg[k]["learner_surv"]["mean"] for k in kinds]
        sd = [lg[k]["learner_surv"]["std"] for k in kinds]
        ax.bar(xs, mu, 0.6, color="#4477aa", alpha=0.85, yerr=sd, capsize=3)
        ax.set_xticks(xs); ax.set_xticklabels(kinds, fontsize=8, rotation=15)
        ax.set_ylabel("champion-civ survival (12 vs 24)")
        ax.set_title("Flagship model vs each opponent family — league eval, 3 seeds mean$\\pm$sd", fontsize=9.5)
        ax.spines[["top", "right"]].set_visible(False)
        save(fig, "fig_18_v30_champion_league")

if __name__ == "__main__":
    main()
