"""fig_01_strategy_emergence — action frequencies & dark-forest signature over training.

Data: runs/seed{42,43,44}_main/metrics.jsonl (3 training seeds, per-episode).
Panel a: action frequency evolution (mean ± per-seed range).
Panel b: P(strike within 10 ticks of FIRST detection) + mean survival fraction.
"""
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "..", "runs")

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9.5,
    "legend.fontsize": 8, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
})
COL = {"SILENT": "#2563eb", "DEVELOP": "#16a34a", "STRIKE": "#dc2626",
       "BROADCAST_SELF": "#d97706", "BROADCAST_OTHER": "#8b5cf6"}
SHOW = ["DEVELOP", "SILENT", "STRIKE", "BROADCAST_SELF", "BROADCAST_OTHER"]


def load_seed(fp):
    recs = [json.loads(l) for l in open(fp) if l.strip()]
    ep = np.array([r["episode"] for r in recs])
    fr = {k: np.array([r["action_freqs"][k] for r in recs]) for k in COL}
    pim = np.array([r["p_strike_after_detect"] if r["p_strike_after_detect"] is not None
                    else np.nan for r in recs])
    alive = np.array([r["mean_alive_frac"] for r in recs])
    return ep, fr, pim, alive


def smooth(x, w=25):
    if len(x) < w:
        return x
    k = np.ones(w) / w
    out = np.convolve(np.nan_to_num(x, nan=np.nanmean(x)), k, mode="same")
    out[: w // 2] = x[: w // 2]; out[-(w // 2):] = x[-(w // 2):]
    return out


def main():
    files = sorted(glob.glob(os.path.join(RUNS, "seed*_main", "metrics.jsonl")))
    assert files, "no main-run metrics found"
    data = [load_seed(f) for f in files]
    ep = data[0][0]

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    ax = axes[0]
    for k in SHOW:
        stack = np.vstack([smooth(d[1][k]) for d in data])
        m = stack.mean(0)
        ax.plot(ep, m, color=COL[k], lw=1.6, label=k.replace("_", " ").title())
        ax.fill_between(ep, stack.min(0), stack.max(0), color=COL[k], alpha=0.14, lw=0)
    ax.set_xlabel("training episode")
    ax.set_ylabel("action frequency")
    ax.set_title("(a) Action frequencies during self-play training (main)")
    ax.set_ylim(-0.02, 1.0)
    ax.legend(ncol=2, frameon=False, loc="upper right")

    ax = axes[1]
    pim = np.vstack([np.nan_to_num(smooth(d[2], 51), nan=np.nan) for d in data])
    alv = np.vstack([smooth(d[3], 51) for d in data])
    ax.plot(ep, np.nanmean(pim, 0), color="#dc2626", lw=1.8,
            label="P(strike ≤10t after 1st detection)")
    ax.fill_between(ep, np.nanmin(pim, 0), np.nanmax(pim, 0), color="#dc2626", alpha=0.14, lw=0)
    ax.plot(ep, alv.mean(0), color="#0f172a", lw=1.8, ls="--", label="mean survival fraction")
    ax.fill_between(ep, alv.min(0), alv.max(0), color="#0f172a", alpha=0.10, lw=0)
    ax.set_xlabel("training episode")
    ax.set_ylabel("probability / fraction")
    ax.set_ylim(0, 1.02)
    ax.set_title("(b) Dark-forest signature vs. survival (main)")
    ax.legend(frameon=False, loc="center right")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"fig_01_strategy_emergence.{ext}"), dpi=300)
    print("fig_01 saved")


if __name__ == "__main__":
    main()
