"""v3.1 real-flagship analysis: v3.1 (curriculum) vs v3.0 (uniform league) head-to-head.
Reads results/{v31_champion,v30_champion}/seed{s}/{solo,league/<kind>}/solo_eval.json."""
import json, os
import numpy as np

RUN = os.path.abspath(os.path.join(".", ".."))
KINDS = ["random", "df", "pacifist", "a_classic", "b_terminal", "c_res", "d_malintent"]
SEEDS = [42, 43, 44]

def load(base, seed, kind):
    p = os.path.join(RUN, "results", base, f"seed{seed}", "league" if kind else "solo",
                     (kind or "") if kind else "", "solo_eval.json")
    p = os.path.join(RUN, "results", base, f"seed{seed}", "league", kind, "solo_eval.json") if kind \
        else os.path.join(RUN, "results", base, f"seed{seed}", "solo", "solo_eval.json")
    return json.load(open(p)) if os.path.exists(p) else None

def agg(vals):
    a = np.array([v for v in vals if v is not None], dtype=float)
    return {"mean": round(float(a.mean()), 4), "std": round(float(a.std(ddof=0)), 4), "n": int(a.size)} if a.size else None

def series(base, key):
    out = {}
    for kind in KINDS:
        vs = [load(base, s, kind) for s in SEEDS]
        out[kind] = agg([d.get(key) if d else None for d in vs])
    return out

def solo_stats(base):
    fs = [load(base, s, None) for s in SEEDS]
    fs = [f for f in fs if f]
    if not fs: return None
    keys = ["survival_fraction_of_lifetime", "strike_per_100_ticks", "blackdomain_per_100_ticks",
            "invade_per_100_ticks", "p_strike_within_10_of_detect"]
    o = {k: agg([f.get(k) for f in fs]) for k in keys}
    acts = {}
    for name in fs[0]["action_freqs"]:
        acts[name] = agg([f["action_freqs"].get(name) for f in fs])
    o["action_freqs"] = acts
    return o

report = {"seeds": SEEDS,
          "league_v31": series("v31_champion", "survival_fraction_of_lifetime"),
          "league_v30": series("v30_champion", "survival_fraction_of_lifetime"),
          "solo_v31": solo_stats("v31_champion"),
          "solo_v30": solo_stats("v30_champion")}
# uplift table
up = {}
for k in KINDS:
    a, b = report["league_v31"].get(k), report["league_v30"].get(k)
    if a and b:
        up[k] = {"v31": a["mean"], "v30": b["mean"], "delta": round(a["mean"] - b["mean"], 4)}
report["uplift"] = up
arm_kinds = ["a_classic", "b_terminal", "c_res", "d_malintent"]
arm_means = [report["league_v31"][k]["mean"] for k in arm_kinds if report["league_v31"].get(k)]
heur_means = [report["league_v31"][k]["mean"] for k in ("random", "df", "pacifist") if report["league_v31"].get(k)]
report["verdict"] = {
    "v31_mean_vs_arm_ckpts": round(float(np.mean(arm_means)), 4) if arm_means else None,
    "v30_mean_vs_arm_ckpts": round(float(np.mean([report["league_v30"][k]["mean"] for k in arm_kinds if report["league_v30"].get(k)])), 4) if any(report["league_v30"].get(k) for k in arm_kinds) else None,
    "all_arm_uplift_positive": bool(arm_means) and all(up[k]["delta"] > 0 for k in arm_kinds if k in up),
    "beats_half_vs_arm_ckpts": bool(arm_means) and all(m > 0.5 for m in arm_means),
}
os.makedirs(os.path.join(RUN, "results", "v3", "v31_champion"), exist_ok=True)
with open(os.path.join(RUN, "results", "v3", "v31_champion", "analysis_v31.json"), "w") as f:
    json.dump(report, f, indent=1)
print("-> analysis_v31.json")
for k in KINDS:
    u = report["uplift"].get(k)
    if u: print(f"  {k:12s} v31 {u['v31']:.3f} vs v30 {u['v30']:.3f}  delta {u['delta']:+.3f}")
print("verdict:", report["verdict"])
