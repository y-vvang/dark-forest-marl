#!/usr/bin/env python3
"""v2.5 multi-seed aggregation: per-arm survival mean+-std across seeds (both
heuristic suites), seed-averaged convergence curves with std bands, late-train
action mixes. Output: results/v25m_arms/analysis_multiseed.json"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.dirname(HERE)
ARMS = ["a_classic", "b_terminal", "c_res", "d_malintent"]
SEEDS = [42, 43, 44]
KINDS = ["learner", "random", "df", "pacifist"]

def meanstd(vals):
    v = np.array(vals, dtype=float)
    return {"mean": round(float(v.mean()), 4), "std": round(float(v.std(ddof=1)), 4), "n": len(v)} if len(v) > 1 \
        else {"mean": round(float(v.mean()), 4), "std": 0.0, "n": len(v)}

def buckets(rows, k=100):
    out = []
    for i in range(0, len(rows), k):
        ch = rows[i:i + k]
        out.append({"ep": int((ch[0]["episode"] + ch[-1]["episode"]) / 2),
                    "return": float(np.mean([r["return"] for r in ch])),
                    "loss": float(np.mean([r["loss"] for r in ch])),
                    "alive_frac": float(np.mean([r["mean_alive_frac"] for r in ch])),
                    "invade_per_100": float(np.mean([r["invade_per_100"] for r in ch])),
                    "strike_per_100": float(np.mean([r["strike_per_100"] for r in ch])),
                    "SILENT": float(np.mean([r["action_freqs"]["SILENT"] for r in ch])),
                    "BROADCAST_OTHER": float(np.mean([r["action_freqs"]["BROADCAST_OTHER"] for r in ch])),
                    "INVADE": float(np.mean([r["action_freqs"]["INVADE"] for r in ch]))})
    return out

def main():
    res = {"survival": {}, "convergence_mean_curve": {}, "late_train_mix": {}, "n_episodes": {}}
    for arm in ARMS:
        # survival aggregation per suite
        res["survival"][arm] = {}
        for suite in ("old", "v2b"):
            per_kind = {k: [] for k in KINDS}
            for s in SEEDS:
                p = os.path.join(RUN, "results", "v25m_arms", arm, f"seed{s}", f"stageb_{suite}", "stage_b.json")
                if not os.path.exists(p):
                    continue
                j = json.load(open(p))["survival_by_kind"]
                for k in KINDS:
                    per_kind[k].append(j[k]["survival_fraction_of_lifetime"] if isinstance(j[k], dict) else j[k])
            res["survival"][arm][suite] = {k: meanstd(v) for k, v in per_kind.items() if v}
        # convergence aggregation
        curves, mixes, eps_done = [], [], []
        for s in SEEDS:
            mp = os.path.join(RUN, "runs", "v25m_arms", arm, "run", f"seed{s}_main", "metrics.jsonl")
            if not os.path.exists(mp):
                continue
            rows = [json.loads(l) for l in open(mp) if l.strip()]
            curves.append(buckets(rows))
            eps_done.append(rows[-1]["episode"])
            last = rows[-50:]
            mixes.append({a: float(np.mean([r["action_freqs"][a] for r in last]))
                          for a in ("SILENT", "DEVELOP", "BROADCAST_ALL", "BROADCAST_OTHER",
                                    "STRIKE", "BLACK_DOMAIN", "INVADE", "OBSERVE")})
        if curves:
            L = min(len(c) for c in curves)
            keys = [k for k in curves[0][0] if k != "ep"]
            res["convergence_mean_curve"][arm] = [
                {"ep": curves[0][i]["ep"],
                 **{k: {"mean": round(float(np.mean([c[i][k] for c in curves])), 4),
                        "std": round(float(np.std([c[i][k] for c in curves])), 4)} for k in keys}}
                for i in range(L)]
            res["late_train_mix"][arm] = {a: round(float(np.mean([m[a] for m in mixes])), 4) for a in mixes[0]}
            res["n_episodes"][arm] = eps_done
    os.makedirs(os.path.join(RUN, "results", "v25m_arms"), exist_ok=True)
    outp = os.path.join(RUN, "results", "v25m_arms", "analysis_multiseed.json")
    json.dump(res, open(outp, "w"), indent=2, ensure_ascii=False)
    print("-> analysis_multiseed.json")
    for arm in ARMS:
        v2b = res["survival"][arm].get("v2b", {})
        top = max(v2b, key=lambda k: v2b[k]["mean"]) if v2b else "?"
        print(f" {arm:12s} v2b top={top} {v2b.get(top)}  eps={res['n_episodes'].get(arm)}")

if __name__ == "__main__":
    main()
