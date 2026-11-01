#!/usr/bin/env python3
"""v2.5 analysis: (1) four-arm action profiles (StageB v2b + late-train mix),
(2) training convergence per arm (50-ep buckets from metrics.jsonl),
(3) heuristic-update shift (v2b vs v2.4-heuristic pool survival)."""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.dirname(HERE)
ARMS = ["a_classic", "b_terminal", "c_res", "d_malintent"]

def buckets(rows, k=50):
    out = []
    for i in range(0, len(rows), k):
        ch = rows[i:i + k]
        af = {a: float(np.mean([r["action_freqs"][a] for r in ch])) for a in ch[0]["action_freqs"]}
        out.append({"ep_range": [ch[0]["episode"], ch[-1]["episode"]],
                    "return": float(np.mean([r["return"] for r in ch])),
                    "loss": float(np.mean([r["loss"] for r in ch])),
                    "alive_frac": float(np.mean([r["mean_alive_frac"] for r in ch])),
                    "invade_per_100": float(np.mean([r["invade_per_100"] for r in ch])),
                    "strike_per_100": float(np.mean([r["strike_per_100"] for r in ch])),
                    **af})
    return out

def main():
    res = {"profile_by_arm": {}, "survival_v2b": {}, "survival_old_heuristics": {},
           "heuristic_shift": {}, "convergence_by_arm": {}, "late_train_mix": {}}
    for tag in ARMS:
        p2 = os.path.join(RUN, "results", "v25_arms_h2", tag, "stage_b.json")
        if os.path.exists(p2):
            d = json.load(open(p2))
            res["profile_by_arm"][tag] = d["learner_action_profile"]
            res["survival_v2b"][tag] = {k: v["survival_fraction_of_lifetime"] if isinstance(v, dict) else v
                                         for k, v in d["survival_by_kind"].items()}
        p1 = os.path.join(RUN, "results", "v25_arms", f"summary_{tag}.json")
        if os.path.exists(p1):
            d1 = json.load(open(p1))
            res["survival_old_heuristics"][tag] = d1["survival_by_kind"]
        mp = os.path.join(RUN, "runs", "v25_arms", tag, "run", "seed42_main", "metrics.jsonl")
        if os.path.exists(mp):
            rows = [json.loads(l) for l in open(mp) if l.strip()]
            b = buckets(rows)
            res["convergence_by_arm"][tag] = b
            last, prev = b[-1], b[-2] if len(b) > 1 else b[0]
            res["late_train_mix"][tag] = {a: last[a] for a in
                ("SILENT", "DEVELOP", "BROADCAST_ALL", "BROADCAST_OTHER", "STRIKE", "BLACK_DOMAIN", "INVADE", "OBSERVE")}
            res.setdefault("convergence_summary", {})[tag] = {
                "episodes_done": rows[-1]["episode"],
                "return_last50": round(last["return"], 1),
                "return_delta_last_vs_prev": round(last["return"] - prev["return"], 1),
                "loss_last50": round(last["loss"], 4),
                "invade_per_100_last50": round(last["invade_per_100"], 2),
                "strike_per_100_last50": round(last["strike_per_100"], 2)}
    for tag in ARMS:
        if tag in res["survival_v2b"] and tag in res["survival_old_heuristics"]:
            res["heuristic_shift"][tag] = {k: round(res["survival_v2b"][tag].get(k, 0) -
                                                    res["survival_old_heuristics"][tag].get(k, 0), 3)
                                           for k in ("learner", "random", "df", "pacifist")}
    outp = os.path.join(RUN, "results", "v25_arms_h2", "analysis.json")
    json.dump(res, open(outp, "w"), indent=2, ensure_ascii=False)
    print("-> analysis.json")
    print(json.dumps(res.get("convergence_summary", {}), indent=1))
    print("heuristic_shift:", json.dumps(res["heuristic_shift"]))

if __name__ == "__main__":
    main()
