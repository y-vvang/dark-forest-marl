#!/usr/bin/env python3
"""v3.0 aggregation: solo evals of 4 arms x 3 seeds (dark-forest emergence
profile per reward design) + champion solo/league. Figures fig_16/17/18."""
import json, os
import numpy as np
RUN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARMS = ["a_classic", "b_terminal", "c_res", "d_malintent"]
SEEDS = [42, 43, 44]
ACTS = ["SILENT", "DEVELOP", "BROADCAST_ALL", "BROADCAST_OTHER", "STRIKE", "BLACK_DOMAIN", "INVADE", "OBSERVE"]

def ms(vals):
    v = np.array(vals, float)
    return {"mean": round(float(v.mean()), 4), "std": round(float(v.std(ddof=1)), 4), "n": len(v)}

def load_solo(base, arm, s):
    p = os.path.join(RUN, "results", base, arm, f"seed{s}", "solo_eval.json")
    return json.load(open(p)) if os.path.exists(p) else None

def main():
    res = {"arms": {}, "champion": {"solo": {}, "league": {}}}
    keys = ["survival_fraction_of_lifetime", "strike_per_100_ticks", "invade_per_100_ticks",
            "blackdomain_per_100_ticks", "observe_per_100_ticks", "p_strike_within_10_of_detect",
            "final_alive_per_ep", "ever_born_per_ep", "mean_detect_to_strike"]
    for arm in ARMS:
        per = []
        for s in SEEDS:
            j = load_solo("v30_solo", arm, s)
            if j: per.append(j)
        if not per: continue
        res["arms"][arm] = {k: ms([p[k] for p in per if p.get(k) is not None]) for k in keys}
        res["arms"][arm]["action_freqs"] = {a: ms([p["action_freqs"][a] for p in per]) for a in ACTS}
        res["arms"][arm]["n_seeds"] = len(per)
    # champion solo lives at results/v3/v30_champion/seed<s>/solo/solo_eval.json
    csolo = []
    for s in SEEDS:
        p = os.path.join(RUN, "results", "v3", "v30_champion", f"seed{s}", "solo", "solo_eval.json")
        if os.path.exists(p): csolo.append(json.load(open(p)))
    csolo = [j for j in csolo if j]
    if csolo:
        res["champion"]["solo"] = {k: ms([j[k] for j in csolo if j.get(k) is not None]) for k in keys}
        res["champion"]["solo"]["action_freqs"] = {a: ms([j["action_freqs"][a] for j in csolo]) for a in ACTS}
    kinds = ["random", "df", "pacifist", "a_classic", "b_terminal", "c_res", "d_malintent"]
    for kind in kinds:
        per = []
        for s in SEEDS:
            p = os.path.join(RUN, "results", "v3", "v30_champion", f"seed{s}", "league", kind, "solo_eval.json")
            if os.path.exists(p): per.append(json.load(open(p)))
        if per:
            res["champion"]["league"][kind] = {
                "learner_surv": ms([p["survival_fraction_of_lifetime"] for p in per]),
                "champ_strike_per_100": ms([p["strike_per_100_ticks"] for p in per]),
                "champ_silent_frac": ms([p["action_freqs"]["SILENT"] for p in per])}
    os.makedirs(os.path.join(RUN, "results", "v30_solo"), exist_ok=True)
    json.dump(res, open(os.path.join(RUN, "results", "v30_solo", "analysis_v30_solo.json"), "w"),
              indent=2, ensure_ascii=False)
    print("-> analysis_v30_solo.json")
    for arm in ARMS:
        if arm in res["arms"]:
            a = res["arms"][arm]
            print(f" {arm:12s} surv {a['survival_fraction_of_lifetime']['mean']:.3f} "
                  f"strike {a['strike_per_100_ticks']['mean']:.2f} p_imm {a['p_strike_within_10_of_detect']['mean']}")
    if res["champion"]["solo"]:
        c = res["champion"]["solo"]
        print(f" CHAMPION     surv {c['survival_fraction_of_lifetime']['mean']:.3f} "
              f"strike {c['strike_per_100_ticks']['mean']:.2f}")

if __name__ == "__main__":
    main()
