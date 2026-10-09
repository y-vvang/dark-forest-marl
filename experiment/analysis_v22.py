#!/usr/bin/env python3
"""v2.2s analysis layer: per-kind action profiles + binned analyses.
Bins: (1) strike resolution vs defender/striker tech ratio, (2) survival by
birth tick (endogenous cohorts), (3) survival by peak tech (the tech trap).
Run AFTER stage_b; does not touch training artifacts."""
import json, sys
from collections import defaultdict
import numpy as np
import data2 as dm
from model import DQNAgent
from train2 import act_all

A_NAMES = ["SILENT", "DEVELOP", "BROADCAST_SELF", "BROADCAST_OTHER", "STRIKE", "BLACK_DOMAIN", "INVADE", "OBSERVE"]

def run(cfg, ckpt, out_prefix, worlds, eps_per_world):
    agent = DQNAgent(obs_dim=dm.OBS_DIM, n_actions=dm.N_ACTIONS, hidden=[64, 64],
                     lr=float(cfg["train"]["lr"]), gamma=float(cfg["train"]["gamma"]),
                     batch_size=128, replay_size=150000, target_sync=2000, seed=1001)
    agent.load(ckpt)
    KIND = ["learner", "random", "df", "pacifist"] * 3
    acts_by_kind = {k: np.zeros(dm.N_ACTIONS) for k in KIND}
    ticks_by_kind = {k: 0 for k in KIND}
    strike_events, civs = [], []
    for es in worlds:
        rng = np.random.default_rng(777)
        env = dm.make_env(cfg, seed=es)
        rngs = [np.random.default_rng(9000 + es + i) for i in range(env.n)]
        for _ in range(eps_per_world):
            obs, valid = env.reset()
            spawn_tick = env.spawn_tick.copy(); spawn_tech = env.tech.copy(); spawn_res = env.res.copy()
            peak = env.tech.copy()
            alive = env.alive.copy(); cause = [""] * env.n
            counts = np.zeros((env.n, dm.N_ACTIONS))
            done = False
            while not done:
                actions, targets = act_all(agent, KIND, obs, valid, 0.05, rngs, env, env.t)
                env.begin_tick(actions, targets)
                obs, valid, r, alive_now, done, info, dead_now = env.resolve(actions)
                acts = actions
                for i in range(env.n):
                    if alive_now[i] or env.spawn_tick[i] <= env.t:
                        counts[i, acts[i]] += 1
                peak = np.where(env.alive, np.maximum(peak, env.tech), peak)
                for v in np.where(dead_now)[0]:
                    cause[v] = env.death_cause[v]
                strike_events.extend([(es, *ev) for ev in info["strike_events"]])
            alive_end = env.alive.copy()
            for i in range(env.n):
                k = KIND[i]
                acts_by_kind[k] += counts[i]; ticks_by_kind[k] += counts[i].sum()
                civs.append({"kind": k, "world": es, "spawn_tick": int(spawn_tick[i]),
                             "spawn_tech": float(spawn_tech[i]), "peak_tech": float(peak[i]),
                             "survived": bool(alive_end[i] and env.spawn_tick[i] < 10**8),
                             "cause": cause[i] or ("survived" if alive_end[i] else "unborn")})
    # (1) strike resolution vs tech ratio defender/striker
    bins = [(1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, float("inf"))]
    strike_bins = {f"{lo:g}-{hi:g}": defaultdict(int) for lo, hi in bins}
    for (_es, i, j, outcome, ti, tj, _take) in strike_events:
        ratio = tj / max(ti, 1e-9)
        for lo, hi in bins:
            if lo <= ratio < hi:
                strike_bins[f"{lo:g}-{hi:g}"][outcome] += 1
                break
    strike_bins = {b: {o: int(n) for o, n in d.items()} for b, d in strike_bins.items()}
    # (2) survival by birth tick
    tbins = [(0, 25), (25, 75), (75, 150), (150, 251)]
    birth_bins = {f"[{lo},{hi})": defaultdict(int) for lo, hi in tbins}
    for c in civs:
        for lo, hi in tbins:
            if lo <= c["spawn_tick"] < hi:
                birth_bins[f"[{lo},{hi})"][c["kind"]] += 1
                birth_bins[f"[{lo},{hi})"][f"{c['kind']}_surv"] += int(c["survived"])
    birth_bins = {b: dict(d) for b, d in birth_bins.items()}
    # (3) survival by peak tech
    pbins = [(0, 10), (10, 50), (50, 200), (200, float("inf"))]
    tech_bins = {f"[{lo:g},{hi:g})": defaultdict(int) for lo, hi in pbins}
    for c in civs:
        for lo, hi in pbins:
            if lo <= c["peak_tech"] < hi:
                tech_bins[f"[{lo:g},{hi:g})"]["n"] += 1
                tech_bins[f"[{lo:g},{hi:g})"]["surv"] += int(c["survived"])
    tech_bins = {b: dict(d) for b, d in tech_bins.items()}
    # action profiles
    profiles = {}
    for k in KIND:
        tot = max(1.0, ticks_by_kind[k])
        profiles[k] = {A_NAMES[a]: round(float(acts_by_kind[k][a] / tot), 4) for a in range(dm.N_ACTIONS)}
    out = {"worlds": worlds, "episodes_per_world": eps_per_world,
           "action_profiles_by_kind": profiles,
           "strike_resolution_by_tech_ratio": strike_bins,
           "survival_by_birth_tick": birth_bins,
           "survival_by_peak_tech": tech_bins}
    json.dump(out, open(f"{out_prefix}_analysis.json", "w"), indent=2)
    print(f"-> {out_prefix}_analysis.json")
    for k in ("learner", "random", "df", "pacifist"):
        print(f"  {k:9s}", {a: v for a, v in profiles[k].items() if v >= 0.01})
    print("  strike bins:", json.dumps(strike_bins))
    return out

if __name__ == "__main__":
    arm = sys.argv[1]           # "v22" or "v22s"
    B = "/home/z/my-project/output/experiment-suite/does-the-dark-forest-strategy-emerge-as-ecec922e/2026-09-04_221544"
    cfg = dm.load_config(f"config_{arm}.yaml")
    if arm == "v22s":
        run(cfg, f"../runs_{arm}/seed42_main/checkpoint.npz", f"../results_{arm}", [10556, 10557, 10558], 40)
    else:
        run(cfg, f"../runs_{arm}/seed42_main/checkpoint.npz", f"../results_{arm}", [10556], 40)
