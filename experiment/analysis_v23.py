#!/usr/bin/env python3
"""v2.3 analysis layer: per-kind action profiles + binned analyses.
Adapted for v2.3: 10 actions, 36 slots (pattern length follows slot count).
Bins: (1) strike resolution vs defender/striker tech ratio, (2) survival by
birth tick, (3) survival by peak tech."""
import json, sys
from collections import defaultdict
import numpy as np
import data2 as dm
from model import DQNAgent
from train2 import act_all

A_NAMES = ["SILENT", "DEVELOP", "BROADCAST_SELF", "BROADCAST_OTHER(retired)", "STRIKE",
           "BLACK_DOMAIN", "INVADE", "OBSERVE", "STRIKE_WEAK", "STRIKE_STRONG"]

def run(cfg, ckpt, out_prefix, worlds, eps_per_world):
    agent = DQNAgent(obs_dim=dm.OBS_DIM, n_actions=dm.N_ACTIONS, hidden=[64, 64],
                     lr=float(cfg["train"]["lr"]), gamma=float(cfg["train"]["gamma"]),
                     batch_size=128, replay_size=150000, target_sync=2000, seed=1001)
    agent.load(ckpt)
    KIND = (["learner", "random", "df", "pacifist"] * 9)[:36]
    acts_by_kind = {k: np.zeros(dm.N_ACTIONS) for k in set(KIND)}
    ticks_by_kind = {k: 0 for k in set(KIND)}
    strike_events, civs = [], []
    for es in worlds:
        rng = np.random.default_rng(777)
        env = dm.make_env(cfg, seed=es)
        rngs = [np.random.default_rng(9000 + es + i) for i in range(env.n)]
        for _ in range(eps_per_world):
            obs, valid = env.reset()
            spawn_tick = env.spawn_tick.copy(); peak = env.tech.copy()
            alive = env.alive.copy(); cause = [""] * env.n
            counts = np.zeros((env.n, dm.N_ACTIONS))
            done = False
            while not done:
                actions, targets = act_all(agent, KIND, obs, valid, 0.05, rngs, env, env.t)
                env.begin_tick(actions, targets)
                obs, valid, r, alive_now, done, info, dead_now = env.resolve(actions)
                for i in range(env.n):
                    if env.alive[i]:
                        counts[i, actions[i]] += 1
                peak = np.where(env.alive, np.maximum(peak, env.tech), peak)
                for v in np.where(dead_now)[0]:
                    cause[v] = env.death_cause[v]
                strike_events.extend([(es, *ev) for ev in info["strike_events"]])
            for i in range(env.n):
                if env.spawn_tick[i] > 10**8:
                    continue                     # never born this episode
                k = KIND[i]
                acts_by_kind[k] += counts[i]
                ticks_by_kind[k] += 1
                civs.append({"kind": k, "world": es, "spawn_tick": int(spawn_tick[i]),
                             "peak_tech": float(peak[i]), "cause": cause[i],
                             "survived": bool(env.alive[i])})
    profiles = {k: {A_NAMES[a]: round(float(acts_by_kind[k][a] / max(1, ticks_by_kind[k])), 4)
                    for a in range(dm.N_ACTIONS) if acts_by_kind[k][a] > 0}
                for k in sorted(acts_by_kind)}
    # bin 1: strike resolution vs tech ratio (defender/striker)
    strike_bins = defaultdict(lambda: defaultdict(int))
    for (_es, i, j, outcome, ti, tj, _take) in strike_events:
        ratio = tj / max(ti, 1e-9)
        b = ("1-1.5" if ratio < 1.5 else "1.5-2" if ratio < 2 else
             "2-3" if ratio < 3 else "3-5" if ratio < 5 else "5-inf")
        strike_bins[b][outcome] += 1
    strike_bins = {b: dict(v) for b, v in sorted(strike_bins.items())}
    # bin 2: survival by birth tick
    birth_bins = defaultdict(lambda: defaultdict(int))
    for c in civs:
        t = c["spawn_tick"]
        b = ("[0,25)" if t < 25 else "[25,75)" if t < 75 else "[75,150)" if t < 150 else "[150,251)")
        birth_bins[b][c["kind"]] += 1
        if c["survived"]:
            birth_bins[b][f"{c['kind']}_surv"] += 1
    birth_bins = {b: dict(v) for b, v in sorted(birth_bins.items())}
    # bin 3: survival by peak tech
    tech_bins = defaultdict(lambda: {"n": 0, "surv": 0})
    for c in civs:
        p = c["peak_tech"]
        b = ("[0,10)" if p < 10 else "[10,50)" if p < 50 else "[50,200)" if p < 200 else "[200,inf)")
        tech_bins[b]["n"] += 1
        tech_bins[b]["surv"] += int(c["survived"])
    out = {"action_profiles_by_kind": profiles,
           "strike_resolution_by_tech_ratio": strike_bins,
           "survival_by_birth_tick": birth_bins,
           "survival_by_peak_tech": {b: dict(v) for b, v in tech_bins.items()},
           "n_strike_events": len(strike_events), "n_civs": len(civs)}
    json.dump(out, open(f"{out_prefix}_analysis.json", "w"), indent=2)
    print(f"-> {out_prefix}_analysis.json", flush=True)
    for k in ("learner", "random", "df", "pacifist"):
        print(f"  {k:9s}", profiles.get(k, {}), flush=True)
    return out

if __name__ == "__main__":
    B = "/home/z/my-project/output/experiment-suite/does-the-dark-forest-strategy-emerge-as-ecec922e/2026-09-04_221544"
    import os
    os.chdir(f"{B}/experiment")
    cfg = dm.load_config("config_v23.yaml")
    run(cfg, "../runs/v23_fix/seed42_main/checkpoint.npz", "../results/v23_fix",
        [10556, 10557, 10558], 40)
