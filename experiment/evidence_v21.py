#!/usr/bin/env python3
"""Evidence run (v2.1, user request 2026-09-09): measure how often civilizations
(A) reach strike-capable conditions (peak tech > gate 7) and (B) die of resource
exhaustion, in two worlds: homogeneous frozen self-play policy and the mixed
strategy pool. Saves per-civilization records + aggregate frequencies."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

import data2 as data_mod
from model import DQNAgent
from train2 import heuristic_action

KIND_PATTERN = ["learner", "random", "df", "pacifist"] * 3


def run_episode(env, agent, rng, eps, kinds):
    obs, valid = env.reset()
    n = env.n
    peak = env.peak_tech.copy()
    spawn_t = env.spawn_tech.copy()
    spawn_tick = env.spawn_tick.copy()
    born = np.zeros(n, dtype=bool)
    lifetime = np.zeros(n, dtype=np.int64)
    cause = [""] * n
    done = False
    while not done:
        actions = np.zeros(n, dtype=np.int64)
        for i in range(n):
            if not env.alive[i]:
                continue
            born[i] = True
            lifetime[i] += 1
            k_i = kinds[i]
            if k_i == "learner":
                actions[i] = agent.act(obs[i:i + 1], valid[i:i + 1], eps)[0]
            else:
                actions[i] = heuristic_action(k_i, obs, valid, env, i, rng, env.t)
        tmask = env.select_targets(actions)
        targets = np.full(n, -1, dtype=np.int64)
        for i in np.where((actions == data_mod.env_mod.A_INVADE) | (actions == data_mod.env_mod.A_BOTHER))[0]:
            cand = np.where(tmask[i])[0]
            if cand.size:
                targets[i] = pick_target(tmask[i], kinds[i], env.tech, actions[i]) if kinds[i] == "df" else int(rng.choice(cand))
        invaded = env.begin_tick(actions, targets)
        re_actions = actions.copy()
        if invaded:
            re_obs, re_valid = env._obs(), env._valid()
            for v in invaded:
                if kinds[v] == "learner":
                    re_actions[v] = agent.act(re_obs[v:v + 1], re_valid[v:v + 1], eps)[0]
                else:
                    re_actions[v] = heuristic_action(kinds[v], re_obs, re_valid, env, v, rng, env.t)
        obs, valid, r, alive, done, info, dead = env.resolve(re_actions)
        peak = np.maximum(peak, env.peak_tech)
        for (v, c) in info["died_this_tick"]:
            cause[v] = c
        # newly born this tick (spawn_tick updated at activation)
        newly = np.where((env.spawn_tick <= env.t) & ~born & env.alive)[0]
        for i in newly:
            born[i] = True
            spawn_t[i] = env.spawn_tech[i]
    return {"kind": kinds, "spawn_tick": spawn_tick.tolist(), "spawn_tech": spawn_t.tolist(),
            "peak_tech": peak.tolist(), "reached_strike_cap": (peak > env.gate_strike_tech).tolist(),
            "born": born.tolist(), "lifetime": lifetime.tolist(), "death_cause": cause,
            "survived": env.alive.tolist()}


def aggregate(records):
    civs = [dict(kind=r["kind"][i], spawn_tick=r["spawn_tick"][i], spawn_tech=r["spawn_tech"][i],
                 peak_tech=r["peak_tech"][i], reached=r["reached_strike_cap"][i],
                 lifetime=int(r["lifetime"][i]), cause=r["death_cause"][i],
                 survived=r["survived"][i])
            for r in records for i in range(len(r["born"])) if r["born"][i]]
    n = len(civs)
    reached = sum(c["reached"] for c in civs)
    deaths = [c for c in civs if c["cause"]]
    by_cause = {}
    for c in deaths:
        by_cause[c["cause"]] = by_cause.get(c["cause"], 0) + 1
    long_lived = [c for c in civs if c["lifetime"] >= 50]
    out = {
        "n_civs": n,
        "reached_strike_cap_count": reached,
        "reached_strike_cap_rate": round(reached / max(1, n), 3),
        "reached_strike_cap_rate_longlived": round(sum(c["reached"] for c in long_lived) / max(1, len(long_lived)), 3)
                                             if long_lived else None,
        "n_deaths": len(deaths),
        "deaths_by_cause": by_cause,
        "starve_share_of_deaths": round(by_cause.get("starve", 0) / max(1, len(deaths)), 3),
        "survivors": sum(c["survived"] for c in civs),
    }
    return out, civs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="../archive_incident_20260922/runs/v21/seed42_main")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--out", default="../archive_incident_20260922/results/v21")
    args = ap.parse_args()
    cfg = data_mod.load_config("config_v21.yaml")
    agent = DQNAgent(obs_dim=data_mod.OBS_DIM, n_actions=data_mod.N_ACTIONS,
                     hidden=[int(h) for h in cfg["model"]["hidden"]],
                     lr=float(cfg["train"]["lr"]), gamma=float(cfg["train"]["gamma"]),
                     batch_size=int(cfg["train"]["batch_size"]), replay_size=int(cfg["train"]["replay_size"]),
                     target_sync=int(cfg["train"]["target_sync"]), seed=1001)
    agent.load(os.path.join(args.run_dir, "checkpoint.npz"))
    rng = np.random.default_rng(777)
    worlds = {"selfplay_frozen": [data_mod.env_mod and "learner"] * 12,
              "mixed_pool": KIND_PATTERN}
    evidence = {}
    for world, kinds in worlds.items():
        env = data_mod.make_env(cfg, seed=10000 + (555 if world.startswith("self") else 556))
        recs = [run_episode(env, agent, rng, 0.05, kinds) for _ in range(args.episodes)]
        agg, civs = aggregate(recs)
        evidence[world] = {"aggregate": agg, "per_civ": civs}
        print(f"[{world}] {agg}")
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "evidence_strike_starve.json"), "w") as f:
        json.dump(evidence, f, indent=2, ensure_ascii=False)
    print(f"-> {args.out}/evidence_strike_starve.json")


if __name__ == "__main__":
    main()
