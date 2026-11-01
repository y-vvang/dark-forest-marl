#!/usr/bin/env python3
"""Stage B (v2.1): deploy the frozen self-play policy into a mixed strategy pool
(vs random / DF-heuristic / pacifist) and produce a strategy analysis.

Outputs results_v21/stage_b.json + figures/fig_07 + fig_08."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

import data2_v25 as data_mod
from env2_v25 import A_BDOMAIN, A_BOTHER, A_INVADE, A_OBSERVE, A_SILENT, A_STRIKE
from model import DQNAgent
from train2_v25 import heuristic_action, pick_target

def build_pattern(n: int) -> list[str]:
    base = ["learner", "random", "df", "pacifist"]
    return (base * (n // 4 + 1))[:n]        # v2.3: pattern length follows slot count


def run_episode(env, agent, rng, eps):
    obs, valid = env.reset()
    n = env.n
    kinds = build_pattern(env.n)
    alive_ticks = np.zeros(n)
    loot = np.zeros(n)
    kills_by = np.zeros(n, dtype=np.int64)
    act_by = np.zeros((n, data_mod.N_ACTIONS), dtype=np.int64)
    first_detect = -np.ones(n, dtype=np.int64)
    n_births_seen = 0
    n_strikes_by_learner = n_invade_by_learner = 0
    strike_after_detect_by_learner = 0
    done = False
    while not done:
        actions = np.zeros(n, dtype=np.int64)
        live_learn = [i for i in range(n) if env.alive[i] and kinds[i] == "learner"]
        if live_learn:                       # v2.3 perf: batched learner forward
            li = np.array(live_learn)
            actions[li] = agent.act(obs[li], valid[li], eps)
        for i in range(n):
            if not env.alive[i] or kinds[i] == "learner":
                continue
            actions[i] = heuristic_action(kinds[i], obs, valid, env, i, rng, env.t)
        tmask = env.select_targets(actions)
        targets = np.full(n, -1, dtype=np.int64)
        for i in np.where(np.isin(actions, (A_INVADE, A_STRIKE, A_BOTHER, A_OBSERVE)))[0]:
            cand = np.where(tmask[i])[0]
            if cand.size:
                targets[i] = (pick_target(tmask[i], kinds[i], env.tech, actions[i])
                              if kinds[i] == "df" else int(rng.choice(cand)))
        invaded = env.begin_tick(actions, targets)
        re_actions = actions.copy()
        if invaded:
            re_obs, re_valid = env._obs(), env._valid()
            for v in invaded:
                if kinds[v] == "learner":
                    re_actions[v] = agent.act(re_obs[v:v + 1], re_valid[v:v + 1], eps)[0]
                else:
                    re_actions[v] = heuristic_action(kinds[v], re_obs, re_valid, env, v, rng, env.t)
        nobs, nvalid, r, alive, done, info, dead_now = env.resolve(re_actions)
        # v2.5: dead-slot recycling means a slot can host several generations per
        # episode. Statistics belong to the CURRENT occupant — reset the lifetime
        # counters of slots (re)born this tick before accumulating.
        for (_bt, bslot, _bres) in env.birth_log[n_births_seen:]:
            alive_ticks[bslot] = 0
            loot[bslot] = 0.0
            kills_by[bslot] = 0
            first_detect[bslot] = -1
        n_births_seen = len(env.birth_log)
        alive_ticks += env.alive
        for i in range(n):
            act_by[i, re_actions[i]] += 1
        for (i, j, killed) in info["strikes"]:
            kills_by[i] += int(killed)
            if kinds[i] == "learner":
                n_strikes_by_learner += 1
                if first_detect[i] >= 0 and env.t - first_detect[i] <= 10:
                    strike_after_detect_by_learner += 1
        for (i, j, mutual) in info["invasions"]:
            if kinds[i] == "learner":
                n_invade_by_learner += 1
        for i in np.where(info.get("loot_to", np.zeros(n)) > 0)[0]:
            loot[i] += info["loot_to"][i]
        for (i, _j) in info["detections"]:
            if first_detect[i] < 0:
                first_detect[i] = env.t
        obs, valid = nobs, nvalid
    return {
        "alive_ticks": alive_ticks.tolist(), "spawn_tech": env.spawn_tech.tolist(),
        "peak_tech": env.peak_tech.tolist(),
        "spawn_tick": env.spawn_tick.tolist(), "kinds": kinds, "loot": loot.tolist(),
        "kills": kills_by.tolist(),
        "learner_action_freqs": {data_mod.ACTION_NAMES[k]: round(float(act_by[np.array(kinds) == "learner", k].sum() /
                                                                         max(1, act_by[np.array(kinds) == "learner"].sum())), 4)
                                 for k in range(data_mod.N_ACTIONS)},
        "learner_strikes": n_strikes_by_learner, "learner_invades": n_invade_by_learner,
        "learner_strike_after_detect": strike_after_detect_by_learner,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--episodes", type=int, default=150)
    ap.add_argument("--out", default="../archive_incident_20260922/results/v21")
    ap.add_argument("--config", default="config_v21.yaml")
    args = ap.parse_args()
    cfg = data_mod.load_config(args.config)
    env = data_mod.make_env(cfg, seed=21001)
    tc = cfg["train"]
    agent = DQNAgent(obs_dim=data_mod.OBS_DIM, n_actions=data_mod.N_ACTIONS,
                     hidden=[int(h) for h in cfg["model"]["hidden"]], lr=float(tc["lr"]),
                     gamma=float(tc["gamma"]), batch_size=int(tc["batch_size"]),
                     replay_size=int(tc["replay_size"]), target_sync=int(tc["target_sync"]), seed=1001)
    agent.load(os.path.join(args.run_dir, "checkpoint.npz"))
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(21001)
    per_ep = [run_episode(env, agent, rng, eps=0.05) for _ in range(args.episodes)]

    kinds = build_pattern(env.n)
    by_kind = {}
    for k in ["learner", "random", "df", "pacifist"]:
        idx = np.array([i for i, kk in enumerate(kinds) if kk == k])
        surv = float(np.mean([np.mean([r["alive_ticks"][i] / max(1, env.t_max - r["spawn_tick"][i]) for i in idx]) for r in per_ep]))
        by_kind[k] = {"survival_fraction_of_lifetime": round(surv, 3),
                      "loot_mean": round(float(np.mean([np.sum([r["loot"][i] for i in idx]) for r in per_ep])), 2),
                      "kills_mean": round(float(np.mean([np.sum([r["kills"][i] for i in idx]) for r in per_ep])), 2)}
    laf = {k: round(float(np.mean([r["learner_action_freqs"][k] for r in per_ep])), 3)
           for k in per_ep[0]["learner_action_freqs"]}
    # v2.4: no tech inheritance → spawn classes are degenerate (all 1.0).
    # Stratify by EARNED tech instead (peak tech reached while alive).
    class_surv = {}
    for cls in ["low", "mid", "high"]:
        vals = []
        for r in per_ep:
            idx = [i for i in range(env.n) if data_mod.tech_class(r["peak_tech"][i]) == cls]
            if idx:
                vals.append(float(np.mean([r["alive_ticks"][i] / max(1, env.t_max - r["spawn_tick"][i]) for i in idx])))
        if vals:
            class_surv[cls] = round(float(np.mean(vals)), 3)
    out = {
        "stage": "B — frozen self-play policy deployed into mixed strategy pool",
        "n_episodes": args.episodes, "kinds_pattern": build_pattern(36),
        "survival_by_kind": by_kind,
        "learner_action_profile": laf,
        "learner_strikes_per_episode": round(float(np.mean([r["learner_strikes"] for r in per_ep])), 2),
        "learner_invades_per_episode": round(float(np.mean([r["learner_invades"] for r in per_ep])), 2),
        "learner_strike_after_detect_rate": (round(float(np.mean([r["learner_strike_after_detect"] for r in per_ep]) /
                                                   max(1, np.mean([r["learner_strikes"] for r in per_ep])))), ),
        "survival_by_earned_tech_class": class_surv,
        "per_episode": per_ep,
    }
    with open(os.path.join(args.out, "stage_b.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("[stage B] survival by kind:", {k: v["survival_fraction_of_lifetime"] for k, v in by_kind.items()})
    print("[stage B] learner action profile:", laf)
    print("[stage B] class survival:", class_surv)
    print(f"-> {args.out}/stage_b.json")


if __name__ == "__main__":
    main()
