#!/usr/bin/env python3
"""Frozen-policy evaluation for DarkForestGrid v2.0, with spawn-tech class analysis."""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import data2 as data_mod
from model import DQNAgent
from train2 import heuristic_action, pick_target


def run_episode(env, agent, kind, rng, eps=0.0, agent_kinds=None):
    obs, valid = env.reset()
    n = env.n
    alive0_tech = env.spawn_tech.copy()
    alive_ticks = np.zeros(n, dtype=np.int64)
    act_counts = np.zeros(data_mod.N_ACTIONS, dtype=np.int64)
    n_strikes = n_invasions = n_mutual = n_loot = 0
    first_detect = -np.ones(n, dtype=np.int64)
    det_events = imm_count = 0
    loot_total = 0.0
    done = False
    while not done:
        actions = np.zeros(n, dtype=np.int64)
        agent_kinds = agent_kinds or [kind] * n
        for i in range(n):
            k_i = agent_kinds[i]
            if k_i == "learner" or kind == "dqn":
                actions[i] = agent.act(obs[i:i + 1], valid[i:i + 1], eps)[0]
            else:
                actions[i] = heuristic_action(k_i, obs, valid, env, i, rng, env.t)
        tmask = env.select_targets(actions)
        targets = np.full(n, -1, dtype=np.int64)
        for i in np.where((actions == data_mod.env_mod.A_INVADE) | (actions == data_mod.env_mod.A_BOTHER))[0]:
            cand = np.where(tmask[i])[0]
            if cand.size:
                if agent_kinds[i] in ("df",):
                    targets[i] = pick_target(tmask[i], agent_kinds[i], env.tech, actions[i])
                else:
                    targets[i] = int(rng.choice(cand))
        invaded = env.begin_tick(actions, targets)
        re_actions = actions.copy()
        if invaded:
            re_obs, re_valid = env._obs(), env._valid()
            for v in invaded:
                if agent_kinds[v] == "learner":
                    re_actions[v] = agent.act(re_obs[v:v + 1], re_valid[v:v + 1], eps)[0]
                else:
                    re_actions[v] = heuristic_action(agent_kinds[v], re_obs, re_valid, env, v, rng, env.t)
        res_before = env.res.copy()
        nobs, nvalid, r, alive, done, info, dead_now = env.resolve(re_actions)
        eff_a = re_actions
        act_counts += np.bincount(eff_a, minlength=data_mod.N_ACTIONS)
        alive_ticks += (alive | dead_now)
        n_strikes += len(info["strikes"])
        n_invasions += len(info["invasions"])
        n_mutual += sum(1 for m in info["invasions"] if m[2])
        for (i, _j) in info["detections"]:
            if first_detect[i] < 0:
                first_detect[i] = env.t
                det_events += 1
        for (i, _j, _k) in info["strikes"]:
            if first_detect[i] >= 0 and env.t - first_detect[i] <= 10:
                imm_count += 1
        loot_total += float(np.clip(res_before - env.res, 0, None).sum() * 0)  # loot tracked below
        obs, valid = nobs, nvalid
    return {
        "alive_ticks": alive_ticks, "final_alive": int(alive.sum()),
        "act_counts": act_counts, "n_strikes": n_strikes, "n_invasions": n_invasions,
        "n_mutual": n_mutual, "det_events": det_events, "imm_count": imm_count,
        "spawn_tech": alive0_tech,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, choices=["dqn", "random", "df_heuristic", "pacifist", "mixed"])
    ap.add_argument("--ablation", default="main")
    ap.add_argument("--policy-seed", type=int, default=1001)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--out", default="../archive_incident_20260922/results/v2")
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--epsilon", type=float, default=0.05)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = data_mod.apply_ablation(data_mod.load_config(args.config), args.ablation)
    env = data_mod.make_env(cfg, seed=10000 + args.policy_seed)
    agent = None
    kind = {"df_heuristic": "df"}.get(args.policy, args.policy)
    tc = cfg["train"]
    if args.policy == "dqn":
        agent = DQNAgent(obs_dim=data_mod.OBS_DIM, n_actions=data_mod.N_ACTIONS,
                         hidden=[int(h) for h in cfg["model"]["hidden"]],
                         lr=float(tc["lr"]), gamma=float(tc["gamma"]),
                         batch_size=int(tc["batch_size"]), replay_size=int(tc["replay_size"]),
                         target_sync=int(tc["target_sync"]), seed=args.policy_seed)
        agent.load(os.path.join(args.run_dir, "checkpoint.npz"))
    if args.policy == "mixed":
        agent = DQNAgent(obs_dim=data_mod.OBS_DIM, n_actions=data_mod.N_ACTIONS,
                         hidden=[int(h) for h in cfg["model"]["hidden"]],
                         lr=float(tc["lr"]), gamma=float(tc["gamma"]),
                         batch_size=int(tc["batch_size"]), replay_size=int(tc["replay_size"]),
                         target_sync=int(tc["target_sync"]), seed=args.policy_seed)
        agent.load(os.path.join(args.run_dir, "checkpoint.npz"))

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.policy_seed)
    per_ep, alive_by_class = [], {"low": [], "mid": [], "high": []}
    for ep in range(args.episodes):
        if kind == "mixed":
            eval_kinds = ["random"] * 3 + ["df"] * 3 + ["pacifist"] * 3 + ["learner"] * 3
            r = run_episode(env, agent, "mixed", rng, eps=args.epsilon, agent_kinds=eval_kinds)
        else:
            r = run_episode(env, agent, kind, rng, eps=args.epsilon)
        per_ep.append(r)
        alive_mask_ep = None
        for cls in alive_by_class:
            idx = np.array([i for i in range(env.n) if data_mod.tech_class(r["spawn_tech"][i]) == cls], dtype=np.int64)
            if idx.size == 0:
                continue
            # survivors by class: final alive is global; approximate per-class survival via alive_ticks share
            alive_by_class[cls].append(float(r["alive_ticks"][idx].sum() / (idx.size * env.t_max)))
    n = env.n
    surv = float(np.mean([r["alive_ticks"].sum() / (n * env.t_max) for r in per_ep]))
    metrics = {
        "survival_by_kind": (
            {k: round(float(np.mean([r["alive_ticks"][kk * 3:(kk + 1) * 3].sum() / (3 * env.t_max) for r in per_ep
                                      if r.get("kinds") and r["kinds"][kk * 3] == k])), 3)
             for kk, k in enumerate(["random", "df", "pacifist", "learner"])}
            if per_ep[0].get("kinds") else None),
        "survival_mean": round(surv, 4),
        "final_alive": round(float(np.mean([r["final_alive"] for r in per_ep])), 2),
        "extinction_rate": round(float(np.mean([1.0 if r["final_alive"] == 0 else 0.0 for r in per_ep])), 3),
        "strike_per_100": round(100.0 * sum(r["n_strikes"] for r in per_ep) / max(1, sum(r["alive_ticks"].sum() for r in per_ep)), 3),
        "invade_per_100": round(100.0 * sum(r["n_invasions"] for r in per_ep) / max(1, sum(r["alive_ticks"].sum() for r in per_ep)), 3),
        "mutual_per_100": round(100.0 * sum(r["n_mutual"] for r in per_ep) / max(1, sum(r["alive_ticks"].sum() for r in per_ep)), 3),
        "p_strike_after_detect": round(float(np.mean([r["imm_count"] for r in per_ep])) /
                                        max(1, float(np.mean([r["det_events"] for r in per_ep]))), 3),
        "action_freqs": {data_mod.ACTION_NAMES[k]: round(float(np.mean([r["act_counts"][k] for r in per_ep])
                                                          / max(1, np.mean([r["act_counts"].sum() for r in per_ep]))), 4)
                         for k in range(data_mod.N_ACTIONS)},
        "survival_by_spawn_tech_class": {cls: round(float(np.mean(v)), 3) for cls, v in alive_by_class.items()},
    }
    rec = {
        "policy": args.policy, "ablation": args.ablation, "policy_seed": args.policy_seed,
        "n_episodes": args.episodes, "n_agents": env.n, "t_max": env.t_max,
        "env_config": {k: v for k, v in cfg["env"].items()},
        "metrics": metrics,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out = os.path.join(args.out, f"eval_{args.policy}_{args.ablation}_s{args.policy_seed}.json")
    with open(out, "w") as f:
        json.dump(rec, f, indent=2, ensure_ascii=False)
    m = metrics
    print(f"[{kind}/{args.ablation}/s{args.policy_seed}] surv {m['survival_mean']:.3f} | alive {m['final_alive']:.1f} | "
          f"strike/100 {m['strike_per_100']:.2f} invade/100 {m['invade_per_100']:.2f} | P(imm) {m['p_strike_after_detect']} | "
          f"class {m['survival_by_spawn_tech_class']} -> {out}")


if __name__ == "__main__":
    main()
