#!/usr/bin/env python3
"""v2.5.1 probe: how often did the (old) 36-slot ceiling actually bind?
Runs greedy rollouts with the MAIN-RUN checkpoints + random rollouts, and
reports max concurrent alive / ever-born / grow-triggers. If max alive < 36 in
every episode, the old recycle-only implementation was tick-equivalent to the
new fully-uncapped one (same seed -> identical rng consumption), and the v2.5
600ep results carry over unchanged."""
import json, os
import numpy as np
import data2_v25 as dm
from env2_v25 import A_SILENT
from model import DQNAgent

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.dirname(HERE)
cfg = dm.load_config()

def rollout(env, agent, rng, eps=0.05, kind_mix=None):
    obs, valid = env.reset()
    n0 = env.n
    max_alive, births0, grows = int(env.alive.sum()), 0, 0
    last_len = env.n
    for t in range(env.t_max):
        act = np.zeros(env.n, dtype=np.int64)
        cand = np.where(valid[0])[0]
        # cheap mixed behavior: random actions (worst-case turnover) or greedy net
        for i in range(env.n):
            cc = np.where(valid[i])[0]
            if cc.size:
                act[i] = int(rng.choice(cc)) if agent is None else agent.act(obs[i:i+1], valid[i:i+1], eps)[0]
        tm = env.select_targets(act)
        tg = np.full(env.n, -1, dtype=np.int64)
        for i in np.where(np.isin(act, (5, 6, 3, 7)))[0]:
            cc = np.where(tm[i])[0]
            if cc.size: tg[i] = int(rng.choice(cc))
        env.begin_tick(act, tg)
        env.resolve(act)
        max_alive = max(max_alive, int(env.alive.sum()))
        if env.n > last_len:
            grows += env.n - last_len; last_len = env.n
        if env.t >= env.t_max - 1:
            break
        obs, valid = env._obs(), env._valid()
    return max_alive, len(env.birth_log), grows

res = {"random": [], "greedy_a": [], "greedy_c": []}
for seed in (0, 1, 2):
    env = dm.make_env(cfg, seed=100 + seed)
    env.reset()
    ma, nb, gr = rollout(env, None, np.random.default_rng(seed))
    res["random"].append({"seed": seed, "max_alive": ma, "births": nb, "grow_slots": gr})
for tag in ("a_classic", "c_res"):
    ck = os.path.join(RUN, "runs", "v25_arms", tag, "run", "seed42_main", "checkpoint.npz")
    agent = DQNAgent(obs_dim=dm.OBS_DIM, n_actions=dm.N_ACTIONS, hidden=[64, 64],
                     lr=6e-4, gamma=0.99, batch_size=256, replay_size=50000,
                     target_sync=500, seed=1)
    agent.load(ck)
    for seed in (0, 1):
        env = dm.make_env(cfg, seed=200 + seed)
        ma, nb, gr = rollout(env, agent, np.random.default_rng(seed))
        res[f"greedy_{tag[0]}"].append({"seed": seed, "max_alive": ma, "births": nb, "grow_slots": gr})
out = os.path.join(RUN, "results", "v25_arms_h2", "probe_cap.json")
json.dump(res, open(out, "w"), indent=2)
worst = max(v["max_alive"] for lst in res.values() for v in lst)
tot_grow = sum(v["grow_slots"] for lst in res.values() for v in lst)
print(json.dumps(res, indent=1))
print(f"WORST_MAX_ALIVE={worst} (cap=36)  TOTAL_GROW_SLOTS={tot_grow}")
print("VERDICT:", "NEVER_BOUND — old results carry over" if worst < 36 else "BOUND — v2.5 results differ under new semantics")
