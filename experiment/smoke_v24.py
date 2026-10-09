#!/usr/bin/env python3
"""v2.4 smoke: shapes, masks, reveal semantics, observe targeting, reward arms."""
import numpy as np
import data2_v24 as dm
from env2_v24 import A_BALL, A_BOTHER, A_STRIKE, A_OBSERVE, A_DEVELOP, A_SILENT, ACTION_NAMES

cfg = dm.load_config()
env = dm.make_env(cfg, seed=42)
assert dm.OBS_DIM == 36 and dm.N_ACTIONS == 8, (dm.OBS_DIM, dm.N_ACTIONS)
print("shapes OK:", dm.OBS_DIM, dm.N_ACTIONS, ACTION_NAMES)
obs, valid = env.reset()
print("spawn tech:", np.unique(np.round(env.tech, 3)), "(no inheritance)")
assert obs.shape == (env.n, 36) and valid.shape == (env.n, 8)
for t in range(300):
    act = np.zeros(env.n, dtype=np.int64)
    for i in range(env.n):
        cand = np.where(valid[i])[0]
        act[i] = int(np.random.default_rng(t * 100 + i).choice(cand)) if cand.size else 0
    tm = env.select_targets(act)
    tg = np.full(env.n, -1, dtype=np.int64)
    for i in np.where(np.isin(act, (A_BOTHER, A_STRIKE, dm.env_mod.A_INVADE, A_OBSERVE)))[0]:
        c = np.where(tm[i])[0]
        tg[i] = c[0] if c.size else -1
    env.begin_tick(act, tg)
    obs, valid, r, alive, done, info, dead = env.resolve(act)
    assert valid.shape == (env.n, 8), t
    if t == 0:
        # BROADCAST_OTHER must reveal target next tick
        bi = None
        for i in range(env.n):
            if act[i] == A_BOTHER:
                bi = i; break
        if bi is not None and tg[bi] >= 0:
            j = int(tg[bi]); print(f"bother: sender {bi} target {j} -> public[{j}]={env.public[j]}")
    if done:
        break
print("300-tick random rollout OK; births:", len(env.birth_log), "alive:", int(env.alive.sum()))

# reward arm spot checks
base = float(cfg["env"]["death_penalty"])
e = dm.make_env({**cfg, "env": {**cfg["env"], "res_reward_scale": 0.0025, "attack_reward": 0.5}}, seed=7)
o, v = e.reset()
act = np.full(e.n, A_DEVELOP, dtype=np.int64); tg = np.full(e.n, -1, dtype=np.int64)
e.begin_tick(act, tg); _, _, r, _, _, _, _ = e.resolve(act)
exp = e.softsign_reward_scale * e.tech[:4] / (1 + e.softsign_a * e.tech[:4]) + e.res_reward_scale * e.res[:4]
assert np.allclose(r[:4], exp), (r[:4], exp)  # slots 0-3: tick-0 cohort (later slots activate at tick end)
print("develop-tick reward arm (c) formula OK; r[0]=", round(float(r[0]), 4))
print("SMOKE_OK")
