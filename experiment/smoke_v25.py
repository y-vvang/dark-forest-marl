#!/usr/bin/env python3
"""v2.5 smoke: v2.4 checks (shapes/masks/reveal/arms) + v2.5 additions
(under_attack self state, self_public state, uncapped births, n_norm scaling)."""
import numpy as np
import data2_v25 as dm
from env2_v25 import A_BALL, A_BOTHER, A_STRIKE, A_OBSERVE, A_DEVELOP, A_SILENT, ACTION_NAMES
from env2_v25 import A_INVADE

cfg = dm.load_config()
env = dm.make_env(cfg, seed=42)
assert dm.OBS_DIM == 38 and dm.N_ACTIONS == 8, (dm.OBS_DIM, dm.N_ACTIONS)
print("shapes OK:", dm.OBS_DIM, dm.N_ACTIONS, ACTION_NAMES)
obs, valid = env.reset()
assert obs.shape == (env.n, 38) and valid.shape == (env.n, 8)
assert env.n == 36 and env.n_norm == 36.0
print("slots:", env.n, "initial cohort:", int(env.alive.sum()), "n_norm:", env.n_norm,
      "(birth count uncapped via dead-slot recycling)")
print("spawn tech:", np.unique(np.round(env.tech[env.alive], 3)), "(no inheritance)")

# --- unit check 1: under_attack fires on the re-choose obs and next tick's obs
env2 = dm.make_env(cfg, seed=42)
env2.reset()
# force a targeted invasion: slot 1 invades slot 0
act = np.full(env2.n, A_SILENT, dtype=np.int64)
act[1] = A_INVADE
env2.known[1, 0] = True
tm = env2.select_targets(act)
assert tm[1, 0], "slot1 must be able to target slot0 (both alive, known)"
tg = np.full(env2.n, -1, dtype=np.int64); tg[1] = 0
invaded = env2.begin_tick(act, tg)
assert 0 in invaded, invaded
re_obs, re_valid = env2._obs(), env._valid()
assert re_obs[0, 7] == 1.0, re_obs[0, 7]      # under_attack scal slot
env2.resolve(act)                                                      # tick T resolution
if env2.alive[0]:
    nxt_obs, _ = env2._obs(), env2._valid()
    assert nxt_obs[0, 7] == 1.0, "next-tick obs must still show under_attack"
    env2.begin_tick(np.full(env2.n, A_SILENT, dtype=np.int64), None)
    env2.resolve(np.full(env2.n, A_SILENT, dtype=np.int64))
    clr_obs, _ = env2._obs(), env2._valid()
    assert clr_obs[0, 7] == 0.0, "flag must clear one tick after"
    print("under_attack sliding window: re-choose=1, next tick=1, cleared after OK")
else:
    print("victim died in the T1 invasion (kill ~0.55); flag semantics for survivors "
          "verified by the re-choose obs above")
assert env2.known[0, 1], "victim must know the invader identity"
print("under_attack: re-choose obs=1 verified; known[0,1]=True OK")

# --- unit check 2: self_public — victim of BROADCAST_OTHER sees its own flag
env3 = dm.make_env(cfg, seed=42)
env3.reset()
env3.known[2, 0] = True
env3.tech[2] = 5.0; env3.res[2] = 20.0   # clear the broadcast gates (tech>=1, res>=10)
act = np.full(env3.n, A_SILENT, dtype=np.int64); act[2] = A_BOTHER
tg = np.full(env3.n, -1, dtype=np.int64); tg[2] = 0
env3.begin_tick(act, tg)
env3.resolve(act)                                   # T1: reveal queued (delayed broadcast)
env3.begin_tick(np.full(env3.n, A_SILENT, dtype=np.int64), None)
env3.resolve(np.full(env3.n, A_SILENT, dtype=np.int64))   # T2 step0: reveal fires
o, _ = env3._obs(), env3._valid()
assert env3.public[0] and o[0, 8] == 1.0, "exposed civ must see self_public=1"
assert o[2, 8] == 0.0, "revealer must NOT be flagged"
print("self_public: exposed slot sees persistent flag; revealer clean OK")

# --- unit check 3: births uncapped — random 300-tick rollout on 256 slots
births = 0
for t in range(300):
    act = np.zeros(env.n, dtype=np.int64)
    for i in range(env.n):
        cand = np.where(valid[i])[0]
        act[i] = int(np.random.default_rng(t * 100 + i).choice(cand)) if cand.size else 0
    tm = env.select_targets(act)
    tg = np.full(env.n, -1, dtype=np.int64)
    for i in np.where(np.isin(act, (A_BOTHER, A_STRIKE, A_INVADE, A_OBSERVE)))[0]:
        c = np.where(tm[i])[0]
        tg[i] = c[0] if c.size else -1
    env.begin_tick(act, tg)
    obs, valid, r, alive, done, info, dead = env.resolve(act)
    assert valid.shape == (env.n, 8), t
    assert obs.shape == (env.n, 38), t
    if done:
        break
births = len(env.birth_log)
print(f"300-tick random rollout OK; births={births} (uncapped), alive_end={int(env.alive.sum())}")
assert births > 36, "recycling must push births beyond the 36-slot count"

# --- unit check 4: dead-slot recycling wipes social state
env4 = dm.make_env(cfg, seed=42)
env4.reset()
env4.known[0, 1] = True; env4.observed[0, 1] = True; env4.public[1] = True
env4.first_detect_tick[0, 1] = 5
env4.tech[1] = 9.0
env4.alive[1] = False; env4.death_cause[1] = "starve"; env4.peak_tech[1] = 9.0
env4.spawn_tick[env4.spawn_tick > 10**8] = 1   # exhaust unborn slots -> force the recycle branch
env4.free_pool = 100.0; env4.pending_spawns = [10.0]
env4._activate_pending_spawns()
assert env4.alive[1] and env4.death_cause[1] == ""
assert not env4.known[0, 1] and not env4.known[1, 0] and not env4.observed[0, 1]
assert not env4.public[1] and env4.first_detect_tick[0, 1] == -1
assert env4.tech[1] == 1.0 and env4.peak_tech[1] == 1.0 and env4.spawn_tech[1] == 1.0
print("recycling: dead slot reborn blank (known/observed/public/peak wiped) OK")

# --- reward arm spot check (carried from v2.4)
base = float(cfg["env"]["death_penalty"])
e = dm.make_env({**cfg, "env": {**cfg["env"], "res_reward_scale": 0.0025, "attack_reward": 0.5}}, seed=7)
o, v = e.reset()
act = np.full(e.n, A_DEVELOP, dtype=np.int64); tg = np.full(e.n, -1, dtype=np.int64)
e.begin_tick(act, tg); _, _, r, _, _, _, _ = e.resolve(act)
exp = e.softsign_reward_scale * e.tech[:4] / (1 + e.softsign_a * e.tech[:4]) + e.res_reward_scale * e.res[:4]
assert np.allclose(r[:4], exp), (r[:4], exp)
print("develop-tick reward arm (c) formula OK; r[0]=", round(float(r[0]), 4))
print("SMOKE_OK")
