#!/usr/bin/env python3
"""v2.5b heuristic smoke: df counter-bite targets the last invader; pacifist
hides (BLACK_DOMAIN) under under_attack / self_public; random unchanged."""
import numpy as np
import data2_v25 as dm
from env2_v25 import A_BDOMAIN, A_DEVELOP, A_INVADE, A_SILENT
from stage_b_v25b import heuristic_action_v2, pick_target_v2

cfg = dm.load_config()
env = dm.make_env(cfg, seed=42)
obs, valid = env.reset()
rng = np.random.default_rng(0)

# unit 1: df counter-bite — slot1 invades slot0 (df), then slot0 must invade back
env.known[1, 0] = True
act = np.full(env.n, A_SILENT, dtype=np.int64); act[1] = A_INVADE
tg = np.full(env.n, -1, dtype=np.int64); tg[1] = 0
invaded = env.begin_tick(act, tg)
assert 0 in invaded
obs, valid = env._obs(), env._valid()   # re-choose sees the post-notice world
a0 = heuristic_action_v2("df", obs, valid, env, 0, rng, env.t)
assert a0 == A_INVADE, a0
env.resolve(act)
env.known[0, 1] = True  # victim knows the invader (set by the notice anyway)
t0 = pick_target_v2(np.ones(env.n, dtype=bool), "df", env.tech, A_INVADE, 0, env)
assert t0 == 1, t0
print("df counter-bite: action=INVADE, target=last_invader OK")

# unit 2: pacifist hides on under_attack and self_public
env2 = dm.make_env(cfg, seed=7); env2.reset()
env2.tech[3] = 8.0; env2.tech[4] = 8.0   # clear the domain tech gate for the victims
o2, v2 = env2._obs(), env2._valid()
env2.under_attack[3] = True
a = heuristic_action_v2("pacifist", o2, v2, env2, 3, rng, 5)
assert a == A_BDOMAIN, a
env2.under_attack[3] = False; env2.public[3] = True
a = heuristic_action_v2("pacifist", o2, v2, env2, 3, rng, 5)
assert a == A_BDOMAIN, a
env2.under_attack[4] = True   # low-tech victim: domain gate shut -> falls back to SILENT
a = heuristic_action_v2("pacifist", o2, v2, env2, 4, rng, 5)
assert a == A_SILENT, f"low-tech pacifist falls back to SILENT, got {a}"
env2.public[3] = False
a = heuristic_action_v2("pacifist", o2, v2, env2, 3, rng, 5)
assert a in (A_DEVELOP, A_SILENT), a
print("pacifist hide/hunt-free responses OK")

# unit 3: one v2b episode runs clean (shapes + no crash)
from stage_b_v25b import run_episode
from model import DQNAgent
agent = DQNAgent(obs_dim=dm.OBS_DIM, n_actions=dm.N_ACTIONS, hidden=[64, 64], lr=6e-4, gamma=0.99, batch_size=256, replay_size=50000, target_sync=500, seed=0)
env3 = dm.make_env(cfg, seed=42)
r = run_episode(env3, agent, np.random.default_rng(1), eps=0.05)
print("v2b episode OK; keys:", sorted(r.keys())[:6], "...")
print("SMOKE_V25B_OK")
