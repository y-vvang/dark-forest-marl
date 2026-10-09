#!/usr/bin/env python3
"""Regression test for the 2026-09-22 strike-kill bug fix (env2.py).

Bug: strike kills were recorded in strike_events/kills but the victim's
alive flag was never cleared (invasion path had it, strike path did not).
Fix mirrors invasion semantics; this test locks the behaviour in.

Run from anywhere:  python3 experiment/tests/test_strike_fix.py
"""
import os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(_HERE, ".."))  # configs are resolved relative to experiment/
sys.path.insert(0, os.path.join(_HERE, ".."))

import data2 as dm  # noqa: E402

A_STRIKE = 4


def _force_strike(seed, tech_i, tech_j):
    cfg = dm.load_config("config_v23.yaml")
    env = dm.make_env(cfg, seed=seed)
    env.reset()
    n = env.n
    ai = np.where(env.alive)[0]
    i, j = int(ai[0]), int(ai[1])
    env.known[i, j] = True
    env.tech[i], env.tech[j] = float(tech_i), float(tech_j)
    actions = np.zeros(n, dtype=int)
    actions[i] = A_STRIKE
    tg = np.zeros(n, dtype=int)
    tg[i] = j
    env.begin_tick(actions, tg)
    obs, valid, r, alive_now, done, info, dead_now = env.resolve(actions)
    ev = [e for e in info["strike_events"] if e[0] == i]
    return env, j, ev, info


def main():
    fails = 0
    for seed in range(10):
        # KILL: ratio 20 -> p_kill clipped 0.95
        env, j, ev, info = _force_strike(100 + seed, 20.0, 1.0)
        if not ((not env.alive[j]) and env.death_cause[j] == "strike"
                and ev and ev[0][2] == "killed"):
            fails += 1
            print(f"seed{seed}: KILL FAILED {ev} alive={env.alive[j]}")
        if abs(info["res_conservation_residual"]) > 1e-6:
            fails += 1
            print(f"seed{seed}: CONSERVATION {info['res_conservation_residual']}")
        # BLOCK: ratio 2.5 >= 2, both above strike tech gate (7.0)
        env, j, ev, info = _force_strike(200 + seed, 10.0, 25.0)
        if not (env.alive[j] and ev and ev[0][2] == "blocked"):
            fails += 1
            print(f"seed{seed}: BLOCK FAILED {ev} alive={env.alive[j]}")
    print("REGRESSION", "PASS" if fails == 0 else f"FAIL({fails})")
    return fails


if __name__ == "__main__":
    sys.exit(main())
