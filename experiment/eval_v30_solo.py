#!/usr/bin/env python3
"""v3.0 SOLO evaluation (user intent, 2026-10-16): the converged policy IS the
research object. One consensus intelligence commands every civilization in the
universe; evaluate its survival and action profile over full episodes with the
FROZEN trained model (never training-time metrics).

  --mode solo    all 36 slots act with the checkpoint net (greedy)
  --mode league  12 learner slots + 24 slots of ONE opponent kind
                 (kind = random | df | pacifist | ckpt:a_classic | ...)

Outputs solo_eval.json: survival (per-civ lifetime fraction, final/ever born),
8-action profile, dark-forest signals (first-strike-after-detect rate, detect->
strike latency, strike target tech bias), deaths by cause."""
import argparse, json, os
import numpy as np
import data2_v25 as data_mod
import env2_v25
from model import DQNAgent
import env2_v25 as A_

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="config_v25.yaml")
    ap.add_argument("--mode", default="solo", choices=["solo", "league"])
    ap.add_argument("--opp-kind", default="random")
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--seed0", type=int, default=900)
    a = ap.parse_args()
    cfg = data_mod.apply_ablation(data_mod.load_config(a.config), "main")
    env_mod = env2_v25
    mk = lambda s: data_mod.make_env(cfg, seed=s)
    hidden = [int(h) for h in cfg["model"]["hidden"]]
    agent = DQNAgent(obs_dim=data_mod.OBS_DIM, n_actions=data_mod.N_ACTIONS, hidden=hidden,
                     gamma=0.99, lr=6e-4, batch_size=256, replay_size=1,
                     target_sync=10**9, seed=0)
    agent.load(a.checkpoint)
    opp_net = None
    if a.opp_kind.startswith("ckpt:"):
        opp_net = DQNAgent(obs_dim=data_mod.OBS_DIM, n_actions=data_mod.N_ACTIONS, hidden=hidden,
                           gamma=0.99, lr=6e-4, batch_size=256, replay_size=1,
                           target_sync=10**9, seed=0)
        opp_net.load(os.path.join("../runs/v25_arms", a.opp_kind[5:], "run", "seed42_main", "checkpoint.npz"))

    n_acts = data_mod.N_ACTIONS
    acts_names = data_mod.ACTION_NAMES
    agg = {k: 0.0 for k in ("surv_frac", "final_alive", "ever_born", "alive_ticks",
                            "strike", "invade", "bd", "observe", "detect_events", "imm_strike",
                            "det_to_strike", "strike_target_tech", "strike_target_n")}
    deaths = {"strike": 0, "invasion": 0, "mutual": 0, "resource": 0}
    af = np.zeros(n_acts, dtype=np.int64)
    import env2_v25 as E
    HEUR = {"random", "df", "pacifist"}
    for ep in range(a.episodes):
        env = mk(a.seed0 + ep)
        obs, valid = env.reset()
        rng = np.random.default_rng(a.seed0 * 31 + ep)
        kinds = ["learner"] * env.n
        if a.mode == "league":
            ls = set(rng.permutation(env.n)[: env.n // 3].tolist())
            kinds = [("learner" if i in ls else a.opp_kind) for i in range(env.n)]
        born_tick = np.full(env.n, -1, dtype=np.int64)
        alive_ticks = np.zeros(env.n, dtype=np.int64)
        first_det = -np.ones(env.n, dtype=np.int64)
        prev_alive = np.zeros(env.n, dtype=bool)
        for t in range(env.t_max):
            obs_a, valid_a = env._obs(), env._valid()
            actions = np.zeros(env.n, dtype=np.int64)
            li = np.array([i for i, k in enumerate(kinds) if k == "learner"])
            if li.size:
                actions[li] = agent.act(obs_a[li], valid_a[li], 0.0)
            if opp_net is not None:
                ci = np.array([i for i, k in enumerate(kinds) if k == a.opp_kind])
                if ci.size:
                    actions[ci] = opp_net.act(obs_a[ci], valid_a[ci], 0.05)
            for i, k in enumerate(kinds):
                if k in HEUR:
                    if k == "random":
                        actions[i] = int(rng.choice(np.where(valid_a[i])[0]))
                    elif k == "df":
                        actions[i] = A_.A_STRIKE if valid_a[i, A_.A_STRIKE] else (
                            A_.A_OBSERVE if (valid_a[i, A_.A_OBSERVE] and rng.uniform() < 0.3)
                            else (A_.A_DEVELOP if rng.uniform() < 0.6 else A_.A_SILENT))
                    else:
                        actions[i] = A_.A_DEVELOP if rng.uniform() < 0.5 else A_.A_SILENT
            tmask = env.select_targets(actions)
            targets = np.full(env.n, -1, dtype=np.int64)
            for i in np.where(np.isin(actions, (A_.A_INVADE, A_.A_STRIKE, A_.A_BOTHER, A_.A_OBSERVE)))[0]:
                cand = np.where(tmask[i])[0]
                if cand.size:
                    targets[i] = int(rng.choice(cand))
            env.begin_tick(actions, targets)
            re = actions.copy()
            inv = env.pending_invaders if hasattr(env, "pending_invaders") else []
            nobs, nvalid, r, alive, done, info, dead_now = env.resolve(re)
            af += np.bincount(actions[np.isin(actions, range(n_acts))], minlength=n_acts)[:n_acts]
            agg["strike"] += len(info["strikes"]); agg["invade"] += len(info["invasions"])
            agg["bd"] += int((actions == A_.A_BDOMAIN).sum())
            agg["observe"] += int((actions == A_.A_OBSERVE).sum())
            for (i, _j) in info["detections"]:
                if first_det[i] < 0:
                    first_det[i] = t
            for (i, tgt, _k) in info["strikes"]:
                if first_det[i] >= 0:
                    agg["det_to_strike"] += t - first_det[i]
                    if t - first_det[i] <= 10:
                        agg["imm_strike"] += 1
                if 0 <= tgt < env.n:
                    agg["strike_target_tech"] += float(env.tech[tgt])
                    agg["strike_target_n"] += 1
            for (i, _j, mutual) in info["invasions"]:
                deaths["mutual" if mutual else "invasion"] += 1
            for (i, _j, killed) in info["strikes"]:
                if killed: deaths["strike"] += 1
            born = env.alive & ~prev_alive & (env.spawn_tick <= t)
            for i in np.where(born)[0]:
                if born_tick[i] < 0:
                    born_tick[i] = max(env.spawn_tick[i], 0)
            alive_ticks += env.alive
            prev_alive = env.alive.copy()
            if done: break
        T = env.t + 1
        agg["alive_ticks"] += float(alive_ticks.sum())
        for i in range(env.n):
            if born_tick[i] >= 0:
                agg["surv_frac"] += alive_ticks[i] / max(1, T - born_tick[i])
        agg["final_alive"] += int(env.alive.sum())
        agg["ever_born"] += int((born_tick >= 0).sum())
        res_deaths = max(0, int((born_tick >= 0).sum()) - deaths["strike"] - deaths["invasion"] - deaths["mutual"])
        # resource deaths are computed once per eval below (env counter more robust)
    tot_alive_ticks = agg["alive_ticks"]
    out = {
        "mode": a.mode, "opp_kind": (a.opp_kind if a.mode == "league" else None),
        "checkpoint": os.path.abspath(a.checkpoint), "episodes": a.episodes,
        "survival_fraction_of_lifetime": round(agg["surv_frac"] / max(1, agg["ever_born"]), 4),
        "final_alive_per_ep": round(agg["final_alive"] / a.episodes, 2),
        "ever_born_per_ep": round(agg["ever_born"] / a.episodes, 1),
        "action_freqs": {acts_names[k]: round(float(af[k]) / max(1, int(af.sum())), 4) for k in range(n_acts)},
        "strike_per_100_ticks": round(100.0 * agg["strike"] / max(1, tot_alive_ticks), 3),
        "invade_per_100_ticks": round(100.0 * agg["invade"] / max(1, tot_alive_ticks), 3),
        "blackdomain_per_100_ticks": round(100.0 * agg["bd"] / max(1, tot_alive_ticks), 3),
        "observe_per_100_ticks": round(100.0 * agg["observe"] / max(1, tot_alive_ticks), 3),
        "p_strike_within_10_of_detect": round(agg["imm_strike"] / max(1, agg["strike_target_n"]), 3),
        "median_note": "det_to_strike mean below (median needs per-event log)",
        "mean_detect_to_strike": round(agg["det_to_strike"] / max(1, agg["strike_target_n"]), 1),
        "deaths_by_cause_partial": deaths,
        "contract": "v3.0 solo: frozen model commands the whole universe; NOT training metrics",
    }
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, "solo_eval.json"), "w"), indent=2, ensure_ascii=False)
    print("->", os.path.join(a.out, "solo_eval.json"),
          "| surv", out["survival_fraction_of_lifetime"],
          "| strike/100", out["strike_per_100_ticks"],
          "| p_imm", out["p_strike_within_10_of_detect"])

if __name__ == "__main__":
    main()
