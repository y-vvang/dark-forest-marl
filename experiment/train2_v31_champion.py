#!/usr/bin/env python3
"""Training for DarkForestGrid v2.0.

Modes:
  main   homogeneous self-play (all 12 civilizations learn, shared DQN)
  mixed  heterogeneous population (branch experiment): 3 random + 3 DF-heuristic
         + 3 pacifist + 3 learning civilizations (shared DQN) — fixed-policy
         agents do NOT learn (ckpt arms act with their own frozen nets, eps .05);
learners (12 of 36 slots) train from their own transitions. This is the v3.0
"flagship" league training: every episode re-draws the opponent mix, so the
champion sees every strategy family (heuristics + all four arm models).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import data2_v25 as data_mod
from model import DQNAgent

A = data_mod.env_mod  # action-constant alias (constants live in env2)


def pick_target(mask_row: np.ndarray, kind: str, tech: np.ndarray, act: int) -> int:
    """Second-pass target choice. Fixed rules (disclosed): learners and random
    pick uniformly among candidates; DF-heuristic invades the LOWEST-tech
    candidate (safe kill) and broadcasts the HIGHEST-tech one (curse the leader);
    pacifists never target (their valid mask excludes targeting actions)."""
    cand = np.where(mask_row)[0]
    if cand.size == 0:
        return -1
    if kind in ("random", "learner"):
        return int(cand[0]) if act == A.A_INVADE else int(cand[np.argmax(tech[cand])])
    if kind == "df":
        return int(cand[np.argmin(tech[cand])]) if act == A.A_INVADE else int(cand[np.argmax(tech[cand])])
    return int(cand[0])


def heuristic_action(kind: str, obs: np.ndarray, valid: np.ndarray, env, i: int, rng, t: int) -> int:
    if kind == "random":
        cand = np.where(valid[i])[0]
        return int(rng.choice(cand))
    if kind == "df":
        if valid[i, A.A_STRIKE]:
            return A.A_STRIKE
        if valid[i, A.A_OBSERVE] and rng.uniform() < 0.3:
            return A.A_OBSERVE
        return A.A_DEVELOP if rng.uniform() < 0.6 else A.A_SILENT
    if kind == "pacifist":
        return A.A_DEVELOP if rng.uniform() < 0.5 else A.A_SILENT
    raise ValueError(kind)


def act_all(agent, kinds, obs, valid, eps, rngs, env, t, opp_nets=None):
    """Phase-1 actions for all agents + (learner) target choice.
    v2.3 perf: learners share one network — act on them in a SINGLE batched
    forward (36 single-row forwards per tick were the v2.3 bottleneck).
    v3.0: kinds may be "ckpt:<arm>" — those slots act with the frozen arm net
    (single batched forward per arm, light eps=0.05 exploration)."""
    actions = np.zeros(env.n, dtype=np.int64)
    learners = [i for i in range(env.n) if kinds[i] == "learner"]
    if learners:
        li = np.array(learners)
        actions[li] = agent.act(obs[li], valid[li], eps)
    if opp_nets:
        for arm, net in opp_nets.items():
            idx = [i for i in range(env.n) if kinds[i] == "ckpt:" + arm]
            if idx:
                ii = np.array(idx)
                actions[ii] = net.act(obs[ii], valid[ii], 0.05)
    for i in range(env.n):
        k = kinds[i]
        if k != "learner" and not (isinstance(k, str) and k.startswith("ckpt:")):
            actions[i] = heuristic_action(k, obs, valid, env, i, rngs[i], t)
    tmask = env.select_targets(actions)
    targets = np.full(env.n, -1, dtype=np.int64)
    for i in np.where(np.isin(actions, (A.A_INVADE, A.A_STRIKE, A.A_BOTHER, A.A_OBSERVE)))[0]:
        kind = kinds[i]
        if kind == "learner" or (isinstance(kind, str) and kind.startswith("ckpt:")):
            cand = np.where(tmask[i])[0]
            targets[i] = int(rngs[i].choice(cand)) if cand.size else -1
        else:
            targets[i] = pick_target(tmask[i], kind, env.tech, actions[i])
    return actions, targets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", default="champion", choices=["main", "mixed", "champion", "v31"])
    ap.add_argument("--pool-dir", default="../runs/v25_arms")
    ap.add_argument("--out", default="../runs/v3/v31_champion")
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--init-from", default=None, help="warm-start checkpoint.npz (v3.1 curriculum)")
    ap.add_argument("--n-heur", type=int, default=1, help="heuristic slots kept in OPP_POOL (0..2)")
    ap.add_argument("--oversample", type=int, default=3, help="repeat factor for each arm ckpt in OPP_POOL")
    ap.add_argument("--boost-b", action="store_true", help="double b_terminal weight in arm block")
    ap.add_argument("--lr", type=float, default=None, help="override config lr")
    ap.add_argument("--eps-start", type=float, default=None, help="override config eps_start")
    ap.add_argument("--eps-end", type=float, default=None, help="override config eps_end")
    ap.add_argument("--gamma", type=float, default=None, help="override config gamma")
    ap.add_argument("--replay", type=int, default=None, help="override config replay_size")
    ap.add_argument("--target-sync", type=int, default=None, help="override config target_sync")
    ap.add_argument("--batch", type=int, default=None, help="override config batch_size")
    ap.add_argument("--tag", default=None, help="run_name suffix: seed<s>_<tag> (branch isolation)")
    args = ap.parse_args()

    cfg = data_mod.apply_ablation(data_mod.load_config(args.config), args.mode if args.mode == "mixed" else "main")
    env = data_mod.make_env(cfg, seed=args.seed)
    tc = cfg["train"]
    episodes = args.episodes or int(tc["episodes"])

    run_name = f"seed{args.seed}_{args.tag or args.mode}"
    out_dir = os.path.join(args.out, run_name)
    os.makedirs(out_dir, exist_ok=True)
    metrics_path = os.path.join(out_dir, "metrics.jsonl")
    log_path = os.path.join(out_dir, "train_log.txt")
    mf = open(metrics_path, "w")
    lf = open(log_path, "a")

    n = env.n
    opp_nets = None
    champ_rng = None
    # v3.2 sweep-parameterized pool: heuristics kept for open-world anchoring,
    # arm ckpts oversampled; arms come from the SAME-seed v3.0 champion batch
    arms = ["ckpt:a_classic", "ckpt:b_terminal", "ckpt:c_res", "ckpt:d_malintent"]
    if args.boost_b:
        arms = ["ckpt:a_classic", "ckpt:b_terminal", "ckpt:b_terminal", "ckpt:c_res", "ckpt:d_malintent"]
    OPP_POOL = (["random", "df", "pacifist"])[:max(0, min(2, args.n_heur))] + arms * args.oversample
    if args.mode in ("champion", "v31"):
        from model import DQNAgent as _DQ
        opp_nets = {}
        for arm in ("a_classic", "b_terminal", "c_res", "d_malintent"):
            ck = os.path.join(args.pool_dir, arm, "run", "seed42_main", "checkpoint.npz")
            net = _DQ(obs_dim=data_mod.OBS_DIM, n_actions=data_mod.N_ACTIONS,
                      hidden=[int(h) for h in cfg["model"]["hidden"]],
                      gamma=float(tc["gamma"]), lr=float(tc["lr"]), batch_size=int(tc["batch_size"]),
                      replay_size=1, target_sync=10**9, seed=0)
            net.load(ck)
            opp_nets[arm] = net
        champ_rng = np.random.default_rng(args.seed * 7 + 1)
    if args.mode == "mixed":
        kinds = (["random"] * 3 + ["df"] * 3 + ["pacifist"] * 3 + ["learner"] * 3)
        rng = np.random.default_rng(args.seed)
        kinds = list(rng.permutation(kinds))          # shuffle positions
    else:
        kinds = ["learner"] * n
    learners = [i for i, k in enumerate(kinds) if k == "learner"]

    if args.lr is not None: tc["lr"] = args.lr
    if args.eps_start is not None: tc["eps_start"] = args.eps_start
    if args.eps_end is not None: tc["eps_end"] = args.eps_end
    if args.gamma is not None: tc["gamma"] = args.gamma
    if args.replay is not None: tc["replay_size"] = args.replay
    if args.target_sync is not None: tc["target_sync"] = args.target_sync
    if args.batch is not None: tc["batch_size"] = args.batch
    agent = DQNAgent(
        obs_dim=data_mod.OBS_DIM, n_actions=data_mod.N_ACTIONS,
        hidden=[int(h) for h in cfg["model"]["hidden"]],
        gamma=float(tc["gamma"]), lr=float(tc["lr"]),
        batch_size=int(tc["batch_size"]), replay_size=int(tc["replay_size"]),
        target_sync=int(tc["target_sync"]),
        seed=args.seed,
    )
    if args.init_from:
        agent.load(args.init_from)
        print(f"[v31] warm start from {args.init_from}", flush=True)
    eps_start, eps_end, eps_frac = float(tc["eps_start"]), float(tc["eps_end"]), float(tc["eps_frac"])
    rngs = [np.random.default_rng(args.seed * 100 + i) for i in range(n)]
    t0 = time.time()
    loss_ema = None

    for ep in range(episodes):
        if args.mode in ("champion", "v31"):
            ls = champ_rng.permutation(n)[: n // 3]
            kinds = np.array(["x"] * n, dtype=object)
            kinds[ls] = "learner"
            rest = np.where(kinds == "x")[0]
            kinds[rest] = [OPP_POOL[int(champ_rng.integers(len(OPP_POOL)))] for _ in rest]
        kinds_l = list(kinds) if not isinstance(kinds, np.ndarray) else list(kinds)
        kinds = kinds_l
        learners = [i for i, k in enumerate(kinds) if k == "learner"]
        obs, valid = env.reset()
        alive_mask = np.ones(n, dtype=bool)
        ep_return = 0.0
        alive_ticks = np.zeros(n, dtype=np.int64)
        act_counts = np.zeros(data_mod.N_ACTIONS, dtype=np.int64)
        n_strikes = n_invasions = n_counter = n_mutual = 0
        strikes_by_class = {"low": 0, "mid": 0, "high": 0}
        deaths_by_cause = {"strike": 0, "invasion": 0, "mutual": 0, "resource": 0}
        first_detect = -np.ones(n, dtype=np.int64)
        imm_count = det_events = 0
        done = False
        while not done:
            eps = eps_schedule((ep + env.t / env.t_max) / (eps_frac * episodes), eps_start, eps_end)
            actions, targets = act_all(agent, kinds, obs, valid, eps, rngs, env, env.t, opp_nets)
            invaded = env.begin_tick(actions, targets)
            # phase-2: invaded civilizations re-choose with knowledge of the invasion
            re_actions = actions.copy()
            if invaded:
                re_obs, re_valid = env._obs(), env._valid()
                for v in invaded:
                    if kinds[v] == "learner":
                        re_actions[v] = agent.act(re_obs[v:v + 1], re_valid[v:v + 1], eps)[0]
                    elif isinstance(kinds[v], str) and kinds[v].startswith("ckpt:"):
                        re_actions[v] = opp_nets[kinds[v][5:]].act(re_obs[v:v + 1], re_valid[v:v + 1], 0.05)[0]
                    else:
                        re_actions[v] = heuristic_action(kinds[v], re_obs, re_valid, env, v, rngs[v], env.t)
                    if re_actions[v] == A.A_INVADE:
                        re_actions[v] = A.A_INVADE   # implicit target = the invader (see env2 docstring)
            nobs, nvalid, r, alive, done, info, dead_now = env.resolve(re_actions)

            # --- learner-only transition storage (invaded learners store re-actions)
            stored_a = actions.copy()
            for v in invaded:
                if kinds[v] == "learner":
                    stored_a[v] = re_actions[v]
            lm = alive_mask[np.array(learners)]
            if lm.any():
                idx = np.array(learners)[lm]
                agent.buffer.add(obs[idx], stored_a[idx], r[idx], nobs[idx], dead_now[idx].astype(np.float32),
                             nvalid[idx].astype(np.float32))
            # --- metrics (all agents)
            larr = np.array(learners)
            lm2 = alive_mask[larr]
            if lm2.any():
                act_counts += np.bincount(stored_a[larr[lm2]], minlength=data_mod.N_ACTIONS)
            ep_return += float(r[alive_mask].sum())
            alive_ticks += alive_mask
            n_strikes += len(info["strikes"])
            n_invasions += len(info["invasions"])
            n_counter += info.get("counter_detections", 0)
            n_mutual += sum(1 for m in info["invasions"] if m[2])
            for (att, _vic, mutual) in info["invasions"]:
                if not mutual:
                    cls = data_mod.tech_class(env.spawn_tech[att])
                    strikes_by_class[cls] = strikes_by_class.get(cls, 0) + 1
            for (i, _j, killed) in info["strikes"]:
                if killed:
                    deaths_by_cause["strike"] += 1
            for (_inv, _vic, mutual) in info["invasions"]:
                deaths_by_cause["mutual" if mutual else "invasion"] += 1
            for (i, _j) in info["detections"]:
                if first_detect[i] < 0:
                    first_detect[i] = env.t
                    det_events += 1
            for (i, _j, _killed) in info["strikes"]:
                if first_detect[i] >= 0 and env.t - first_detect[i] <= 10:
                    imm_count += 1
            for (i, _j, _mutual) in info["invasions"]:
                if first_detect[i] >= 0 and env.t - first_detect[i] <= 10:
                    imm_count += 1
            alive_mask = alive
            loss = agent.update()
            if loss is not None:
                loss_ema = loss if loss_ema is None else 0.98 * loss_ema + 0.02 * loss
            obs, valid = nobs, nvalid

        # deaths by cause (from env counters): resource deaths = dead agents not killed
        rec = {
            "episode": ep + 1, "eps": round(eps, 3),
            "return": round(ep_return, 2),
            "mean_alive_frac": round(float(alive_ticks.sum() / (n * env.t_max)), 4),
            "final_alive": int(alive.sum()),
            "action_freqs": {data_mod.ACTION_NAMES[k]: round(float(act_counts[k] / max(1, act_counts.sum())), 4)
                             for k in range(data_mod.N_ACTIONS)},
            "strike_per_100": round(100.0 * n_strikes / max(1, alive_ticks.sum()), 3),
            "invade_per_100": round(100.0 * n_invasions / max(1, alive_ticks.sum()), 3),
            "observe_per_100": round(100.0 * int(act_counts[A.A_OBSERVE]) / max(1, alive_ticks.sum()), 3),
            "mutual_per_100": round(100.0 * n_mutual / max(1, alive_ticks.sum()), 3),
            "p_strike_after_detect": (round(imm_count / det_events, 3) if det_events else None),
            "strikes_by_invader_class": strikes_by_class,
            "loss": (round(loss_ema, 4) if loss_ema is not None else None),
            "wall_s": round(time.time() - t0, 1),
            "kinds": ({k: int(sum(1 for x in kinds if x == k)) for k in set(kinds)}
                      if args.mode in ("champion", "v31") else (kinds if ep == 0 else None)),
        }
        mf.write(json.dumps(rec) + "\n")
        if (ep + 1) % 50 == 0 or ep + 1 == episodes:
            af = rec["action_freqs"]
            line = (f"ep {ep+1}/{episodes} | ret {rec['return']:7.1f} | alive {rec['mean_alive_frac']:.3f} | "
                    f"silent {af['SILENT']:.2f} strike {af['STRIKE']:.2f} invade {af['INVADE']:.2f} obs {af['OBSERVE']:.2f} bd {af['BLACK_DOMAIN']:.2f} | "
                    f"eps {eps:.2f} | loss {rec['loss']} | {rec['wall_s']:.0f}s")
            print(f"[{time.strftime('%H:%M:%S')}] {line}", flush=True)
            lf.write(line + "\n")
            agent.save(os.path.join(out_dir, "checkpoint.npz"))
            if (ep + 1) % 250 == 0:
                agent.save(os.path.join(out_dir, f"checkpoint_ep{ep+1}.npz"))
    agent.save(os.path.join(out_dir, "checkpoint.npz"))
    mf.close()
    print(f"=== done in {(time.time()-t0)/60:.1f} min -> {out_dir} ===")


def eps_schedule(progress: float, eps_start: float, eps_end: float) -> float:
    return float(eps_start - (eps_start - eps_end) * min(1.0, max(0.0, progress)))


if __name__ == "__main__":
    main()
