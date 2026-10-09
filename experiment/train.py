"""Train DarkForestGrid civilizations with a shared-policy Double-DQN (self-play).

Usage:
    python train.py --config config.yaml --seed 42 --ablation main --out ../runs
    python train.py --episodes 8            # smoke test

Outputs per run directory `--out/seed{seed}_{ablation}/`:
    checkpoint.npz   final Q-network weights (+ update counter)
    metrics.jsonl    one JSON object per episode (action freqs, survival, strikes…)
    train_log.txt    human-readable progress log
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import data as data_mod
from model import DQNAgent

A_SILENT, A_DEVELOP, A_BSELF, A_BOTHER, A_STRIKE = range(5)


def eps_schedule(progress: float, eps_start: float, eps_end: float) -> float:
    """Anneal on EPISODE progress (robust to episodes ending early), not on an
    assumed fixed number of ticks per episode."""
    return float(eps_start - (eps_start - eps_end) * min(1.0, max(0.0, progress)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ablation", default="main")  # validated against config.ablations in data.make_env
    ap.add_argument("--out", default="../runs")
    ap.add_argument("--episodes", type=int, default=None, help="override config (smoke tests)")
    args = ap.parse_args()

    cfg = data_mod.load_config(args.config)
    tc = cfg["train"]
    episodes = args.episodes if args.episodes is not None else int(tc["episodes"])
    t_max = int(cfg["env"]["t_max"])

    run_dir = os.path.join(args.out, f"seed{args.seed}_{args.ablation}")
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "train_log.txt")
    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    log_f = open(log_path, "a")

    def log(msg: str):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")
        log_f.flush()

    log(f"=== train seed={args.seed} ablation={args.ablation} episodes={episodes} ===")

    env = data_mod.make_env(cfg, seed=args.seed, ablation=args.ablation)
    agent = DQNAgent(
        obs_dim=data_mod.OBS_DIM,
        n_actions=data_mod.N_ACTIONS,
        hidden=list(cfg["model"]["hidden"]),
        lr=float(tc["lr"]),
        gamma=float(tc["gamma"]),
        batch_size=int(tc["batch_size"]),
        replay_size=int(tc["replay_size"]),
        target_sync=int(tc["target_sync"]),
        seed=args.seed,
    )
    update_every = int(tc["update_every"])

    t0 = time.time()
    global_step = 0
    loss_ema = None
    recent = []
    mf = open(metrics_path, "w")

    for ep in range(episodes):
        obs, valid = env.reset()
        alive_mask = np.ones(env.n, dtype=bool)
        first_detect_tick = -np.ones(env.n, dtype=np.int64)
        ep_return = 0.0
        alive_ticks = np.zeros(env.n, dtype=np.int64)
        act_counts = np.zeros(data_mod.N_ACTIONS, dtype=np.int64)
        n_strikes = 0
        imm_flag = np.zeros(env.n, dtype=bool)  # agent struck within 10t of its FIRST detection
        n_detect_events = 0
        done = False
        while not done:
            # epsilon anneals on episode progress — robust to early-ending episodes
            eps = eps_schedule(
                (ep + env.t / t_max) / (float(tc["eps_frac"]) * episodes),
                float(tc["eps_start"]), float(tc["eps_end"]))
            a = agent.act(obs, valid, eps)
            nobs, nvalid, r, alive, done, info, dead_now = env.step(a)

            mask = alive_mask
            agent.buffer.add(obs[mask], a[mask], r[mask], nobs[mask], dead_now[mask].astype(np.float32))

            # --- episode statistics ------------------------------------
            act_counts += np.bincount(a[mask], minlength=data_mod.N_ACTIONS)
            ep_return += float(r[mask].sum())
            alive_ticks += alive_mask.astype(np.int64)
            for (i, _j) in info["detections"]:
                if first_detect_tick[i] < 0:      # unified metric: FIRST detection per agent
                    first_detect_tick[i] = env.t
                    n_detect_events += 1
            for (i, _j, _k) in info["strikes"]:
                n_strikes += 1
                if first_detect_tick[i] >= 0 and env.t - first_detect_tick[i] <= 10:
                    imm_flag[i] = True

            if global_step % update_every == 0:
                loss = agent.update()
                if loss is not None:
                    loss_ema = loss if loss_ema is None else 0.995 * loss_ema + 0.005 * loss

            global_step += 1
            obs, valid = nobs, nvalid
            alive_mask = alive

        denom = max(1, act_counts.sum())
        rec = {
            "episode": ep,
            "epsilon": round(eps, 4),
            "mean_return_per_agent": round(ep_return / env.n, 3),
            "mean_alive_frac": round(float(alive_ticks.mean()) / t_max, 4),
            "final_alive": int(alive.sum()),
            "action_freqs": {data_mod.ACTION_NAMES[k]: round(float(act_counts[k]) / denom, 4)
                             for k in range(data_mod.N_ACTIONS)},
            "strike_per_100": round(100.0 * n_strikes / max(1, alive_ticks.sum()), 4),
            "p_strike_after_detect": round(
                float((imm_flag & (first_detect_tick >= 0)).sum()) / n_detect_events, 4
            ) if n_detect_events else None,
            "n_detect_events": n_detect_events,
            "loss_ema": None if loss_ema is None else round(float(loss_ema), 5),
            "wall_s": round(time.time() - t0, 1),
        }
        mf.write(json.dumps(rec) + "\n")
        recent.append(rec)
        if len(recent) > 50:
            recent.pop(0)

        if (ep + 1) % 100 == 0 or ep == episodes - 1:
            agent.save(os.path.join(run_dir, "checkpoint.npz"))
        if (ep + 1) % 50 == 0 or ep == episodes - 1:
            mr = float(np.mean([x["mean_return_per_agent"] for x in recent]))
            ma = float(np.mean([x["mean_alive_frac"] for x in recent]))
            st = float(np.mean([x["action_freqs"]["STRIKE"] for x in recent]))
            si = float(np.mean([x["action_freqs"]["SILENT"] for x in recent]))
            pim = [x["p_strike_after_detect"] for x in recent if x["p_strike_after_detect"] is not None]
            pim = float(np.mean(pim)) if pim else float("nan")
            log(f"ep {ep+1}/{episodes} | ret {mr:7.1f} | alive {ma:.3f} | "
                f"silent {si:.2f} strike {st:.2f} | P(imm-strike) {pim:.2f} | "
                f"eps {eps:.2f} | loss {loss_ema if loss_ema is None else round(loss_ema,4)} | "
                f"{time.time()-t0:.0f}s")

    agent.save(os.path.join(run_dir, "checkpoint.npz"))
    mf.close()
    log(f"=== done in {(time.time()-t0)/60:.1f} min -> {run_dir} ===")
    log_f.close()


if __name__ == "__main__":
    main()
