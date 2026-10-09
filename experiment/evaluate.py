"""Evaluate policies on DarkForestGrid v1 and aggregate results.json.

Policies
    dqn          — loads checkpoint.npz from a training run dir, epsilon-greedy (eval eps)
    random       — uniform over valid actions
    df_heuristic — hand-written "textbook" Dark Forest: STRIKE nearest if any known, else SILENT
    pacifist     — hand-written peaceful developer: 50% DEVELOP / 50% SILENT, never strikes

Usage
    python evaluate.py --policy dqn --ablation main --policy-seed 1001 \
                       --run-dir ../runs/seed42_main --config config.yaml --out ../results
    python evaluate.py --policy random --ablation main --policy-seed 1001 --out ../results
    python evaluate.py --aggregate --config config.yaml --out ../results
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np

import data as data_mod
from model import DQNAgent

A_SILENT, A_DEVELOP, A_BSELF, A_BOTHER, A_STRIKE = range(5)
POLICIES = ["dqn", "random", "df_heuristic", "pacifist"]


def heuristic_actions(policy: str, valid: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = valid.shape[0]
    if policy == "random":
        return np.array([rng.choice(np.where(valid[i])[0]) for i in range(n)])
    if policy == "df_heuristic":
        a = np.full(n, A_SILENT, dtype=np.int64)
        a[valid[:, A_STRIKE]] = A_STRIKE
        return a
    if policy == "pacifist":
        a = np.where(rng.uniform(size=n) < 0.5, A_DEVELOP, A_SILENT).astype(np.int64)
        return a
    raise ValueError(policy)


def run_eval(cfg, policy: str, ablation: str, policy_seed: int, n_episodes: int,
             run_dir: str | None = None) -> dict:
    ecfg = cfg["eval"]
    env = data_mod.make_env(cfg, seed=policy_seed, ablation=ablation)
    agent = None
    if policy == "dqn":
        agent = DQNAgent(
            obs_dim=data_mod.OBS_DIM, n_actions=data_mod.N_ACTIONS,
            hidden=list(cfg["model"]["hidden"]), lr=1e-4, gamma=0.99,
            batch_size=128, replay_size=1000, target_sync=10**9,
            seed=policy_seed,
        )
        agent.load(os.path.join(run_dir, "checkpoint.npz"))
    rng = np.random.default_rng(policy_seed)
    t_max = int(cfg["env"]["t_max"])

    per_ep = {"survival_mean": [], "final_alive": [], "extinction": [], "tech_surv": [],
              "strike_per_100": [], "p_strike_after_detect": []}
    freqs = np.zeros(data_mod.N_ACTIONS, dtype=np.int64)
    total_actions = 0

    for ep in range(n_episodes):
        obs, valid = env.reset(seed=int(ecfg["env_seed_base"]) + ep)
        alive_mask = np.ones(env.n, dtype=bool)
        first_detect = -np.ones(env.n, dtype=np.int64)
        first_strike_after = np.zeros(env.n, dtype=bool)
        alive_ticks = np.zeros(env.n, dtype=np.int64)
        alive_curve = np.zeros(t_max, dtype=np.int64)
        done = False
        ep_freqs = np.zeros(data_mod.N_ACTIONS, dtype=np.int64)
        while not done:
            if policy == "dqn":
                a = agent.act(obs, valid, float(ecfg["epsilon"]))
            else:
                a = heuristic_actions(policy, valid, rng)
            obs, valid, r, alive, done, info, dead_now = env.step(a)

            mask = alive_mask
            ep_freqs += np.bincount(a[mask], minlength=data_mod.N_ACTIONS)
            alive_ticks += mask.astype(np.int64)
            for (i, _j) in info["detections"]:
                if first_detect[i] < 0:
                    first_detect[i] = env.t
            for (i, _j, _k) in info["strikes"]:
                if first_detect[i] >= 0 and env.t - first_detect[i] <= 10:
                    first_strike_after[i] = True
            alive_mask = alive
            alive_curve[env.t - 1] += int(alive_mask.sum())
        denom = max(1, int(alive_ticks.sum()))
        detect_agents = int((first_detect >= 0).sum())
        imm = int((first_strike_after & (first_detect >= 0)).sum())
        per_ep["survival_mean"].append(float(alive_ticks.mean()) / t_max)
        per_ep["final_alive"].append(int(alive.sum()))
        per_ep["extinction"].append(float(int(alive.sum()) == 0))
        per_ep["tech_surv"].append(float(env.tech[alive].mean()) if alive.any() else float("nan"))
        per_ep["strike_per_100"].append(100.0 * int(ep_freqs[A_STRIKE]) / denom)
        per_ep["p_strike_after_detect"].append(imm / detect_agents if detect_agents else float("nan"))
        freqs += ep_freqs
        total_actions += int(ep_freqs.sum())

    out = {
        "policy": policy,
        "ablation": ablation,
        "policy_seed": policy_seed,
        "n_episodes": n_episodes,
        "metrics": {
            "survival_mean": float(np.nanmean(per_ep["survival_mean"])),
            "final_alive": float(np.mean(per_ep["final_alive"])),
            "extinction_rate": float(np.mean(per_ep["extinction"])),
            "tech_surv_mean": float(np.nanmean(per_ep["tech_surv"])),
            "strike_per_100": float(np.nanmean(per_ep["strike_per_100"])),
            "p_strike_after_detect": float(np.nanmean(per_ep["p_strike_after_detect"])),
        },
        "action_freqs": {data_mod.ACTION_NAMES[k]: float(freqs[k]) / max(1, total_actions)
                         for k in range(data_mod.N_ACTIONS)},
        "alive_curve": (alive_curve / float(n_episodes)).round(4).tolist(),
    }
    return out


def aggregate(cfg, results_dir: str, runs_dir: str) -> dict:
    """Merge per-(policy, ablation, seed) eval JSONs into the results.json schema."""
    files = sorted(glob.glob(os.path.join(results_dir, "eval_*.json")))
    groups: dict[tuple, list[dict]] = {}
    for fp in files:
        with open(fp) as f:
            rec = json.load(f)
        groups.setdefault((rec["policy"], rec["ablation"]), []).append(rec)

    def cell(recs: list[dict], key: str, sub: str | None = None):
        vals = [r["metrics"][key] if sub is None else r["action_freqs"][key] for r in recs]
        return {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "n_seeds": len(vals)}

    metric_keys = ["survival_mean", "final_alive", "extinction_rate", "tech_surv_mean",
                   "strike_per_100", "p_strike_after_detect"]
    label = {"dqn": "DQN-trained", "random": "random-policy",
             "df_heuristic": "DF-heuristic", "pacifist": "pacifist-heuristic"}

    summary: dict[str, dict] = {}
    for (pol, abl), recs in sorted(groups.items()):
        if abl != "main":
            continue
        m = {k: cell(recs, k) for k in metric_keys}
        m.update({f"freq_{name}": cell(recs, name, sub="freq") for name in data_mod.ACTION_NAMES})
        summary[label[pol]] = m

    ablation_block: dict[str, dict] = {}
    base = summary.get("DQN-trained", {})
    for abl in data_mod.ablation_names(cfg):
        recs = [v for (pol, a), vs in groups.items() if pol == "dqn" and a == abl for v in vs]
        if not recs:
            continue
        m = {k: cell(recs, k) for k in metric_keys}
        m.update({f"freq_{name}": cell(recs, name, sub="freq") for name in data_mod.ACTION_NAMES})
        ablation_block[abl] = {"metrics": m, "delta_vs_full": {}}
        for k in metric_keys:
            if k in base and k in m:
                ablation_block[abl]["delta_vs_full"][k] = {
                    "abs": round(m[k]["mean"] - base[k]["mean"], 4),
                    "rel": round((m[k]["mean"] - base[k]["mean"]) / base[k]["mean"], 4)
                    if abs(base[k]["mean"]) > 1e-9 else None,
                }

    # training curves: 3-seed mean of main runs, subsampled
    curves_raw: dict[str, list] = {}
    for sd in cfg["train"]["seeds"]:
        fp = os.path.join(runs_dir, f"seed{sd}_main", "metrics.jsonl")
        if not os.path.exists(fp):
            continue
        with open(fp) as f:
            recs = [json.loads(line) for line in f if line.strip()]
        for r in recs:
            curves_raw.setdefault(r["episode"], {}).setdefault(sd, []).append(r)
    episodes = sorted(curves_raw)
    step = max(1, len(episodes) // 100)

    def curve(field: str, mapper) -> list[float]:
        out = []
        for e in episodes[::step]:
            vals = []
            for sd_rec in curves_raw[e].values():
                v = mapper(sd_rec[0])
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    vals.append(v)
            out.append(round(float(np.mean(vals)), 4) if vals else None)
        return out

    get = lambda key: (lambda r: r[key])
    training_curves = {
        "episodes": episodes[::step],
        "silent": curve("silent", lambda r: r["action_freqs"]["SILENT"]),
        "develop": curve("develop", lambda r: r["action_freqs"]["DEVELOP"]),
        "broadcast_self": curve("bself", lambda r: r["action_freqs"]["BROADCAST_SELF"]),
        "broadcast_other": curve("bother", lambda r: r["action_freqs"]["BROADCAST_OTHER"]),
        "strike": curve("strike", lambda r: r["action_freqs"]["STRIKE"]),
        "p_strike_after_detect": curve("pim", get("p_strike_after_detect")),
        "mean_alive_frac": curve("alive", get("mean_alive_frac")),
    }

    return {
        "task": "Does the Dark Forest strategy emerge as the dominant survival policy in multi-agent RL?",
        "dataset": "DarkForestGrid v1.1 (self-built synthetic multi-agent environment; see data_contract.md)",
        "metrics": ["survival_mean", "final_alive", "extinction_rate", "tech_surv_mean",
                    "strike_per_100", "p_strike_after_detect"],
        "seeds": list(cfg["train"]["seeds"]),
        "eval": {"episodes_per_seed": int(cfg["eval"]["episodes"]),
                 "epsilon": float(cfg["eval"]["epsilon"]),
                 "env_seed_base": int(cfg["eval"]["env_seed_base"])},
        "provenance": {
            "mode": "measured",
            "source": "real execution of experiment/train.py + experiment/evaluate.py (agent sandbox)",
            "hardware": "2x Intel Xeon Platinum vCPU, 4 GB RAM, no GPU; pure numpy, 1 BLAS thread",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "summary": summary,
        "ablation": ablation_block,
        "training_curves": training_curves,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--policy", choices=POLICIES, default=None)
    ap.add_argument("--ablation", default="main")
    ap.add_argument("--policy-seed", type=int, default=None)
    ap.add_argument("--run-dir", default=None, help="training run dir containing checkpoint.npz (for dqn)")
    ap.add_argument("--out", default="../results")
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()

    cfg = data_mod.load_config(args.config)
    os.makedirs(args.out, exist_ok=True)

    if args.aggregate:
        res = aggregate(cfg, args.out, os.path.join(os.path.dirname(args.out), "runs"))
        with open(os.path.join(args.out, "results.json"), "w") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        print(f"aggregated -> {os.path.join(args.out, 'results.json')}")
        return

    n_ep = args.episodes if args.episodes is not None else int(cfg["eval"]["episodes"])
    rec = run_eval(cfg, args.policy, args.ablation, args.policy_seed, n_ep, args.run_dir)
    fp = os.path.join(args.out, f"eval_{args.policy}_{args.ablation}_s{args.policy_seed}.json")
    with open(fp, "w") as f:
        json.dump(rec, f, indent=2)
    m = rec["metrics"]
    print(f"[{args.policy}/{args.ablation}/s{args.policy_seed}] "
          f"surv {m['survival_mean']:.3f} | final_alive {m['final_alive']:.1f} | "
          f"strike/100 {m['strike_per_100']:.2f} | P(imm) {m['p_strike_after_detect']:.2f} -> {fp}")


if __name__ == "__main__":
    main()
