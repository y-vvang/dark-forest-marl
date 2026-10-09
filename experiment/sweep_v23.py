#!/usr/bin/env python3
"""v2.3 parameter verification sweep: short training arms (150 eps) over the
parameters never evaluated before, + DQN training health check (loss level).
Arms: shield_cost_frac {0.25, 0.75}, softsign_reward_scale {0.005, 0.02},
res_total_budget {100, 200}, lr {3e-4, 1.2e-3}, gamma {0.99, 0.999}.
Baseline (0.5 / 0.01 / 150 / 6e-4 / 0.995) known from v2.2s."""
import json, os, subprocess, sys
import numpy as np
import yaml

B = "/home/z/my-project/output/experiment-suite/does-the-dark-forest-strategy-emerge-as-ecec922e/2026-09-04_221544"
os.chdir(f"{B}/experiment")
sys.path.insert(0, ".")
import data2 as dm
from model import DQNAgent
import stage_b

ARMS = [
    {"name": "base_control"},
    {"name": "shield025", "env": {"shield_cost_frac": 0.25}},
    {"name": "shield075", "env": {"shield_cost_frac": 0.75}},
    {"name": "scale005", "env": {"softsign_reward_scale": 0.005}},
    {"name": "scale02", "env": {"softsign_reward_scale": 0.02}},
    {"name": "budget100", "env": {"res_total_budget": 100.0}},
    {"name": "budget200", "env": {"res_total_budget": 200.0}},
    {"name": "lr3e4", "train": {"lr": 0.0003}},
    {"name": "lr12e3", "train": {"lr": 0.0012}},
    {"name": "gamma099", "train": {"gamma": 0.99}},
    {"name": "gamma0999", "train": {"gamma": 0.999}},
]

def evaluate(ckpt, cfg, n_eps=40):
    agent = DQNAgent(obs_dim=dm.OBS_DIM, n_actions=dm.N_ACTIONS, hidden=[64, 64],
                     lr=1e-4, gamma=float(cfg["train"]["gamma"]), batch_size=128,
                     replay_size=10000, target_sync=2000, seed=1001)
    agent.load(ckpt)
    surv = {"learner": [], "random": [], "df": [], "pacifist": []}
    for es in (10556, 10557):
        env = dm.make_env(cfg, seed=es)
        rng = np.random.default_rng(777)
        for _ in range(n_eps):
            rec = stage_b.run_episode(env, agent, rng, 0.05)
            tmax = float(cfg["env"]["t_max"])
            for i, k in enumerate(rec["kinds"]):
                born = rec["spawn_tick"][i] < 1e8
                if born and k in surv:
                    span = max(1.0, tmax - rec["spawn_tick"][i])
                    surv[k].append(rec["alive_ticks"][i] / span)
    return {k: round(float(np.mean(v)), 3) for k, v in surv.items()}

# evidence chain: resumable — arms already recorded in the results file are
# skipped; every arm's config/metrics/checkpoint persist under sweep_v23_arms_fix/
RES = "../results/v23_fix/sweep.json"
results = json.load(open(RES)) if os.path.exists(RES) else {}
ARMS_DIR = "../runs/v23_fix_sweep_arms"
os.makedirs(ARMS_DIR, exist_ok=True)
base = yaml.safe_load(open("config_v23.yaml"))
for arm in ARMS:
    tag = arm["name"]
    if tag in results:
        print(f"[{tag}] already recorded — skip (evidence chain preserved)", flush=True)
        continue
    cfg = json.loads(json.dumps(base))
    for sec, kv in arm.items():
        if sec in ("env", "train"):
            cfg[sec].update(kv)
    arm_dir = f"{ARMS_DIR}/{tag}"
    os.makedirs(arm_dir, exist_ok=True)
    cfgp = f"{arm_dir}/config.yaml"
    yaml.safe_dump(cfg, open(cfgp, "w"))
    out = f"{arm_dir}/run"
    r = subprocess.run([sys.executable, "train2.py", "--seed", "42", "--mode", "main",
                        "--out", out, "--config", cfgp, "--episodes", "150"],
                       capture_output=True, text=True, env={**os.environ, "OMP_NUM_THREADS": "1"})
    if r.returncode != 0:
        results[tag] = {"error": r.stderr[-400:]}
        print(f"[{tag}] TRAIN FAILED", flush=True)
        json.dump(results, open(RES, "w"), indent=2)
        continue
    tr = [json.loads(l) for l in open(f"{out}/seed42_main/metrics.jsonl")]
    kl = [k for k in tr[0].keys() if "loss" in k.lower()][0]
    loss50 = float(np.mean([x[kl] for x in tr[-50:]]))
    surv = evaluate(f"{out}/seed42_main/checkpoint.npz", cfg)
    results[tag] = {"override": {s: v for s, v in arm.items() if s != "name"},
                    "loss_last50": round(loss50, 4),
                    "pool_survival_2worlds40eps": surv,
                    "arm_dir": arm_dir}
    print(f"[{tag}] loss50={loss50:.4f} surv={surv}", flush=True)
    json.dump(results, open(RES, "w"), indent=2)
print("SWEEP DONE")
