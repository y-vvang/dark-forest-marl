#!/usr/bin/env python3
"""v2.3-fix ablation arm: terminal-reward-only (2026-10-02).

Only reward signal = death penalty (-10). Removed:
  - softsign tech shaping (env.softsign_reward_scale: 0.01 -> 0.0)
  - black-domain entry penalty (env.bd_entry_penalty: 0.5 -> 0.0)
Same seed / episodes / architecture as runs/v23_fix main arm.
Idempotent: skips training if checkpoint exists, skips Stage B if json exists.
"""
import json, os, subprocess, sys
import yaml

B = "/home/z/my-project/output/experiment-suite/does-the-dark-forest-strategy-emerge-as-ecec922e/2026-09-04_221544"
os.chdir(f"{B}/experiment")

RUN_DIR = "../runs/v23_fix_term_only"
RES_DIR = "../results/v23_fix_term_only"
os.makedirs(RUN_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

base = yaml.safe_load(open("config_v23.yaml"))
base["env"]["softsign_reward_scale"] = 0.0
base["env"]["bd_entry_penalty"] = 0.0
cfgp = f"{RUN_DIR}/config.yaml"
yaml.safe_dump(base, open(cfgp, "w"))

ck = f"{RUN_DIR}/run/seed42_main/checkpoint.npz"
if not os.path.exists(ck):
    r = subprocess.run([sys.executable, "train2.py", "--seed", "42", "--mode", "main",
                        "--out", f"{RUN_DIR}/run", "--config", cfgp, "--episodes", "600"],
                       env={**os.environ, "OMP_NUM_THREADS": "1"})
    print(f"TRAIN_EXIT={r.returncode}", flush=True)
else:
    print("TRAIN skip (checkpoint exists)", flush=True)

sb = f"{RES_DIR}/stage_b.json"
if not os.path.exists(sb):
    r = subprocess.run([sys.executable, "stage_b.py", "--run-dir", f"{RUN_DIR}/run/seed42_main",
                        "--episodes", "150", "--out", RES_DIR, "--config", cfgp],
                       env={**os.environ, "OMP_NUM_THREADS": "1"})
    print(f"STAGEB_EXIT={r.returncode}", flush=True)
else:
    print("STAGEB skip", flush=True)

if os.path.exists(sb):
    d = json.load(open(sb))
    print("SURV:", d.get("survival_by_kind"), flush=True)
    print("PROFILE:", d.get("learner_action_profile"), flush=True)
    print("CLASS:", d.get("class_survival"), flush=True)
print("TERM_ONLY_DONE", flush=True)
