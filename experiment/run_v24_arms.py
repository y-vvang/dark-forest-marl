#!/usr/bin/env python3
"""v2.4 reward arms (user spec 2026-09-10... 2026-10-09): a classic / b terminal_only
/ c res_reward / d malintent. Two parallel workers (2-core box), idempotent per arm."""
import json, os, subprocess, sys, time
import yaml

B = "/home/z/my-project/output/experiment-suite/does-the-dark-forest-strategy-emerge-as-ecec922e/2026-09-04_221544"
os.chdir(f"{B}/experiment")

ARMS = [
    ("a_classic", {}),
    ("b_terminal", {"softsign_reward_scale": 0.0, "bd_entry_penalty": 0.0}),
    ("c_res", {"res_reward_scale": 0.0025}),
    ("d_malintent", {"res_reward_scale": 0.0025, "attack_reward": 0.5}),
]
RES = "../results/v24_arms"
RUNS = "../runs/v24_arms"
os.makedirs(RES, exist_ok=True)
os.makedirs(RUNS, exist_ok=True)
summary_path = f"{RES}/summary.json"

def merge_summaries():
    """Main process only: merge per-arm summaries into summary.json.
    (fork workers must NOT write the shared file directly — v1 of this driver
    lost arms to a write race on 2026-10-09; summaries were rebuilt from the
    per-arm stage_b.json files.)"""
    merged = {}
    for tag, _ in ARMS:
        p = f"{RES}/summary_{tag}.json"
        if os.path.exists(p):
            merged[tag] = json.load(open(p))
    json.dump(merged, open(summary_path, "w"), indent=2)
    return merged

def run_arm(tag, ov):
    arm_runs, arm_res = f"{RUNS}/{tag}", f"{RES}/{tag}"
    os.makedirs(arm_runs, exist_ok=True); os.makedirs(arm_res, exist_ok=True)
    base = yaml.safe_load(open("config_v24.yaml"))
    base["env"].update(ov)
    cfgp = f"{arm_runs}/config.yaml"
    yaml.safe_dump(base, open(cfgp, "w"))
    arm_summary = None
    ck = f"{arm_runs}/run/seed42_main/checkpoint.npz"
    if not os.path.exists(ck):
        r = subprocess.run([sys.executable, "train2_v24.py", "--seed", "42", "--mode", "main",
                            "--out", f"{arm_runs}/run", "--config", cfgp, "--episodes", "600"],
                           env={**os.environ, "OMP_NUM_THREADS": "1"},
                           stdout=open(f"{arm_runs}/train.log", "w"), stderr=subprocess.STDOUT)
        print(f"[{tag}] TRAIN_EXIT={r.returncode}", flush=True)
        if r.returncode != 0:
            arm_summary = {"error": open(f"{arm_runs}/train.log").read()[-400:]}
    else:
        print(f"[{tag}] train skip", flush=True)
    sb = f"{arm_res}/stage_b.json"
    if arm_summary is None and not os.path.exists(sb):
        r = subprocess.run([sys.executable, "stage_b_v24.py", "--run-dir", f"{arm_runs}/run/seed42_main",
                            "--episodes", "150", "--out", arm_res, "--config", cfgp],
                           env={**os.environ, "OMP_NUM_THREADS": "1"},
                           stdout=open(f"{arm_runs}/stageb.log", "w"), stderr=subprocess.STDOUT)
        print(f"[{tag}] STAGEB_EXIT={r.returncode}", flush=True)
        if r.returncode != 0:
            arm_summary = {"error": open(f"{arm_runs}/stageb.log").read()[-400:]}
    if arm_summary is None and os.path.exists(sb):
        d = json.load(open(sb))
        arm_summary = {"override": ov,
                       "survival_by_kind": {k: v["survival_fraction_of_lifetime"] for k, v in d["survival_by_kind"].items()},
                       "learner_loot_mean": d["survival_by_kind"]["learner"]["loot_mean"],
                       "learner_kills_mean": d["survival_by_kind"]["learner"]["kills_mean"],
                       "learner_action_profile": d["learner_action_profile"],
                       "survival_by_earned_tech_class": d.get("survival_by_earned_tech_class")}
        print(f"[{tag}] DONE surv={arm_summary['survival_by_kind']}", flush=True)
    json.dump(arm_summary, open(f"{RES}/summary_{tag}.json", "w"), indent=2)

# two batches of two (2 cores)
for batch in (ARMS[:2], ARMS[2:]):
    t0 = time.time()
    procs = []
    for tag, ov in batch:
        pid = os.fork()
        if pid == 0:
            run_arm(tag, ov)
            os._exit(0)
        procs.append(pid)
    for p in procs:
        os.waitpid(p, 0)
    print(f"batch {batch} finished in {(time.time()-t0)/60:.1f} min", flush=True)
merge_summaries()
print("V24_ARMS_DONE", flush=True)
