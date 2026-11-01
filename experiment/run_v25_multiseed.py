#!/usr/bin/env python3
"""v2.5 multi-seed: 4 arms x 3 seeds x 1000ep training (2-way parallel on
2 vCPU), then StageB double-suite (old heuristics + v2b) per (arm, seed).
Idempotent per unit (checkpoint / stage_b.json existence)."""
import argparse, json, os, subprocess, sys, yaml
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.dirname(HERE)
ARMS = [
    ("a_classic", {}),
    ("b_terminal", {"softsign_reward_scale": 0.0, "bd_entry_penalty": 0.0}),
    ("c_res", {"res_reward_scale": 0.0025}),
    ("d_malintent", {"res_reward_scale": 0.0025, "attack_reward": 0.5}),
]
SEEDS = [42, 43, 44]
EPISODES = 1000

def sh(cmd, logp):
    r = subprocess.run(cmd, env={**os.environ, "OMP_NUM_THREADS": "1"}, cwd=HERE,
                       stdout=open(logp, "w"), stderr=subprocess.STDOUT, text=True)
    return r.returncode, (open(logp).read()[-600:] if r.returncode else "")

def train_done(runs, seed, need):
    """Idempotency by metrics tail (fix 2026-10-16): a mid-training checkpoint
    (saved every 50ep) must NOT count as done — b43/44 were killed at ep818/764
    and the checkpoint-existence check wrongly skipped them."""
    mp = os.path.join(runs, "run", f"seed{seed}_main", "metrics.jsonl")
    if not os.path.exists(mp):
        return False
    last = None
    with open(mp) as f:
        for line in f:
            if line.strip():
                last = line
    try:
        return last is not None and json.loads(last)["episode"] >= need
    except Exception:
        return False

def train_one(arm, ov, seed):
    runs = os.path.join(RUN, "runs", "v25m_arms", arm)
    os.makedirs(runs, exist_ok=True)
    cfgp = os.path.join(runs, "config.yaml")
    if not os.path.exists(cfgp):
        base = yaml.safe_load(open(os.path.join(HERE, "config_v25.yaml")))
        base["env"].update(ov)
        yaml.safe_dump(base, open(cfgp, "w"))
    if train_done(runs, seed, EPISODES):
        return f"[{arm}/s{seed}] train skip"
    rc, tail = sh([sys.executable, "train2_v25.py", "--seed", str(seed), "--mode", "main",
                   "--out", os.path.join(runs, "run"), "--config", cfgp,
                   "--episodes", str(EPISODES)],
                  os.path.join(runs, f"train_seed{seed}.log"))
    return f"[{arm}/s{seed}] TRAIN_EXIT={rc}" + ("" if rc == 0 else " TAIL:" + tail)

def stageb_one(arm, ov, seed, suite):
    script = "stage_b_v25.py" if suite == "old" else "stage_b_v25b.py"
    out = os.path.join(RUN, "results", "v25m_arms", arm, f"seed{seed}",
                       "stageb_old" if suite == "old" else "stageb_v2b")
    os.makedirs(out, exist_ok=True)   # fix 2026-10-16: was missing -> all 24 FileNotFoundError
    sb = os.path.join(out, "stage_b.json")
    if os.path.exists(sb):
        return f"[{arm}/s{seed}/{suite}] skip"
    cfgp = os.path.join(RUN, "runs", "v25m_arms", arm, "config.yaml")
    rc, tail = sh([sys.executable, script, "--run-dir",
                   os.path.join(RUN, "runs", "v25m_arms", arm, "run", f"seed{seed}_main"),
                   "--episodes", "150", "--out", out, "--config", cfgp],
                  os.path.join(out, "stageb.log"))
    return f"[{arm}/s{seed}/{suite}] STAGEB_EXIT={rc}" + ("" if rc == 0 else " TAIL:" + tail)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["train", "stageb"], required=True)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    if a.stage == "train":
        jobs = [(arm, ov, s) for arm, ov in ARMS for s in SEEDS]
        with ThreadPoolExecutor(a.workers) as ex:
            for msg in ex.map(lambda j: train_one(*j), jobs):
                print(msg, flush=True)
    else:
        jobs = [(arm, ov, s, suite) for arm, ov in ARMS for s in SEEDS for suite in ("old", "v2b")]
        with ThreadPoolExecutor(a.workers) as ex:
            for msg in ex.map(lambda j: stageb_one(*j), jobs):
                print(msg, flush=True)
        # merge into summary_multiseed.json
        merged = {}
        for arm, _ov in ARMS:
            merged[arm] = {}
            for s in SEEDS:
                d = {}
                for suite in ("old", "v2b"):
                    p = os.path.join(RUN, "results", "v25m_arms", arm, f"seed{s}",
                                     f"stageb_{suite}", "stage_b.json")
                    if os.path.exists(p):
                        j = json.load(open(p))
                        d[suite] = {"survival_by_kind": j["survival_by_kind"],
                                    "learner_action_profile": j["learner_action_profile"],
                                    "learner_loot_mean": j.get("learner_loot_mean"),
                                    "learner_kills_mean": j.get("learner_kills_mean")}
                merged[arm][str(s)] = d
        os.makedirs(os.path.join(RUN, "results", "v25m_arms"), exist_ok=True)
        json.dump(merged, open(os.path.join(RUN, "results", "v25m_arms", "summary_multiseed.json"), "w"),
                  indent=2, ensure_ascii=False)
        print("-> summary_multiseed.json", flush=True)

if __name__ == "__main__":
    main()
