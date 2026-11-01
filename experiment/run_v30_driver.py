#!/usr/bin/env python3
"""v3.0 pipeline driver: (1) solo-eval all 12 v2.5.1 arm checkpoints,
(2) champion league training (3 seeds x 1000ep), (3) champion solo+league eval.
Idempotent per unit (output json / metrics-tail)."""
import argparse, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.dirname(HERE)
ARMS = ["a_classic", "b_terminal", "c_res", "d_malintent"]
SEEDS = [42, 43, 44]

def sh(cmd, logp):
    os.makedirs(os.path.dirname(logp), exist_ok=True)
    r = subprocess.run(cmd, env={**os.environ, "OMP_NUM_THREADS": "1"}, cwd=HERE,
                       stdout=open(logp, "w"), stderr=subprocess.STDOUT, text=True)
    return r.returncode, (open(logp).read()[-500:] if r.returncode else "")

def solo_eval(ck, out, logp, episodes=16, mode="solo", opp=None):
    if os.path.exists(os.path.join(out, "solo_eval.json")):
        return f"[eval {out.split('/')[-2:]}] skip"
    cmd = [sys.executable, "eval_v30_solo.py", "--checkpoint", ck, "--out", out,
           "--episodes", str(episodes), "--mode", mode]
    if opp: cmd += ["--opp-kind", opp]
    rc, tail = sh(cmd, logp)
    return f"[eval {out.split('results/')[-1]}] EXIT={rc}" + ("" if rc == 0 else " TAIL:" + tail)

def stage_solo25m():
    jobs = []
    for arm in ARMS:
        for s in SEEDS:
            ck = os.path.join(RUN, "runs", "v25m_arms", arm, "run", f"seed{s}_main", "checkpoint.npz")
            out = os.path.join(RUN, "results", "v30_solo", arm, f"seed{s}")
            jobs.append((ck, out, os.path.join(out, "eval.log")))
    with ThreadPoolExecutor(2) as ex:
        for m in ex.map(lambda j: solo_eval(*j), jobs):
            print(m, flush=True)

def train_done(runs, seed, need, tag="main"):
    mp = os.path.join(runs, "run", f"seed{seed}_{tag}", "metrics.jsonl")
    if not os.path.exists(mp): return False
    last = None
    with open(mp) as f:
        for line in f:
            if line.strip(): last = line
    try: return json.loads(last)["episode"] >= need
    except Exception: return False

def champion_train(seed):
    runs = os.path.join(RUN, "runs", "v3", "v30_champion")
    if train_done(runs, seed, 1000, "champion"):
        return f"[champ s{seed}] skip"
    rc, tail = sh([sys.executable, "train2_v30_champion.py", "--seed", str(seed),
                   "--mode", "champion", "--out", os.path.join(runs, "run"),
                   "--config", os.path.join(HERE, "config_v25.yaml"), "--episodes", "1000"],
                  os.path.join(runs, f"train_seed{seed}.log"))
    return f"[champ s{seed}] TRAIN_EXIT={rc}" + ("" if rc == 0 else " TAIL:" + tail)

def champion_eval(seed):
    ck = os.path.join(RUN, "runs", "v3", "v30_champion", "run", f"seed{seed}_champion", "checkpoint.npz")
    if not os.path.exists(ck):
        return f"[champ s{seed}] no ckpt, skip"
    msgs = [solo_eval(ck, os.path.join(RUN, "results", "v3", "v30_champion", f"seed{seed}", "solo"),
                      os.path.join(RUN, "results", "v3", "v30_champion", f"seed{seed}", "eval.log"), episodes=24)]
    jobs = []
    for kind in ["random", "df", "pacifist", "ckpt:a_classic", "ckpt:b_terminal", "ckpt:c_res", "ckpt:d_malintent"]:
        out = os.path.join(RUN, "results", "v3", "v30_champion", f"seed{seed}", "league", kind.replace("ckpt:", ""))
        jobs.append((ck, out, os.path.join(out, "eval.log"), 8, "league", kind))
    with ThreadPoolExecutor(2) as ex:
        for m in ex.map(lambda j: solo_eval(*j), jobs):
            msgs.append(m)
    return "\n".join(msgs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["solo25m", "champion_train", "champion_eval"])
    a = ap.parse_args()
    if a.stage == "solo25m": stage_solo25m()
    elif a.stage == "champion_train":
        with ThreadPoolExecutor(2) as ex:
            for m in ex.map(champion_train, SEEDS): print(m, flush=True)
    else:
        with ThreadPoolExecutor(3) as ex:
            for m in ex.map(champion_eval, SEEDS): print(m, flush=True)

if __name__ == "__main__":
    main()
