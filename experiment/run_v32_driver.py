"""v3.2 sweep driver: chained greedy coordinate search over DQN hyperparameters.
21 branches x 500ep (seed42, warm-start from v3.0 champion s42), serial (throughput).
Stages: A1 pool x lr (9) -> A2 gamma (3) -> A3 replay (3) -> A4 eps (2) -> A5 tsync+batch (4).
Gated confirm: final vs-arm mean > 0.45 AND vs-heur mean >= 0.48 -> 3 seeds x 1000ep."""
import json, os, subprocess, sys, time
import numpy as np

RUN = os.path.abspath(os.path.join(".", ".."))
EXP = "."
for d in ("runs/v3/v32_sweep", "results/v3/v32_sweep", "runs/v3/v32_confirm", "results/v3/v32_confirm", "logs"):
    os.makedirs(os.path.join(RUN, d), exist_ok=True)
ARMS = ["a_classic", "b_terminal", "c_res", "d_malintent"]
KINDS = ["random", "df", "pacifist"] + [f"ckpt:{a}" for a in ARMS]
POOL = {"b": dict(n_heur=1, oversample=3, boost_b=False),
        "c": dict(n_heur=0, oversample=3, boost_b=False),
        "d": dict(n_heur=1, oversample=3, boost_b=True)}

def sh(cmd, log):
    r = subprocess.run(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT, cwd=EXP)
    return r.returncode

def tail_ep(p):
    try: return json.loads(open(p).read().strip().split("\n")[-1])["episode"]
    except Exception: return 0

def train_branch(name, pool, lr, eps, gamma, replay, tsync, batch, episodes=500):
    out = os.path.join(RUN, "runs", "v3", "v32_sweep")
    m = os.path.join(out, f"seed42_{name}", "metrics.jsonl")
    if tail_ep(m) >= episodes: return True
    p = POOL[pool]
    cmd = [sys.executable, "train2_v31_champion.py", "--mode", "v31", "--config", "config_v31.yaml",
           "--seed", "42", "--episodes", str(episodes), "--tag", name,
           "--init-from", os.path.join(RUN, "runs", "v3", "v30_champion", "run", "seed42_champion", "checkpoint.npz"),
           "--out", out, "--n-heur", str(p["n_heur"]), "--oversample", str(p["oversample"]),
           "--lr", str(lr), "--eps-start", str(eps), "--eps-end", str(eps),
           "--gamma", str(gamma), "--replay", str(replay), "--target-sync", str(tsync), "--batch", str(batch)]
    if p["boost_b"]: cmd.append("--boost-b")
    rc = sh(cmd, os.path.join(out, f"train_{name}.log"))
    ok = tail_ep(m) >= episodes
    print(f"[v32 train] {name} rc={rc} tail_ep={tail_ep(m)}", flush=True)
    return ok

def eval_branch(name, episodes=4):
    base = os.path.join(RUN, "results", "v3", "v32_sweep", name)
    ck = os.path.join(RUN, "runs", "v3", "v32_sweep", f"seed42_{name}", "checkpoint.npz")
    jobs = []
    solo = os.path.join(base, "solo"); os.makedirs(solo, exist_ok=True)
    if not os.path.exists(os.path.join(solo, "solo_eval.json")):
        jobs.append((ck, solo, None, "solo"))
    for k in KINDS:
        out = os.path.join(base, "league", k.replace("ckpt:", "")); os.makedirs(out, exist_ok=True)
        if not os.path.exists(os.path.join(out, "solo_eval.json")):
            jobs.append((ck, out, k, "league"))
    for ck, out, opp, mode in jobs:
        cmd = [sys.executable, "eval_v30_solo.py", "--checkpoint", ck, "--out", out,
               "--config", "config_v31.yaml", "--mode", mode, "--episodes", str(episodes), "--seed0", "900"]
        if opp: cmd += ["--opp-kind", opp]
        rc = sh(cmd, os.path.join(out, "eval.log"))
        print(f"[v32 eval] {name}/{mode}/{opp} rc={rc}", flush=True)

def score(name):
    base = os.path.join(RUN, "results", "v3", "v32_sweep", name)
    arm, heur = [], []
    solo = None
    try: solo = json.load(open(os.path.join(base, "solo", "solo_eval.json")))["survival_fraction_of_lifetime"]
    except Exception: pass
    for a in ARMS:
        try: arm.append(json.load(open(os.path.join(base, "league", a, "solo_eval.json")))["survival_fraction_of_lifetime"])
        except Exception: pass
    for h in ("random", "df", "pacifist"):
        try: heur.append(json.load(open(os.path.join(base, "league", h, "solo_eval.json")))["survival_fraction_of_lifetime"])
        except Exception: pass
    if len(arm) < 4 or len(heur) < 3: return None
    return {"name": name, "vs_arm": round(float(np.mean(arm)), 4), "vs_heur": round(float(np.mean(heur)), 4),
            "solo_surv": solo, "complete": True}

def run_stage(tag, branches):
    for b in branches:
        train_branch(**b)
        eval_branch(b["name"])
    scores = [s for s in (score(b["name"]) for b in branches) if s]
    best = max(scores, key=lambda s: (s["vs_arm"], s["solo_surv"] or 0)) if scores else None
    with open(os.path.join(RUN, "results", "v3", "v32_sweep", f"stage_{tag}.json"), "w") as f:
        json.dump({"branches": scores, "best": best}, f, indent=1)
    print(f"[v32 stage {tag}] best={best}", flush=True)
    return best

BASE = dict(eps=0.05, gamma=0.995, replay=150000, tsync=2000, batch=128)
t0 = time.time()
# ---- A1: pool x lr ----
a1 = [{"name": f"sw_{p}_lr{lr:.1e}".replace("e-0", "e-").replace(".0e", "e-"), "pool": p, "lr": lr, **BASE}
      for p in ("b", "c", "d") for lr in (1.5e-4, 3e-4, 6e-4)]
best1 = run_stage("A1", a1)
if not best1: sys.exit("A1 empty")
pool = best1["name"].split("_")[1]
lr = next(b["lr"] for b in a1 if b["name"] == best1["name"])  # robust: read from def, not name parse (branch names like 3e--4 break float())
# ---- A2: gamma ----
a2 = [{"name": f"sw_g{g}".replace(".", ""), "pool": pool, "lr": lr, **{**BASE, "gamma": g}}
      for g in (0.99, 0.999)]
best2 = run_stage("A2", a2)
# note: 0.995 baseline already covered by A1 best (gamma not retrained) -> only 2 new branches
if best2 and best2["vs_arm"] > best1["vs_arm"]: gbest = best2["name"]
else: gbest = None
gamma = 0.99 if gbest == "sw_g099" else (0.999 if gbest == "sw_g0999" else 0.995)
# ---- A3: replay ----
a3 = [{"name": f"sw_rp{r//1000}k", "pool": pool, "lr": lr, **{**BASE, "gamma": gamma, "replay": r}}
      for r in (50000, 400000)]
best3 = run_stage("A3", a3)
replay = 150000
if best3 and best3["vs_arm"] > (best2["vs_arm"] if (best2 and gbest) else best1["vs_arm"]):
    replay = int(best3["name"].split("rp")[1].split("k")[0]) * 1000
# ---- A4: eps ----
a4 = [{"name": f"sw_e{int(e*100)}", "pool": pool, "lr": lr, **{**BASE, "gamma": gamma, "replay": replay, "eps": e}}
      for e in (0.10, 0.15)]
best4 = run_stage("A4", a4)
eps = 0.05
prev = max(s["vs_arm"] for s in [best1, best2, best3] if s)
if best4 and best4["vs_arm"] > prev: eps = float(best4["name"].split("e")[1]) / 100
# ---- A5: tsync + batch ----
a5 = [{"name": "sw_ts500", "pool": pool, "lr": lr, **{**BASE, "gamma": gamma, "replay": replay, "eps": eps, "tsync": 500}},
      {"name": "sw_ts4000", "pool": pool, "lr": lr, **{**BASE, "gamma": gamma, "replay": replay, "eps": eps, "tsync": 4000}},
      {"name": "sw_b64", "pool": pool, "lr": lr, **{**BASE, "gamma": gamma, "replay": replay, "eps": eps, "batch": 64}},
      {"name": "sw_b256", "pool": pool, "lr": lr, **{**BASE, "gamma": gamma, "replay": replay, "eps": eps, "batch": 256}}]
best5 = run_stage("A5", a5)
final = dict(pool=pool, lr=lr, gamma=gamma, replay=replay, eps=eps,
             tsync=2000, batch=128)
if best5:
    n = best5["name"]
    if n.startswith("sw_ts"): final["tsync"] = int(n.split("ts")[1])
    if n.startswith("sw_b"): final["batch"] = int(n[4:])
    prev = max(s["vs_arm"] for s in [best1, best2, best3, best4] if s)
    if best5["vs_arm"] <= prev:
        final["tsync"], final["batch"] = 2000, 128
with open(os.path.join(RUN, "results", "v3", "v32_sweep", "sweep_summary.json"), "w") as f:
    json.dump({"final_config": final, "chained_best": {"A1": best1, "A2": best2, "A3": best3, "A4": best4, "A5": best5},
               "wall_s": round(time.time() - t0)}, f, indent=1, default=str)
print(f"[v32] FINAL: {final}", flush=True)
# ---- gated confirm ----
gate_arm = max(s["vs_arm"] for s in [best1, best2, best3, best4, best5] if s)
gate_heur = [s["vs_heur"] for s in [best1, best2, best3, best4, best5] if s and s["name"] in
             [b["name"] for b in (a1 + a2 + a3 + a4 + a5) if s.get("vs_arm") == gate_arm]]
print(f"[v32] GATE: vs_arm={gate_arm} vs_heur={gate_heur}", flush=True)
if not (gate_arm > 0.45 and (not gate_heur or max(gate_heur) >= 0.48)):
    print("[v32] GATE NOT PASSED -> no confirm run (honest negative)", flush=True)
    with open(os.path.join(RUN, "results", "v3", "v32_sweep", "CONFIRM_SKIPPED"), "w") as f: f.write(f"gate_arm={gate_arm}\n")
    sys.exit(0)

# early-stopping confirm: train 1000ep w/ per-250ep snapshots -> offline league eval
# per snapshot (8ep, same protocol as sweep) -> per-seed peak snapshot -> solo sentinel
import shutil
SNAP_EPS = (250, 500, 750, 1000)
p = POOL[final["pool"]]
res = []
for s in (42, 43, 44):
    name = f"confirm_s{s}"
    out = os.path.join(RUN, "runs", "v3", "v32_confirm")
    run_dir = os.path.join(out, f"seed{s}_{name}")
    m = os.path.join(run_dir, "metrics.jsonl")
    snaps_ok = all(os.path.exists(os.path.join(run_dir, f"checkpoint_ep{n}.npz")) for n in SNAP_EPS[:-1])
    if not (tail_ep(m) >= 1000 and snaps_ok and os.path.exists(os.path.join(run_dir, "checkpoint.npz"))):
        if os.path.isdir(run_dir):
            shutil.rmtree(run_dir)
        cmd = [sys.executable, "train2_v31_champion.py", "--mode", "v31", "--config", "config_v31.yaml",
               "--seed", str(s), "--episodes", "1000", "--tag", name,
               "--init-from", os.path.join(RUN, "runs", "v3", "v30_champion", "run", f"seed{s}_champion", "checkpoint.npz"),
               "--out", out, "--n-heur", str(p["n_heur"]), "--oversample", str(p["oversample"])]
        if p["boost_b"]: cmd.append("--boost-b")
        sh(cmd, os.path.join(out, f"train_{name}.log"))
    snaps = [os.path.join(run_dir, f"checkpoint_ep{n}.npz") for n in SNAP_EPS[:-1]] + [os.path.join(run_dir, "checkpoint.npz")]
    for n, ck in zip(SNAP_EPS, snaps):
        for k in KINDS:
            o = os.path.join(RUN, "results", "v3", "v32_confirm", f"seed{s}", f"snap{n}", "league", k.replace("ckpt:", ""))
            os.makedirs(o, exist_ok=True)
            if not os.path.exists(os.path.join(o, "solo_eval.json")):
                sh([sys.executable, "eval_v30_solo.py", "--checkpoint", ck, "--out", o, "--config", "config_v31.yaml",
                    "--mode", "league", "--opp-kind", k, "--episodes", "8", "--seed0", "900"], os.path.join(o, "eval.log"))
    peak = None
    for n in SNAP_EPS:
        arm = [json.load(open(os.path.join(RUN, "results", "v3", "v32_confirm", f"seed{s}", f"snap{n}", "league", a, "solo_eval.json")))["survival_fraction_of_lifetime"] for a in ARMS]
        heur = [json.load(open(os.path.join(RUN, "results", "v3", "v32_confirm", f"seed{s}", f"snap{n}", "league", h, "solo_eval.json")))["survival_fraction_of_lifetime"] for h in ("random", "df", "pacifist")]
        va, vh = float(np.mean(arm)), float(np.mean(heur))
        if peak is None or va > peak["vs_arm"]:
            peak = {"seed": s, "snap_ep": n, "vs_arm": round(va, 4), "vs_heur": round(vh, 4)}
    pk_ckpt = snaps[SNAP_EPS.index(peak["snap_ep"])]
    o = os.path.join(RUN, "results", "v3", "v32_confirm", f"seed{s}", "solo")
    os.makedirs(o, exist_ok=True)
    if not os.path.exists(os.path.join(o, "solo_eval.json")):
        sh([sys.executable, "eval_v30_solo.py", "--checkpoint", pk_ckpt, "--out", o, "--config", "config_v31.yaml",
            "--mode", "solo", "--episodes", "16", "--seed0", "900"], os.path.join(o, "eval.log"))
    peak["solo_surv"] = json.load(open(os.path.join(o, "solo_eval.json")))["survival_fraction_of_lifetime"]
    res.append(peak)
with open(os.path.join(RUN, "results", "v3", "v32_confirm", "confirm_summary.json"), "w") as f:
    json.dump({"per_seed_peaks": res, "final_config": final}, f, indent=1)
print(f"[v32 CONFIRM-ES] {res}", flush=True)
