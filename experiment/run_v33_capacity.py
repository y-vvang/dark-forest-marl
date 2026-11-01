"""v3.3 capacity-axis driver: hidden-width sweep on the v3.2 FINAL config.
Base config: pool=b lr6e-4 gamma0.995 replay150k eps.05 tsync500 batch128
(v3.2 FINAL, gate 0.4667 single-seed / ES-confirm peak mean 0.4132).
Baseline hidden [64,64] = v3.2 branch sw_ts500 (identical 8ep league protocol).
First wave: seeds [42] x hiddens [[128,128],[192,192]] x 500ep, terminal-ckpt
eval (comparable to sweep branches). Snapshots kept for later ES analysis.
Smoke mode (--smoke): 2ep training, 10ep snap interval, 1 kind x 1ep evals.
Idempotent: tail_ep + checkpoint existence gates; evals skip on solo_eval.json."""
import json, os, subprocess, sys

RUN = os.path.abspath(os.path.join(".", ".."))
EXP = "."
for d in ("runs/v3/v33_capacity", "results/v3/v33_capacity", "logs"):
    os.makedirs(os.path.join(RUN, d), exist_ok=True)
ARMS = ["a_classic", "b_terminal", "c_res", "d_malintent"]
KINDS = ["random", "df", "pacifist"] + [f"ckpt:{a}" for a in ARMS]
FINAL = dict(lr="6e-4", gamma="0.995", replay="150000", eps_start="0.05",
             eps_end="0.05", tsync="500", batch="128", n_heur="1", oversample="3")


def sh(cmd, log):
    return subprocess.run(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT, cwd=EXP).returncode


def tail_ep(p):
    try:
        return json.loads(open(p).read().strip().split("\n")[-1])["episode"]
    except Exception:
        return 0


def main(smoke=False):
    episodes = 2 if smoke else 500
    snap = 10 if smoke else 250
    hiddens = ["64,64"] if smoke else ["64,64", "128,128", "192,192"]
    eval_eps = "1" if smoke else "8"
    solo_eps = "1" if smoke else "16"
    kinds = KINDS[:1] if smoke else KINDS
    out = os.path.join(RUN, "runs", "v3", "v33_capacity")
    results = []
    for h in hiddens:
        tag = "cap_h" + h.replace(",", "x")
        run_dir = os.path.join(out, f"seed42_{tag}")
        m = os.path.join(run_dir, "metrics.jsonl")
        ck = os.path.join(run_dir, "checkpoint.npz")
        if tail_ep(m) < episodes or not os.path.exists(ck):
            cmd = [sys.executable, "train3_champion.py", "--mode", "v31", "--config", "config_v33.yaml",
                   "--seed", "42", "--episodes", str(episodes), "--tag", tag, "--hidden", h,
                   "--snap-every", str(snap),
                   "--out", out, "--n-heur", FINAL["n_heur"], "--oversample", FINAL["oversample"],
                   "--lr", FINAL["lr"], "--gamma", FINAL["gamma"], "--replay", FINAL["replay"],
                   "--target-sync", FINAL["tsync"], "--batch", FINAL["batch"],
                   "--eps-start", FINAL["eps_start"], "--eps-end", FINAL["eps_end"]]
            rc = sh(cmd, os.path.join(out, f"train_{tag}.log"))
            if rc != 0:
                print(f"[v30] TRAIN FAIL {tag} rc={rc}", flush=True)
                sys.exit(1)
        rows = {}
        for k in kinds:
            o = os.path.join(RUN, "results", "v3", "v33_capacity", tag, "league", k.replace("ckpt:", ""))
            os.makedirs(o, exist_ok=True)
            if not os.path.exists(os.path.join(o, "solo_eval.json")):
                rc = sh([sys.executable, "eval_v30_solo.py", "--checkpoint", ck, "--out", o,
                         "--config", "config_v33.yaml", "--mode", "league", "--opp-kind", k,
                         "--episodes", eval_eps, "--seed0", "900"], os.path.join(o, "eval.log"))
                if rc != 0:
                    print(f"[v30] EVAL FAIL {tag}/{k} rc={rc}", flush=True)
                    sys.exit(1)
            rows[k] = json.load(open(os.path.join(o, "solo_eval.json")))["survival_fraction_of_lifetime"]
        arm = [rows[f"ckpt:{a}"] for a in ARMS if f"ckpt:{a}" in rows]
        heur = [rows[x] for x in ("random", "df", "pacifist") if x in rows]
        rec = {"hidden": h, "tag": tag,
               "vs_arm": round(sum(arm) / len(arm), 4) if arm else None,
               "vs_heur": round(sum(heur) / len(heur), 4) if heur else None}
        o = os.path.join(RUN, "results", "v3", "v33_capacity", tag, "solo")
        os.makedirs(o, exist_ok=True)
        if not os.path.exists(os.path.join(o, "solo_eval.json")):
            sh([sys.executable, "eval_v30_solo.py", "--checkpoint", ck, "--out", o,
                "--config", "config_v33.yaml", "--mode", "solo", "--episodes", solo_eps,
                "--seed0", "900"], os.path.join(o, "eval.log"))
        rec["solo_surv"] = json.load(open(os.path.join(o, "solo_eval.json")))["survival_fraction_of_lifetime"]
        print(f"[v33 cap] {tag}: {rec}", flush=True)
        results.append(rec)
    base_path = os.path.join(RUN, "results", "v3", "v32_sweep", "stage_A5.json")
    base = json.load(open(base_path))
    branches = base if isinstance(base, list) else base.get("branches", [])
    ts500 = next((b for b in branches if b["name"] == "sw_ts500"), None)
    for r in results:
        r["delta_vs_arm_vs_h64"] = round(r["vs_arm"] - ts500["vs_arm"], 4) if (r["vs_arm"] and ts500) else None
    with open(os.path.join(RUN, "results", "v3", "v33_capacity", "capacity_summary.json"), "w") as f:
        json.dump({"baseline_h64": ts500, "capacity": results}, f, indent=1)
    print(f"[v33 CAPACITY] {results}", flush=True)


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
