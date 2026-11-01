#!/bin/bash
# v3.1 real-flagship curriculum stage-2 (warm start from v3.0 champion + 80% arm oversampling)
# idempotent: train done = metrics tail episode>=1000; eval done = solo_eval.json exists
set -u
cd "$(dirname "$0")"
RUN=$(cd .. && pwd)
exec 9>"$RUN/logs/pipeline_v31.lock"; flock -n 9 || { echo "[v31] another instance holds the lock"; exit 0; }
L="$RUN/logs/pipeline_v31.log"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$L"; }

train_done(){ python3 - "$1" <<'PY'
import sys, json
try:
    last = open(sys.argv[1]).read().strip().split("\n")[-1]
    print(1 if json.loads(last)["episode"] >= 1000 else 0)
except Exception: print(0)
PY
}

mkdir -p "$RUN/runs/v3/v31_champion" "$RUN/results/v3/v31_champion" "$RUN/logs"
log "PIPELINE_V27 START"
SEEDS="42 43 44"
# ---- stage 1: curriculum training (batch1: 42+43 parallel, batch2: 44) ----
PIDS=""
for s in 42 43; do
  M="$RUN/runs/v3/v31_champion/run/seed${s}_v31/metrics.jsonl"
  [ "$(train_done "$M")" = "1" ] && { log "V27 s$s train skip (done)"; continue; }
  ( python3 train2_v31_champion.py --mode v31 --config config_v31.yaml --seed $s \
      --init-from "$RUN/runs/v3/v30_champion/run/seed${s}_champion/checkpoint.npz" \
      --out "$RUN/runs/v3/v31_champion/run" > "$RUN/runs/v3/v31_champion/train_seed$s.log" 2>&1 \
      && echo "V27_TRAIN_${s}_EXIT=0" || echo "V27_TRAIN_${s}_EXIT=1" ) >> "$L" 2>&1 &
  PIDS="$PIDS $!"
done
for p in $PIDS; do wait $p; done
M="$RUN/runs/v3/v31_champion/run/seed44_v31/metrics.jsonl"
if [ "$(train_done "$M")" != "1" ]; then
  python3 train2_v31_champion.py --mode v31 --config config_v31.yaml --seed 44 \
    --init-from "$RUN/runs/v3/v30_champion/run/seed44_champion/checkpoint.npz" \
    --out "$RUN/runs/v3/v31_champion/run" > "$RUN/runs/v3/v31_champion/train_seed44.log" 2>&1
  echo "V27_TRAIN_44_EXIT=$?" >> "$L"; log "V27 s44 train done"
else log "V27 s44 train skip (done)"; fi
grep -q "V27_TRAIN_42_EXIT=1\|V27_TRAIN_43_EXIT=1\|V27_TRAIN_44_EXIT=1" "$L" && { log "V27_TRAIN_FAIL"; echo "PIPELINE_V27_DONE $(date '+%F %T')" >> "$L"; exit 1; }
log "V27_TRAIN_EXIT=0"

# ---- stage 2: eval (solo 16ep + league 7 kinds x 8ep, 6-way parallel) ----
python3 - <<'PY' >> "$L" 2>&1
import os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
RUN = os.path.dirname(os.path.dirname(os.path.abspath(".")))
RUN = os.path.abspath(os.path.join(".", ".."))
KINDS = ["random", "df", "pacifist", "ckpt:a_classic", "ckpt:b_terminal", "ckpt:c_res", "ckpt:d_malintent"]
jobs = []
for s in (42, 43, 44):
    ck = os.path.join(RUN, "runs", "v31_champion", "run", f"seed{s}_v31", "checkpoint.npz")
    solo = os.path.join(RUN, "results", "v31_champion", f"seed{s}", "solo")
    os.makedirs(solo, exist_ok=True)
    if not os.path.exists(os.path.join(solo, "solo_eval.json")):
        jobs.append((ck, solo, os.path.join(solo, "eval.log"), 16, "solo", None))
    for k in KINDS:
        out = os.path.join(RUN, "results", "v31_champion", f"seed{s}", "league", k.replace("ckpt:", ""))
        os.makedirs(out, exist_ok=True)
        if not os.path.exists(os.path.join(out, "solo_eval.json")):
            jobs.append((ck, out, os.path.join(out, "eval.log"), 8, "league", k))
def run(j):
    ck, out, logp, eps, mode, opp = j
    cmd = [sys.executable, "eval_v30_solo.py", "--checkpoint", ck, "--out", out,
           "--config", "config_v31.yaml", "--mode", mode, "--episodes", str(eps), "--seed0", "900"]
    if opp: cmd += ["--opp-kind", opp]
    r = subprocess.run(cmd, stdout=open(logp, "w"), stderr=subprocess.STDOUT, cwd=".")
    return f"{out} exit={r.returncode}"
with ThreadPoolExecutor(6) as ex:
    for m in ex.map(run, jobs): print("[v31 eval]", m, flush=True)
bad = sum(1 for j in jobs if not os.path.exists(os.path.join(j[1], "solo_eval.json")))
print(f"V27_EVAL_MISSING={bad}")
PY
grep -q "V27_EVAL_MISSING=0" "$L" || { log "V27_EVAL_FAIL"; echo "PIPELINE_V27_DONE $(date '+%F %T')" >> "$L"; exit 1; }
log "V27_EVAL_EXIT=0"

# ---- stage 3: analysis + plots ----
python3 analysis_v31.py >> "$L" 2>&1 && log "V27_ANALYSIS_EXIT=0" || { log "V27_ANALYSIS_EXIT=1"; echo "PIPELINE_V27_DONE $(date '+%F %T')" >> "$L"; exit 1; }
python3 plot_v31.py >> "$L" 2>&1 && log "V27_PLOT_EXIT=0" || { log "V27_PLOT_EXIT=1"; echo "PIPELINE_V27_DONE $(date '+%F %T')" >> "$L"; exit 1; }
log "PIPELINE_V27_DONE"
