#!/usr/bin/env bash
# Run a batch of evaluate.py jobs in two parallel streams.
# Usage: bash run_evals.sh <jobsfile>   (one job per line: policy ablation policy_seed [run_dir])
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/../experiment"
export OMP_NUM_THREADS=1
JOBS="$1"
LOG="$HERE/evals.log"
: > "$LOG"

run_job() {
  read -r policy ablation pseed rundir <<< "$1"
  extra=()
  if [ -n "${rundir:-}" ] && [ "$rundir" != "-" ]; then extra=(--run-dir "$rundir"); fi
  echo "[eval start] $policy $ablation s$pseed" >> "$LOG"
  python3 evaluate.py --policy "$policy" --ablation "$ablation" --policy-seed "$pseed" \
      "${extra[@]}" --out ../results >> "$LOG" 2>&1
}

n=0
while IFS= read -r line; do
  [ -z "$line" ] && continue
  if [ $((n % 2)) -eq 0 ]; then
    run_job "$line" &
  else
    ( sleep 1; run_job "$line" ) &
  fi
  n=$((n+1))
  while [ "$(jobs -rp | wc -l)" -ge 2 ]; do sleep 2; done
done < "$JOBS"
wait
echo "ALL_EVALS_DONE" >> "$LOG"
