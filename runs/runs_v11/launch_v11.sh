#!/usr/bin/env bash
# Launch all v1.1 training runs in two parallel streams (one python per vCPU).
# Usage: bash launch_v11.sh
set -u
RUN="$(cd "$(dirname "$0")/.." && pwd)"
cd "$RUN/experiment"
export OMP_NUM_THREADS=1

stream() {  # $1 = log tag, rest = "seed ablation" pairs
  local tag="$1"; shift
  for spec in "$@"; do
    set -- $spec
    echo "[$(date +%H:%M:%S)] stream$tag start seed$1 $2" >> "$RUN/runs/stream$tag.log"
    python3 train.py --seed "$1" --ablation "$2" --out ../runs >> "$RUN/runs/stream$tag.log" 2>&1
    echo "[$(date +%H:%M:%S)] stream$tag done  seed$1 $2" >> "$RUN/runs/stream$tag.log"
  done
}

stream A "42 main" "43 main" "44 main" "42 no_matter_scarcity" "43 no_matter_scarcity" "44 no_matter_scarcity" &
PA=$!
stream B "42 no_tech_explosion" "43 no_tech_explosion" "44 no_tech_explosion" "42 costly_strike" "43 costly_strike" "44 costly_strike" &
PB=$!
wait $PA $PB
echo "ALL_V11_RUNS_DONE" >> "$RUN/runs/streamA.log"
