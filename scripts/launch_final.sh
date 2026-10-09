#!/usr/bin/env bash
# Launch v1.3 (12 runs) + v1.2 unified rerun (3 runs) in two parallel streams.
# Spec format: seed ablation out config
set -u
RUN="$(cd "$(dirname "$0")/.." && pwd)"
cd "$RUN/experiment"
export OMP_NUM_THREADS=1

stream() {
  local tag="$1"; shift
  for spec in "$@"; do
    set -- $spec
    echo "[$(date +%H:%M:%S)] stream$tag start seed$1 $2 -> $3" >> "$RUN/runs/stream$tag.log"
    python3 train.py --seed "$1" --ablation "$2" --out "$3" --config "$4" >> "$RUN/runs/stream$tag.log" 2>&1
    echo "[$(date +%H:%M:%S)] stream$tag done  seed$1 $2 -> $3" >> "$RUN/runs/stream$tag.log"
  done
}

stream A \
  "42 main ../runs config.yaml" \
  "43 main ../runs config.yaml" \
  "44 main ../runs config.yaml" \
  "42 no_black_domain ../runs config.yaml" \
  "43 no_black_domain ../runs config.yaml" \
  "44 no_black_domain ../runs config.yaml" \
  "42 main ../runs/v12_rerun config_v12.yaml" \
  "43 main ../runs/v12_rerun config_v12.yaml" &

stream B \
  "42 no_tech_explosion ../runs config.yaml" \
  "43 no_tech_explosion ../runs config.yaml" \
  "44 no_tech_explosion ../runs config.yaml" \
  "42 costly_strike ../runs config.yaml" \
  "43 costly_strike ../runs config.yaml" \
  "44 costly_strike ../runs config.yaml" \
  "44 main ../runs/v12_rerun config_v12.yaml" &
wait
echo "ALL_RUNS_DONE" >> "$RUN/runs/streamA.log"
