#!/bin/bash
# v3.3 capacity-axis pipeline (flock-protected, idempotent).
# Usage: bash pipeline_v33.sh [--smoke]
set -u
cd "$(dirname "$0")"
L=../logs/pipeline_v30.log
exec 9>../logs/pipeline_v30.lock
flock -n 9 || { echo "[$(date '+%F %T')] [v30] lock held, exit"; exit 0; }
echo "[$(date '+%F %T')] PIPELINE_V30 START $*"
python3 run_v33_capacity.py "$@" >> "$L" 2>&1
rc=$?
echo "[$(date '+%F %T')] PIPELINE_V30 EXIT=$rc"
exit $rc
