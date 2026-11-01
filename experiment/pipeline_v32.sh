#!/bin/bash
# v3.2 sweep wrapper: flock + markers; body = run_v32_driver.py (chained 19-branch sweep + gated confirm)
set -u
cd "$(dirname "$0")"
RUN=$(cd .. && pwd)
exec 9>"$RUN/logs/pipeline_v32.lock"; flock -n 9 || { echo "[v32] lock held"; exit 0; }
L="$RUN/logs/pipeline_v32.log"
echo "[$(date '+%F %T')] PIPELINE_V28 START" >> "$L"
python3 run_v32_driver.py >> "$L" 2>&1
echo "V28_BODY_EXIT=$? [$T]" >> "$L" 2>/dev/null || echo "V28_BODY_EXIT=$?" >> "$L"
echo "[$(date '+%F %T')] PIPELINE_V28_DONE" >> "$L"
