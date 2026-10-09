#!/bin/bash
# v2.3 post-conversation pipeline: finish sweep -> Stage B -> analysis.
# Self-chaining; safe to re-run (every stage is resumable / idempotent).
RUN=/home/z/my-project/output/experiment-suite/does-the-dark-forest-strategy-emerge-as-ecec922e/2026-09-04_221544
LOG=$RUN/pipeline_v23.log
echo "PIPELINE START $(date)" >> $LOG

# 0) wait for any already-running sweep to finish (avoid double-running)
while pgrep -f "sweep_v23.py" > /dev/null; do sleep 30; done
echo "SWEEP_SLOT_FREE $(date)" >> $LOG

# 1) sweep to completion (resumable: skips arms already recorded)
cd $RUN/experiment
OMP_NUM_THREADS=1 python3 -u sweep_v23.py >> $RUN/sweep_v23_resume.log 2>&1
echo "SWEEP_EXIT=$? $(date)" >> $LOG

# 2) Stage B: pool evaluation of the v2.3 learner (150 eps)
OMP_NUM_THREADS=1 python3 -u stage_b.py --run-dir ../runs_v23/seed42_main \
  --episodes 150 --out ../results_v23 --config config_v23.yaml >> $LOG 2>&1
echo "STAGEB_EXIT=$? $(date)" >> $LOG

# 3) analysis: profiles + binned analyses (3 worlds x 40 eps)
OMP_NUM_THREADS=1 python3 -u analysis_v23.py >> $LOG 2>&1
echo "ANALYSIS_EXIT=$? $(date)" >> $LOG

echo "PIPELINE_DONE $(date)" >> $LOG
