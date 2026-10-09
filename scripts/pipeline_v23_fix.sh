#!/bin/bash
# v2.3-FIX pipeline (strike-kill bug fixed in env2.py 2026-09-22): full rerun.
# Stages are idempotent; re-running this script skips completed work.
RUN=/home/z/my-project/output/experiment-suite/does-the-dark-forest-strategy-emerge-as-ecec922e/2026-09-04_221544
LOG=$RUN/logs/pipeline_v23_fix.log
echo "PIPELINE_FIX START $(date)" >> $LOG

# guard: no duplicate fix-pipeline
while pgrep -f "pipeline_v23_fix.sh" | grep -v $$ > /dev/null; do sleep 30; done

cd $RUN/experiment
# 2026-09-23 repo reorg: runs_v23_fix -> runs/v23_fix, results_v23_fix -> results/v23_fix
mkdir -p ../runs/v23_fix ../results/v23_fix

# 1) headline: main training 600 eps (fixed env)
if [ ! -f ../runs/v23_fix/seed42_main/checkpoint.npz ]; then
  OMP_NUM_THREADS=1 python3 -u train2.py --seed 42 --mode main \
    --out ../runs/v23_fix --config config_v23.yaml --episodes 600 >> $LOG 2>&1
fi
echo "TRAIN_EXIT=$? $(date)" >> $LOG

# 2) Stage B: frozen policy into mixed pool (150 eps)
if [ ! -f ../results/v23_fix/stage_b.json ]; then
  OMP_NUM_THREADS=1 python3 -u stage_b.py --run-dir ../runs/v23_fix/seed42_main \
    --episodes 150 --out ../results/v23_fix --config config_v23.yaml >> $LOG 2>&1
fi
echo "STAGEB_EXIT=$? $(date)" >> $LOG

# 3) analysis: profiles + bins (3 worlds x 40 eps)
if [ ! -f ../results/v23_fix/analysis.json ]; then
  OMP_NUM_THREADS=1 python3 -u analysis_v23.py >> $LOG 2>&1
fi
echo "ANALYSIS_EXIT=$? $(date)" >> $LOG

# 4) sweep: 12 arms incl base_control (resumable, skips recorded arms)
OMP_NUM_THREADS=1 python3 -u sweep_v23.py >> $RUN/logs/sweep_v23_fix.log 2>&1
echo "SWEEP_EXIT=$? $(date)" >> $LOG

echo "PIPELINE_FIX_DONE $(date)" >> $LOG
