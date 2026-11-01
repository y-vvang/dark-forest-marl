#!/bin/bash
# v2.5 multi-seed pipeline (user order 2026-10-15 20:23): 4 arms x 3 seeds x
# 1000ep + StageB double suite + aggregation + figures. Fully idempotent.
RUN=/home/z/my-project/output/experiment-suite/does-the-dark-forest-strategy-emerge-as-ecec922e/2026-09-04_221544
LOG=$RUN/logs/pipeline_v25m.log
echo "PIPELINE_V25M START $(date)" >> $LOG
# anti double-run: flock (pgrep self-match proved unreliable 2026-10-16)
exec 9>$RUN/logs/pipeline_v25m.lock
flock -n 9 || { echo "LOCKED_SKIP $(date)" >> $LOG; exit 0; }
cd $RUN/experiment
# stage 1: 12 trainings, 2-way parallel (idempotent per checkpoint)
python3 -u run_v25_multiseed.py --stage train > $RUN/logs/v25m_train.log 2>&1
echo "V25M_TRAIN_EXIT=$? $(date)" >> $LOG
# stage 2: 24 StageB (double suite), 2-way parallel (idempotent per stage_b.json)
python3 -u run_v25_multiseed.py --stage stageb > $RUN/logs/v25m_stageb.log 2>&1
echo "V25M_STAGEB_EXIT=$? $(date)" >> $LOG
# stage 3: multi-seed aggregation
python3 -u analysis_v25m.py > $RUN/logs/analysis_v25m.log 2>&1
echo "V25M_ANALYSIS_EXIT=$? $(date)" >> $LOG
# stage 4: figures
python3 -u plot_v25m.py > $RUN/logs/plot_v25m.log 2>&1
echo "V25M_PLOT_EXIT=$? $(date)" >> $LOG
echo "PIPELINE_V25M_DONE $(date)" >> $LOG
