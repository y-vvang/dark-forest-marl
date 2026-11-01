#!/bin/bash
# v3.0 pipeline (user intent 2026-10-16): solo evals + flagship champion.
RUN=/home/z/my-project/output/experiment-suite/does-the-dark-forest-strategy-emerge-as-ecec922e/2026-09-04_221544
LOG=$RUN/logs/pipeline_v30.log
echo "PIPELINE_V26 START $(date)" >> $LOG
cd $RUN/experiment
exec 9>$RUN/logs/pipeline_v30.lock
flock -n 9 || { echo "V26_LOCKED_SKIP" >> $LOG; exit 0; }
# wait for the v2.5.1 fix pipeline (2nd DONE marker), timeout 2.5h
TO=0
until [ "$(grep -c PIPELINE_V25M_DONE $RUN/logs/pipeline_v25m.log 2>/dev/null)" -ge 2 ]; do
  sleep 120; TO=$((TO+1)); [ $TO -ge 75 ] && { echo "V26_WAIT_TIMEOUT" >> $LOG; exit 1; }
done
echo "V26_FIX_DETECTED $(date)" >> $LOG
# stage 1: solo evals of 12 arm checkpoints
python3 -u run_v30_driver.py --stage solo25m > $RUN/logs/v30_solo25m.log 2>&1
echo "V26_SOLO25M_EXIT=$? $(date)" >> $LOG
# stage 2: champion training (3 seeds x 1000ep, league mix)
python3 -u run_v30_driver.py --stage champion_train > $RUN/logs/v30_champ_train.log 2>&1
echo "V26_CHAMP_TRAIN_EXIT=$? $(date)" >> $LOG
# stage 3: champion solo + league evals
python3 -u run_v30_driver.py --stage champion_eval > $RUN/logs/v30_champ_eval.log 2>&1
echo "V26_CHAMP_EVAL_EXIT=$? $(date)" >> $LOG
# stage 4: aggregation + figures
python3 -u analysis_v30_solo.py > $RUN/logs/analysis_v30.log 2>&1
echo "V26_ANALYSIS_EXIT=$? $(date)" >> $LOG
python3 -u plot_v30m.py > $RUN/logs/plot_v30.log 2>&1
echo "V26_PLOT_EXIT=$? $(date)" >> $LOG
echo "PIPELINE_V26_DONE $(date)" >> $LOG
