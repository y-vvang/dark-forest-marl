#!/bin/bash
# v2.5 async pipeline: wait for the in-flight four-arm driver, then StageB-v2b,
# analysis, figures. Idempotent — re-running skips completed stages.
RUN=/home/z/my-project/output/experiment-suite/does-the-dark-forest-strategy-emerge-as-ecec922e/2026-09-04_221544
LOG=$RUN/logs/pipeline_v25.log
echo "PIPELINE_V25 START $(date)" >> $LOG
# anti double-run: wait for the in-flight run_v25_arms.py driver to finish
while pgrep -f "run_v25_arms.py" > /dev/null; do sleep 20; done
echo "ARMS_WAIT_DONE $(date)" >> $LOG
# stage A: four arms completeness (re-fire idempotently if the driver died mid-way)
cd $RUN/experiment
python3 -u run_v25_arms.py >> $RUN/logs/v25_arms.log 2>&1
echo "V25_ARMS_EXIT=$? $(date)" >> $LOG
# stage B: v2b heuristic regression gate
python3 -u smoke_v25b.py > $RUN/logs/smoke_v25b.log 2>&1
SM=$?
echo "SMOKEB_EXIT=$SM $(date)" >> $LOG
if [ $SM -ne 0 ]; then echo "PIPELINE_V25_ABORTED_AT_SMOKE $(date)" >> $LOG; exit 1; fi
# stage C: StageB with v2b heuristics, all four arms (idempotent per arm)
python3 -u run_v25_stageb_h2.py >> $RUN/logs/stageb_h2.log 2>&1
echo "STAGEB_H2_EXIT=$? $(date)" >> $LOG
# stage D: profiles + convergence analysis
python3 -u analysis_v25.py >> $RUN/logs/analysis_v25.log 2>&1
echo "ANALYSIS_EXIT=$? $(date)" >> $LOG
# stage E: figures
python3 -u plot_v25.py >> $RUN/logs/plot_v25.log 2>&1
echo "PLOT_EXIT=$? $(date)" >> $LOG
echo "PIPELINE_V25_DONE $(date)" >> $LOG
