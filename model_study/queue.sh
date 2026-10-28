#!/bin/bash
# Runs the queued experiments one after another, after the 2-D sigma-point study exits.
cd /home/kphawkins/dev/fbsde_traj_opt-ilqr-mppi
while pgrep -f "bazel-bin/examples/sigma_point_study" > /dev/null; do sleep 10; done
echo "STEP sigma_2d done $(date +%H:%M)" >> model_study/queue.log
B=bazel-bin/examples
R="leg6:rough=1e-6;leg6:rough=1e-5;leg8:rough=1e-7;leg8:rough=1e-6;leg8:rough=1e-5;leg8:rough=1e-4;leg10:rough=1e-6;leg10:rough=1e-5;poly6:rough=1e-6;poly8:rough=1e-6;spline7:rough=1e-6;rbf9:rough=1e-6"
(cd model_study && ../$B/value_model_study --phase dt --only integrator,pendulum --seeds 3 --time-budget 30 --models "$R" --csv dt_rough.csv > dt_rough.log 2>&1)
echo "STEP roughness done $(date +%H:%M) exit $?" >> model_study/queue.log
D="softmin4;softmin2;softmin8;poly2;poly4;poly6;leg4;rbf4;spline2;tanh32"
(cd model_study && ../$B/value_model_study --phase dt --only double-pendulum --seeds 3 --dip-time-budget 180 --models "$D" --csv dt_dip.csv > dt_dip.log 2>&1)
echo "STEP dip_models done $(date +%H:%M) exit $?" >> model_study/queue.log
S="random;ut;cut;cub5;gh3;random:model=poly4;ut:model=poly4"
(cd sigma_study && ../$B/sigma_point_study --only double-pendulum --seeds 3 --dip-time-budget 180 --configs "$S" --csv sigma_dip.csv --summary-csv sigma_dip_runs.csv > sigma_dip.log 2>&1)
echo "STEP dip_sigma done $(date +%H:%M) exit $?" >> model_study/queue.log
echo "QUEUE FINISHED" >> model_study/queue.log
