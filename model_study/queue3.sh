#!/bin/bash
# Final 2-D confirmations, after the second queue finishes.
cd /home/kphawkins/dev/fbsde_traj_opt-ilqr-mppi
until grep -q "QUEUE2 FINISHED" model_study/queue.log; do sleep 15; done
C="poly4;poly6;poly2;poly8;rbf5;spline5;leg6:rough=1e-6"
(cd model_study && ../bazel-bin/examples/value_model_study --phase dt --only integrator,pendulum --seeds 8 --time-budget 60 --models "$C" --csv dt_confirm.csv > dt_confirm.log 2>&1)
echo "STEP confirm done $(date +%H:%M) exit $?" >> model_study/queue.log
echo "QUEUE3 FINISHED" >> model_study/queue.log
