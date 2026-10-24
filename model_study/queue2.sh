#!/bin/bash
# The randomized-UT experiments, after the first queue finishes.
cd /home/kphawkins/dev/fbsde_traj_opt-ilqr-mppi
until grep -q "QUEUE FINISHED" model_study/queue.log; do sleep 15; done
B=bazel-bin/examples
S2="ut:rotate=random;cub5:rotate=random;cut:rotate=random;gh3:rotate=random;ut:rotate=random,probes=0"
(cd sigma_study && ../$B/sigma_point_study --only integrator,pendulum --seeds 5 --time-budget 30 --configs "$S2" --csv sigma_2d_random.csv --summary-csv sigma_2d_random_runs.csv > sigma_2d_random.log 2>&1)
echo "STEP sigma_2d_random done $(date +%H:%M) exit $?" >> model_study/queue.log
S4="ut:rotate=random;cut:rotate=random"
(cd sigma_study && ../$B/sigma_point_study --only double-pendulum --seeds 3 --dip-time-budget 180 --configs "$S4" --csv sigma_dip_random.csv --summary-csv sigma_dip_random_runs.csv > sigma_dip_random.log 2>&1)
echo "STEP sigma_dip_random done $(date +%H:%M) exit $?" >> model_study/queue.log
echo "QUEUE2 FINISHED" >> model_study/queue.log
