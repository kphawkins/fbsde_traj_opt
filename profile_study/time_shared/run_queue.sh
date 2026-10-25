#!/bin/bash
# Profiles the 2-D recipe without slope targets (the default) under the per-stage value-function
# table and the time-shared ones (time_shared_study/), one run at a time so the timings don't
# compete for cores. The 2-D problems use the degree-6 polynomial, the double inverted pendulum the
# degree-4 one at a batch of 128 (time_shared_study/dt_dip_queue.sh), plus method_comparison's
# soft-min sweep configuration, the DIP default.
# usage: run_queue.sh <value_model_study binary> <method_comparison binary> [seeds] [budget] [dip seeds] [dip budget]
VMS=$1; MC=$2; SEEDS=${3:-3}; BUDGET=${4:-30}; DIP_SEEDS=${5:-2}; DIP_BUDGET=${6:-150}
SHARED="rank=8,time=linear,knots=16,balance=1,sweeps=2"
for problem in integrator pendulum; do
  $VMS --phase dt --only $problem --seeds $SEEDS --time-budget $BUDGET --held-out 1024 --csv ${problem}.csv \
    --runs-csv ${problem}_runs.csv --profile-csv ${problem}_profile.csv \
    --models "poly6;poly6:time=linear,knots=16;poly6:$SHARED" > ${problem}.log 2>&1
done
COMMON="batch=128,inflation=6,uphill=1"
$VMS --phase dt --only double-pendulum --seeds $DIP_SEEDS --dip-time-budget $DIP_BUDGET --held-out 1024 --csv dip.csv \
  --runs-csv dip_runs.csv --profile-csv dip_profile.csv \
  --models "poly4:$COMMON;poly4:$COMMON,time=linear,knots=16;poly4:$COMMON,$SHARED" > dip.log 2>&1
$MC --methods dt-fbsde --only double-pendulum --seeds $DIP_SEEDS --dip-time-budget $DIP_BUDGET \
  --mpc-seeds 0 --csv dip_softmin.csv --mpc-csv dip_softmin_mpc.csv --reference-csv dip_softmin_reference.csv \
  --profile-csv dip_softmin_profile.csv > dip_softmin.log 2>&1
