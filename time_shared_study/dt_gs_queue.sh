#!/bin/bash
# Block Gauss-Seidel within the pass with a 0.1 weight on the previous pass's equations (pass=gs), the
# fix for the fresh-only variant's failures, on the two time-shared shapes dt_queue.sh found best.
# usage: dt_gs_queue.sh <vms binary> <problem> [seeds] [budget]
BIN=$1; PROBLEM=$2; SEEDS=${3:-4}; BUDGET=${4:-30}
$BIN --phase dt --only $PROBLEM --seeds $SEEDS --time-budget $BUDGET --held-out 1024 --csv dt_gs_$PROBLEM.csv \
  --runs-csv dt_gs_${PROBLEM}_runs.csv \
  --models "poly6:time=linear,knots=16,pass=gs,stale=0.1;poly6:rank=8,time=linear,knots=16,balance=1,sweeps=2,pass=gs,stale=0.1" \
  > dt_gs_$PROBLEM.log 2>&1
