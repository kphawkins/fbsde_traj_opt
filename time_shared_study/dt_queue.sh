#!/bin/bash
# The in-solver arm of the time-shared study: DT-FBSDE on each 2-D benchmark with the per-stage
# degree-6 table against time-shared ones, each with the default slope targets and without them. The
# CSVs label the pass=gs,stale=0 runs pass=fresh, their name in the earlier build that made them.
# usage: dt_queue.sh <vms binary> <problem> [seeds] [budget]
BIN=$1; PROBLEM=$2; SEEDS=${3:-4}; BUDGET=${4:-30}
SHAPES="poly6
poly6:time=linear,knots=16
poly6:rank=8,time=linear,knots=16,balance=1,sweeps=2
poly6:rank=6,time=linear,knots=16,balance=1,sweeps=2
poly6:time=linear,knots=16,pass=gs,stale=0
poly6:rank=8,time=linear,knots=16,balance=1,sweeps=2,pass=gs,stale=0"
models=""
for shape in $SHAPES; do
  models="$models;$shape"
  if [[ $shape == *:* ]]; then models="$models;$shape,slope=0"; else models="$models;$shape:slope=0"; fi
done
$BIN --phase dt --only $PROBLEM --seeds $SEEDS --time-budget $BUDGET --held-out 1024 --csv dt_$PROBLEM.csv \
  --runs-csv dt_${PROBLEM}_runs.csv --models "${models#;}" > dt_$PROBLEM.log 2>&1
