#!/bin/bash
# The oracle sweep of the time-shared study: every model fitted to the grid DP value function at 8,
# 32 and 512 training samples per stage (8 disjoint subsets for the small counts), two ridge weights.
# usage: oracle_queue.sh <vms binary> <problem>
BIN=$1; PROBLEM=$2
SHAPES="time=stage time=linear,knots=4 time=linear,knots=8 time=linear,knots=16 time=const,knots=8 time=const,knots=16
rank=3 rank=4 rank=6 rank=8 rank=12 rank=4,time=linear,knots=16 rank=6,time=linear,knots=16 rank=8,time=linear,knots=16"
models=""
for samples in 8 32 512; do
  reps=8; [ $samples = 512 ] && reps=1
  for ((rep = 0; rep < reps; ++rep)); do
    for ridge in 1e-6 1e-3; do
      for shape in $SHAPES; do
        models="$models;poly6:samples=$samples,offset=$((rep * samples)),ridge=$ridge,$shape"
      done
    done
  done
done
$BIN --phase oracle --only $PROBLEM --held-out 512 --csv oracle_$PROBLEM.csv --models "${models#;}" > oracle_$PROBLEM.log 2>&1
