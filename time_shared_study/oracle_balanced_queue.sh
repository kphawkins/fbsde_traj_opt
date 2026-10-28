#!/bin/bash
# The second oracle sweep: the shapes worth keeping from the first, each stage weighted by one over its
# targets' variance (balance=1), at 8, 32 and 512 samples per stage, without and with label noise of
# 0.2 of a stage's spread of targets.
# usage: oracle_balanced_queue.sh <vms binary> <problem>
BIN=$1; PROBLEM=$2
SHAPES="time=stage time=linear,knots=8 time=linear,knots=16 rank=6 rank=8 rank=12 rank=6,time=linear,knots=16
rank=8,time=linear,knots=8 rank=8,time=linear,knots=16 rank=12,time=linear,knots=16"
models=""
for samples in 8 32 512; do
  reps=8; [ $samples = 512 ] && reps=1
  for ((rep = 0; rep < reps; ++rep)); do
    for noise in 0 0.2; do
      for shape in $SHAPES; do
        models="$models;poly6:samples=$samples,offset=$((rep * samples)),noise=$noise,balance=1,$shape"
      done
    done
  done
done
$BIN --phase oracle --only $PROBLEM --held-out 512 --csv oracle_balanced_$PROBLEM.csv --models "${models#;}" \
  > oracle_balanced_$PROBLEM.log 2>&1
