#!/bin/bash
# DT-FBSDE on the double inverted pendulum (4-D, 100 control stages) with a degree-4 polynomial value
# function (70 features): per stage against time-shared, at a batch of 128 with the 2-D recipe's
# inflation and uphill stop, without slope targets (the 4-D default).
# usage: dt_dip_queue.sh <vms binary> [seeds] [budget]
BIN=$1; SEEDS=${2:-2}; BUDGET=${3:-180}
COMMON="batch=128,inflation=6,uphill=1"
$BIN --phase dt --only double-pendulum --seeds $SEEDS --dip-time-budget $BUDGET --held-out 1024 --csv dt_dip.csv \
  --runs-csv dt_dip_runs.csv \
  --models "poly4:$COMMON;poly4:$COMMON,time=linear,knots=16;poly4:$COMMON,rank=8,time=linear,knots=16,balance=1,sweeps=2" \
  > dt_dip.log 2>&1
