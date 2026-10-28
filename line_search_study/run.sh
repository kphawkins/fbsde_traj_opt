#!/bin/bash
cd /tmp/claude-0/-home-claude-fbsde-traj-opt/acf48e39-f26f-5f86-9262-e2569ea77311/scratchpad/exp
run() { name=$1; shift; for p in integrator pendulum double-pendulum; do
  [ -f $name.$p.done ] && continue
  rm -f $name.$p.it.csv
  LS_TRACE=$name.$p.it.csv /tmp/claude-0/-home-claude-fbsde-traj-opt/acf48e39-f26f-5f86-9262-e2569ea77311/scratchpad/mc2 --only $p --methods dt-fbsde --seeds 4 --mpc-seeds 0 --csv $name.$p.ckpt.csv --mpc-csv /dev/null --reference-csv $name.$p.ref.csv "$@" > $name.$p.log 2>&1 && touch $name.$p.done
done; }
run base
run half --dt-initial-step 0.5
run adapt --dt-adaptive-step on
run pool4 --dt-pool 4
run pool16 --dt-pool 16
run anchor03 --dt-anchor-weight 0.3
run anchor3 --dt-anchor-weight 3
echo ALLDONE
