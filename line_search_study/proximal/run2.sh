#!/bin/bash
cd /tmp/claude-0/-home-claude-fbsde-traj-opt/acf48e39-f26f-5f86-9262-e2569ea77311/scratchpad/exp3
run() { name=$1; shift; p=double-pendulum
  [ -f $name.$p.done ] && return
  rm -f $name.$p.it.csv
  LS_TRACE=$name.$p.it.csv /tmp/claude-0/-home-claude-fbsde-traj-opt/acf48e39-f26f-5f86-9262-e2569ea77311/scratchpad/mc4 --only $p --methods dt-fbsde --seeds 8 --seed 7 --mpc-seeds 0 --csv $name.$p.ckpt.csv --mpc-csv /dev/null --reference-csv $name.$p.ref.csv "$@" > $name.$p.log 2>&1 && touch $name.$p.done
}
run baseB
run proxAB --dt-proximal-weight 1 --dt-proximal-reject 1
run proxLMB --dt-proximal-weight 1
echo ALLDONE
