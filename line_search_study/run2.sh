#!/bin/bash
cd /tmp/claude-0/-home-claude-fbsde-traj-opt/acf48e39-f26f-5f86-9262-e2569ea77311/scratchpad/exp
while pgrep -f "exp/run.sh" >/dev/null; do sleep 20; done
run() { name=$1; shift; for p in integrator pendulum double-pendulum; do
  [ -f $name.$p.done ] && continue
  rm -f $name.$p.it.csv
  LS_TRACE=$name.$p.it.csv /tmp/claude-0/-home-claude-fbsde-traj-opt/acf48e39-f26f-5f86-9262-e2569ea77311/scratchpad/mc2 --only $p --methods dt-fbsde --seeds 8 --seed 7 --mpc-seeds 0 --csv $name.$p.ckpt.csv --mpc-csv /dev/null --reference-csv $name.$p.ref.csv "$@" > $name.$p.log 2>&1 && touch $name.$p.done
done; }
run baseB
run halfB --dt-initial-step 0.5
echo ALLDONE
