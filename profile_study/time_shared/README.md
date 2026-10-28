# Profile: the 2-D recipe without slope targets, per stage against time-shared

The lead section of `../report.html`. DT-FBSDE runs the batch sweep's 2-D recipe with slope weight 0, now
the default. Each problem is run with a separate value function per stage and with the time-shared models
of `time_shared_study/`. The double inverted pendulum also gets the soft-min sweep configuration that
`method_comparison` runs by default.

## Findings (2026-10-25, clang 21 -O2, 4 vCPU)

- The time-shared closing joint solve (`backward.time_share`) takes 16–25% of an iteration: 12–14 ms in 2-D
  and 110–150 ms on the DIP. Shared models fit 10–25% fewer iterations into a budget.
- In 2-D, single-threaded screening is still 64–72% of a per-stage iteration and about 50% of a shared one.
- On the DIP with the degree-4 polynomial, the backward targets take 33–40% and screening 25–33%. The
  soft-min default spends 45% in its SGD fit. All four configurations run at 0.65–0.75 s per iteration.
- Rank 8 + 16 knots had the lowest median final cost on all three problems (integrator +0.92%, pendulum
  +10.2%, DIP 7.6). The DIP result rests on 2 seeds.

## Reproducing

```
cd results
../run_queue.sh <value_model_study> <method_comparison>   # 3 seeds x 30 s in 2-D, 2 x 150 s on the DIP
python3 ../analyze.py                                     # -> summary.json
python3 ../gen_section.py summary.json                    # the report's chart and table, as HTML
```
