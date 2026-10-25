# DT-FBSDE profiling study

Where the default DT-FBSDE configuration spends its time, on the two 2-D benchmarks (double
integrator, pendulum swing-up) and on the double inverted pendulum. `report.html` is the write-up.

## Update (2026-10-25)

The report now leads with the default after slope targets were dropped, profiled per stage and with
the time-shared value functions; see `time_shared/README.md`. The findings below are the earlier
profile, which ran with slope targets.

## Findings (2026-10-24)

- **2-D:** the line search takes 70–80% of an iteration, and most of that is the 32-rollout
  screening pass. That pass runs on one core, because `ParallelFor` won't split fewer than 64 items.
  `Decide` is about 13 µs and rebuilds the degree-6 polynomial's derivatives 19 times, which makes
  `PolynomialValueFunctionApprox::Derivatives<true>` 76–88% of CPU samples.
- With a grain of 4 rollouts per thread in `EvaluatePolicyByRollouts`, the 2-D iterations fall from
  112–122 ms to 61–63 ms, and the per-iteration costs stay identical (`results/rollout_grain_4`).
  This is not merged.
- **Double inverted pendulum:** the backward pass takes 85% of an iteration under the comparison
  configuration, and the single-threaded SGD fit alone takes 66% (200 fits per iteration, about 30 ms
  each). The 2-D recipe, now the default, runs at 1.3 s per iteration against 8.9 s and has the same
  bottleneck.

## Profiling a run

`method_comparison --methods dt-fbsde --profile-csv profile.csv` turns on the phase timers in
`fbsde_traj_opt/utils/phase_profiler.hpp`. It writes one row per DT-FBSDE run and phase, with
seconds and call counts, plus a `charged_total` row. The held-out evaluations are excluded. Phases
timed inside parallel loops (`*.sample`, `*.policy`, `backward.argmin`, ...) add up thread-seconds,
not wall time. The timers cost under 2% (0.130 s against 0.132 s per iteration on the same seed).

Any other program can do the same with `profiling::SetEnabled(true)`, then `profiling::Reset()` and
`profiling::Snapshot()` around the part it wants to measure.

## Reproducing the report

The CSVs in `results/` came from a scratch driver that ran the comparison's DT-FBSDE configurations
without held-out checkpoints. Its columns are those `agg.py` reads: `*_it.csv` holds each iteration's
`IterationReport` timings, and `*_ph.csv` holds each seed's phase totals, which match
`--profile-csv`. Those runs used 4 seeds × 60 s for each 2-D problem, and 3 seeds (comparison
configuration) and 2 seeds (2-D recipe) × 120 s for the DIP. They were built with g++ 13.3 `-O2` on
a 4-vCPU container, not the repository's clang toolchain, so expect the absolute times to differ.

```
python3 agg.py integrator pendulum dip dip-sweep   # results/*.csv -> results/agg.json
perf record -g -F 499 -o dip.perf.data -- <binary> ...
PERF=perf python3 perfx.py integrator pendulum dip # *.perf.data -> results/perf.json
python3 gen.py                                     # -> report.html
```
