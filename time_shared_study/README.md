# Time-shared value functions

DT-FBSDE fits one value function per stage, so a table of K stages over P basis functions has K P
free coefficients, each stage's estimated from that stage's samples alone (8 per half-batch in the
2-D recipe). This study ties the stages together with `TimeSharedLinearFit`
(`fbsde_traj_opt/value_function/time_shared_linear_fit.hpp`):

    theta_k = U C b(k)      U: P x r shared basis (or I),  C: r x R,  b(k): R time functions

with `b` per stage (today's table), piecewise constant, or piecewise linear between R knots, and
U either the identity or a rank-r basis found by alternating least squares. Each stage keeps its
own ridge. Optionally each stage is weighted by one over its targets' variance (`balance=1`), so
that a shared basis is not spent only on the stages with the largest values.

Everything below uses the linear-basis degree-6 polynomial (28 features) on the two 2-D benchmarks
(64 control stages), and the degree-4 polynomial (70 features) on the double inverted pendulum
(100 control stages).

## Findings (2026-10-24)

- **Pick: 16 linear time knots with a rank-8 shared basis, stages balanced** (`rank=8,time=linear,
  knots=16,balance=1`). 352 parameters against 1,792 per stage (5.1x fewer). Inside DT-FBSDE, 4
  seeds x 30 s, median held-out gap to the grid optimum: integrator +0.56% against +0.56% per
  stage, pendulum +6.3% against +7.9%. 16 knots without a basis (448, 4x fewer) ties on the
  integrator (+0.53%) and is worse on the pendulum (+10.6%). Rank 6 with 16 knots (264, 6.8x fewer)
  is +0.63% and +10.7%.
- **The parameters must be shared at the end of the backward pass, not inside it.** The pass fits
  each stage on its own as before, keeps its normal equations, and the closing joint solve replaces
  the table the line search blends toward (`DtFbsdeTimeSharingPass::kAfterPass`). Re-solving the
  knots around each stage during the pass (block Gauss-Seidel) either lags, when the stages not yet
  reached count with their previous pass's equations, or diverges to a non-finite fit within a few
  hundred iterations when they count for little or nothing (`dt_gs_*`, and the `pass=fresh` rows).
- **Why sharing helps: it averages label noise across stages.** In the oracle (each stage fitted to
  the grid DP value), with plenty of exact labels every shape with 16 knots ties per-stage. With
  label noise of 0.2 of a stage's spread the time-shared fits win clearly: at 32 samples per stage
  the gap is +34% (rank 8 + 16 knots) and +41% (16 knots) against +65% per stage on the pendulum,
  and +55% to +61% against +129% on the integrator, with half the value RMSE.
- **A shared basis without time knots barely averages noise** (ranks 6, 8, 12 with a free mix per
  stage: +92% to +132% against +129% per stage, integrator, 32 noisy samples), and **without
  balancing it breaks the pendulum**: unweighted, the basis is spent on the early stages' targets in
  the hundreds, and the policy gap at 512 exact samples is +850% to +1,160% at ranks 3 to 8. Balanced, rank 8 + 16 knots is +18.6%
  against +18.2% per stage. Rank 6 stays too small on the pendulum (+58%).
- **Piecewise-constant knots are worse than linear ones** at every count (const x16 +12%, linear x16
  +9.1%, per stage +9.4%, integrator at 512 samples); 8 knots is too few for the pendulum.
- **A ridge toward zero is not a usable prior**: in the first oracle sweep `ridge=1e-3` takes the
  integrator's per-stage gap at 8 samples from +2.5% to +571% (and +9.4% to +25% at 512), and does
  the same to every shared shape. Slope targets matter on both 2-D problems: without them the gap is
  1 to 1.5 points worse on the integrator and 4 to 7 points worse on the pendulum, for every shape.
- **Double inverted pendulum (2 seeds x 180 s, inconclusive).** Final held-out cost, per-stage
  degree 4 (7,000 parameters) 11.2 and 16.8; rank 8 + 16 knots (688, 10x fewer) 5.7 and 18.0; 16
  knots (1,120) 22.7 and 23.7. All three run about 2 s per iteration at a batch of 128, so the linear
  models' closed-form fit removes the SGD bottleneck but the line search's policy evaluations take
  its place. None beats the soft-min table the comparison uses (median cost about 7.6 at 305 s).
- **Cost.** Shared runs fit fewer iterations into 30 s: 280 per stage against 231–277 on the
  integrator, 322 against 263–280 on the pendulum. The closing alternating least squares and joint
  solve account for part of that; it was not profiled separately.

## How the parameter count scales

K = 64 stages in 2-D and 100 beyond, R = 16 knots, P = binomial(n + d, d) features of total degree
d in n dimensions.

| n, degree | P | per stage (K P) | 16 knots (R P) | rank 8 + 16 knots (P r + R r) |
|---|---|---|---|---|
| 2, 6 | 28 | 1,792 | 448 | 352 |
| 4, 4 | 70 | 7,000 | 1,120 | 688 |
| 4, 6 | 210 | 21,000 | 3,360 | 1,808 |
| 6, 4 | 210 | 21,000 | 3,360 | 1,808 |
| 6, 6 | 924 | 92,400 | 14,784 | 7,520 |

Knots divide the count by K / R (6 in 4-D) whatever the dimension; the shared basis makes it grow
like P r rather than K P. The work per iteration is the normal equations, S P^2 per stage (S samples,
the same as today), a block-tridiagonal solve over the knots, R P^3 (16 x 210^3 = 0.15 GFLOP in
6-D at degree 4), and for the basis a dense (P r)^2 system, (P r)^3 / 3 = 1.6 GFLOP at P = 210,
r = 8. At P = 924 that last solve (7,392 unknowns) needs an iterative solver. The feature count
itself is the limit at 6-D degree 6: sharing in time cannot reduce P, so 6-D also needs a sparser
state basis (for example, a quadratic plus per-axis higher-order terms), which the shared fit
supports unchanged.

## Reproducing

The study binary is `//examples:value_model_study`; the scripts take its path.

```
./oracle_queue.sh <binary> integrator          # first oracle sweep -> oracle_<problem>.csv
./oracle_balanced_queue.sh <binary> pendulum   # balanced, with label noise -> oracle_balanced_*.csv
./dt_queue.sh <binary> pendulum                # in-solver, 4 seeds x 30 s -> dt_<problem>*.csv
./dt_gs_queue.sh <binary> integrator           # in-pass Gauss-Seidel -> dt_gs_<problem>*.csv
./dt_dip_queue.sh <binary>                     # double inverted pendulum -> dt_dip*.csv
python3 analyze_oracle.py oracle_integrator.csv oracle_pendulum.csv   # -> oracle_summary.{json,txt}
python3 analyze_dt.py integrator pendulum gs_integrator gs_pendulum dip   # -> dt_summary.{json,txt}
```

The runs were built with g++ 13.3 `-O2` on a 4-vCPU container, not the repository's clang
toolchain, and some ran while the next build compiled at low priority, so absolute times are rough.
