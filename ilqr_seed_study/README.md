# Seeding DT-FBSDE with iLQR

The method comparison (`comparison_runs/`) ended by recommending one experiment first: warm-start
DT-FBSDE from an iLQR solution, since iLQR covers the ground from a cost of 99 to about 10 in a second
on the double inverted pendulum, which is where DT-FBSDE spends almost all of its budget. This study
runs that experiment on the three L1 benchmarks against plain DT-FBSDE and iLQR.

`report.html` has the convergence curves and every table; `summary.txt` the tables alone.

## What was built

- `BoxIlqrSolver::ValueExpansions(smoothing)`: the backward pass's quadratic model of the cost-to-go
  about the nominal trajectory at every stage (value, gradient, Hessian), from a pass at a chosen
  pseudo-Huber smoothing.
- `examples/ilqr_seeded_table.hpp`: turns those expansions into a DT-FBSDE value function table. The
  polynomial model holds each quadratic exactly. The soft-min model holds only convex bowls, so the
  Hessian's eigenvalues are floored, the bowl is centered so that the value and slope at the nominal
  state are iLQR's, and its components are spread slightly along the flattest direction so that the
  SGD fit can move them apart.
- `DtFbsdeIterativeSolver::SeedValueFunctions()` installs such a table, and `SeededCost()` scores one
  on the solver's own evaluation rollouts without installing it.
- A `DT-FBSDE+iLQR` method in `//examples:method_comparison` (`--methods dt-fbsde-ilqr`). Each run
  starts iLQR from that seed's random controls. At geometrically spaced iLQR iterations, and at the
  last, the expansions become a candidate table, which is scored on DT-FBSDE's 256 evaluation
  rollouts (never the held-out ones). The best candidate is installed. iLQR stops when it converges
  (or after 10 s or 2,000 iterations), and DT-FBSDE then iterates from the installed table for the
  rest of the budget, with its initial exploration probability set to the regular 0.03 so that it
  samples around the seeded policy from the start. Everything is charged: iLQR, building and
  scoring the candidates, and DT-FBSDE's iterations.

Scoring several candidates matters because iLQR's best policies on the pendulums are mid-run, not
final (the comparison's "found and then lost" observation). The scoring uses only the solver's
rollouts, so the method is still a real offline method.

## Setup

The comparison's protocol and configurations, unchanged: 60 s per run and 10 seeds on the 2-D
problems, 300 s and 6 seeds on the double inverted pendulum. DT-FBSDE uses the batch sweep's
polynomial recipe in 2-D and the FBRRT comparison's soft-min configuration in 4-D, and the seeded
method uses exactly the same solver configuration. Every policy is scored on the same 1,024
held-out rollouts as the comparison, so held-out costs are directly comparable with it.

Two settings of the seed's smoothing were run, since a pilot showed it matters on the double
pendulum: 1 (the smoothing the comparison's iLQR tracking gains use) and 0.03 (`_s003`
directories). A smaller smoothing keeps less artificial control curvature in the value function, so
the seeded policy is closer to iLQR's own bang-off-bang plan.

All runs were on this cloud machine's 4 cores, not the comparison's 8, so plain DT-FBSDE got 37
iterations in 300 s on the double pendulum rather than 67. Plain DT-FBSDE and iLQR were rerun here so
that every time in this study is from the same machine.

## Results

Gap = held-out cost / reference - 1. The 2-D reference is the grid DP optimum. The double pendulum's
is 2.342, the lowest held-out cost of any policy seen in this study or the comparison (a mid-run
iLQR iterate here).

| problem | method | final gap, median [IQR] | worst | median time to within 10% |
| --- | --- | --- | --- | --- |
| double integrator | DT-FBSDE | +0.3% [+0.2, +0.4] | +0.5% | 0.54 s |
| | DT-FBSDE+iLQR | +0.3% [+0.2, +0.3] | +0.6% | **0.17 s** |
| | DT-FBSDE+iLQR (s=0.03) | +0.2% [+0.2, +0.4] | +0.6% | 0.31 s |
| | iLQR | +7.6% | +7.7% | 0.3 ms (to +7.6%) |
| pendulum | DT-FBSDE | +3.6% [+3.2, +4.2] | +5.8% | 9.7 s |
| | DT-FBSDE+iLQR | +4.2% [+3.7, +4.7] | +8.4% | 5.4 s |
| | DT-FBSDE+iLQR (s=0.03) | **+3.3%** [+3.1, +3.7] | **+4.4%** | **2.1 s** |
| | iLQR | +34% [+27, +38] | +39% | never |
| double inverted pendulum | DT-FBSDE | +260% [+222, +335] | +616% | never |
| | DT-FBSDE+iLQR | **+77%** [+36, +156] | +311% | never (1 of 6 seeds) |
| | DT-FBSDE+iLQR (s=0.03) | **+77%** [+19, +148] | +406% | never (2 of 6 seeds) |
| | iLQR | +293% [+219, +425] | +659% | never (final); best-seen median +22% |

On the double pendulum, per seed (held-out cost at the end of 300 s; plain DT-FBSDE's median is 8.42):

| seed | DT-FBSDE | DT-FBSDE+iLQR | DT-FBSDE+iLQR (s=0.03) | iLQR final |
| --- | --- | --- | --- | --- |
| 0 | 7.39 | 4.58 | 2.64 | 7.19 |
| 1 | 16.77 | 3.69 | **2.41** | 10.11 |
| 2 | 8.84 | 9.63 | 11.86 | 5.11 |
| 3 | 8.00 | 3.01 | 3.24 | 17.77 |
| 4 | 6.23 | 6.46 | 6.07 | 8.29 |
| 5 | 10.64 | **2.37** | 5.07 | 13.03 |

## Conclusions

- **On the double inverted pendulum, seeding is a large improvement.** The median final cost falls
  from +260% to +77% above the best known, and in 4 of 6 seeds (both settings) the seeded method beats
  plain DT-FBSDE's 300-second result within 5 s of charged time. The best seeded runs end at 2.37 and
  2.41, within 3% of the best policy known, and as offline policies. Plain DT-FBSDE's best run is at
  6.23.
- **The seed is most of the result.** Most of the gain arrives with the table itself in the first few
  seconds. DT-FBSDE then improves on it in some runs (seed 3's seed scores 5.04 on the solver's
  rollouts and the run ends at 3.01 held out; seed 4's, 13.9 and 6.46), but in others it rejects
  nearly every update:
  its soft-min fits to 4,096 samples are worse than the iLQR quadratic they started from. The 2-D
  sweep's recipe (small batches, six times the noise) was tried from the seed in a pilot and rejected
  every update.
- **The failures are iLQR's.** Seeds 2 and 4 are the runs where iLQR itself stays in a poor basin
  (seed 2 converges at iteration 42 to a 5.1 plan), and the seeded method inherits it. Seeding from
  several iLQR starts, and keeping the best table on the solver's rollouts, is the obvious fix.
- **On the 2-D problems seeding only speeds up the start.** It reaches within 10% of optimal three to
  five times sooner (double integrator 0.17 s vs 0.54 s; pendulum 2.1 s vs 9.7 s at s=0.03) and ends
  at about the same accuracy (at s=1 the pendulum ends slightly worse, +4.2% against +3.6%). At s=0.03 it is also the most reliable configuration on the pendulum: all ten
  seeds end within 5% (worst +4.4%), against nine of ten for plain DT-FBSDE.
- **The seed's smoothing matters, and no single value is best.** s=0.03 is better on the pendulum and
  gives the two best double-pendulum runs but also the worst one; s=1 has the smaller spread on the
  double pendulum. Both are a clear win over no seed.

Next steps, in order: seed from several iLQR starts and keep the best; find out why the soft-min
refits lose to their own seed in 4-D (the fit, not the sampling, is the suspect: a tiny blend step is
already uphill); then retune DT-FBSDE from the seeded start, as the comparison's step 2 suggested.

## Caveats

- **The seed's smoothing was chosen from a one-seed, 35-second pilot on the double pendulum's seed 0**,
  scored on held-out cost, before the full runs. That pilot also showed the solver's 256-rollout
  scores can be optimistic against held-out (2.30 vs 3.43 at s=0.1): selecting among ~25 candidates
  and line-search steps on the same fixed rollouts overfits them.
- **Six seeds on the double pendulum, with a wide spread.** The medians are robust to one seed either
  way; the means are not.
- **4 cores, not 8.** Times are not comparable with `comparison_runs/`; held-out costs are.
- **No retuning.** The DT-FBSDE configurations are the comparison's, chosen for cold starts.

## Reproducing

```sh
bazel build -c opt //examples:method_comparison
cd ilqr_seed_study/full_2d && ../../bazel-bin/examples/method_comparison --mpc-seeds 0 --mpc-csv /dev/null \
    --only integrator,pendulum --seeds 10 --methods dt-fbsde,dt-fbsde-ilqr,ilqr > log.txt
# full_2d_s003: --methods dt-fbsde-ilqr --seed-smoothing 0.03
# full_dip:      --only double-pendulum --seeds 6 --dip-time-budget 300 --methods dt-fbsde,dt-fbsde-ilqr,ilqr
# full_dip_s003: --only double-pendulum --seeds 6 --dip-time-budget 300 --methods dt-fbsde-ilqr --seed-smoothing 0.03
python3 tools/analyze_ilqr_seed_study.py ilqr_seed_study --prior comparison_runs/full_dip
```
