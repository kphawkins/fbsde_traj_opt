# Line search study: pooling, a function-space trust region, and a later first step

Three ways to get more out of DT-FBSDE's line search, tried on the double integrator, the pendulum swing-up and the
double inverted pendulum (DIP) under the current defaults (degree-4 polynomial, Taylor noiseless target, screened
stop-uphill line search). Only one helps: **starting the line search at a half step makes the DIP converge about
twice as fast and much more reliably, and is neutral in 2-D.** Pooling batches and anchoring the fit both make every
problem worse.

| Option | DT-FBSDE flag | Verdict |
| --- | --- | --- |
| Start the line search at 1/2 (`initial_step = 0.5`) | `--dt-initial-step`, `--dip-initial-step` | DIP: clear win. 2-D: neutral. Now the DIP default. |
| Start at twice the last accepted step (`adaptive_initial_step`) | `--dt-adaptive-step on` | Worse in 2-D, no gain on the DIP |
| Pool the batches since the last acceptance (`pool_batches_since_acceptance`) | `--dt-pool N` | Worse everywhere, and slower per iteration |
| Function-space trust region (`anchor_weight`) | `--dt-anchor-weight W` | Much worse everywhere, worse as the weight grows |

## Results

`python3 analyze.py` prints these tables from `results/`. Each cell is the median held-out cost (1024 rollouts) at
that charged time. Baseline and start-at-1/2 have 12 seeds each (4 at `--seed 0`, 8 at `--seed 7`). Start-at-1/4
has 8 seeds and the other options have 4. Threshold: 2% above the grid optimum in 2-D, and 4.5 on the DIP.

Double inverted pendulum (120 s):

| Config | Runs | 20 s | 40 s | 60 s | 120 s | Final range | Reached 4.5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline (start at 1) | 12 | 5.83 | 5.52 | 5.44 | 4.81 | 4.09 to 6.21 | 3/12 |
| **Start at 1/2** | 12 | 5.69 | **4.45** | **4.21** | **4.09** | 3.23 to 6.58 | **11/12** |
| Start at 1/4 | 8 | 7.16 | 4.58 | 4.41 | 4.27 | 3.69 to 4.52 | 7/8 |
| Adaptive start | 4 | 5.54 | 5.44 | 5.09 | 4.79 | 4.28 to 5.05 | 1/4 |
| Pool 4 | 4 | 6.17 | 5.46 | 5.46 | 5.46 | 4.65 to 5.52 | 0/4 |
| Pool 16 | 4 | 6.61 | 5.46 | 5.46 | 5.46 | 5.04 to 6.43 | 0/4 |
| Anchor 0.3 | 4 | 16.8 | 15.9 | 15.9 | 14.4 | 12.0 to 16.3 | 0/4 |
| Anchor 3 | 4 | 41.4 | 35.6 | 35.6 | 25.5 | 19.1 to 42.6 | 0/4 |

Starting at a half step reaches the baseline's 120 s cost by about 40 s. A start-at-1/2 run ends below the baseline
run it is compared against 85% of the time (all 144 pairs of the 12 seeds). One start-at-1/2 seed ends at 6.58,
which is about as bad as the baseline's worst (6.21).

Double integrator (grid optimum 4.137) and pendulum (2.298), 60 s:

| Config | Runs | Integrator 5 s | Integrator 60 s | Pendulum 5 s | Pendulum 60 s | ms per iteration (int. / pend.) |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | 12 | 4.220 | 4.172 | 2.734 | 2.417 | 42 / 50 |
| Start at 1/2 | 12 | 4.231 | 4.170 | 2.812 | 2.434 | 46 / 52 |
| Adaptive start | 4 | 4.247 | 4.187 | 2.732 | 2.492 | 66 / 61 |
| Pool 4 | 4 | 4.525 | 4.283 | 3.058 | 2.949 | 58 / 77 |
| Pool 16 | 4 | 5.189 | 5.187 | 3.406 | 3.340 | 131 / 169 |
| Anchor 0.3 | 4 | 10.14 | 4.240 | 3.247 | 2.483 | 53 / 54 |
| Anchor 3 | 4 | 16.0 | 14.5 | 10.8 | 5.61 | 52 / 67 |

In 2-D, starting at 1/2 is within seed noise of the baseline. It ends slightly lower on the integrator (better in 68%
of pairs) and slightly higher on the pendulum (better in 31%). It is a little slower in the first few seconds on both.

## Why

- **Starting at 1/2 (DIP).** A full step is almost never accepted on the DIP: 10 of 301 iterations in the profiling
  run, at a median cost 5 times the current one. Skipping that trial saves a rollout batch per iteration, and the
  accepted steps get larger on average because the uphill test now sees 1/2 and 1/4, where the cost curve is closer
  to quadratic. Accepted iterations rise from 20% to 24%. In 2-D full steps are accepted more often (about 30% of
  accepts), so skipping them is a wash.
- **Adaptive start.** Accepted steps in 2-D are often small (1/16 to 1/4), so the next search starts small. That
  makes accepts more frequent but each one smaller, and the extra halvings cost line-search time: 66 ms per
  iteration against 42 ms on the integrator.
- **Pooling.** Pooling lowers the acceptance rate (integrator: 27% to 16% in the first 100 iterations, and 0% late
  with 16 batches). It also multiplies backward-pass time by the pool size. A lower-variance estimate of the current
  policy's value gives nearly the same greedy policy every iteration, and once that policy is rejected it keeps being
  rejected. The fresh 16-path fits are noisy, and the line search accepts the occasional noisy proposal that happens
  to improve. So the noise is acting as a random search, and the line search as the selection step. Removing the
  noise removes the search.
- **Anchors.** Holding the value function at the current policy's own states blocks exactly the change that improves
  the policy there. From the starting table this makes the first few hundred iterations crawl (integrator: 10.1
  against 4.22 at 5 s), and on the DIP it never recovers within the budget.

## What would be worth trying next

The pooling result says DT-FBSDE in its current form improves by selecting among noisy proposals. That suggests
leaning into it: fit several independent small-batch proposals per iteration (or bootstrap resamples of one batch),
score them all on the screening rollouts in one parallel batch, and keep the best. The line search's guarantee is
unchanged, because a candidate is still accepted only if it beats the current policy on every evaluation rollout.

## Reproducing

The runs used a scratch build of `method_comparison` (g++ 13.3, `-O2 -DNDEBUG`, outside Bazel) on a 4-vCPU container,
with a few lines added to log each `IterationReport`. `results/iterations_summary.csv` holds those logs summarized per
run. `run.sh` (round 1) and `run2.sh` (round 2) are the scripts, with the scratch paths left as they were. Round 2's
start-at-1/4 run was:

```
method_comparison --only double-pendulum --methods dt-fbsde --seeds 8 --seed 7 --mpc-seeds 0 --dip-initial-step 0.25
```

At the time of the runs the DIP's first step defaulted to 1, so `base*` used no flags and `half*` used
`--dt-initial-step 0.5`. Now that the DIP defaults to a half step, reproduce its baseline with `--dip-initial-step 1`.
