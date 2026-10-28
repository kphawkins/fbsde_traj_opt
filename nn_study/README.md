# Nearest-neighbor DT-FBSDE (DT-NN): one Y per sample, the same noise every iteration

DT-NN replaces the fitted value function with the samples themselves. Every Y_s^i is its own parameter. The policy
reads V at an unexplored state from the nearest previous sample, and every iteration redraws exactly the same noise,
with no regression, blending, or line search (`fbsde_traj_opt/solvers/nearest_neighbor_dt_fbsde_solver.hpp`).
**It converges in about a second in 2-D, but to a policy 18–40% worse than the current default, and it is 2.4×
worse on the double inverted pendulum (DIP). Overfitting to the fixed noise is small (held-out within about ±6% of
the training cost), and fixed noise beats fresh noise.**

## Results (4 seeds, held-out cost on 1,024 fresh rollouts)

`python3 analyze.py` prints these from `full/` (also in `summary.txt`). Budgets are 60 s in 2-D and 120 s on the DIP,
on a 4-vCPU container, with the same harness and held-out noise as `line_search_study`.

| Method | Integrator (grid opt 4.137) | Pendulum (grid opt 2.298) | DIP |
| --- | --- | --- | --- |
| Default DT-FBSDE | **4.17** (4.16–4.17) | **2.40** (2.39–2.43) | **3.96** (3.67–4.14) |
| `nn`: 4-NN mean, pathwise Y, lookahead 16, memory 4 | 4.92 (4.84–5.13) | 3.34 (3.30–3.47) | 9.34 (7.36–12.9) |
| `ll`: local linear readout, greedy backup, lookahead 4 | 6.56 (4.71–8.37) | 3.57 (3.09–3.96) | 36.5 (24.3–41.6) |
| `llgrad`: local linear gradient policy, one step | 8.97 (8.00–9.26) | 6.20 (5.86–7.38) | 49.7 (46.9–56.3) |

Medians at checkpoints (integrator, 1 s / 5 s / 60 s): default 4.34 / 4.19 / 4.17, `nn` 4.99 / 4.92 / 4.92. DT-NN
reaches its plateau almost at once, and then the iteration only reshuffles controls. The default is ahead at every
checkpoint on every benchmark.

**Overfitting.** The held-out cost of the kept policy against its own batch cost on the fixed noise was
−3.0% to +0.6% (`nn`, integrator), −3.3% to +1.4% (pendulum), and −1.3% to +6.3% (DIP). The pilots' fresh-noise
ablation (`--nn-fresh-noise on`) was worse than fixed noise: integrator 4.95 against 4.55, DIP 24.6 against 14.2
(greedy, 1 seed), so common random numbers do help here.

## What it took to make it work at all (pilots, 1 seed, 20 s in 2-D and 40 s on the DIP)

| Variant | Integrator | Pendulum | DIP |
| --- | --- | --- | --- |
| The spec: 1-NN, pathwise Y, one-step Q | 235 | 46.5 | 67.2 |
| + 16-stage lookahead | 4.60 | 3.52 | 20.0 |
| 4-NN, greedy backup, lookahead 16 | 5.14 | 3.35 | 24.0 |
| ... + memory of 4 batches | 4.55 | 3.41 | 14.2 |
| 4-NN, pathwise, lookahead 16, memory 4 | 4.84 | 3.33 | 11.3 |
| 16-NN, pathwise, lookahead 16, memory 4 | 5.37 | 3.29 | 10.4 |

- **One-step Q fails on resolution.** One control step moves the integrator's velocity by about 0.06, far less than
  the spacing of 256 samples. The three candidates {−1, 0, 1} land on the same sample (a tie, so the policy coasts) or
  on neighbors whose pathwise Y differ only by their independent future noise. About half of all controls flip every
  iteration. k-NN averaging and the greedy backup reduce the noise, but they do not fix the resolution.
- **The lookahead** (`lookahead_stages`) holds each candidate for h noise-free stages before reading the table, which
  spreads the landings h times further apart in velocity. h = 16 fixes 2-D. h = 64 (the whole horizon, table never
  read) gets 37 on the integrator, so the table matters. It multiplies the policy's model calls by h, which is why
  the DIP manages only about 100 iterations in 40 s.
- **Memory** (`memory_batches`) keeps earlier batches' samples in the table, with the greedy backup re-scoring them
  each iteration. It steadies the iteration (controls changed per iteration 4% against 60% on the integrator) and helps
  the DIP most.
- **The local linear fit** (`readout = kLocalLinear`, `policy_rule = kGradient`) adds a gradient from a least-squares
  plane through the nearest samples. The gradient rule works without a lookahead (8–9 on the integrator, against 235
  for 1-NN), but it is far worse than the lookup with a lookahead, and erratic in the number of fit samples (with 24
  or more samples it extrapolates badly at the edge of the cloud). The local linear readout with a short lookahead
  was 4.7 in a pilot but varies from 4.7 to 8.4 across seeds.
- Even with the lookahead, the batch cost swings widely from one iteration to the next. Keeping the policy whose
  batch cost was lowest does most of the work.

## Reproducing

The runs used a scratch build of `method_comparison` (g++ 13.3, `-O2 -DNDEBUG`, Eigen 3.4, outside Bazel). `full/`:

```
method_comparison --only P --methods dt-fbsde --seeds 4 --mpc-seeds 0                                   # default
method_comparison --only P --methods dt-nn --seeds 4 --mpc-seeds 0 --nn-samples 256 --nn-neighbors 4 \
  --nn-lookahead 16 --dip-nn-lookahead 16 --nn-memory 4                                                 # nn
method_comparison ... --nn-target greedy --nn-readout linear --nn-fit-neighbors 12 --nn-lookahead 4 \
  --dip-nn-lookahead 4                                                                                  # ll
method_comparison ... --nn-target greedy --nn-readout linear --nn-policy gradient --nn-fit-neighbors 8  # llgrad
```

`pilot/` holds pilots 2 and 3 with the same flags, at `--seeds 1 --time-budget 20 --dip-time-budget 40`. Each
run's `*.nn.csv` has DT-NN's per-iteration batch cost on the fixed noise, the best so far, and the fraction of
controls that changed. Some pilot-3 runs overlapped with side runs on the same machine, so their timings are rough.
