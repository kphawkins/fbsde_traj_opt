# Proximal fit: a trust region inside the fit

This study tests whether a trust region built into the fit, rather than coefficient blending, makes DT-FBSDE better. **It
does not.** In 2-D every variant converges more slowly and ends higher. On the double inverted pendulum (DIP) the best
variants end level with the current default, but they are 2 to 3 times worse in the first 20 s and spread more widely
across seeds.

## The option

`DtFbsdeIterativeOptions::proximal_weight` adds this proximal term to every stage's least squares:

```
lambda * W * (theta - theta_current)' (G / w_G) (theta - theta_current)
```

- `W` is the batch's total sample weight.
- `G` is the Gram matrix of the earlier batches, decayed by `proximal_memory` (0.9) per iteration, and `w_G` is its
  total weight.

Directions that earlier data pinned down are held near the current table, and directions it never constrained follow
the new batch. This is a Kalman-style update. `lambda` adapts like a Levenberg-Marquardt damping: it is divided by 2
after an update accepted at the first candidate, multiplied by 2 after one accepted later, and multiplied by
`proximal_rejection_adaptation` after a rejected one. It is clamped to [1e-3, 1e3]. With `proximal_line_search`, the
line search refits at `lambda * 4^h` instead of blending. All of this is off by default.

| Config | Flags (all with `--dt-proximal-weight`) | Rejection factor |
| --- | --- | --- |
| `proxA`: proximal fit, blend line search | `1 --dt-proximal-reject 1` | 1 (weight left alone) |
| `proxLM`: classic Levenberg-Marquardt | `1` | 2 |
| `proxB`: line search over the weight | `1 --dt-proximal-reject 1 --dt-proximal-search on` | 1 |
| `proxFix`: fixed weight | `0.3 --dt-proximal-adapt 1 --dt-proximal-reject 1` | 1 |

## Results

`python3 analyze.py results` prints the full tables. Round 1 ran 4 seeds of every config on every problem (`run.sh`).
Round 2 ran 8 more DIP seeds of `base`, `proxA` and `proxLM` (`run2.sh`, `--seed 7`). The baseline is the current
default, which starts the DIP's line search at a half step.

Double inverted pendulum, median held-out cost (12 seeds; `proxB` and `proxFix` 4 seeds):

| Config | 10 s | 20 s | 40 s | 120 s | Final range | Better than baseline at 120 s |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | 6.11 | 5.06 | 4.31 | 4.09 | 3.23 to 4.41 | |
| proxA | 18.8 | 9.04 | 4.83 | 3.99 | 3.10 to 5.28 | 43% of pairs |
| proxLM | 18.1 | 8.56 | 4.72 | 4.11 | 3.02 to 4.93 | 46% of pairs |
| proxB | 17.2 | 7.42 | 7.42 | 7.39 | 6.32 to 9.74 | |
| proxFix | 27.5 | 18.6 | 11.4 | 4.63 | 4.44 to 5.75 | |

2-D, median held-out cost at 60 s (4 seeds):

| Config | Integrator (optimum 4.137) | Pendulum (optimum 2.298) |
| --- | --- | --- |
| Baseline | 4.172 | 2.415 |
| proxA | 4.172 | 3.138 |
| proxLM | 6.073 | 5.246 |
| proxB | 4.208 | 2.715 |
| proxFix | 4.398 | 2.888 |

## Why it does not help

- **Rejections are noise, not overshoot, in 2-D.** About 97% of 2-D iterations are rejected because the fresh
  proposal is worse, not because it went too far. Classic Levenberg-Marquardt doubles the weight on each one and
  pins it at the cap within a few dozen iterations. That freezes the table at whatever it had reached (proxLM: 6.07 on
  the integrator). Leaving the weight alone on rejection (proxA) avoids the freeze, but any weight above zero still
  shrinks the noisy proposals that the line search was selecting from. The pooling study found the same thing.
- **On the DIP the weight collapses to its floor.** Under proxA the median weight is 0.002, so after the first few
  dozen iterations the fit is essentially the plain one. The early iterations, with the weight still near 1, are
  held near the poor starting table, which is the slow start in the table above. The 4-seed hint of a gain (3.90
  against 4.20) did not survive 8 more seeds.
- **Searching over the weight instead of blending (proxB) has no way back to small steps.** A large weight only
  holds the directions earlier data constrained, so no candidate gets close to the current table, and the DIP stalls.

The half-step start (`../README.md`) stays the only line search change that helped.
