# Slope targets on the 2-D problems, with and without the iLQR seed

The DIP ablation (`../slope_ablation`) found that slope targets cost half of each iteration and did not help.
This study asks the same question on the double integrator and the pendulum, for both plain DT-FBSDE and
DT-FBSDE+iLQR (seed smoothing 0.03, the iLQR seed study's best 2-D setting).

Each arm ran 6 seeds × 60 s with the batch sweep's recipe. The slopes-on arm used each problem's default slope
weight (10 for the integrator, 1 for the pendulum). The slopes-off arm passes `--integrator-slope-weight 0
--pendulum-slope-weight 0`. The build was clang 21.1 at `-O2` on a 4-vCPU container, the same toolchain as
`ilqr_seed_study/`, so iteration counts are comparable with it.

```
method_comparison --only integrator,pendulum --methods dt-fbsde,dt-fbsde-ilqr --seed-smoothing 0.03 \
    --seeds 6 --mpc-seeds 0 --csv slopes_on.csv --profile-csv profile_on.csv
# slopes_off: add --integrator-slope-weight 0 --pendulum-slope-weight 0
python3 analyze.py
```

The gap is measured against the grid DP optimum. Times are charged seconds, and the median time is shown only
when every seed got there (otherwise the table gives how many seeds did).

| problem | method | slopes | median gap | worst | to within 10% | to within 5% |
|---|---|---|---|---|---|---|
| double integrator | DT-FBSDE | on | **+0.26%** | +0.47% | 0.54 s | 0.72 s |
| | | off | +1.47% | +1.68% | 0.46 s | 1.07 s |
| | DT-FBSDE+iLQR | on | **+0.23%** | +0.58% | 0.26 s | 0.30 s |
| | | off | +1.03% | +1.23% | 0.25 s | 0.28 s |
| pendulum | DT-FBSDE | on | **+4.2%** | +6.2% | 5.8 s | 5 of 6 |
| | | off | +9.8% | +18.8% | 3 of 6 | 0 of 6 |
| | DT-FBSDE+iLQR | on | **+3.3%** | +3.5% | 1.7 s | 7.2 s |
| | | off | +4.9% | +5.6% | 1.7 s | 3 of 6 |

The 2-D problems need the slopes. Every arm ends worse without them: by about 1 point on the integrator, by 5.6
points on the pendulum without the seed, and by 1.6 points with it. Dropping them also saves no time.
Computing them is 0.6% of wall time on the integrator and 11% on the pendulum. Without slopes, the line search
rejects more candidates, so iterations are no faster (integrator 34 → 49 ms, pendulum 44 → 46 ms). The iLQR
seed shrinks the pendulum's loss but doesn't remove it. This agrees with the oracle fits in
`time_shared_study/`, which were 1 to 7 points worse without slopes.

The double inverted pendulum is different. There, slopes are expensive (a finite-difference Jacobian through
the RK step, plus an SGD fit that runs twice as long), and six seeds showed no benefit.
