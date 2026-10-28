# Slope targets on the double inverted pendulum

Does the DIP need the slope-matching targets? `method_comparison` was run on the double inverted pendulum with
its default 2-D recipe, varying only `--dip-slope-weight` between 10 (the default) and 0. Each arm ran 6 seeds
for 120 s with `--profile-csv`. The machine was a 4-vCPU container, and the build was g++ 13 at `-O2`.

```
method_comparison --only double-pendulum --methods dt-fbsde --seeds 6 --dip-time-budget 120 --mpc-seeds 0 \
    --dip-slope-weight {10,0} --csv slope_w{10,0}.csv --profile-csv profile_w{10,0}.csv
python3 analyze.py
```

| | slope weight 10 | slope weight 0 |
|---|---|---|
| Seconds per iteration | 1.36 | 0.69 |
| Median held-out cost at 30 s | 37.9 | 16.0 |
| Median held-out cost at 120 s | 28.9 | 10.6 |
| Mean held-out cost at 120 s | 27.5 | 15.5 |
| Median cost at iteration 40 | 29.1 | 16.0 |

The slope targets cost about half of each iteration. One part is computing them, 62% of the backward targets'
thread time. The other part is the SGD fit, which takes 0.84 s per iteration with slopes and 0.36 s without.
At this sample size they also don't improve the result: without slopes the median cost is lower at every
checkpoint, and it is lower at equal iteration counts too. Six seeds is not decisive, though. At 120 s the
no-slope run wins 26 of the 36 seed pairs (Mann-Whitney U = 10, p ≈ 0.15), and both arms have a seed stuck
above 40.
