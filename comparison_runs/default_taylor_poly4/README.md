# DT-FBSDE default switch: degree-4 polynomial + Taylor noiseless target

`method_comparison --seeds 5 --methods dt-fbsde,dt-fbsde-ilqr --mpc-seeds 0`, full budgets (60 s on the 2-D
benchmarks, 120 s on the DIP), 1024 held-out rollouts, run once with each binary on the same 4-core machine:

* `old`: research at 7ab0270 -- double-greedy targets, degree-6 polynomial in 2-D, soft-min (4 components) on the DIP.
* `new`: this change -- Taylor noiseless targets, degree-4 polynomial on every benchmark.

Built with clang 20 at `-O2 -DNDEBUG` outside Bazel (the cloud session could not download Bazel).
`python3 analyze.py .` prints the table below; `summary.json` is its `--json` output.

Median held-out cost over 5 seeds at the end of the budget (range in brackets):

| Problem | Method | Old | New |
|---|---|---|---|
| Double integrator (grid optimum 4.137) | DT-FBSDE | 4.197 [4.172, 4.206] | 4.162 [4.157, 4.165] |
| | DT-FBSDE+iLQR | 4.187 [4.162, 4.192] | 4.161 [4.152, 4.170] |
| Pendulum (grid optimum 2.298) | DT-FBSDE | 2.517 [2.419, 2.729] | 2.383 [2.376, 2.417] |
| | DT-FBSDE+iLQR | 2.458 [2.415, 2.601] | 2.396 [2.356, 2.431] |
| Double inverted pendulum | DT-FBSDE | 12.47 [3.85, 41.17] | 4.78 [4.28, 5.03] |
| | DT-FBSDE+iLQR | 4.60 [2.90, 9.50] | 3.99 [2.84, 5.63] |

No regression at the end of the budget on any problem. The new defaults run 2-4x more iterations in 2-D. The one
place the new defaults trail is DT-FBSDE+iLQR on the DIP during its first ~30 s, most of which is the iLQR phase
(seeding took up to 10 s against up to 9 s before).
