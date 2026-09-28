# Example screenshots

One full-page capture of each report written by `//examples:lqr_experiments`. Every image shows
the same three things for one model: the sampled state distributions under the LQR policy and
under the baseline controller, a phase portrait where one is informative, and the expected
cost-to-go of both policies over the horizon.

| Model | Baseline | How the baseline fails | Cost relative to LQR |
|---|---|---|---|
| [Double integrator](double-integrator.png) | proportional position feedback | closed-loop poles on the imaginary axis, so the mass oscillates forever instead of settling | 36x |
| [Inverted pendulum on a cart](inverted-pendulum-on-a-cart.png) | pole-angle PD | rights the pole, but every newton that does also pushes the cart, and nothing watches the cart | 35x |
| [Two-mass spring](two-mass-spring.png) | colocated PD on the driven mass | pins the driven mass, and a pinned mass is a wall, so the other rings against it undamped | 7.6x |

The cost-to-go figure is the one that settles each comparison, and it is on a logarithmic axis
because the two policies differ by more than an order of magnitude at every stage.

## Regenerating them

Write the reports, then capture each page at a height that fits all of its figures:

```sh
bazel run //examples:lqr_experiments -- --output-dir /tmp/lqr_reports

chrome --headless --disable-gpu --hide-scrollbars --virtual-time-budget=15000 \
  --window-size=1180,2520 \
  --screenshot=/tmp/lqr_reports/double-integrator.png \
  file:///tmp/lqr_reports/double-integrator.html
```

The `--virtual-time-budget` matters: the page fetches plotly.js from a CDN and draws nothing until
it arrives, so a capture without it catches an empty page. The inverted pendulum report has one
figure fewer than the other two and fits in a window height of 1960.
