# fbsde_traj_opt

Trajectory optimization via forward-backward stochastic differential equations (FBSDEs).

This is a research library for algorithms that rapidly produce closed-loop optimal trajectory
controllers for cost functions that are defined arbitrarily and encountered online. The approach
rests on the connection between FBSDEs and stochastic optimal control problems established by the
Feynman-Kac equations, which turns the value function of a control problem into the solution of a
backward SDE that can be estimated by forward sampling.

The methods implemented here follow the thesis below, in particular the discrete-time FBSDE
estimators (DT-FBSDE) and Forward-Backward RRT (FBRRT).

## Reference

Kelsey P. Hawkins, *Feynman-Kac Numerical Techniques for Stochastic Optimal Control*,
Ph.D. thesis, Georgia Institute of Technology, 2021.
<https://repository.gatech.edu/entities/publication/0011748b-4460-4292-b861-eca1a976eb0e>

## Status

Early development. The public API is unstable.

## What is here

The library is built out of small functor types -- drift, diffusion, cost, policy -- that are
composed into models, with concepts rather than base classes stating what each one has to provide.
Everything is fixed-size at compile time and allocates nothing on the sampling path.

- **Forward SDE models.** A discrete-time, control-affine SDE assembled from an uncontrolled
  drift, a control drift matrix, and a diffusion term.
- **Trajectory batches.** `TrajectoryBatch` samples a fixed number of closed-loop rollouts over a
  fixed horizon and answers cross-sectional questions about them -- the state distribution at a
  stage, its mean, and the expected cost-to-go from every stage.
- **Reproducible noise.** The noise comes from a stateless, coordinate-addressed sampler, so two
  batches with the same seed and shape see exactly the same random numbers even if their
  dynamics, cost, or policy differ. That is what makes two batches comparable: a difference
  between them is a difference between the things that changed.
- **Finite-horizon LQR.** `SolveFiniteHorizonLqr` runs the backward Riccati recursion over a
  linear model and a quadratic cost and returns the optimal stage-varying feedback policy, ready
  to be handed straight back to the sampler. `SolveFiniteHorizonLqrWithCostToGo` also returns the
  cost-to-go Hessian at every stage, which makes an LQR problem a ground truth an estimator can be
  checked against rather than only compared with.
- **Value function approximation.** A value function at one stage is anything that can report its
  value, its state gradient, and its state Hessian -- stated as a concept, so models can be
  swapped without touching the algorithms that use them. `SoftMinQuadraticValueFunctionApprox` is
  the first: a smooth minimum over several convex quadratics, exact on the LQR case and able to
  represent a multi-basin landscape. Every evaluation is batched over a compile-time number of
  samples held one per column.
- **Backward-step estimation.** `TaylorNoiselessBackwardStepEstimator` implements the off-policy
  drifted Taylor noiseless estimator of the thesis: given a state, the drift the forward pass used
  to leave it, and the next stage's value function, it estimates this stage's. It depends on no
  noise realization, so its targets are fixed labels; and where the true value function is
  quadratic it is exact.
- **Fitting.** `ValueFunctionSgdFitter` drives a value function toward those targets by minibatch
  Adam, damped so the representation improves without lurching -- principally by a trust region on
  the change the step makes to the function's own values, which means the same thing for every
  model.
- **Visualization.** `//fbsde_traj_opt/viz` turns a batch into Plotly figures and writes them as a
  self-contained HTML page, which the example binary opens for you.

## Worked examples

```sh
bazel run //examples:lqr_experiments
```

This solves three LQR problems from the control literature, rolls out both the optimal policy and
a plausible fixed-gain baseline over identical noise, and writes one HTML report per model to open
in your browser. Each report shows the sampled state distributions with the mean trajectory
over them, a phase portrait, and the expected cost-to-go of both policies.

[![The two-mass spring report](docs/screenshots/two-mass-spring.png)](docs/screenshots/)

The models are the double integrator, the inverted pendulum on a cart, and the Wie-Bernstein
two-mass spring benchmark; the baselines cost between 7 and 36 times what the optimal policy does.
See [docs/screenshots](docs/screenshots/) for each one and for why its baseline fails.

Pass `--output-dir` to put the reports somewhere other than `lqr_reports/`, and `--seed` to change
the noise both policies share.

```sh
bazel run //examples:value_function_fitting_experiments
```

This runs the value function approximation suite and writes three more reports. The first fits the
soft-minimum model to a one-dimensional target that is nowhere a quadratic, and charts the
approach along with how often the trust region shortened a step. The second sweeps the Taylor
Noiseless estimator backward over an LQR problem, whose value function is known in closed form, and
takes the error apart into the estimator's own -- which comes out at the level of rounding -- and
the regression's. The third fits a two-dimensional target and draws the target, the approximation,
and the signed error as heatmaps over the state plane.

It takes the same `--output-dir` and `--seed` flags.

## Prerequisites

- [Bazelisk](https://github.com/bazelbuild/bazelisk) (it reads `.bazelversion` and fetches the
  matching Bazel release)
- Clang with C++23 support, including `<expected>`
- [pre-commit](https://pre-commit.com/), for contributors

Dependencies -- Eigen, nlohmann/json, and GoogleTest -- are fetched by Bazel from the Bazel
Central Registry; there is nothing to install by hand.

## Build and test

```sh
bazel build //...
bazel test //...
```

## Contributing

Install the hooks once after cloning:

```sh
pre-commit install
```

Formatting is `clang-format` (Google style, 2-space indent, 120 columns, west-const) and static
analysis is `clang-tidy`; both are enforced on commit.

## License

MIT. See [LICENSE](LICENSE).
