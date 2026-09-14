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

## Prerequisites

- [Bazelisk](https://github.com/bazelbuild/bazelisk) (it reads `.bazelversion` and fetches the
  matching Bazel release)
- Clang with C++23 support, including `<expected>`
- [pre-commit](https://pre-commit.com/), for contributors

Dependencies -- Eigen and GoogleTest -- are fetched by Bazel from the Bazel Central Registry;
there is nothing to install by hand.

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
