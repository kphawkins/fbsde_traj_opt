// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#ifndef FBSDE_TRAJ_OPT_EXAMPLES_VALUE_FUNCTION_FITTING_HPP_
#define FBSDE_TRAJ_OPT_EXAMPLES_VALUE_FUNCTION_FITTING_HPP_

#include <cstdint>
#include <string>
#include <vector>

#include "fbsde_traj_opt/utils/result.hpp"
#include "fbsde_traj_opt/viz/plotly_figure.hpp"

namespace fbsde_traj_opt::examples {

// Three experiments that show the value function approximation suite working, each producing a
// page of figures.
//
// They are arranged so that each one isolates a different claim:
//
//   * The generic fit exercises the model and the fitter alone, with no FBSDE machinery at all.
//     Its claim is the plain one -- that a soft minimum of quadratics can be driven to a function
//     that is nothing like a quadratic, gradually, by stochastic gradient descent.
//
//   * The LQR backward pass exercises the whole pipeline against an answer known in closed form.
//     Its claim is sharper: the value function of an LQR problem is exactly quadratic, so the
//     Taylor Noiseless estimator is exact there, and the backward recursion has a right answer to
//     be held to at every stage rather than merely a plausible shape.
//
//   * The two-dimensional fit exercises the parts a one-dimensional problem cannot reach: the
//     full state Hessian, and a soft minimum whose components actually separate into distinct
//     basins.

// One experiment's output: a page of figures and the numbers worth printing beside them.
struct ValueFunctionFittingReport {
  std::string name{};

  // A sentence or two of prose: what the reader is looking at, and what it is supposed to show.
  std::string description{};

  // One line naming what this run showed, for the terminal. Each experiment's headline claim is
  // a different quantity, so each writes its own rather than the caller guessing from the numbers
  // below.
  std::string summary{};

  std::vector<viz::PlotlyFigure> figures{};

  // The root mean squared residual over the fitted samples at the end of the run, in the units of
  // the value function.
  double final_root_mean_squared_residual = 0.0;

  // The headline accuracy of the run, relative to the quantity being approximated. For the LQR
  // experiment this is measured against the exact Riccati value function; for the other two,
  // against the target function being fitted.
  double worst_relative_error = 0.0;

  // The error attributable to the backward-step estimator alone -- that is, with the estimator
  // reading an exact next-stage value function rather than a fitted one, so that no regression
  // error is folded in. Zero for the experiments that run no estimator.
  double estimator_relative_error = 0.0;
};

// Fits the soft-min-of-quadratics model to a one-dimensional multi-basin target by SGD, with the
// function-space trust region in force, and charts the approach.
//
// Fails if any model, fitter, or figure the experiment builds is rejected.
[[nodiscard]] auto RunGenericFunctionFittingExperiment(std::uint64_t seed) noexcept
    -> Result<ValueFunctionFittingReport>;

// Sweeps the Taylor Noiseless estimator backward over an LQR problem, fitting a value function
// approximation at each stage to the targets it produces, and charts the result against the exact
// Riccati value function.
//
// The forward samples are drawn under a deliberately suboptimal drift, so the estimator is
// genuinely off-policy and the Girsanov terms it carries are doing work rather than vanishing.
//
// Fails if the Riccati recursion, the rollout, or any fit is rejected.
[[nodiscard]] auto RunLqrBackwardPassExperiment(std::uint64_t seed) noexcept -> Result<ValueFunctionFittingReport>;

// Fits the model to a two-dimensional multi-basin target and charts the target, the
// approximation, and the signed error as heatmaps over the state plane.
//
// Fails if any model, fitter, or figure the experiment builds is rejected.
[[nodiscard]] auto RunTwoDimensionalFittingExperiment(std::uint64_t seed) noexcept
    -> Result<ValueFunctionFittingReport>;

}  // namespace fbsde_traj_opt::examples

#endif  // FBSDE_TRAJ_OPT_EXAMPLES_VALUE_FUNCTION_FITTING_HPP_
