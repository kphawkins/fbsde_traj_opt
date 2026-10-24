// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#ifndef FBSDE_TRAJ_OPT_VALUE_FUNCTION_VALUE_FUNCTION_SLOPE_TARGETS_HPP_
#define FBSDE_TRAJ_OPT_VALUE_FUNCTION_VALUE_FUNCTION_SLOPE_TARGETS_HPP_

#include <algorithm>
#include <array>
#include <cmath>

#include <Eigen/Core>

#include "fbsde_traj_opt/value_function/value_function_approx_concepts.hpp"

namespace fbsde_traj_opt {

// Targets for a value function's slopes, alongside the usual value targets: for every sample, `D`
// directions `d` and the slope `d' grad V` the fit should have along each.
//
// Why slopes. A policy that minimizes a Q-function over the next stage's value function compares
// that function across the few states its candidate controls land on, which differ along the
// control columns. What decides the control is how the value function changes along those
// columns, not its level, and a fit to values alone leaves that change unconstrained. A fit that
// also matches `b' grad V` for each control column `b` constrains exactly what the policy reads;
// see docs/derivations/gradient_target.html.
//
// `directions[k].col(s)` is sample `s`'s `k`-th direction and `slopes[k][s]` its target slope.
template <typename State, int Count, int D>
struct ValueFunctionSlopeTargets {
  std::array<ValueFunctionStateBatch<State, Count>, static_cast<std::size_t>(D)> directions{};
  std::array<ValueFunctionValueBatch<State, Count>, static_cast<std::size_t>(D)> slopes{};
};

// The half-width of the central difference a fitter takes a model's slope by, along a direction
// `d` from `state`: the step `h d / |d|`, with `h` relative to the state's size, so that it means
// the same thing in any units and the slope `|d| (V(x + h d^) - V(x - h d^)) / 2h` is accurate to
// `O(h^2)` and to rounding.
template <typename State>
[[nodiscard]] auto SlopeDifferenceStep(const State& state) noexcept -> typename State::Scalar {
  using Scalar = typename State::Scalar;
  return Scalar{1e-4} * std::max(Scalar{1}, state.norm());
}

}  // namespace fbsde_traj_opt

#endif  // FBSDE_TRAJ_OPT_VALUE_FUNCTION_VALUE_FUNCTION_SLOPE_TARGETS_HPP_
