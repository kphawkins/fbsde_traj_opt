// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#ifndef FBSDE_TRAJ_OPT_SOLVERS_TAYLOR_GRADIENT_TARGET_HPP_
#define FBSDE_TRAJ_OPT_SOLVERS_TAYLOR_GRADIENT_TARGET_HPP_

#include <algorithm>
#include <cmath>
#include <cstddef>

#include <Eigen/Core>

namespace fbsde_traj_opt {

// The gradient of the noiseless Taylor target (4.37) at `state`, for the target taken with
// `control` against `next_value_function`, the next stage's table:
//
//   G^_i(x) = J(x, u)' grad V~_{i+1}(x + F_i(x, u)),     J = d (x + F_i(x, u)) / dx.
//
// It is the derivative of `Q~_i(x, u*(x))` with two simplifications. The control's own derivative
// is left out: almost everywhere each control coordinate either sits at a bound or at zero, and so
// is locally constant, or is interior, where the Q-function is stationary in it. And the
// noise-curvature term's derivative, which involves the third derivatives of `V~_{i+1}` and
// vanishes when it is quadratic, is dropped. For linear dynamics and a quadratic `V~_{i+1}` the
// result is exact. docs/derivations/gradient_target.html derives it.
//
// The Jacobian `J` is taken by central differences of the model's own noise-free step, of relative
// half-width `relative_step`, so the model need not supply derivatives.
template <typename ForwardModelT, typename ValueFunctionT, typename State, typename Control>
[[nodiscard]] auto TaylorGradientTarget(const ForwardModelT& forward_model,
                                        std::size_t stage,
                                        const State& state,
                                        const Control& control,
                                        const ValueFunctionT& next_value_function,
                                        typename State::Scalar relative_step = 1e-5) noexcept -> State {
  using Scalar = typename State::Scalar;
  constexpr int kStateDim = State::RowsAtCompileTime;
  const State landing = forward_model(stage, state, control, State::Zero());
  Eigen::Matrix<Scalar, kStateDim, kStateDim> jacobian;
  for (Eigen::Index axis = 0; axis < kStateDim; ++axis) {
    const Scalar step = relative_step * std::max(Scalar{1}, std::abs(state[axis]));
    const State offset = step * State::Unit(axis);
    jacobian.col(axis) = (forward_model(stage, State(state + offset), control, State::Zero()) -
                          forward_model(stage, State(state - offset), control, State::Zero())) /
                         (Scalar{2} * step);
  }
  return jacobian.transpose() * next_value_function.Gradient(landing);
}

}  // namespace fbsde_traj_opt

#endif  // FBSDE_TRAJ_OPT_SOLVERS_TAYLOR_GRADIENT_TARGET_HPP_
