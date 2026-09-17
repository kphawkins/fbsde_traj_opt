// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#ifndef FBSDE_TRAJ_OPT_QUADRATIC_REGULATOR_SDE_TERMINAL_COST_TERM_HPP_
#define FBSDE_TRAJ_OPT_QUADRATIC_REGULATOR_SDE_TERMINAL_COST_TERM_HPP_

#include <utility>

#include <Eigen/Core>

namespace fbsde_traj_opt {

// An SdeTerminalCostTerm (see sde_term_concepts.hpp) whose terminal cost is the constant
// quadratic form
//
//   phi(x_K) = x_K^T F x_K,
//
// the standard terminal cost of a linear-quadratic regulator.
//
// `N` is the compile-time state dimension; `Scalar` defaults to `double`.
template <int N, typename Scalar = double>
class QuadraticRegulatorSdeTerminalCostTerm {
 public:
  using State = Eigen::Matrix<Scalar, N, 1>;

  // Builds a QuadraticRegulatorSdeTerminalCostTerm whose terminal cost is x^T f x, for the
  // constant matrix `f`.
  explicit QuadraticRegulatorSdeTerminalCostTerm(Eigen::Matrix<Scalar, N, N> f) noexcept : f_(std::move(f)) {}

  // Returns x^T F x.
  auto operator()(const State& state) const noexcept -> Scalar { return state.dot(f_ * state); }

 private:
  Eigen::Matrix<Scalar, N, N> f_;
};

}  // namespace fbsde_traj_opt

#endif  // FBSDE_TRAJ_OPT_QUADRATIC_REGULATOR_SDE_TERMINAL_COST_TERM_HPP_
