// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#ifndef FBSDE_TRAJ_OPT_QUADRATIC_REGULATOR_SDE_RUNNING_COST_TERM_HPP_
#define FBSDE_TRAJ_OPT_QUADRATIC_REGULATOR_SDE_RUNNING_COST_TERM_HPP_

#include <cstddef>
#include <utility>

#include <Eigen/Core>

namespace fbsde_traj_opt {

// An SdeRunningCostTerm (see sde_term_concepts.hpp) whose running cost is the constant quadratic
// form
//
//   l(k, x_k, u_k) = x_k^T Q x_k + u_k^T R u_k + 2 x_k^T N u_k,
//
// the standard stage cost of a linear-quadratic regulator.
//
// `N` is the compile-time state dimension, `M` is the compile-time control dimension, and
// `Scalar` defaults to `double`.
template <int N, int M, typename Scalar = double>
class QuadraticRegulatorSdeRunningCostTerm {
 public:
  using State = Eigen::Matrix<Scalar, N, 1>;
  using Control = Eigen::Matrix<Scalar, M, 1>;

  // Builds a QuadraticRegulatorSdeRunningCostTerm whose running cost is
  // x^T q x + u^T r u + 2 x^T n u, for the constant matrices `q`, `r`, and `n`.
  QuadraticRegulatorSdeRunningCostTerm(Eigen::Matrix<Scalar, N, N> q,
                                       Eigen::Matrix<Scalar, M, M> r,
                                       Eigen::Matrix<Scalar, N, M> n) noexcept
      : q_(std::move(q)), r_(std::move(r)), n_(std::move(n)) {}

  // Returns x^T Q x + u^T R u + 2 x^T N u, independent of `stage`.
  auto operator()([[maybe_unused]] std::size_t stage, const State& state, const Control& control) const noexcept
      -> Scalar {
    return state.dot(q_ * state) + control.dot(r_ * control) + static_cast<Scalar>(2) * state.dot(n_ * control);
  }

 private:
  Eigen::Matrix<Scalar, N, N> q_;
  Eigen::Matrix<Scalar, M, M> r_;
  Eigen::Matrix<Scalar, N, M> n_;
};

}  // namespace fbsde_traj_opt

#endif  // FBSDE_TRAJ_OPT_QUADRATIC_REGULATOR_SDE_RUNNING_COST_TERM_HPP_
