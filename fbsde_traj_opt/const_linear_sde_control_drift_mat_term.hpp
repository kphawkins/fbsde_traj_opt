// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#ifndef FBSDE_TRAJ_OPT_CONST_LINEAR_SDE_CONTROL_DRIFT_MAT_TERM_HPP_
#define FBSDE_TRAJ_OPT_CONST_LINEAR_SDE_CONTROL_DRIFT_MAT_TERM_HPP_

#include <cstddef>
#include <utility>

#include <Eigen/Core>

namespace fbsde_traj_opt {

// An SdeControlDriftMatTerm (see sde_term_concepts.hpp) whose control drift matrix `B(k, x_k)` is
// the constant matrix `B`, so that the control-affine drift term is `B * u_k`.
//
// `N` is the compile-time state dimension, `M` is the compile-time control dimension, and
// `Scalar` defaults to `double`.
template <int N, int M, typename Scalar = double>
class ConstLinearSdeControlDriftMatTerm {
 public:
  using State = Eigen::Matrix<Scalar, N, 1>;

  // Builds a ConstLinearSdeControlDriftMatTerm whose control drift matrix is `b`.
  explicit ConstLinearSdeControlDriftMatTerm(Eigen::Matrix<Scalar, N, M> b) noexcept : b_(std::move(b)) {}

  // Returns `B`, independent of `stage` and `state`.
  auto operator()([[maybe_unused]] std::size_t stage, [[maybe_unused]] const State& state) const noexcept
      -> Eigen::Matrix<Scalar, N, M> {
    return b_;
  }

 private:
  Eigen::Matrix<Scalar, N, M> b_;
};

}  // namespace fbsde_traj_opt

#endif  // FBSDE_TRAJ_OPT_CONST_LINEAR_SDE_CONTROL_DRIFT_MAT_TERM_HPP_
