// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#ifndef FBSDE_TRAJ_OPT_CONST_LINEAR_SDE_STATE_DRIFT_TERM_HPP_
#define FBSDE_TRAJ_OPT_CONST_LINEAR_SDE_STATE_DRIFT_TERM_HPP_

#include <cstddef>
#include <utility>

#include <Eigen/Core>

namespace fbsde_traj_opt {

// An SdeStateDriftTerm (see sde_term_concepts.hpp) whose uncontrolled drift is the constant
// linear map `f(k, x_k) = A * x_k`.
//
// `N` is the compile-time state dimension; `Scalar` defaults to `double`.
template <int N, typename Scalar = double>
class ConstLinearSdeStateDriftTerm {
 public:
  using State = Eigen::Matrix<Scalar, N, 1>;

  // Builds a ConstLinearSdeStateDriftTerm whose drift is `A * x_k`.
  explicit ConstLinearSdeStateDriftTerm(Eigen::Matrix<Scalar, N, N> a) noexcept : a_(std::move(a)) {}

  // Returns `A * state`, independent of `stage`.
  auto operator()([[maybe_unused]] std::size_t stage, const State& state) const noexcept -> State { return a_ * state; }

 private:
  Eigen::Matrix<Scalar, N, N> a_;
};

}  // namespace fbsde_traj_opt

#endif  // FBSDE_TRAJ_OPT_CONST_LINEAR_SDE_STATE_DRIFT_TERM_HPP_
