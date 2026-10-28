// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#ifndef FBSDE_TRAJ_OPT_CONST_LINEAR_FEEDBACK_SDE_CONTROL_POLICY_TERM_HPP_
#define FBSDE_TRAJ_OPT_CONST_LINEAR_FEEDBACK_SDE_CONTROL_POLICY_TERM_HPP_

#include <cstddef>
#include <utility>

#include <Eigen/Core>

namespace fbsde_traj_opt {

// An SdeControlPolicyTerm (see sde_term_concepts.hpp) whose feedback control policy is the
// constant affine map `pi(k, x_k) = K * x_k + l`.
//
// `N` is the compile-time state dimension, `M` is the compile-time control dimension, and
// `Scalar` defaults to `double`.
template <int N, int M, typename Scalar = double>
class ConstLinearFeedbackSdeControlPolicyTerm {
 public:
  using State = Eigen::Matrix<Scalar, N, 1>;
  using Control = Eigen::Matrix<Scalar, M, 1>;

  // Builds a ConstLinearFeedbackSdeControlPolicyTerm whose control policy is `k * x_k + l`.
  ConstLinearFeedbackSdeControlPolicyTerm(Eigen::Matrix<Scalar, M, N> k, Eigen::Matrix<Scalar, M, 1> l) noexcept
      : k_(std::move(k)), l_(std::move(l)) {}

  // Returns `K * state + l`, independent of `stage`.
  auto operator()([[maybe_unused]] std::size_t stage, const State& state) const noexcept -> Control {
    return k_ * state + l_;
  }

 private:
  Eigen::Matrix<Scalar, M, N> k_;
  Eigen::Matrix<Scalar, M, 1> l_;
};

}  // namespace fbsde_traj_opt

#endif  // FBSDE_TRAJ_OPT_CONST_LINEAR_FEEDBACK_SDE_CONTROL_POLICY_TERM_HPP_
