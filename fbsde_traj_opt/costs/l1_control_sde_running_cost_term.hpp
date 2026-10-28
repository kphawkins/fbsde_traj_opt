// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#ifndef FBSDE_TRAJ_OPT_L1_CONTROL_SDE_RUNNING_COST_TERM_HPP_
#define FBSDE_TRAJ_OPT_L1_CONTROL_SDE_RUNNING_COST_TERM_HPP_

#include <cmath>
#include <cstddef>

#include <Eigen/Core>

#include "fbsde_traj_opt/utils/result.hpp"

namespace fbsde_traj_opt {

// An SdeRunningCostTerm (see sde_term_concepts.hpp) that charges the L1 norm of the control,
//
//   l(k, x_k, u_k) = c || u_k ||_1 = c sum_j | u_k[j] |,
//
// and nothing for the state -- the running cost of a minimum-fuel problem (Hawkins (2021),
// Section 5.5, where it is written `a |u|` in continuous time and multiplied by the time step to
// give the stage cost).
//
// Why a separate term rather than a special case of the quadratic one. The L1 norm is what makes
// the optimal control *sparse*: with the control confined to a box, the minimizer of an L1 cost
// plus anything smooth and convex in the control is pushed to the faces of the box or to exactly
// zero, never loitering at small nonzero values, so the optimal policy is bang-off-bang. A
// quadratic penalty cannot produce that shape at all -- it makes small controls nearly free -- and
// the shape is the whole point of the problem class. Algorithms that exploit it (see
// taylor_q_function_l1_policy.hpp) read the weight through L1Weight().
//
// `N` is the compile-time state dimension, `M` is the compile-time control dimension, and
// `Scalar` defaults to `double`.
template <int N, int M, typename Scalar = double>
class L1ControlSdeRunningCostTerm {
 public:
  using State = Eigen::Matrix<Scalar, N, 1>;
  using Control = Eigen::Matrix<Scalar, M, 1>;

  // Builds the term with stage cost `l1_weight * || u ||_1`.
  //
  // `l1_weight` is the per-stage weight -- already multiplied by the time step, if the problem
  // was posed in continuous time.
  //
  // Fails unless `l1_weight` is finite and non-negative. Zero is allowed: it is the pure
  // terminal-cost problem, whose optimal control is unconstrained in effort.
  static auto Make(Scalar l1_weight) noexcept -> Result<L1ControlSdeRunningCostTerm> {
    RESULT_ASSERT(std::isfinite(l1_weight) && l1_weight >= Scalar{0},
                  "L1ControlSdeRunningCostTerm::Make: the L1 weight must be finite and non-negative.");
    return SuccessResult(L1ControlSdeRunningCostTerm(l1_weight));
  }

  // Returns `c || u ||_1`, independent of `stage` and `state`.
  auto operator()([[maybe_unused]] std::size_t stage,
                  [[maybe_unused]] const State& state,
                  const Control& control) const noexcept -> Scalar {
    return l1_weight_ * control.template lpNorm<1>();
  }

  // Returns `c`, the per-stage weight on the L1 norm of the control.
  [[nodiscard]] auto L1Weight() const noexcept -> Scalar { return l1_weight_; }

 private:
  explicit L1ControlSdeRunningCostTerm(Scalar l1_weight) noexcept : l1_weight_(l1_weight) {}

  Scalar l1_weight_;
};

}  // namespace fbsde_traj_opt

#endif  // FBSDE_TRAJ_OPT_L1_CONTROL_SDE_RUNNING_COST_TERM_HPP_
