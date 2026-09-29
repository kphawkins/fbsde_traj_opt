// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#ifndef FBSDE_TRAJ_OPT_STAGE_VARYING_LINEAR_FEEDBACK_SDE_CONTROL_POLICY_TERM_HPP_
#define FBSDE_TRAJ_OPT_STAGE_VARYING_LINEAR_FEEDBACK_SDE_CONTROL_POLICY_TERM_HPP_

#include <array>
#include <cstddef>
#include <utility>

#include <Eigen/Core>

namespace fbsde_traj_opt {

// An SdeControlPolicyTerm (see sde_term_concepts.hpp) whose feedback control policy is linear
// with a gain that changes from stage to stage:
//
//   pi(k, x_k) = K_k * x_k.
//
// This is the shape of the optimal policy of a finite-horizon linear-quadratic regulator, whose
// gain genuinely does vary over the horizon -- it approaches the infinite-horizon gain in the
// early stages and then swings away near the terminal stage, where there is no longer enough time
// left to recover the control effort an aggressive correction would cost.
//
// The policy is linear rather than affine. A quadratic regulator cost has no linear term, so its optimal policy has no
// offset; adding one that every producer would set to zero would be dead weight. An affine stage-varying policy belongs
// with the affine cost that would motivate it.
//
// `N` is the compile-time state dimension, `M` the compile-time control dimension,
// `NumControlStages` the number of stages at which a control is applied, and `Scalar` defaults to
// `double`. Note that `NumControlStages` is one less than the number of stages of a trajectory
// over the same horizon: no control is applied at the terminal stage.
template <int N, int M, std::size_t NumControlStages, typename Scalar = double>
class StageVaryingLinearFeedbackSdeControlPolicyTerm {
  static_assert(NumControlStages >= 1, "A stage-varying policy needs at least one control stage.");

 public:
  using State = Eigen::Matrix<Scalar, N, 1>;
  using Control = Eigen::Matrix<Scalar, M, 1>;
  using Gain = Eigen::Matrix<Scalar, M, N>;
  using GainArray = std::array<Gain, NumControlStages>;

  static constexpr std::size_t kNumControlStages = NumControlStages;

  // Builds a StageVaryingLinearFeedbackSdeControlPolicyTerm from one gain per control stage,
  // indexed by stage.
  explicit StageVaryingLinearFeedbackSdeControlPolicyTerm(GainArray gains) noexcept : gains_(std::move(gains)) {}

  // Returns `K_stage * state`.
  //
  // Precondition: `stage < kNumControlStages`. A trajectory over `kNumControlStages + 1` stages
  // applies a control at exactly the stages this policy has a gain for, so a caller rolling one
  // out satisfies this by construction.
  auto operator()(std::size_t stage, const State& state) const noexcept -> Control { return gains_[stage] * state; }

  // Returns the gain `K_stage` applied at `stage`.
  //
  // Precondition: `stage < kNumControlStages`.
  [[nodiscard]] auto GainAtStage(std::size_t stage) const noexcept -> const Gain& { return gains_[stage]; }

 private:
  GainArray gains_;
};

}  // namespace fbsde_traj_opt

#endif  // FBSDE_TRAJ_OPT_STAGE_VARYING_LINEAR_FEEDBACK_SDE_CONTROL_POLICY_TERM_HPP_
