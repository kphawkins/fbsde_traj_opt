// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#include "fbsde_traj_opt/stage_varying_linear_feedback_sde_control_policy_term.hpp"

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "fbsde_traj_opt/sde_term_concepts.hpp"

namespace fbsde_traj_opt {
namespace {

using Term3x2 = StageVaryingLinearFeedbackSdeControlPolicyTerm<3, 2, 4>;

static_assert(SdeControlPolicyTerm<Term3x2, Eigen::Vector3d, Eigen::Vector2d>);

// Builds a policy whose stage-k gain is k times the same pattern, so a test can tell the stages
// apart by inspection.
auto MakeTestPolicy() noexcept -> Term3x2 {
  Eigen::Matrix<double, 2, 3> pattern;
  pattern << 1.0, 0.0, 2.0,  //
      0.0, 1.0, -1.0;

  Term3x2::GainArray gains{};
  for (std::size_t stage = 0; stage < Term3x2::kNumControlStages; ++stage) {
    gains[stage] = pattern * static_cast<double>(stage + 1);
  }
  return Term3x2(gains);
}

TEST(StageVaryingLinearFeedbackSdeControlPolicyTermTest, OperatorAppliesTheGainForTheGivenStage) {
  const Term3x2 policy = MakeTestPolicy();

  const Eigen::Vector3d state(1.0, 2.0, 3.0);

  // The stage-0 gain is the pattern itself: K x = (1 + 6, 2 - 3) = (7, -1).
  EXPECT_TRUE(policy(0, state).isApprox(Eigen::Vector2d(7.0, -1.0)));
  // The stage-2 gain is three times the pattern.
  EXPECT_TRUE(policy(2, state).isApprox(Eigen::Vector2d(21.0, -3.0)));
}

TEST(StageVaryingLinearFeedbackSdeControlPolicyTermTest, GainAtStageReturnsTheConstructorGains) {
  const Term3x2 policy = MakeTestPolicy();

  Eigen::Matrix<double, 2, 3> pattern;
  pattern << 1.0, 0.0, 2.0,  //
      0.0, 1.0, -1.0;

  for (std::size_t stage = 0; stage < Term3x2::kNumControlStages; ++stage) {
    EXPECT_TRUE(policy.GainAtStage(stage).isApprox(pattern * static_cast<double>(stage + 1))) << "at stage " << stage;
  }
}

TEST(StageVaryingLinearFeedbackSdeControlPolicyTermTest, ZeroStateGivesZeroControl) {
  const Term3x2 policy = MakeTestPolicy();

  EXPECT_TRUE(policy(1, Eigen::Vector3d::Zero()).isZero());
}

}  // namespace
}  // namespace fbsde_traj_opt
