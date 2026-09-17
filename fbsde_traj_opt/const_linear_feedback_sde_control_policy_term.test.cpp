// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#include "fbsde_traj_opt/const_linear_feedback_sde_control_policy_term.hpp"

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "fbsde_traj_opt/sde_term_concepts.hpp"

namespace fbsde_traj_opt {
namespace {

using Term3x2 = ConstLinearFeedbackSdeControlPolicyTerm<3, 2>;

static_assert(SdeControlPolicyTerm<Term3x2, Eigen::Vector3d, Eigen::Vector2d>);

TEST(ConstLinearFeedbackSdeControlPolicyTermTest, OperatorComputesAffineFeedback) {
  Eigen::Matrix<double, 2, 3> k;
  k << 1.0, 0.0, 2.0,  //
      0.0, 1.0, -1.0;
  const Eigen::Vector2d l(0.5, -0.5);
  const Term3x2 term(k, l);

  const Eigen::Vector3d state(1.0, 2.0, 3.0);

  // K * x = (1 + 6, 2 - 3) = (7, -1); plus l = (7.5, -1.5).
  const Eigen::Vector2d control = term(0, state);

  EXPECT_TRUE(control.isApprox(Eigen::Vector2d(7.5, -1.5)));
}

TEST(ConstLinearFeedbackSdeControlPolicyTermTest, OperatorReturnsOffsetForZeroState) {
  const Eigen::Matrix<double, 2, 3> k = Eigen::Matrix<double, 2, 3>::Zero();
  const Eigen::Vector2d l(3.0, -2.0);
  const Term3x2 term(k, l);

  const Eigen::Vector2d control = term(0, Eigen::Vector3d::Zero());

  EXPECT_TRUE(control.isApprox(l));
}

TEST(ConstLinearFeedbackSdeControlPolicyTermTest, OperatorIgnoresStage) {
  Eigen::Matrix<double, 2, 3> k;
  k << 1.0, 0.0, 2.0,  //
      0.0, 1.0, -1.0;
  const Eigen::Vector2d l(0.5, -0.5);
  const Term3x2 term(k, l);

  const Eigen::Vector3d state(1.0, 2.0, 3.0);

  const Eigen::Vector2d control_stage_zero = term(0, state);
  const Eigen::Vector2d control_stage_far = term(1000, state);

  EXPECT_TRUE(control_stage_zero.isApprox(control_stage_far));
}

}  // namespace
}  // namespace fbsde_traj_opt
