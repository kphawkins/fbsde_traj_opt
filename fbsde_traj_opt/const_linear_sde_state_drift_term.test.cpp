// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#include "fbsde_traj_opt/const_linear_sde_state_drift_term.hpp"

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "fbsde_traj_opt/sde_term_concepts.hpp"

namespace fbsde_traj_opt {
namespace {

using Term3 = ConstLinearSdeStateDriftTerm<3>;

static_assert(SdeStateDriftTerm<Term3, Eigen::Vector3d>);

TEST(ConstLinearSdeStateDriftTermTest, OperatorReturnsAxForTheIdentityMatrix) {
  const Term3 term(Eigen::Matrix3d::Identity());
  const Eigen::Vector3d state(1.0, -2.0, 3.0);

  const Eigen::Vector3d drift = term(0, state);

  EXPECT_TRUE(drift.isApprox(state));
}

TEST(ConstLinearSdeStateDriftTermTest, OperatorReturnsAxForAnArbitraryMatrix) {
  Eigen::Matrix3d a;
  a << 1.0, 2.0, 0.0,  //
      0.0, 1.0, -1.0,  //
      3.0, 0.0, 2.0;
  const Term3 term(a);
  const Eigen::Vector3d state(1.0, 2.0, 3.0);

  const Eigen::Vector3d drift = term(0, state);

  EXPECT_TRUE(drift.isApprox(a * state));
}

TEST(ConstLinearSdeStateDriftTermTest, OperatorIgnoresStage) {
  const Term3 term(Eigen::Matrix3d::Identity());
  const Eigen::Vector3d state(1.0, -2.0, 3.0);

  const Eigen::Vector3d drift_stage_zero = term(0, state);
  const Eigen::Vector3d drift_stage_far = term(1000, state);

  EXPECT_TRUE(drift_stage_zero.isApprox(drift_stage_far));
}

}  // namespace
}  // namespace fbsde_traj_opt
