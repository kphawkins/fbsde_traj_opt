// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#include "fbsde_traj_opt/l1_control_sde_running_cost_term.hpp"

#include <limits>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "fbsde_traj_opt/sde_term_concepts.hpp"
#include "fbsde_traj_opt/utils/result.hpp"

namespace fbsde_traj_opt {
namespace {

using Term3x2 = L1ControlSdeRunningCostTerm<3, 2>;

static_assert(SdeRunningCostTerm<Term3x2, Eigen::Vector3d, Eigen::Vector2d>);

TEST(L1ControlSdeRunningCostTermTest, OperatorChargesWeightedL1NormOfControl) {
  const Result<Term3x2> term = Term3x2::Make(0.5);
  ASSERT_TRUE(term.has_value()) << term.error();

  // 0.5 * (|1.5| + |-2|) = 1.75.
  EXPECT_DOUBLE_EQ((*term)(0, Eigen::Vector3d(1.0, 2.0, 3.0), Eigen::Vector2d(1.5, -2.0)), 1.75);
}

TEST(L1ControlSdeRunningCostTermTest, OperatorIgnoresStageAndState) {
  const Result<Term3x2> term = Term3x2::Make(2.0);
  ASSERT_TRUE(term.has_value()) << term.error();
  const Eigen::Vector2d control(-0.25, 0.75);

  EXPECT_DOUBLE_EQ((*term)(0, Eigen::Vector3d::Zero(), control), (*term)(99, Eigen::Vector3d(4.0, -5.0, 6.0), control));
}

TEST(L1ControlSdeRunningCostTermTest, ZeroControlIsFree) {
  const Result<Term3x2> term = Term3x2::Make(3.0);
  ASSERT_TRUE(term.has_value()) << term.error();

  EXPECT_DOUBLE_EQ((*term)(0, Eigen::Vector3d(1.0, 1.0, 1.0), Eigen::Vector2d::Zero()), 0.0);
}

TEST(L1ControlSdeRunningCostTermTest, L1WeightReturnsConstructorWeight) {
  const Result<Term3x2> term = Term3x2::Make(0.125);
  ASSERT_TRUE(term.has_value()) << term.error();

  EXPECT_DOUBLE_EQ(term->L1Weight(), 0.125);
}

TEST(L1ControlSdeRunningCostTermTest, MakeAcceptsZeroWeight) {
  EXPECT_TRUE(Term3x2::Make(0.0).has_value());
}

TEST(L1ControlSdeRunningCostTermTest, MakeRejectsNegativeOrNonFiniteWeight) {
  EXPECT_FALSE(Term3x2::Make(-1.0).has_value());
  EXPECT_FALSE(Term3x2::Make(std::numeric_limits<double>::infinity()).has_value());
  EXPECT_FALSE(Term3x2::Make(std::numeric_limits<double>::quiet_NaN()).has_value());
}

}  // namespace
}  // namespace fbsde_traj_opt
