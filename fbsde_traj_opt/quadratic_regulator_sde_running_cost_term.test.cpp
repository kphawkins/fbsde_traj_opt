// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#include "fbsde_traj_opt/quadratic_regulator_sde_running_cost_term.hpp"

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "fbsde_traj_opt/sde_term_concepts.hpp"

namespace fbsde_traj_opt {
namespace {

using Term3x2 = QuadraticRegulatorSdeRunningCostTerm<3, 2>;

static_assert(SdeRunningCostTerm<Term3x2, Eigen::Vector3d, Eigen::Vector2d>);

TEST(QuadraticRegulatorSdeRunningCostTermTest, OperatorComputesQuadraticForm) {
  Eigen::Matrix3d q = Eigen::Matrix3d::Identity();
  Eigen::Matrix2d r = 2.0 * Eigen::Matrix2d::Identity();
  Eigen::Matrix<double, 3, 2> n = Eigen::Matrix<double, 3, 2>::Zero();
  const Term3x2 term(q, r, n);

  const Eigen::Vector3d state(1.0, 2.0, 3.0);
  const Eigen::Vector2d control(1.0, -1.0);

  // x^T Q x = 1 + 4 + 9 = 14; u^T R u = 2*(1 + 1) = 4; cross term is zero.
  const double cost = term(0, state, control);

  EXPECT_DOUBLE_EQ(cost, 18.0);
}

TEST(QuadraticRegulatorSdeRunningCostTermTest, OperatorIncludesCrossTerm) {
  const Eigen::Matrix3d q = Eigen::Matrix3d::Zero();
  const Eigen::Matrix2d r = Eigen::Matrix2d::Zero();
  Eigen::Matrix<double, 3, 2> n;
  n << 1.0, 0.0,  //
      0.0, 1.0,   //
      0.0, 0.0;
  const Term3x2 term(q, r, n);

  const Eigen::Vector3d state(1.0, 1.0, 1.0);
  const Eigen::Vector2d control(1.0, 1.0);

  // 2 * x^T N u = 2 * (1*1 + 1*1) = 4.
  const double cost = term(0, state, control);

  EXPECT_DOUBLE_EQ(cost, 4.0);
}

TEST(QuadraticRegulatorSdeRunningCostTermTest, OperatorIgnoresStage) {
  const Eigen::Matrix3d q = Eigen::Matrix3d::Identity();
  const Eigen::Matrix2d r = Eigen::Matrix2d::Identity();
  const Eigen::Matrix<double, 3, 2> n = Eigen::Matrix<double, 3, 2>::Zero();
  const Term3x2 term(q, r, n);

  const Eigen::Vector3d state(1.0, 2.0, 3.0);
  const Eigen::Vector2d control(1.0, -1.0);

  const double cost_stage_zero = term(0, state, control);
  const double cost_stage_far = term(1000, state, control);

  EXPECT_DOUBLE_EQ(cost_stage_zero, cost_stage_far);
}

}  // namespace
}  // namespace fbsde_traj_opt
