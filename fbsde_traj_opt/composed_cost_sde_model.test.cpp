// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#include "fbsde_traj_opt/composed_cost_sde_model.hpp"

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "fbsde_traj_opt/quadratic_regulator_sde_running_cost_term.hpp"
#include "fbsde_traj_opt/quadratic_regulator_sde_terminal_cost_term.hpp"

namespace fbsde_traj_opt {
namespace {

using RunningCost3x2 = QuadraticRegulatorSdeRunningCostTerm<3, 2>;
using TerminalCost3 = QuadraticRegulatorSdeTerminalCostTerm<3>;
using CostModel3x2 = ComposedCostSdeModel<3, 2, RunningCost3x2, TerminalCost3>;

// Builds a cost model with Q = I, R = 2 I, N = 0, and F = 10 I.
auto MakeTestCostModel() noexcept -> CostModel3x2 {
  const RunningCost3x2 running_cost(
      Eigen::Matrix3d::Identity(), Eigen::Matrix2d::Identity() * 2.0, Eigen::Matrix<double, 3, 2>::Zero());
  const TerminalCost3 terminal_cost(Eigen::Matrix3d::Identity() * 10.0);
  return {running_cost, terminal_cost};
}

TEST(ComposedCostSdeModelTest, RunningCostAccessorEvaluatesTheRunningCost) {
  const CostModel3x2 cost_model = MakeTestCostModel();

  const Eigen::Vector3d state(1.0, 2.0, 3.0);
  const Eigen::Vector2d control(1.0, 1.0);

  // x^T x = 14; u^T (2 I) u = 4; cross term is zero.
  EXPECT_DOUBLE_EQ(cost_model.running_cost()(0, state, control), 18.0);
}

TEST(ComposedCostSdeModelTest, TerminalCostAccessorEvaluatesTheTerminalCost) {
  const CostModel3x2 cost_model = MakeTestCostModel();

  const Eigen::Vector3d state(1.0, 2.0, 3.0);

  // x^T (10 I) x = 140.
  EXPECT_DOUBLE_EQ(cost_model.terminal_cost()(state), 140.0);
}

TEST(ComposedCostSdeModelTest, AccessorsExposeTheUnderlyingTermMatrices) {
  const CostModel3x2 cost_model = MakeTestCostModel();

  EXPECT_TRUE(cost_model.running_cost().state_cost_mat().isApprox(Eigen::Matrix3d::Identity()));
  EXPECT_TRUE(cost_model.running_cost().control_cost_mat().isApprox(Eigen::Matrix2d::Identity() * 2.0));
  EXPECT_TRUE(cost_model.terminal_cost().terminal_cost_mat().isApprox(Eigen::Matrix3d::Identity() * 10.0));
}

}  // namespace
}  // namespace fbsde_traj_opt
