// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#include "fbsde_traj_opt/quadratic_regulator_sde_terminal_cost_term.hpp"

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "fbsde_traj_opt/sde_term_concepts.hpp"

namespace fbsde_traj_opt {
namespace {

using Term3 = QuadraticRegulatorSdeTerminalCostTerm<3>;

static_assert(SdeTerminalCostTerm<Term3, Eigen::Vector3d>);

TEST(QuadraticRegulatorSdeTerminalCostTermTest, OperatorComputesQuadraticForm) {
  const Eigen::Matrix3d f = Eigen::Matrix3d::Identity();
  const Term3 term(f);

  const Eigen::Vector3d state(1.0, 2.0, 3.0);

  // x^T F x = 1 + 4 + 9 = 14.
  const double cost = term(state);

  EXPECT_DOUBLE_EQ(cost, 14.0);
}

TEST(QuadraticRegulatorSdeTerminalCostTermTest, OperatorHandlesNonDiagonalF) {
  Eigen::Matrix3d f;
  f << 1.0, 1.0, 0.0,  //
      1.0, 1.0, 0.0,   //
      0.0, 0.0, 1.0;
  const Term3 term(f);

  const Eigen::Vector3d state(1.0, 1.0, 1.0);

  // x^T F x = 1 + 1 + 1 + 1 + 1 = 5.
  const double cost = term(state);

  EXPECT_DOUBLE_EQ(cost, 5.0);
}

}  // namespace
}  // namespace fbsde_traj_opt
