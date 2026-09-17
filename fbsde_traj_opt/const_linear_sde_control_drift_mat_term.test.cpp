// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#include "fbsde_traj_opt/const_linear_sde_control_drift_mat_term.hpp"

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "fbsde_traj_opt/sde_term_concepts.hpp"

namespace fbsde_traj_opt {
namespace {

using Term3x2 = ConstLinearSdeControlDriftMatTerm<3, 2>;

static_assert(SdeControlDriftMatTerm<Term3x2, Eigen::Vector3d>);

TEST(ConstLinearSdeControlDriftMatTermTest, OperatorReturnsTheConstructorArgument) {
  Eigen::Matrix<double, 3, 2> b;
  b << 1.0, 0.0,  //
      0.0, 1.0,   //
      2.0, -1.0;
  const Term3x2 term(b);

  const Eigen::Matrix<double, 3, 2> returned = term(0, Eigen::Vector3d::Zero());

  EXPECT_TRUE(returned.isApprox(b));
}

TEST(ConstLinearSdeControlDriftMatTermTest, OperatorIgnoresStageAndState) {
  Eigen::Matrix<double, 3, 2> b;
  b << 1.0, 0.0,  //
      0.0, 1.0,   //
      2.0, -1.0;
  const Term3x2 term(b);

  const Eigen::Matrix<double, 3, 2> b_stage_zero = term(0, Eigen::Vector3d::Zero());
  const Eigen::Matrix<double, 3, 2> b_stage_far = term(1000, Eigen::Vector3d(5.0, -5.0, 2.0));

  EXPECT_TRUE(b_stage_zero.isApprox(b_stage_far));
}

}  // namespace
}  // namespace fbsde_traj_opt
