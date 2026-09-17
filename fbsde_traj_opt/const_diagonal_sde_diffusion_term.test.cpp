// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#include "fbsde_traj_opt/const_diagonal_sde_diffusion_term.hpp"

#include <cstddef>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "fbsde_traj_opt/sde_term_concepts.hpp"

namespace fbsde_traj_opt {
namespace {

using Term3 = ConstDiagonalSdeDiffusionTerm<3>;

static_assert(SdeDiffusionTerm<Term3, Eigen::Vector3d>);

TEST(ConstDiagonalSdeDiffusionTermTest, MakeSucceedsWithAnOrdinaryDiagonal) {
  const Eigen::Vector3d diagonal(1.0, 2.0, 3.0);

  const auto term = Term3::Make(diagonal);

  EXPECT_TRUE(term.has_value());
}

TEST(ConstDiagonalSdeDiffusionTermTest, MakeSucceedsAtTheInclusiveBoundaryValues) {
  const Eigen::Vector3d diagonal(-1e10, 1e10, 0.0);

  const auto term = Term3::Make(diagonal);

  EXPECT_TRUE(term.has_value());
}

TEST(ConstDiagonalSdeDiffusionTermTest, MakeFailsWhenAnElementExceedsTheUpperBound) {
  const Eigen::Vector3d diagonal(1.0, 1e10 + 1.0, 1.0);

  const auto term = Term3::Make(diagonal);

  EXPECT_FALSE(term.has_value());
}

TEST(ConstDiagonalSdeDiffusionTermTest, MakeFailsWhenAnElementIsBelowTheLowerBound) {
  const Eigen::Vector3d diagonal(-1e10 - 1.0, 1.0, 1.0);

  const auto term = Term3::Make(diagonal);

  EXPECT_FALSE(term.has_value());
}

TEST(ConstDiagonalSdeDiffusionTermTest, OperatorReturnsTheDiagonalMatrixOfTheConstructorArgument) {
  const Eigen::Vector3d diagonal(1.0, 2.0, 3.0);
  const auto term = Term3::Make(diagonal);
  ASSERT_TRUE(term.has_value());

  const Eigen::Matrix3d sigma = (*term)(0, Eigen::Vector3d::Zero());

  EXPECT_TRUE(sigma.isApprox(diagonal.asDiagonal().toDenseMatrix()));
}

TEST(ConstDiagonalSdeDiffusionTermTest, OperatorIgnoresStageAndState) {
  const Eigen::Vector3d diagonal(1.0, 2.0, 3.0);
  const auto term = Term3::Make(diagonal);
  ASSERT_TRUE(term.has_value());

  const Eigen::Matrix3d sigma_stage_zero = (*term)(0, Eigen::Vector3d::Zero());
  const Eigen::Matrix3d sigma_stage_far = (*term)(1000, Eigen::Vector3d(5.0, -5.0, 2.0));

  EXPECT_TRUE(sigma_stage_zero.isApprox(sigma_stage_far));
}

}  // namespace
}  // namespace fbsde_traj_opt
