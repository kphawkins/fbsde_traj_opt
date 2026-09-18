// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#include "fbsde_traj_opt/diagonal_covariance_normal_distribution.hpp"

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "fbsde_traj_opt/normal_distribution_concepts.hpp"

namespace fbsde_traj_opt {
namespace {

using Normal3 = DiagonalCovarianceNormalDistribution<3>;

static_assert(NormalDistribution<Normal3, Eigen::Vector3d>);

TEST(DiagonalCovarianceNormalDistributionTest, MakeSucceedsWithAnOrdinaryMeanAndCovariance) {
  const Eigen::Vector3d mean(1.0, -2.0, 3.0);
  const Eigen::Vector3d covariance_diagonal(1.0, 2.0, 3.0);

  const auto distribution = Normal3::Make(mean, covariance_diagonal);

  EXPECT_TRUE(distribution.has_value());
}

TEST(DiagonalCovarianceNormalDistributionTest, MakeSucceedsAtTheInclusiveBoundaryValues) {
  const Eigen::Vector3d mean(-1e10, 1e10, 0.0);
  const Eigen::Vector3d covariance_diagonal(1e-10, 1e10, 1.0);

  const auto distribution = Normal3::Make(mean, covariance_diagonal);

  EXPECT_TRUE(distribution.has_value());
}

TEST(DiagonalCovarianceNormalDistributionTest, MakeFailsWhenAMeanElementExceedsTheUpperBound) {
  const Eigen::Vector3d mean(1.0, 1e10 + 1.0, 1.0);
  const Eigen::Vector3d covariance_diagonal(1.0, 1.0, 1.0);

  const auto distribution = Normal3::Make(mean, covariance_diagonal);

  EXPECT_FALSE(distribution.has_value());
}

TEST(DiagonalCovarianceNormalDistributionTest, MakeFailsWhenAMeanElementIsBelowTheLowerBound) {
  const Eigen::Vector3d mean(-1e10 - 1.0, 1.0, 1.0);
  const Eigen::Vector3d covariance_diagonal(1.0, 1.0, 1.0);

  const auto distribution = Normal3::Make(mean, covariance_diagonal);

  EXPECT_FALSE(distribution.has_value());
}

TEST(DiagonalCovarianceNormalDistributionTest, MakeFailsWhenACovarianceDiagonalElementExceedsTheUpperBound) {
  const Eigen::Vector3d mean = Eigen::Vector3d::Zero();
  const Eigen::Vector3d covariance_diagonal(1.0, 1e10 + 1.0, 1.0);

  const auto distribution = Normal3::Make(mean, covariance_diagonal);

  EXPECT_FALSE(distribution.has_value());
}

TEST(DiagonalCovarianceNormalDistributionTest, MakeFailsWhenACovarianceDiagonalElementIsZero) {
  const Eigen::Vector3d mean = Eigen::Vector3d::Zero();
  const Eigen::Vector3d covariance_diagonal(1.0, 0.0, 1.0);

  const auto distribution = Normal3::Make(mean, covariance_diagonal);

  EXPECT_FALSE(distribution.has_value());
}

TEST(DiagonalCovarianceNormalDistributionTest, MakeFailsWhenACovarianceDiagonalElementIsBelowTheLowerBound) {
  const Eigen::Vector3d mean = Eigen::Vector3d::Zero();
  const Eigen::Vector3d covariance_diagonal(1.0, 1e-10 / 2.0, 1.0);

  const auto distribution = Normal3::Make(mean, covariance_diagonal);

  EXPECT_FALSE(distribution.has_value());
}

TEST(DiagonalCovarianceNormalDistributionTest, MakeFailsWhenACovarianceDiagonalElementIsNegative) {
  const Eigen::Vector3d mean = Eigen::Vector3d::Zero();
  const Eigen::Vector3d covariance_diagonal(1.0, -1.0, 1.0);

  const auto distribution = Normal3::Make(mean, covariance_diagonal);

  EXPECT_FALSE(distribution.has_value());
}

TEST(DiagonalCovarianceNormalDistributionTest, MeanReturnsTheConstructorArgument) {
  const Eigen::Vector3d mean(1.0, -2.0, 3.0);
  const Eigen::Vector3d covariance_diagonal(1.0, 2.0, 3.0);
  const auto distribution = Normal3::Make(mean, covariance_diagonal);
  ASSERT_TRUE(distribution.has_value());

  const Eigen::Vector3d mu = distribution->Mean();

  EXPECT_TRUE(mu.isApprox(mean));
}

TEST(DiagonalCovarianceNormalDistributionTest, CovarianceReturnsTheDiagonalMatrixOfTheConstructorArgument) {
  const Eigen::Vector3d mean(1.0, -2.0, 3.0);
  const Eigen::Vector3d covariance_diagonal(1.0, 2.0, 3.0);
  const auto distribution = Normal3::Make(mean, covariance_diagonal);
  ASSERT_TRUE(distribution.has_value());

  const Eigen::Matrix3d sigma = distribution->Covariance();

  EXPECT_TRUE(sigma.isApprox(covariance_diagonal.asDiagonal().toDenseMatrix()));
}

}  // namespace
}  // namespace fbsde_traj_opt
