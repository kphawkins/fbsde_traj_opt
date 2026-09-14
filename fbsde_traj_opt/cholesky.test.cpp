// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#include "fbsde_traj_opt/cholesky.hpp"

#include <cmath>

#include <Eigen/Core>
#include <Eigen/LU>
#include <gtest/gtest.h>

namespace fbsde_traj_opt {
namespace {

constexpr double kTolerance = 1e-12;

// A symmetric positive-definite 3x3 matrix, diagonally dominant so that positive definiteness is
// obvious by inspection rather than by assertion.
Eigen::MatrixXd SampleCovariance() {
  Eigen::MatrixXd covariance(3, 3);
  covariance << 4.0, 2.0, 0.6, 2.0, 5.0, 1.0, 0.6, 1.0, 2.0;
  return covariance;
}

TEST(LowerCholeskyFactorTest, ReconstructsTheOriginalCovariance) {
  const Eigen::MatrixXd covariance = SampleCovariance();

  const auto factor = LowerCholeskyFactor(covariance);

  ASSERT_TRUE(factor.has_value());
  EXPECT_TRUE(factor->isLowerTriangular());
  EXPECT_TRUE((*factor * factor->transpose()).isApprox(covariance, kTolerance));
}

TEST(LowerCholeskyFactorTest, FactorsTheIdentityIntoTheIdentity) {
  const Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(4, 4);

  const auto factor = LowerCholeskyFactor(identity);

  ASSERT_TRUE(factor.has_value());
  EXPECT_TRUE(factor->isApprox(identity, kTolerance));
}

TEST(LowerCholeskyFactorTest, FactorsADiagonalCovarianceIntoElementwiseSquareRoots) {
  Eigen::MatrixXd covariance = Eigen::MatrixXd::Zero(2, 2);
  covariance(0, 0) = 9.0;
  covariance(1, 1) = 16.0;

  const auto factor = LowerCholeskyFactor(covariance);

  ASSERT_TRUE(factor.has_value());
  EXPECT_NEAR((*factor)(0, 0), 3.0, kTolerance);
  EXPECT_NEAR((*factor)(1, 1), 4.0, kTolerance);
  EXPECT_NEAR((*factor)(0, 1), 0.0, kTolerance);
}

TEST(LowerCholeskyFactorTest, RejectsANonSquareMatrix) {
  const Eigen::MatrixXd rectangular = Eigen::MatrixXd::Zero(2, 3);

  const auto factor = LowerCholeskyFactor(rectangular);

  ASSERT_FALSE(factor.has_value());
  EXPECT_EQ(factor.error(), CholeskyError::kNotSquare);
}

TEST(LowerCholeskyFactorTest, RejectsAnEmptyMatrix) {
  const Eigen::MatrixXd empty(0, 0);

  const auto factor = LowerCholeskyFactor(empty);

  ASSERT_FALSE(factor.has_value());
  EXPECT_EQ(factor.error(), CholeskyError::kEmpty);
}

TEST(LowerCholeskyFactorTest, RejectsAnAsymmetricMatrix) {
  // Eigen's LLT reads only the lower triangle, so without the explicit symmetry check this would
  // succeed and silently return the factor of a matrix the caller never passed.
  Eigen::MatrixXd asymmetric(2, 2);
  asymmetric << 1.0, 0.50, 0.25, 1.0;

  const auto factor = LowerCholeskyFactor(asymmetric);

  ASSERT_FALSE(factor.has_value());
  EXPECT_EQ(factor.error(), CholeskyError::kNotSymmetric);
}

TEST(LowerCholeskyFactorTest, AcceptsAnAsymmetricMatrixWhenTheCheckIsDisabled) {
  Eigen::MatrixXd asymmetric(2, 2);
  asymmetric << 1.0, 0.50, 0.25, 1.0;

  const auto factor = LowerCholeskyFactor(asymmetric, -1.0);

  // The result is the factor of the symmetrized lower triangle, which is exactly the documented
  // behavior of skipping the check.
  ASSERT_TRUE(factor.has_value());
}

TEST(LowerCholeskyFactorTest, RejectsAnIndefiniteMatrix) {
  // Symmetric with determinant -3, so it has one negative eigenvalue.
  Eigen::MatrixXd indefinite(2, 2);
  indefinite << 1.0, 2.0, 2.0, 1.0;

  const auto factor = LowerCholeskyFactor(indefinite);

  ASSERT_FALSE(factor.has_value());
  EXPECT_EQ(factor.error(), CholeskyError::kNotPositiveDefinite);
}

TEST(LogDeterminantFromFactorTest, MatchesTheLogOfTheDirectDeterminant) {
  const Eigen::MatrixXd covariance = SampleCovariance();
  const auto factor = LowerCholeskyFactor(covariance);
  ASSERT_TRUE(factor.has_value());

  EXPECT_NEAR(LogDeterminantFromFactor(*factor), std::log(covariance.determinant()), 1e-10);
}

TEST(LogDeterminantFromFactorTest, ReturnsZeroForTheIdentity) {
  const auto factor = LowerCholeskyFactor(Eigen::MatrixXd::Identity(5, 5));
  ASSERT_TRUE(factor.has_value());

  EXPECT_NEAR(LogDeterminantFromFactor(*factor), 0.0, kTolerance);
}

TEST(CorrelateSampleTest, IsInvertedByTheTriangularSolve) {
  const auto factor = LowerCholeskyFactor(SampleCovariance());
  ASSERT_TRUE(factor.has_value());
  Eigen::VectorXd standard_normal_sample(3);
  standard_normal_sample << 1.0, -2.0, 0.5;

  const Eigen::VectorXd correlated = CorrelateSample(*factor, standard_normal_sample);
  const Eigen::VectorXd whitened = factor->triangularView<Eigen::Lower>().solve(correlated);

  EXPECT_TRUE(whitened.isApprox(standard_normal_sample, kTolerance));
}

TEST(CorrelateSampleTest, ReproducesTheTargetCovariance) {
  // Pushing the columns of the identity through CorrelateSample() and re-accumulating the outer
  // products recovers Sigma exactly: sum_i (L e_i)(L e_i)^T = L L^T = Sigma. This is the
  // deterministic analogue of checking the sample covariance of a large ensemble.
  const Eigen::MatrixXd covariance = SampleCovariance();
  const auto factor = LowerCholeskyFactor(covariance);
  ASSERT_TRUE(factor.has_value());

  Eigen::MatrixXd accumulated = Eigen::MatrixXd::Zero(3, 3);
  for (Eigen::Index i = 0; i < 3; ++i) {
    const Eigen::VectorXd correlated = CorrelateSample(*factor, Eigen::VectorXd::Unit(3, i));
    accumulated += correlated * correlated.transpose();
  }

  EXPECT_TRUE(accumulated.isApprox(covariance, kTolerance));
}

TEST(CholeskyErrorNameTest, NamesEveryError) {
  EXPECT_EQ(CholeskyErrorName(CholeskyError::kNotSquare), "not square");
  EXPECT_EQ(CholeskyErrorName(CholeskyError::kEmpty), "empty");
  EXPECT_EQ(CholeskyErrorName(CholeskyError::kNotSymmetric), "not symmetric");
  EXPECT_EQ(CholeskyErrorName(CholeskyError::kNotPositiveDefinite), "not positive definite");
}

}  // namespace
}  // namespace fbsde_traj_opt
