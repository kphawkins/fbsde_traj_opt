// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#include "fbsde_traj_opt/normal_distribution_concepts.hpp"

#include <Eigen/Core>
#include <gtest/gtest.h>

namespace fbsde_traj_opt {
namespace {

using State3 = Eigen::Vector3d;

// A conforming type returning a dense fixed-size covariance matrix.
struct DenseIdentityNormal {
  [[nodiscard]] static auto Mean() noexcept -> Eigen::Vector3d { return Eigen::Vector3d::Zero(); }

  [[nodiscard]] static auto Covariance() noexcept -> Eigen::Matrix3d { return Eigen::Matrix3d::Identity(); }
};

static_assert(NormalDistribution<DenseIdentityNormal, State3>);

// A conforming type returning a diagonal covariance matrix -- the shape this concept is designed
// around.
struct DiagonalNormal {
  [[nodiscard]] static auto Mean() noexcept -> Eigen::Vector3d { return Eigen::Vector3d::Zero(); }

  [[nodiscard]] static auto Covariance() noexcept -> Eigen::DiagonalMatrix<double, 3> {
    return Eigen::DiagonalMatrix<double, 3>(Eigen::Vector3d::Ones());
  }
};

static_assert(NormalDistribution<DiagonalNormal, State3>);

// Rejected: Mean()'s returned vector's fixed size doesn't match the state's.
struct WrongMeanSizeNormal {
  [[nodiscard]] static auto Mean() noexcept -> Eigen::Vector2d { return Eigen::Vector2d::Zero(); }

  [[nodiscard]] static auto Covariance() noexcept -> Eigen::Matrix3d { return Eigen::Matrix3d::Identity(); }
};

static_assert(!NormalDistribution<WrongMeanSizeNormal, State3>);

// Rejected: Mean() returns a row vector, not a column vector.
struct RowVectorMeanNormal {
  [[nodiscard]] static auto Mean() noexcept -> Eigen::RowVector3d { return Eigen::RowVector3d::Zero(); }

  [[nodiscard]] static auto Covariance() noexcept -> Eigen::Matrix3d { return Eigen::Matrix3d::Identity(); }
};

static_assert(!NormalDistribution<RowVectorMeanNormal, State3>);

// Rejected: Mean() returns a dynamically sized vector, even though it happens to be size 3 at
// runtime.
struct DynamicSizeMeanNormal {
  [[nodiscard]] static auto Mean() noexcept -> Eigen::VectorXd { return Eigen::VectorXd::Zero(3); }

  [[nodiscard]] static auto Covariance() noexcept -> Eigen::Matrix3d { return Eigen::Matrix3d::Identity(); }
};

static_assert(!NormalDistribution<DynamicSizeMeanNormal, State3>);

// Rejected: Covariance()'s returned matrix's fixed size doesn't match the state's.
struct WrongCovarianceSizeNormal {
  [[nodiscard]] static auto Mean() noexcept -> Eigen::Vector3d { return Eigen::Vector3d::Zero(); }

  [[nodiscard]] static auto Covariance() noexcept -> Eigen::Matrix2d { return Eigen::Matrix2d::Identity(); }
};

static_assert(!NormalDistribution<WrongCovarianceSizeNormal, State3>);

// Rejected: Covariance() returns a fixed-size matrix that isn't square.
struct NonSquareCovarianceNormal {
  [[nodiscard]] static auto Mean() noexcept -> Eigen::Vector3d { return Eigen::Vector3d::Zero(); }

  [[nodiscard]] static auto Covariance() noexcept -> Eigen::Matrix<double, 3, 2> {
    return Eigen::Matrix<double, 3, 2>::Zero();
  }
};

static_assert(!NormalDistribution<NonSquareCovarianceNormal, State3>);

// Rejected: Covariance() returns a dynamically sized matrix, even though it happens to be 3x3 at
// runtime.
struct DynamicSizeCovarianceNormal {
  [[nodiscard]] static auto Mean() noexcept -> Eigen::Vector3d { return Eigen::Vector3d::Zero(); }

  [[nodiscard]] static auto Covariance() noexcept -> Eigen::MatrixXd { return Eigen::MatrixXd::Identity(3, 3); }
};

static_assert(!NormalDistribution<DynamicSizeCovarianceNormal, State3>);

// Rejected: the state type itself is dynamically sized, regardless of the type.
static_assert(!NormalDistribution<DenseIdentityNormal, Eigen::VectorXd>);

// Rejected: missing the Covariance() accessor.
struct MeanOnlyNormal {
  [[nodiscard]] static auto Mean() noexcept -> Eigen::Vector3d { return Eigen::Vector3d::Zero(); }
};

static_assert(!NormalDistribution<MeanOnlyNormal, State3>);

// Rejected: missing the Mean() accessor.
struct CovarianceOnlyNormal {
  [[nodiscard]] static auto Covariance() noexcept -> Eigen::Matrix3d { return Eigen::Matrix3d::Identity(); }
};

static_assert(!NormalDistribution<CovarianceOnlyNormal, State3>);

// Rejected: neither accessor is present.
struct EmptyNormal {};

static_assert(!NormalDistribution<EmptyNormal, State3>);

TEST(NormalDistributionConceptsTest, CompileTimeChecksPassed) {
  // All the interesting checks for this concept are the static_asserts above; this test exists
  // only so the target has a runnable case.
  SUCCEED();
}

}  // namespace
}  // namespace fbsde_traj_opt
