// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#include "fbsde_traj_opt/sde_diffusion_term.hpp"

#include <cstddef>
#include <string>

#include <Eigen/Core>
#include <gtest/gtest.h>

namespace fbsde_traj_opt {
namespace {

using State3 = Eigen::Vector3d;

// A conforming functor returning a dense fixed-size matrix.
struct DenseIdentityDiffusion {
  auto operator()(std::size_t /*stage*/, const State3& /*state*/) const noexcept -> Eigen::Matrix3d {
    return Eigen::Matrix3d::Identity();
  }
};

static_assert(SdeDiffusionTerm<DenseIdentityDiffusion, State3>);

// A conforming functor returning a diagonal matrix -- the shape this concept is designed around.
struct DiagonalDiffusion {
  auto operator()(std::size_t /*stage*/, const State3& /*state*/) const noexcept -> Eigen::DiagonalMatrix<double, 3> {
    return Eigen::DiagonalMatrix<double, 3>(Eigen::Vector3d::Ones());
  }
};

static_assert(SdeDiffusionTerm<DiagonalDiffusion, State3>);

// Rejected: the returned matrix's fixed size doesn't match the state's.
struct WrongSizeDiffusion {
  auto operator()(std::size_t /*stage*/, const State3& /*state*/) const noexcept -> Eigen::Matrix2d {
    return Eigen::Matrix2d::Identity();
  }
};

static_assert(!SdeDiffusionTerm<WrongSizeDiffusion, State3>);

// Rejected: the returned matrix is fixed-size but not square.
struct NonSquareDiffusion {
  auto operator()(std::size_t /*stage*/, const State3& /*state*/) const noexcept -> Eigen::Matrix<double, 3, 2> {
    return Eigen::Matrix<double, 3, 2>::Zero();
  }
};

static_assert(!SdeDiffusionTerm<NonSquareDiffusion, State3>);

// Rejected: the returned matrix is dynamically sized, even though it happens to be 3x3 at runtime.
struct DynamicSizeDiffusion {
  auto operator()(std::size_t /*stage*/, const State3& /*state*/) const noexcept -> Eigen::MatrixXd {
    return Eigen::MatrixXd::Identity(3, 3);
  }
};

static_assert(!SdeDiffusionTerm<DynamicSizeDiffusion, State3>);

// Rejected: the state type itself is dynamically sized, regardless of the functor.
static_assert(!SdeDiffusionTerm<DenseIdentityDiffusion, Eigen::VectorXd>);

// Rejected: not callable with (std::size_t, State) at all.
struct WrongSignatureDiffusion {
  auto operator()(const std::string& /*stage*/, const State3& /*state*/) const noexcept -> Eigen::Matrix3d {
    return Eigen::Matrix3d::Identity();
  }
};

static_assert(!SdeDiffusionTerm<WrongSignatureDiffusion, State3>);

// Rejected: the return type isn't an Eigen expression at all.
struct NonEigenReturnDiffusion {
  auto operator()(std::size_t /*stage*/, const State3& /*state*/) const noexcept -> double { return 1.0; }
};

static_assert(!SdeDiffusionTerm<NonEigenReturnDiffusion, State3>);

TEST(SdeDiffusionTermTest, CompileTimeChecksPassed) {
  // All the interesting checks for this concept are the static_asserts above; this test exists
  // only so the target has a runnable case.
  SUCCEED();
}

}  // namespace
}  // namespace fbsde_traj_opt
