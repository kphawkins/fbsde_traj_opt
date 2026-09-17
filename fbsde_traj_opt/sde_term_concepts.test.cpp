// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#include "fbsde_traj_opt/sde_term_concepts.hpp"

#include <cstddef>
#include <string>

#include <Eigen/Core>
#include <gtest/gtest.h>

namespace fbsde_traj_opt {
namespace {

using State3 = Eigen::Vector3d;

// ---------------------------------------------------------------------------------------------
// SdeStateDriftTerm
// ---------------------------------------------------------------------------------------------

// A conforming functor returning a fixed-size column vector matching the state's dimension.
struct LinearStateDrift {
  auto operator()(std::size_t /*stage*/, const State3& state) const noexcept -> Eigen::Vector3d { return -state; }
};

static_assert(SdeStateDriftTerm<LinearStateDrift, State3>);

// Rejected: the returned vector's fixed size doesn't match the state's.
struct WrongSizeStateDrift {
  auto operator()(std::size_t /*stage*/, const State3& /*state*/) const noexcept -> Eigen::Vector2d {
    return Eigen::Vector2d::Zero();
  }
};

static_assert(!SdeStateDriftTerm<WrongSizeStateDrift, State3>);

// Rejected: the returned type is a row vector, not a column vector.
struct RowVectorStateDrift {
  auto operator()(std::size_t /*stage*/, const State3& /*state*/) const noexcept -> Eigen::RowVector3d {
    return Eigen::RowVector3d::Zero();
  }
};

static_assert(!SdeStateDriftTerm<RowVectorStateDrift, State3>);

// Rejected: the returned vector is dynamically sized, even though it happens to be size 3 at
// runtime.
struct DynamicSizeStateDrift {
  auto operator()(std::size_t /*stage*/, const State3& /*state*/) const noexcept -> Eigen::VectorXd {
    return Eigen::VectorXd::Zero(3);
  }
};

static_assert(!SdeStateDriftTerm<DynamicSizeStateDrift, State3>);

// Rejected: the state type itself is dynamically sized, regardless of the functor.
static_assert(!SdeStateDriftTerm<LinearStateDrift, Eigen::VectorXd>);

// Rejected: not callable with (std::size_t, State) at all.
struct WrongSignatureStateDrift {
  auto operator()(const std::string& /*stage*/, const State3& /*state*/) const noexcept -> Eigen::Vector3d {
    return Eigen::Vector3d::Zero();
  }
};

static_assert(!SdeStateDriftTerm<WrongSignatureStateDrift, State3>);

// Rejected: the return type isn't an Eigen expression at all.
struct NonEigenReturnStateDrift {
  auto operator()(std::size_t /*stage*/, const State3& /*state*/) const noexcept -> double { return 1.0; }
};

static_assert(!SdeStateDriftTerm<NonEigenReturnStateDrift, State3>);

// ---------------------------------------------------------------------------------------------
// SdeControlDriftMatTerm
// ---------------------------------------------------------------------------------------------

// A conforming functor returning a fixed-size matrix whose rows match the state's dimension and
// whose columns (the control dimension) differ from it.
struct RectangularControlDriftMat {
  auto operator()(std::size_t /*stage*/, const State3& /*state*/) const noexcept -> Eigen::Matrix<double, 3, 2> {
    return Eigen::Matrix<double, 3, 2>::Zero();
  }
};

static_assert(SdeControlDriftMatTerm<RectangularControlDriftMat, State3>);

// A conforming functor whose control dimension happens to equal the state dimension.
struct SquareControlDriftMat {
  auto operator()(std::size_t /*stage*/, const State3& /*state*/) const noexcept -> Eigen::Matrix3d {
    return Eigen::Matrix3d::Identity();
  }
};

static_assert(SdeControlDriftMatTerm<SquareControlDriftMat, State3>);

// Rejected: the returned matrix's row count doesn't match the state's dimension.
struct WrongRowCountControlDriftMat {
  auto operator()(std::size_t /*stage*/, const State3& /*state*/) const noexcept -> Eigen::Matrix<double, 2, 2> {
    return Eigen::Matrix<double, 2, 2>::Zero();
  }
};

static_assert(!SdeControlDriftMatTerm<WrongRowCountControlDriftMat, State3>);

// Rejected: the returned matrix has a dynamic column count, even though it happens to have 2
// columns at runtime.
struct DynamicColsControlDriftMat {
  auto operator()(std::size_t /*stage*/, const State3& /*state*/) const noexcept -> Eigen::Matrix<double, 3, -1> {
    return Eigen::Matrix<double, 3, -1>::Zero(3, 2);
  }
};

static_assert(!SdeControlDriftMatTerm<DynamicColsControlDriftMat, State3>);

// Rejected: the state type itself is dynamically sized, regardless of the functor.
static_assert(!SdeControlDriftMatTerm<RectangularControlDriftMat, Eigen::VectorXd>);

// Rejected: not callable with (std::size_t, State) at all.
struct WrongSignatureControlDriftMat {
  auto operator()(const std::string& /*stage*/, const State3& /*state*/) const noexcept -> Eigen::Matrix<double, 3, 2> {
    return Eigen::Matrix<double, 3, 2>::Zero();
  }
};

static_assert(!SdeControlDriftMatTerm<WrongSignatureControlDriftMat, State3>);

// Rejected: the return type isn't an Eigen expression at all.
struct NonEigenReturnControlDriftMat {
  auto operator()(std::size_t /*stage*/, const State3& /*state*/) const noexcept -> double { return 1.0; }
};

static_assert(!SdeControlDriftMatTerm<NonEigenReturnControlDriftMat, State3>);

// ---------------------------------------------------------------------------------------------
// SdeDiffusionTerm
// ---------------------------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------------------------
// SdeRunningCostTerm
// ---------------------------------------------------------------------------------------------

using Control2 = Eigen::Vector2d;

// A conforming functor returning a Scalar matching the state's and control's shared scalar type.
struct QuadraticRunningCost {
  auto operator()(std::size_t /*stage*/, const State3& state, const Control2& control) const noexcept -> double {
    return state.squaredNorm() + control.squaredNorm();
  }
};

static_assert(SdeRunningCostTerm<QuadraticRunningCost, State3, Control2>);

// Rejected: the returned Scalar doesn't match the state's and control's scalar type.
struct WrongScalarRunningCost {
  auto operator()(std::size_t /*stage*/, const State3& /*state*/, const Control2& /*control*/) const noexcept -> float {
    return 0.0F;
  }
};

static_assert(!SdeRunningCostTerm<WrongScalarRunningCost, State3, Control2>);

// Rejected: the state and control types have different scalar types, even though the functor
// itself would otherwise conform.
using FloatControl2 = Eigen::Vector2f;

static_assert(!SdeRunningCostTerm<QuadraticRunningCost, State3, FloatControl2>);

// Rejected: the state type itself is dynamically sized, regardless of the functor.
static_assert(!SdeRunningCostTerm<QuadraticRunningCost, Eigen::VectorXd, Control2>);

// Rejected: the control type itself is dynamically sized, regardless of the functor.
static_assert(!SdeRunningCostTerm<QuadraticRunningCost, State3, Eigen::VectorXd>);

// Rejected: not callable with (std::size_t, State, Control) at all.
struct WrongSignatureRunningCost {
  auto operator()(const std::string& /*stage*/, const State3& /*state*/, const Control2& /*control*/) const noexcept
      -> double {
    return 0.0;
  }
};

static_assert(!SdeRunningCostTerm<WrongSignatureRunningCost, State3, Control2>);

// Rejected: the return type isn't a Scalar at all.
struct NonScalarReturnRunningCost {
  auto operator()(std::size_t /*stage*/, const State3& /*state*/, const Control2& /*control*/) const noexcept
      -> Eigen::Vector3d {
    return Eigen::Vector3d::Zero();
  }
};

static_assert(!SdeRunningCostTerm<NonScalarReturnRunningCost, State3, Control2>);

// ---------------------------------------------------------------------------------------------
// SdeTerminalCostTerm
// ---------------------------------------------------------------------------------------------

// A conforming functor returning a Scalar matching the state's scalar type.
struct QuadraticTerminalCost {
  auto operator()(const State3& state) const noexcept -> double { return state.squaredNorm(); }
};

static_assert(SdeTerminalCostTerm<QuadraticTerminalCost, State3>);

// Rejected: the returned Scalar doesn't match the state's scalar type.
struct WrongScalarTerminalCost {
  auto operator()(const State3& /*state*/) const noexcept -> float { return 0.0F; }
};

static_assert(!SdeTerminalCostTerm<WrongScalarTerminalCost, State3>);

// Rejected: the state type itself is dynamically sized, regardless of the functor.
static_assert(!SdeTerminalCostTerm<QuadraticTerminalCost, Eigen::VectorXd>);

// Rejected: not callable with (State) at all.
struct WrongSignatureTerminalCost {
  auto operator()(const std::string& /*state*/) const noexcept -> double { return 0.0; }
};

static_assert(!SdeTerminalCostTerm<WrongSignatureTerminalCost, State3>);

// Rejected: the return type isn't a Scalar at all.
struct NonScalarReturnTerminalCost {
  auto operator()(const State3& /*state*/) const noexcept -> Eigen::Vector3d { return Eigen::Vector3d::Zero(); }
};

static_assert(!SdeTerminalCostTerm<NonScalarReturnTerminalCost, State3>);

TEST(SdeTermConceptsTest, CompileTimeChecksPassed) {
  // All the interesting checks for these concepts are the static_asserts above; this test exists
  // only so the target has a runnable case.
  SUCCEED();
}

}  // namespace
}  // namespace fbsde_traj_opt
