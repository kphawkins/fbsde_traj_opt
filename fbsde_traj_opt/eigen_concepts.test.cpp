// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#include "fbsde_traj_opt/eigen_concepts.hpp"

#include <Eigen/Core>
#include <gtest/gtest.h>

namespace fbsde_traj_opt {
namespace {

static_assert(EigenExpressionWithCompileTimeShape<Eigen::Matrix3d>);
static_assert(EigenExpressionWithCompileTimeShape<Eigen::DiagonalMatrix<double, 3>>);
static_assert(EigenExpressionWithCompileTimeShape<Eigen::MatrixXd>);
static_assert(!EigenExpressionWithCompileTimeShape<double>);
static_assert(!EigenExpressionWithCompileTimeShape<int>);

static_assert(EigenFixedSizeColumnVector<Eigen::Vector3d>);
static_assert(!EigenFixedSizeColumnVector<Eigen::VectorXd>);     // Dynamic row count.
static_assert(!EigenFixedSizeColumnVector<Eigen::RowVector3d>);  // A row, not a column, vector.
static_assert(!EigenFixedSizeColumnVector<Eigen::Matrix3d>);     // More than one column.

static_assert(EigenFixedSizeColumnVectorOfDimension<Eigen::Vector3d, 3>);
static_assert(!EigenFixedSizeColumnVectorOfDimension<Eigen::Vector3d, 2>);  // Wrong dimension.
static_assert(!EigenFixedSizeColumnVectorOfDimension<Eigen::VectorXd, 3>);  // Dynamically sized.
static_assert(!EigenFixedSizeColumnVectorOfDimension<Eigen::Matrix3d, 3>);  // More than one column.

static_assert(EigenFixedSizeSquareMatrixOfDimension<Eigen::Matrix3d, 3>);
static_assert(EigenFixedSizeSquareMatrixOfDimension<Eigen::DiagonalMatrix<double, 3>, 3>);
static_assert(!EigenFixedSizeSquareMatrixOfDimension<Eigen::Matrix3d, 2>);              // Wrong dimension.
static_assert(!EigenFixedSizeSquareMatrixOfDimension<Eigen::Matrix<double, 3, 2>, 3>);  // Not square.
static_assert(!EigenFixedSizeSquareMatrixOfDimension<Eigen::MatrixXd, 3>);              // Dynamically sized.

static_assert(EigenFixedSizeMatrixWithRowsOfDimension<Eigen::Matrix<double, 3, 2>, 3>);
static_assert(EigenFixedSizeMatrixWithRowsOfDimension<Eigen::Matrix3d, 3>);  // Free columns may equal the rows.
static_assert(!EigenFixedSizeMatrixWithRowsOfDimension<Eigen::Matrix<double, 3, 2>, 2>);   // Wrong row count.
static_assert(!EigenFixedSizeMatrixWithRowsOfDimension<Eigen::Matrix<double, 3, -1>, 3>);  // Dynamic column count.
static_assert(!EigenFixedSizeMatrixWithRowsOfDimension<Eigen::MatrixXd, 3>);               // Dynamically sized.

TEST(EigenConceptsTest, CompileTimeChecksPassed) {
  // All the interesting checks for these concepts are the static_asserts above; this test exists
  // only so the target has a runnable case.
  SUCCEED();
}

}  // namespace
}  // namespace fbsde_traj_opt
