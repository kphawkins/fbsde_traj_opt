// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#ifndef FBSDE_TRAJ_OPT_EIGEN_CONCEPTS_HPP_
#define FBSDE_TRAJ_OPT_EIGEN_CONCEPTS_HPP_

#include <concepts>

#include <Eigen/Core>

namespace fbsde_traj_opt {

// True for Eigen expression types that expose a fixed (non-Dynamic) compile-time shape. This
// holds not just for Eigen::Matrix/Eigen::Array but also for Eigen::DiagonalMatrix and other
// expression templates, since every one of them defines RowsAtCompileTime/ColsAtCompileTime
// through its respective CRTP base class.
template <typename T>
concept EigenExpressionWithCompileTimeShape = requires {
  typename T::Scalar;
  { T::RowsAtCompileTime } -> std::convertible_to<int>;
  { T::ColsAtCompileTime } -> std::convertible_to<int>;
};

// True for fixed-size (non-Dynamic) Eigen column vector types.
template <typename T>
concept EigenFixedSizeColumnVector =
    EigenExpressionWithCompileTimeShape<T> && T::ColsAtCompileTime == 1 && T::RowsAtCompileTime != Eigen::Dynamic;

// True for Eigen matrix types that are square with fixed compile-time dimension `N`.
template <typename T, int N>
concept EigenFixedSizeSquareMatrixOfDimension =
    EigenExpressionWithCompileTimeShape<T> && T::RowsAtCompileTime == N && T::ColsAtCompileTime == N;

}  // namespace fbsde_traj_opt

#endif  // FBSDE_TRAJ_OPT_EIGEN_CONCEPTS_HPP_
