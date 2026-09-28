// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#ifndef FBSDE_TRAJ_OPT_EIGEN_CONCEPTS_HPP_
#define FBSDE_TRAJ_OPT_EIGEN_CONCEPTS_HPP_

#include <concepts>
#include <type_traits>

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

// True when `T`, once any reference and cv-qualification are stripped, is an Eigen expression
// with a fixed compile-time shape.
//
// A return-type requirement of the form `{ expr } -> Concept` tests the declared type of `expr`,
// references and all. A concept written over an accessor that returns `const Eigen::Matrix<...>&`
// -- as the matrix accessors on this project's constant terms do -- therefore needs this form;
// EigenExpressionWithCompileTimeShape itself would be tested against the reference type and
// always fail.
template <typename T>
concept DecaysToEigenExpressionWithCompileTimeShape = EigenExpressionWithCompileTimeShape<std::remove_cvref_t<T>>;

// True for fixed-size (non-Dynamic) Eigen column vector types.
template <typename T>
concept EigenFixedSizeColumnVector =
    EigenExpressionWithCompileTimeShape<T> && T::ColsAtCompileTime == 1 && T::RowsAtCompileTime != Eigen::Dynamic;

// True for fixed-size Eigen column vector types with a specific compile-time dimension `N`.
template <typename T, int N>
concept EigenFixedSizeColumnVectorOfDimension = EigenFixedSizeColumnVector<T> && T::RowsAtCompileTime == N;

// True for Eigen matrix types that are square with fixed compile-time dimension `N`.
template <typename T, int N>
concept EigenFixedSizeSquareMatrixOfDimension =
    EigenExpressionWithCompileTimeShape<T> && T::RowsAtCompileTime == N && T::ColsAtCompileTime == N;

// True for Eigen matrix types with fixed compile-time row count `N` and an unconstrained but
// still fixed (non-Dynamic) column count. Used where the number of rows is pinned by some other
// quantity (e.g. the state dimension) but the number of columns is free (e.g. the control
// dimension).
template <typename T, int N>
concept EigenFixedSizeMatrixWithRowsOfDimension =
    EigenExpressionWithCompileTimeShape<T> && T::RowsAtCompileTime == N && T::ColsAtCompileTime != Eigen::Dynamic;

}  // namespace fbsde_traj_opt

#endif  // FBSDE_TRAJ_OPT_EIGEN_CONCEPTS_HPP_
