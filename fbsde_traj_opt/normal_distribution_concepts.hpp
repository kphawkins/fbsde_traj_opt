// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#ifndef FBSDE_TRAJ_OPT_NORMAL_DISTRIBUTION_CONCEPTS_HPP_
#define FBSDE_TRAJ_OPT_NORMAL_DISTRIBUTION_CONCEPTS_HPP_

#include "fbsde_traj_opt/eigen_concepts.hpp"

namespace fbsde_traj_opt {

// Concept for a type `T` that represents a multivariate normal distribution N(mu, Sigma) over
// `State`.
//
// A conforming `T` exposes const accessors `Mean()` and `Covariance()`: `Mean()` returns mu as a
// fixed-size Eigen column vector of the same compile-time size as `State`, and `Covariance()`
// returns Sigma as a fixed-size N x N Eigen matrix, where N is `State`'s compile-time size. The
// return type of `Covariance()` may be any kind of Eigen matrix expression -- a dense
// Eigen::Matrix, an Eigen::DiagonalMatrix, or any other type that exposes the usual compile-time
// Eigen shape interface -- so that an implementation whose Sigma is structurally diagonal (the
// common case) can return a lightweight Eigen::DiagonalMatrix rather than materializing a dense
// matrix.
//
// Sigma is required to be strictly positive definite for every conforming `T`; this is a semantic
// requirement on implementations that the concept itself cannot check.
template <typename T, typename State>
concept NormalDistribution = EigenFixedSizeColumnVector<State> && requires(const T& normal_distribution) {
  { normal_distribution.Mean() } -> EigenFixedSizeColumnVectorOfDimension<State::RowsAtCompileTime>;
  { normal_distribution.Covariance() } -> EigenFixedSizeSquareMatrixOfDimension<State::RowsAtCompileTime>;
};

}  // namespace fbsde_traj_opt

#endif  // FBSDE_TRAJ_OPT_NORMAL_DISTRIBUTION_CONCEPTS_HPP_
