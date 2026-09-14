// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#ifndef FBSDE_TRAJ_OPT_CHOLESKY_HPP_
#define FBSDE_TRAJ_OPT_CHOLESKY_HPP_

#include <cstdint>
#include <expected>
#include <string_view>

#include <Eigen/Core>

namespace fbsde_traj_opt {

// Lower-triangular Cholesky factorization of noise covariance matrices.
//
// Every estimator in the DT-FBSDE family reduces, at some point, to one of three operations on
// the forward process's noise covariance `Sigma`:
//
//   * drawing a correlated sample `L * z` from a standard normal sample `z ~ N(0, I)`,
//   * evaluating a Gaussian log-density, which needs `log det Sigma`,
//   * whitening a residual, which is a triangular solve against `L`.
//
// All three are expressed in terms of the lower-triangular factor `L` satisfying
// `Sigma = L L^T`, so the factor is computed once per covariance and reused.

// Reasons a matrix cannot be treated as a covariance and factored.
enum class CholeskyError : std::uint8_t {
  kNotSquare,            // The matrix has differing row and column counts.
  kEmpty,                // The matrix is 0x0; there is no factor to return.
  kNotSymmetric,         // The matrix differs from its transpose by more than the given
                         // tolerance.
  kNotPositiveDefinite,  // Eigen's LLT decomposition reported a non-positive pivot.
};

// Returns a short, stable, human-readable name for `error`. Intended for log and assertion
// messages; the returned view points at a string literal and outlives any caller.
std::string_view CholeskyErrorName(CholeskyError error);

// Computes the lower-triangular Cholesky factor `L` of the symmetric positive-definite covariance
// `covariance`, such that `covariance == L * L.transpose()`.
//
// `symmetry_tolerance` bounds the largest permitted absolute entry of
// `covariance - covariance.transpose()`. The check exists because Eigen's LLT reads only the
// lower triangle of its input and would otherwise silently accept a non-symmetric matrix,
// returning the factor of a different matrix than the caller passed. Pass a negative tolerance to
// skip the check when the caller has already established symmetry by construction.
std::expected<Eigen::MatrixXd, CholeskyError> LowerCholeskyFactor(const Eigen::Ref<const Eigen::MatrixXd>& covariance,
                                                                  double symmetry_tolerance = 1e-10);

// Returns `log det(Sigma)` given the lower-triangular factor `L` produced by
// LowerCholeskyFactor().
//
// Evaluated as `2 * sum(log(diag(L)))` rather than by forming a determinant, which for the
// state-dimension-sized covariances used here would underflow to zero long before the log of it
// became inaccurate.
//
// Precondition: `lower_factor` is square with a strictly positive diagonal.
double LogDeterminantFromFactor(const Eigen::Ref<const Eigen::MatrixXd>& lower_factor);

// Maps a standard normal sample `z ~ N(0, I)` to a correlated sample `L * z ~ N(0, Sigma)`.
//
// Precondition: `lower_factor` is square and `standard_normal_sample` has matching size.
Eigen::VectorXd CorrelateSample(const Eigen::Ref<const Eigen::MatrixXd>& lower_factor,
                                const Eigen::Ref<const Eigen::VectorXd>& standard_normal_sample);

}  // namespace fbsde_traj_opt

#endif  // FBSDE_TRAJ_OPT_CHOLESKY_HPP_
