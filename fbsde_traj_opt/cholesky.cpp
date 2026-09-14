// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#include "fbsde_traj_opt/cholesky.hpp"

#include <expected>
#include <string_view>

#include <Eigen/Cholesky>
#include <Eigen/Core>

namespace fbsde_traj_opt {

std::string_view CholeskyErrorName(CholeskyError error) {
  switch (error) {
    case CholeskyError::kNotSquare:
      return "not square";
    case CholeskyError::kEmpty:
      return "empty";
    case CholeskyError::kNotSymmetric:
      return "not symmetric";
    case CholeskyError::kNotPositiveDefinite:
      return "not positive definite";
  }
  return "unknown";
}

std::expected<Eigen::MatrixXd, CholeskyError> LowerCholeskyFactor(const Eigen::Ref<const Eigen::MatrixXd>& covariance,
                                                                  double symmetry_tolerance) {
  if (covariance.rows() != covariance.cols()) {
    return std::unexpected(CholeskyError::kNotSquare);
  }
  // A 0x0 matrix is square, so this check must follow the squareness check above.
  if (covariance.rows() == 0) {
    return std::unexpected(CholeskyError::kEmpty);
  }
  if (symmetry_tolerance >= 0.0 && (covariance - covariance.transpose()).cwiseAbs().maxCoeff() > symmetry_tolerance) {
    return std::unexpected(CholeskyError::kNotSymmetric);
  }

  const Eigen::LLT<Eigen::MatrixXd> decomposition(covariance);
  if (decomposition.info() != Eigen::Success) {
    return std::unexpected(CholeskyError::kNotPositiveDefinite);
  }
  return Eigen::MatrixXd(decomposition.matrixL());
}

double LogDeterminantFromFactor(const Eigen::Ref<const Eigen::MatrixXd>& lower_factor) {
  return 2.0 * lower_factor.diagonal().array().log().sum();
}

Eigen::VectorXd CorrelateSample(const Eigen::Ref<const Eigen::MatrixXd>& lower_factor,
                                const Eigen::Ref<const Eigen::VectorXd>& standard_normal_sample) {
  return lower_factor.triangularView<Eigen::Lower>() * standard_normal_sample;
}

}  // namespace fbsde_traj_opt
