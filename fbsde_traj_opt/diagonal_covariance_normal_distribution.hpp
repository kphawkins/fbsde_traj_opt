// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#ifndef FBSDE_TRAJ_OPT_DIAGONAL_COVARIANCE_NORMAL_DISTRIBUTION_HPP_
#define FBSDE_TRAJ_OPT_DIAGONAL_COVARIANCE_NORMAL_DISTRIBUTION_HPP_

#include <utility>

#include <Eigen/Core>

#include "fbsde_traj_opt/utils/result.hpp"

namespace fbsde_traj_opt {

// A NormalDistribution (see normal_distribution_concepts.hpp) whose Covariance is diag(diagonal)
// for some constant, strictly positive `diagonal`.
//
// `N` is the compile-time state dimension; `Scalar` defaults to `double`.
template <int N, typename Scalar = double>
class DiagonalCovarianceNormalDistribution {
 public:
  using State = Eigen::Matrix<Scalar, N, 1>;

  // The smallest and largest value a mean element may take. Values outside this range are almost
  // certainly a units or configuration error rather than an intentionally extreme mean, so Make()
  // rejects them rather than silently building a distribution that would immediately overflow or
  // underflow downstream computations.
  static constexpr Scalar kMinMeanElement = static_cast<Scalar>(-1e10);
  static constexpr Scalar kMaxMeanElement = static_cast<Scalar>(1e10);

  // The smallest and largest value a covariance diagonal element may take, for the same
  // units-or-configuration-error reason as kMinMeanElement/kMaxMeanElement above. The lower bound
  // is strictly positive (rather than symmetric with the upper bound, as kMinMeanElement is) since
  // that is exactly the condition under which diag(diagonal) is positive definite, which every
  // NormalDistribution's Covariance is required to be (see normal_distribution_concepts.hpp).
  static constexpr Scalar kMinCovarianceDiagonalElement = static_cast<Scalar>(1e-10);
  static constexpr Scalar kMaxCovarianceDiagonalElement = static_cast<Scalar>(1e10);

  // Builds a DiagonalCovarianceNormalDistribution N(mean, diag(covariance_diagonal)).
  //
  // Fails if any element of `mean` falls outside [kMinMeanElement, kMaxMeanElement], or if any
  // element of `covariance_diagonal` falls outside [kMinCovarianceDiagonalElement,
  // kMaxCovarianceDiagonalElement].
  static auto Make(const State& mean, const State& covariance_diagonal) noexcept
      -> Result<DiagonalCovarianceNormalDistribution> {
    RESULT_ASSERT((mean.array() >= kMinMeanElement).all() && (mean.array() <= kMaxMeanElement).all(),
                  "DiagonalCovarianceNormalDistribution::Make: every mean element must lie in [-1e10, 1e10].");
    RESULT_ASSERT((covariance_diagonal.array() >= kMinCovarianceDiagonalElement).all() &&
                      (covariance_diagonal.array() <= kMaxCovarianceDiagonalElement).all(),
                  "DiagonalCovarianceNormalDistribution::Make: every covariance diagonal element must lie in "
                  "[1e-10, 1e10].");
    return SuccessResult(DiagonalCovarianceNormalDistribution(mean, covariance_diagonal));
  }

  // Returns mu = mean, the mean of the distribution.
  [[nodiscard]] auto Mean() const noexcept -> State { return mean_; }

  // Returns Sigma = diag(covariance_diagonal), the covariance of the distribution.
  [[nodiscard]] auto Covariance() const noexcept -> Eigen::DiagonalMatrix<Scalar, N> {
    return Eigen::DiagonalMatrix<Scalar, N>(covariance_diagonal_);
  }

 private:
  DiagonalCovarianceNormalDistribution(State mean, State covariance_diagonal) noexcept
      : mean_(std::move(mean)), covariance_diagonal_(std::move(covariance_diagonal)) {}

  State mean_;
  State covariance_diagonal_;
};

}  // namespace fbsde_traj_opt

#endif  // FBSDE_TRAJ_OPT_DIAGONAL_COVARIANCE_NORMAL_DISTRIBUTION_HPP_
