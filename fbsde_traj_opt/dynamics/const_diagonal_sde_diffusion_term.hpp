// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#ifndef FBSDE_TRAJ_OPT_CONST_DIAGONAL_SDE_DIFFUSION_TERM_HPP_
#define FBSDE_TRAJ_OPT_CONST_DIAGONAL_SDE_DIFFUSION_TERM_HPP_

#include <cstddef>
#include <utility>

#include <Eigen/Core>

#include "fbsde_traj_opt/utils/result.hpp"

namespace fbsde_traj_opt {

// An SdeDiffusionTerm (see sde_term_concepts.hpp) whose Sigma matrix is diagonal and constant
// across every stage and state.
//
// `N` is the compile-time state dimension; `Scalar` defaults to `double`.
template <int N, typename Scalar = double>
class ConstDiagonalSdeDiffusionTerm {
 public:
  using State = Eigen::Matrix<Scalar, N, 1>;

  // The smallest and largest value a diagonal element may take. The lower bound is strictly
  // positive rather than symmetric with the upper bound, since every diagonal element must be
  // nonzero for Sigma to be nonsingular, as SdeDiffusionTerm requires (see
  // sde_term_concepts.hpp); the upper bound guards against the same units-or-configuration-error
  // case as elsewhere in this codebase, where an out-of-range value is almost certainly a mistake
  // rather than an intentionally extreme diffusion coefficient.
  static constexpr Scalar kMinDiagonalElement = static_cast<Scalar>(1e-10);
  static constexpr Scalar kMaxDiagonalElement = static_cast<Scalar>(1e10);

  // Builds a ConstDiagonalSdeDiffusionTerm whose Sigma is diag(diagonal).
  //
  // Fails if any element of `diagonal` falls outside [kMinDiagonalElement, kMaxDiagonalElement].
  static auto Make(const State& diagonal) noexcept -> Result<ConstDiagonalSdeDiffusionTerm> {
    RESULT_ASSERT((diagonal.array() >= kMinDiagonalElement).all() && (diagonal.array() <= kMaxDiagonalElement).all(),
                  "ConstDiagonalSdeDiffusionTerm::Make: every diagonal element must lie in [1e-10, 1e10].");
    return SuccessResult(ConstDiagonalSdeDiffusionTerm(diagonal));
  }

  // Returns Sigma = diag(diagonal), independent of `stage` and `state`.
  auto operator()([[maybe_unused]] std::size_t stage, [[maybe_unused]] const State& state) const noexcept
      -> Eigen::DiagonalMatrix<Scalar, N> {
    return Eigen::DiagonalMatrix<Scalar, N>(diagonal_);
  }

 private:
  explicit ConstDiagonalSdeDiffusionTerm(State diagonal) noexcept : diagonal_(std::move(diagonal)) {}

  State diagonal_;
};

}  // namespace fbsde_traj_opt

#endif  // FBSDE_TRAJ_OPT_CONST_DIAGONAL_SDE_DIFFUSION_TERM_HPP_
