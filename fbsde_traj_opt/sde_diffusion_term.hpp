// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#ifndef FBSDE_TRAJ_OPT_SDE_DIFFUSION_TERM_HPP_
#define FBSDE_TRAJ_OPT_SDE_DIFFUSION_TERM_HPP_

#include <cstddef>

#include "fbsde_traj_opt/eigen_concepts.hpp"

namespace fbsde_traj_opt {

// Concept for a functor type `T` that supplies the diffusion coefficient (the "Sigma" matrix) of a
// discrete-time forward SDE
//
//   x_{k+1} = f(k, x_k) + Sigma(k, x_k) * z_k,     z_k ~ N(0, I),
//
// shaping the Brownian noise increment injected at each stage. `State` is the fixed-size Eigen
// column vector type of `x_k`.
//
// A conforming `T` is callable as `diffusion_term(stage, state)`, where `stage` is a
// `std::size_t` and `state` is a `State`, and returns a fixed-size N x N Eigen matrix, where N is
// `State`'s compile-time size. The return type may be any kind of Eigen matrix expression -- a
// dense Eigen::Matrix, an Eigen::DiagonalMatrix, or any other type that exposes the usual
// compile-time Eigen shape interface -- so that an implementation whose Sigma is structurally
// diagonal (the common case) can return a lightweight Eigen::DiagonalMatrix rather than
// materializing a dense matrix.
//
// Sigma is required to be nonsingular for every stage and state a conforming type is invoked
// with; this is a semantic requirement on implementations that the concept itself cannot check.
template <typename T, typename State>
concept SdeDiffusionTerm =
    EigenFixedSizeColumnVector<State> && requires(const T& diffusion_term, std::size_t stage, const State& state) {
      { diffusion_term(stage, state) } -> EigenFixedSizeSquareMatrixOfDimension<State::RowsAtCompileTime>;
    };

}  // namespace fbsde_traj_opt

#endif  // FBSDE_TRAJ_OPT_SDE_DIFFUSION_TERM_HPP_
