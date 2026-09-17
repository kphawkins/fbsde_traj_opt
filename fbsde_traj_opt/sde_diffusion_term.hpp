// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#ifndef FBSDE_TRAJ_OPT_SDE_DIFFUSION_TERM_HPP_
#define FBSDE_TRAJ_OPT_SDE_DIFFUSION_TERM_HPP_

#include <concepts>
#include <cstddef>

#include <Eigen/Core>

namespace fbsde_traj_opt {

namespace detail {

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

template <typename T>
concept EigenFixedSizeColumnVector =
    EigenExpressionWithCompileTimeShape<T> && T::ColsAtCompileTime == 1 && T::RowsAtCompileTime != Eigen::Dynamic;

template <typename T, int N>
concept EigenFixedSizeSquareMatrixOfDimension =
    EigenExpressionWithCompileTimeShape<T> && T::RowsAtCompileTime == N && T::ColsAtCompileTime == N;

}  // namespace detail

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
concept SdeDiffusionTerm = detail::EigenFixedSizeColumnVector<State> &&
                           requires(const T& diffusion_term, std::size_t stage, const State& state) {
                             {
                               diffusion_term(stage, state)
                             } -> detail::EigenFixedSizeSquareMatrixOfDimension<State::RowsAtCompileTime>;
                           };

}  // namespace fbsde_traj_opt

#endif  // FBSDE_TRAJ_OPT_SDE_DIFFUSION_TERM_HPP_
