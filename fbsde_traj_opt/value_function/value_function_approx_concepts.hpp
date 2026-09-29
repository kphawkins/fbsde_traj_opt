// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#ifndef FBSDE_TRAJ_OPT_VALUE_FUNCTION_APPROX_CONCEPTS_HPP_
#define FBSDE_TRAJ_OPT_VALUE_FUNCTION_APPROX_CONCEPTS_HPP_

#include <concepts>

#include <Eigen/Core>

#include "fbsde_traj_opt/utils/eigen_concepts.hpp"
#include "fbsde_traj_opt/utils/result.hpp"

namespace fbsde_traj_opt {

// Concepts for an approximate value function at one stage of a discrete-time stochastic optimal
// control problem -- the object written `V~_i` in the DT-FBSDE literature, standing in for the
// true stage-i value function `V_i`.
//
// The backward recursion needs three things from `V~_{i+1}` and nothing else: its value, its
// gradient in the state, and its Hessian in the state. That is not an arbitrary choice. Every
// Taylor-expansion estimator of the backward step is built from the second-order expansion of
// `V~_{i+1}` around the conditional mean of the next state, so the second derivative is exactly
// where the expansion stops and exactly what the trace correction needs. A representation that
// cannot produce a Hessian cannot be used with these estimators at all, which is why the Hessian
// is part of the base concept rather than an optional refinement.
//
// Fitting the representation needs one more thing -- the gradient with respect to the parameters
// -- and that is a separate concept, because the two roles are separate. Within one backward step
// `V~_{i+1}` is frozen and only ever read, while `V~_i` is written to; a model used solely as a
// frozen next-stage representation need satisfy only ValueFunctionApprox.
//
// Four concepts, in two pairs:
//
//   ValueFunctionApprox                        one state at a time
//   BatchedValueFunctionApprox                 a whole batch of states at once
//   ParameterizedValueFunctionApprox           one state at a time, plus the parameter gradient
//   BatchedParameterizedValueFunctionApprox    a whole batch, plus the parameter gradients
//
// The batched forms exist for speed, and the shape of the batch types is what buys it. States are
// held one per column of an `N x BatchSize` matrix, so an operation applied to every sample is one
// Eigen expression over a contiguous fixed-size matrix rather than a loop over vectors -- a GEMM
// where the pointwise form would be a sequence of GEMVs. The same layout is what TrajectoryBatch
// already uses for the states at a stage, so a sampled batch feeds these interfaces directly.
//
// `BatchSize` is a compile-time parameter rather than a runtime one, matching the rest of this
// codebase's fixed-shape posture: the column count is known to the vectorizer, and a shape
// mismatch is a compile error rather than a runtime check. The cost is that a caller whose sample
// count is not a multiple of `BatchSize` must arrange full batches itself; see
// ValueFunctionSgdFitter, which does so by drawing a wrap-around index permutation.

// The states of one batch: one column per sample.
template <typename State, int BatchSize>
using ValueFunctionStateBatch = Eigen::Matrix<typename State::Scalar, State::RowsAtCompileTime, BatchSize>;

// One scalar value per sample of a batch, as a row vector so that it lines up column-wise with
// ValueFunctionStateBatch.
template <typename State, int BatchSize>
using ValueFunctionValueBatch = Eigen::Matrix<typename State::Scalar, 1, BatchSize>;

// One state-gradient per sample of a batch, one column per sample.
template <typename State, int BatchSize>
using ValueFunctionGradientBatch = Eigen::Matrix<typename State::Scalar, State::RowsAtCompileTime, BatchSize>;

// One state-Hessian per sample of a batch, laid out as `BatchSize` side-by-side `N x N` blocks:
// sample `s` occupies columns `s * N` through `s * N + N - 1`, which in Eigen's column-major
// storage is one contiguous run of `N * N` scalars. Retrieve it with
// `hessians.template block<N, N>(0, s * N)`.
//
// This grows as `N * N * BatchSize`, so a caller combining a large state dimension with a large
// batch should give the matrix storage of its own rather than leaving it on the stack.
template <typename State, int BatchSize>
using ValueFunctionHessianBatch =
    Eigen::Matrix<typename State::Scalar, State::RowsAtCompileTime, State::RowsAtCompileTime * BatchSize>;

// One parameter-gradient per sample of a batch, one column per sample. Transposed against the
// convention a design matrix usually follows, and deliberately: the gradient of a mean squared
// error over the batch is then `parameter_gradients * residuals`, a single matrix-vector product
// with no transpose in the inner loop.
template <typename State, int NumParameters, int BatchSize>
using ValueFunctionParameterGradientBatch = Eigen::Matrix<typename State::Scalar, NumParameters, BatchSize>;

// Concept for a type `T` representing an approximate value function `V~(x)` over states of type
// `State`, able to report its value, its gradient in the state, and its Hessian in the state.
//
// A conforming `T` is callable as `value_function(state)` returning the same `Scalar` as `State`,
// and offers `Gradient(state)` returning a fixed-size column vector of the state's dimension and
// `Hessian(state)` returning a fixed-size square matrix of that dimension. All three are const:
// reading a value function never changes it.
//
// The Hessian is required to be symmetric for every state a conforming type is invoked with. That
// is a semantic requirement on implementations which the concept cannot check, and callers may
// rely on it -- the trace correction `tr(Sigma' H Sigma)` in the backward estimators is computed
// as a coefficient-wise product on the strength of it.
template <typename T, typename State>
concept ValueFunctionApprox =
    EigenFixedSizeColumnVector<State> && requires(const T& value_function, const State& state) {
      { value_function(state) } -> std::same_as<typename State::Scalar>;
      { value_function.Gradient(state) } -> EigenFixedSizeColumnVectorOfDimension<State::RowsAtCompileTime>;
      { value_function.Hessian(state) } -> EigenFixedSizeSquareMatrixOfDimension<State::RowsAtCompileTime>;
    };

// Concept for a ValueFunctionApprox that can also evaluate a whole batch of `BatchSize` states in
// one call.
//
// A conforming `T` offers three batch evaluations, each writing into caller-supplied output
// matrices rather than returning them, since a batch of Hessians is far too large to return by
// value:
//
//   Values(states, values_out)
//   ValuesAndGradients(states, values_out, gradients_out)
//   ValuesGradientsAndHessians(states, values_out, gradients_out, hessians_out)
//
// They are three entry points rather than one because the three cost very different amounts and
// the callers want different subsets: fitting needs values only, a policy improvement step needs
// values and gradients, and the Taylor estimators need all three. Each is required to produce
// results identical to calling the pointwise form on each column.
template <typename T, typename State, int BatchSize>
concept BatchedValueFunctionApprox =
    ValueFunctionApprox<T, State> && BatchSize >= 1 &&
    requires(const T& value_function,
             const ValueFunctionStateBatch<State, BatchSize>& states,
             ValueFunctionValueBatch<State, BatchSize>& values_out,
             ValueFunctionGradientBatch<State, BatchSize>& gradients_out,
             ValueFunctionHessianBatch<State, BatchSize>& hessians_out) {
      { value_function.Values(states, values_out) } -> std::same_as<void>;
      { value_function.ValuesAndGradients(states, values_out, gradients_out) } -> std::same_as<void>;
      {
        value_function.ValuesGradientsAndHessians(states, values_out, gradients_out, hessians_out)
      } -> std::same_as<void>;
    };

// Concept for a ValueFunctionApprox carrying a fixed-size vector of free parameters that can be
// read, replaced, and differentiated against -- everything a gradient-based fitter needs in order
// to move the representation toward a set of targets.
//
// A conforming `T` declares `kNumParameters` and a `ParameterVector` type of that dimension, and
// offers:
//
//   Parameters()              the current parameter vector
//   SetParameters(parameters) replaces it
//   ParameterGradient(state)  d V~(state) / d parameters, at the current parameters
//
// The type alias is `ParameterVector` rather than `Parameters` because the accessor already has
// that name and a class cannot spell both.
//
// `SetParameters` returns a `Result<>` rather than `void` for two reasons. A model may have
// parameters it must reject -- a non-finite entry, most obviously, which a diverging optimizer
// will eventually hand it -- and reporting that is better than carrying a quietly poisoned
// representation forward. More usefully, it gives every model exactly one place to refresh
// whatever it caches from the raw parameter vector, so that a batch evaluation of a thousand
// states unpacks the parameterization once rather than a thousand times.
template <typename T, typename State>
concept ParameterizedValueFunctionApprox =
    ValueFunctionApprox<T, State> &&
    requires {
      { T::kNumParameters } -> std::convertible_to<int>;
      typename T::ParameterVector;
    } && EigenFixedSizeColumnVectorOfDimension<typename T::ParameterVector, T::kNumParameters> &&
    requires(T& mutable_value_function,
             const T& value_function,
             const typename T::ParameterVector& parameters,
             const State& state) {
      { value_function.Parameters() } -> std::convertible_to<const typename T::ParameterVector&>;
      { mutable_value_function.SetParameters(parameters) } -> std::same_as<Result<>>;
      { value_function.ParameterGradient(state) } -> EigenFixedSizeColumnVectorOfDimension<T::kNumParameters>;
    };

// Concept for a ParameterizedValueFunctionApprox that can evaluate the values and the parameter
// gradients of a whole batch of `BatchSize` states in one call.
//
// A conforming `T` offers `ValuesAndParameterGradients(states, values_out,
// parameter_gradients_out)`, writing one value per column of `values_out` and one parameter
// gradient per column of `parameter_gradients_out`. The two are produced together rather than
// separately because a least-squares step needs both and the parameter gradient's computation
// already produces the value along the way.
template <typename T, typename State, int BatchSize>
concept BatchedParameterizedValueFunctionApprox =
    ParameterizedValueFunctionApprox<T, State> && BatchedValueFunctionApprox<T, State, BatchSize> &&
    requires(const T& value_function,
             const ValueFunctionStateBatch<State, BatchSize>& states,
             ValueFunctionValueBatch<State, BatchSize>& values_out,
             ValueFunctionParameterGradientBatch<State, T::kNumParameters, BatchSize>& parameter_gradients_out) {
      { value_function.ValuesAndParameterGradients(states, values_out, parameter_gradients_out) } -> std::same_as<void>;
    };

}  // namespace fbsde_traj_opt

#endif  // FBSDE_TRAJ_OPT_VALUE_FUNCTION_APPROX_CONCEPTS_HPP_
