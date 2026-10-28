// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#ifndef FBSDE_TRAJ_OPT_VALUE_FUNCTION_NONNEGATIVE_VALUE_FUNCTION_APPROX_HPP_
#define FBSDE_TRAJ_OPT_VALUE_FUNCTION_NONNEGATIVE_VALUE_FUNCTION_APPROX_HPP_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>

#include <Eigen/Core>

#include "fbsde_traj_opt/utils/result.hpp"
#include "fbsde_traj_opt/value_function/value_function_approx_concepts.hpp"

namespace fbsde_traj_opt {

// A value function model that cannot go negative: another model's output `f` passed through a
// smooth softplus,
//
//   V~(x) = s log(1 + exp(f(x) / s)),
//
// with a softness `s > 0`. Where `f` is well above zero, `V~` equals it to within `s exp(-f / s)`;
// where `f` would go negative, `V~` instead tends to zero from above.
//
// Why. The value function of a problem whose costs are all non-negative is non-negative, but a
// fitted model is not, and in an iterative method a negative basin is self-sustaining: the policy
// steers into it, the targets built from it are low, and erosion keeps the particles that sit in it
// because their heuristic looks cheapest. Flooring the targets removes the negative targets, but not
// the negative values the model can still take between and beyond its samples. Constraining the
// model removes those too.
//
// The derivatives follow from the chain rule with `g(f) = s softplus(f / s)`, `g' = sigmoid(f / s)`,
// `g'' = g' (1 - g') / s`:
//
//   grad V~ = g' grad f,     hess V~ = g' hess f + g'' grad f grad f',     dV~/dtheta = g' df/dtheta.
//
// `InnerT` is any BatchedParameterizedValueFunctionApprox; its parameters are this model's.
template <typename InnerT>
class NonnegativeValueFunctionApprox {
 public:
  using Inner = InnerT;
  using State = typename Inner::State;
  using Scalar = typename State::Scalar;
  using StateMat = Eigen::Matrix<Scalar, State::RowsAtCompileTime, State::RowsAtCompileTime>;
  using ParameterVector = typename Inner::ParameterVector;

  static constexpr int kNumParameters = Inner::kNumParameters;
  static constexpr int kStateDim = State::RowsAtCompileTime;

  // Fails unless the softness is finite and positive.
  static auto Make(Inner inner, Scalar softness) noexcept -> Result<NonnegativeValueFunctionApprox> {
    const bool valid_softness = std::isfinite(softness) && softness > Scalar{0};
    RESULT_ASSERT(valid_softness, "NonnegativeValueFunctionApprox::Make: the softness must be finite and positive.");
    return SuccessResult(NonnegativeValueFunctionApprox(std::move(inner), softness));
  }

  auto operator()(const State& state) const noexcept -> Scalar { return Outer(inner_(state)); }

  [[nodiscard]] auto Gradient(const State& state) const noexcept -> State {
    return Slope(inner_(state)) * inner_.Gradient(state);
  }

  [[nodiscard]] auto Hessian(const State& state) const noexcept -> StateMat {
    const Scalar value = inner_(state);
    const State gradient = inner_.Gradient(state);
    return (Slope(value) * inner_.Hessian(state)) + (Curvature(value) * gradient * gradient.transpose());
  }

  template <int BatchSize>
  auto Values(const ValueFunctionStateBatch<State, BatchSize>& states,
              ValueFunctionValueBatch<State, BatchSize>& values_out) const noexcept -> void {
    inner_.template Values<BatchSize>(states, values_out);
    values_out = values_out.unaryExpr([this](Scalar value) { return Outer(value); });
  }

  template <int BatchSize>
  auto ValuesAndGradients(const ValueFunctionStateBatch<State, BatchSize>& states,
                          ValueFunctionValueBatch<State, BatchSize>& values_out,
                          ValueFunctionGradientBatch<State, BatchSize>& gradients_out) const noexcept -> void {
    inner_.template ValuesAndGradients<BatchSize>(states, values_out, gradients_out);
    for (Eigen::Index column = 0; column < BatchSize; ++column) {
      gradients_out.col(column) *= Slope(values_out[column]);
      values_out[column] = Outer(values_out[column]);
    }
  }

  template <int BatchSize>
  auto ValuesGradientsAndHessians(const ValueFunctionStateBatch<State, BatchSize>& states,
                                  ValueFunctionValueBatch<State, BatchSize>& values_out,
                                  ValueFunctionGradientBatch<State, BatchSize>& gradients_out,
                                  ValueFunctionHessianBatch<State, BatchSize>& hessians_out) const noexcept -> void {
    inner_.template ValuesGradientsAndHessians<BatchSize>(states, values_out, gradients_out, hessians_out);
    for (Eigen::Index column = 0; column < BatchSize; ++column) {
      const Scalar value = values_out[column];
      const State gradient = gradients_out.col(column);
      auto hessian = hessians_out.template block<kStateDim, kStateDim>(0, column * kStateDim);
      hessian = (Slope(value) * hessian) + (Curvature(value) * gradient * gradient.transpose());
      gradients_out.col(column) = Slope(value) * gradient;
      values_out[column] = Outer(value);
    }
  }

  [[nodiscard]] auto ParameterGradient(const State& state) const noexcept -> ParameterVector {
    return Slope(inner_(state)) * inner_.ParameterGradient(state);
  }

  template <int BatchSize>
  auto ValuesAndParameterGradients(
      const ValueFunctionStateBatch<State, BatchSize>& states,
      ValueFunctionValueBatch<State, BatchSize>& values_out,
      ValueFunctionParameterGradientBatch<State, kNumParameters, BatchSize>& parameter_gradients_out) const noexcept
      -> void {
    inner_.template ValuesAndParameterGradients<BatchSize>(states, values_out, parameter_gradients_out);
    for (Eigen::Index column = 0; column < BatchSize; ++column) {
      parameter_gradients_out.col(column) *= Slope(values_out[column]);
      values_out[column] = Outer(values_out[column]);
    }
  }

  [[nodiscard]] auto Parameters() const noexcept -> const ParameterVector& { return inner_.Parameters(); }

  auto SetParameters(const ParameterVector& parameters) noexcept -> Result<> {
    return inner_.SetParameters(parameters);
  }

  [[nodiscard]] auto inner() const noexcept -> const Inner& { return inner_; }

  [[nodiscard]] auto softness() const noexcept -> Scalar { return softness_; }

 private:
  NonnegativeValueFunctionApprox(Inner inner, Scalar softness) noexcept
      : inner_(std::move(inner)), softness_(softness) {}

  // s softplus(f / s), computed without overflow for large f.
  [[nodiscard]] auto Outer(Scalar value) const noexcept -> Scalar {
    const Scalar scaled = value / softness_;
    return softness_ * (std::max(scaled, Scalar{0}) + std::log1p(std::exp(-std::abs(scaled))));
  }

  // sigmoid(f / s).
  [[nodiscard]] auto Slope(Scalar value) const noexcept -> Scalar {
    const Scalar scaled = value / softness_;
    return scaled >= Scalar{0} ? Scalar{1} / (Scalar{1} + std::exp(-scaled))
                               : std::exp(scaled) / (Scalar{1} + std::exp(scaled));
  }

  [[nodiscard]] auto Curvature(Scalar value) const noexcept -> Scalar {
    const Scalar slope = Slope(value);
    return slope * (Scalar{1} - slope) / softness_;
  }

  Inner inner_;
  Scalar softness_;
};

}  // namespace fbsde_traj_opt

#endif  // FBSDE_TRAJ_OPT_VALUE_FUNCTION_NONNEGATIVE_VALUE_FUNCTION_APPROX_HPP_
