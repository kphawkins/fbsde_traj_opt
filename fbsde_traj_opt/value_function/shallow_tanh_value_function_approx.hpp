// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#ifndef FBSDE_TRAJ_OPT_VALUE_FUNCTION_SHALLOW_TANH_VALUE_FUNCTION_APPROX_HPP_
#define FBSDE_TRAJ_OPT_VALUE_FUNCTION_SHALLOW_TANH_VALUE_FUNCTION_APPROX_HPP_

#include <cmath>
#include <cstddef>
#include <cstdint>

#include <Eigen/Core>

#include "fbsde_traj_opt/sampling/counter_based_normal_sampler.hpp"
#include "fbsde_traj_opt/utils/result.hpp"
#include "fbsde_traj_opt/value_function/linear_basis_value_function_approx.hpp"
#include "fbsde_traj_opt/value_function/value_function_approx_concepts.hpp"

namespace fbsde_traj_opt {

// A one-hidden-layer tanh network plus a quadratic, of a normalized state,
//
//   V~(x) = sum_i a_i tanh(w_i' z + b_i) + q' psi(z),    z = (x - center) ./ scale,
//
// with `psi` the Legendre basis of total degree two: the neural-network representation of the deep
// BSDE and deep FBSDE literature, at the smallest size that still has its defining property --
// features that move with the data rather than sit where they were put. The quadratic carries the
// terminal cost's bowl and the growth outside the region the units saturate in.
//
// Fitted by gradient descent (ValueFunctionSgdFitter), with no closed form; its value, state
// gradient, and state Hessian are exact, as the Taylor estimators require.
//
// Parameters, in order: the `Width` output weights a, the `Width x N` input weights w (unit by
// unit), the `Width` biases b, and the quadratic's coefficients q.
template <int N, int Width, typename Scalar = double>
class ShallowTanhValueFunctionApprox {
 public:
  using State = Eigen::Matrix<Scalar, N, 1>;
  using StateMat = Eigen::Matrix<Scalar, N, N>;
  using Quadratic = LegendreBasis<N, 2, Scalar>;

  static constexpr int kNumQuadratic = Quadratic::kNumFeatures;
  static constexpr int kNumParameters = (Width * (N + 2)) + kNumQuadratic;
  using ParameterVector = Eigen::Matrix<Scalar, kNumParameters, 1>;

  template <int BatchSize>
  using StateBatch = ValueFunctionStateBatch<State, BatchSize>;
  template <int BatchSize>
  using ValueBatch = ValueFunctionValueBatch<State, BatchSize>;
  template <int BatchSize>
  using GradientBatch = ValueFunctionGradientBatch<State, BatchSize>;
  template <int BatchSize>
  using HessianBatch = ValueFunctionHessianBatch<State, BatchSize>;
  template <int BatchSize>
  using ParameterGradientBatch = ValueFunctionParameterGradientBatch<State, kNumParameters, BatchSize>;

  static auto Make(const ParameterVector& parameters, const State& center, const State& scale) noexcept
      -> Result<ShallowTanhValueFunctionApprox> {
    RESULT_ASSERT(parameters.allFinite(), "ShallowTanhValueFunctionApprox::Make: the parameters must be finite.");
    RESULT_ASSERT(center.allFinite() && scale.allFinite() && (scale.array() > Scalar{0}).all(),
                  "ShallowTanhValueFunctionApprox::Make: the center must be finite and every scale finite and "
                  "positive.");
    return SuccessResult(ShallowTanhValueFunctionApprox(parameters, center, scale));
  }

  // Random hidden units -- input weights of unit norm in random directions, biases spread over
  // [-1, 1], output weights of `output_scale` -- with the quadratic's coefficients `quadratic`.
  [[nodiscard]] static auto RandomParameters(const Eigen::Matrix<Scalar, kNumQuadratic, 1>& quadratic,
                                             Scalar output_scale,
                                             std::uint64_t seed) noexcept -> ParameterVector {
    const CounterBasedNormalSampler sampler(seed);
    ParameterVector parameters = ParameterVector::Zero();
    for (int unit = 0; unit < Width; ++unit) {
      const auto u = static_cast<std::size_t>(unit);
      parameters[unit] = output_scale * static_cast<Scalar>(sampler.Sample(u, 0, 0));
      State direction;
      for (int axis = 0; axis < N; ++axis) {
        direction[axis] = static_cast<Scalar>(sampler.Sample(u, 1, static_cast<std::size_t>(axis)));
      }
      direction /= std::max(direction.norm(), Scalar{1e-9});
      parameters.template segment<N>(Width + (unit * N)) = Scalar{2} * direction;
      parameters[Width + (Width * N) + unit] = static_cast<Scalar>(std::tanh(sampler.Sample(u, 2, 0)));
    }
    parameters.template tail<kNumQuadratic>() = quadratic;
    return parameters;
  }

  auto operator()(const State& state) const noexcept -> Scalar {
    Scalar value;
    State gradient;
    StateMat hessian;
    Evaluate<false, false>(state, value, gradient, hessian);
    return value;
  }

  [[nodiscard]] auto Gradient(const State& state) const noexcept -> State {
    Scalar value;
    State gradient;
    StateMat hessian;
    Evaluate<true, false>(state, value, gradient, hessian);
    return gradient;
  }

  [[nodiscard]] auto Hessian(const State& state) const noexcept -> StateMat {
    Scalar value;
    State gradient;
    StateMat hessian;
    Evaluate<true, true>(state, value, gradient, hessian);
    return hessian;
  }

  [[nodiscard]] auto ParameterGradient(const State& state) const noexcept -> ParameterVector {
    const State z = Normalize(state);
    ParameterVector gradient;
    for (int unit = 0; unit < Width; ++unit) {
      const Scalar t = std::tanh(InputWeights(unit).dot(z) + Bias(unit));
      const Scalar dt = Scalar{1} - (t * t);
      gradient[unit] = t;
      gradient.template segment<N>(Width + (unit * N)) = (parameters_[unit] * dt) * z;
      gradient[Width + (Width * N) + unit] = parameters_[unit] * dt;
    }
    gradient.template tail<kNumQuadratic>() = QuadraticFeatures(z);
    return gradient;
  }

  [[nodiscard]] auto Parameters() const noexcept -> const ParameterVector& { return parameters_; }

  auto SetParameters(const ParameterVector& parameters) noexcept -> Result<> {
    RESULT_ASSERT(parameters.allFinite(), "ShallowTanhValueFunctionApprox::SetParameters: non-finite parameters.");
    parameters_ = parameters;
    return SuccessResult();
  }

  [[nodiscard]] auto QuadraticFeatures(const State& z) const noexcept -> Eigen::Matrix<Scalar, kNumQuadratic, 1> {
    Eigen::Matrix<Scalar, kNumQuadratic, 1> phi;
    Eigen::Matrix<Scalar, N, kNumQuadratic> unused_gradients;
    Eigen::Matrix<Scalar, N * N, kNumQuadratic> unused_hessians;
    Quadratic{}.template Jet<false, false>(z, phi, unused_gradients, unused_hessians);
    return phi;
  }

  [[nodiscard]] auto Normalize(const State& state) const noexcept -> State {
    return (state - center_).cwiseProduct(inverse_scale_);
  }

  template <int BatchSize>
  auto Values(const StateBatch<BatchSize>& states, ValueBatch<BatchSize>& values_out) const noexcept -> void {
    for (Eigen::Index column = 0; column < BatchSize; ++column) {
      values_out[column] = (*this)(states.col(column));
    }
  }

  template <int BatchSize>
  auto ValuesAndGradients(const StateBatch<BatchSize>& states,
                          ValueBatch<BatchSize>& values_out,
                          GradientBatch<BatchSize>& gradients_out) const noexcept -> void {
    for (Eigen::Index column = 0; column < BatchSize; ++column) {
      State gradient;
      StateMat hessian;
      Evaluate<true, false>(states.col(column), values_out[column], gradient, hessian);
      gradients_out.col(column) = gradient;
    }
  }

  template <int BatchSize>
  auto ValuesGradientsAndHessians(const StateBatch<BatchSize>& states,
                                  ValueBatch<BatchSize>& values_out,
                                  GradientBatch<BatchSize>& gradients_out,
                                  HessianBatch<BatchSize>& hessians_out) const noexcept -> void {
    for (Eigen::Index column = 0; column < BatchSize; ++column) {
      State gradient;
      StateMat hessian;
      Evaluate<true, true>(states.col(column), values_out[column], gradient, hessian);
      gradients_out.col(column) = gradient;
      hessians_out.template block<N, N>(0, column * N) = hessian;
    }
  }

  template <int BatchSize>
  auto ValuesAndParameterGradients(const StateBatch<BatchSize>& states,
                                   ValueBatch<BatchSize>& values_out,
                                   ParameterGradientBatch<BatchSize>& parameter_gradients_out) const noexcept -> void {
    for (Eigen::Index column = 0; column < BatchSize; ++column) {
      const ParameterVector gradient = ParameterGradient(states.col(column));
      parameter_gradients_out.col(column) = gradient;
      values_out[column] = (*this)(states.col(column));
    }
  }

 private:
  ShallowTanhValueFunctionApprox(const ParameterVector& parameters, const State& center, const State& scale) noexcept
      : parameters_(parameters), center_(center), inverse_scale_(scale.cwiseInverse()) {}

  [[nodiscard]] auto InputWeights(int unit) const noexcept { return parameters_.template segment<N>(Width + (unit * N)); }
  [[nodiscard]] auto Bias(int unit) const noexcept -> Scalar { return parameters_[Width + (Width * N) + unit]; }

  template <bool WithGradient, bool WithHessian>
  auto Evaluate(const State& state, Scalar& value_out, State& gradient_out, StateMat& hessian_out) const noexcept
      -> void {
    const State z = Normalize(state);
    Scalar value = Scalar{0};
    State gradient_z = State::Zero();
    StateMat hessian_z = StateMat::Zero();
    for (int unit = 0; unit < Width; ++unit) {
      const State w = InputWeights(unit);
      const Scalar t = std::tanh(w.dot(z) + Bias(unit));
      const Scalar a = parameters_[unit];
      value += a * t;
      if constexpr (WithGradient || WithHessian) {
        gradient_z += (a * (Scalar{1} - (t * t))) * w;
      }
      if constexpr (WithHessian) {
        hessian_z += (a * Scalar{-2} * t * (Scalar{1} - (t * t))) * (w * w.transpose());
      }
    }
    Eigen::Matrix<Scalar, kNumQuadratic, 1> phi;
    Eigen::Matrix<Scalar, N, kNumQuadratic> dphi;
    Eigen::Matrix<Scalar, N * N, kNumQuadratic> d2phi;
    Quadratic{}.template Jet<WithGradient || WithHessian, WithHessian>(z, phi, dphi, d2phi);
    const auto q = parameters_.template tail<kNumQuadratic>();
    value_out = value + phi.dot(q);
    if constexpr (WithGradient || WithHessian) {
      gradient_out = (gradient_z + (dphi * q)).cwiseProduct(inverse_scale_);
    }
    if constexpr (WithHessian) {
      const Eigen::Matrix<Scalar, N * N, 1> flat = d2phi * q;
      for (int column = 0; column < N; ++column) {
        for (int row = 0; row < N; ++row) {
          hessian_out(row, column) =
              (hessian_z(row, column) + flat[(column * N) + row]) * inverse_scale_[row] * inverse_scale_[column];
        }
      }
    }
  }

  ParameterVector parameters_;
  State center_;
  State inverse_scale_;
};

}  // namespace fbsde_traj_opt

#endif  // FBSDE_TRAJ_OPT_VALUE_FUNCTION_SHALLOW_TANH_VALUE_FUNCTION_APPROX_HPP_
