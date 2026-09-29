// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#ifndef FBSDE_TRAJ_OPT_SOFT_MIN_QUADRATIC_VALUE_FUNCTION_APPROX_HPP_
#define FBSDE_TRAJ_OPT_SOFT_MIN_QUADRATIC_VALUE_FUNCTION_APPROX_HPP_

#include <array>
#include <cmath>
#include <cstddef>

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include "fbsde_traj_opt/utils/result.hpp"
#include "fbsde_traj_opt/value_function/value_function_approx_concepts.hpp"

namespace fbsde_traj_opt {

// A BatchedParameterizedValueFunctionApprox (see value_function_approx_concepts.hpp) that
// represents a value function as a smooth minimum over `NumComponents` convex quadratics:
//
//   V~(x) = -(1/beta) log( sum_p exp(-beta q_p(x)) ),
//   q_p(x) = || L_p' (x - b_p) ||^2 + c_p,
//
// where `L_p` is lower triangular, `b_p` is a center, and `c_p` is a constant.
//
// Why this family. A value function of a control problem is, near its minimum, a quadratic, and
// is exactly quadratic for the whole of an LQR problem -- so one component already covers the
// baseline case exactly. What one component cannot do is represent a cost landscape with several
// basins, which is what a nonlinear problem or an obstacle field produces. Taking a minimum over
// several quadratics covers that, and taking a *soft* minimum keeps the result twice
// differentiable, which the Taylor estimators require and a hard minimum does not provide along
// the ridges where the active component changes. `beta` interpolates between the two: as it grows
// the soft minimum approaches the hard one, at the cost of curvature that grows just as fast
// along those ridges.
//
// Parameterization. Every free parameter is unconstrained, so a gradient step can never leave the
// feasible set and the fitter needs no projection. The diagonal of `L_p` is the only entry that
// is not free outright: it is `softplus(theta) + eps` for a free `theta` and a positive
// metaparameter `eps`, which keeps it strictly positive and therefore keeps `L_p` nonsingular and
// each `A_p := L_p L_p'` positive definite. `softplus` rather than `exp` because its gradient
// neither explodes for large `theta` nor vanishes for moderately negative `theta`; `eps` rather
// than nothing because `softplus` alone approaches zero, and a component whose metric is
// approaching singular has a gradient that degenerates with it.
//
// The parameter vector is the components' blocks laid end to end. Within one component: the
// lower triangle of `L_p` column by column, then `b_p`, then `c_p`.
//
// Derivatives. With `w_p := softmax_p(-beta q_p)`, `d_p := x - b_p`, `v_p := L_p' d_p`:
//
//   grad q_p  = 2 L_p v_p
//   grad V~   = sum_p w_p grad q_p
//   hess V~   = 2 sum_p w_p A_p - beta ( sum_p w_p (grad q_p)(grad q_p)' - (grad V~)(grad V~)' )
//
// The second Hessian term is `-beta` times the softmax-weighted covariance of the component
// gradients, so it is negative semidefinite: the soft minimum of convex quadratics is itself not
// convex, and this model reports that honestly rather than returning a curvature that is always
// positive. The parameter derivatives are
//
//   d q_p / d c_p       = 1
//   d q_p / d b_p       = -2 L_p v_p
//   d q_p / d L_p[i,j]  = 2 d_p[i] v_p[j]          for i >= j,
//
// the last multiplied by `sigmoid(theta)` on the diagonal, that being the derivative of the
// softplus reparameterization, and each weighted by `w_p` on the way into `d V~ / d theta`.
//
// `N` is the compile-time state dimension, `NumComponents` the number of quadratics, and
// `Scalar` defaults to `double`.
//
// Storage during a batch evaluation: the batch methods hold `NumComponents` intermediate
// `N x BatchSize` matrices on the stack. A caller combining a large state dimension, many
// components, and a large batch should keep that product in mind.
template <int N, int NumComponents, typename Scalar = double>
class SoftMinQuadraticValueFunctionApprox {
  static_assert(N >= 1, "A value function needs at least a one-dimensional state.");
  static_assert(NumComponents >= 1, "A soft minimum needs at least one component to take it over.");

 public:
  using State = Eigen::Matrix<Scalar, N, 1>;
  using StateMat = Eigen::Matrix<Scalar, N, N>;

  static constexpr std::size_t kStateDim = static_cast<std::size_t>(N);
  static constexpr std::size_t kNumComponents = static_cast<std::size_t>(NumComponents);

  // The free parameters of one component: the lower triangle of `L_p`, then `b_p`, then `c_p`.
  static constexpr int kLowerTriangleSize = N * (N + 1) / 2;
  static constexpr int kParametersPerComponent = kLowerTriangleSize + N + 1;

  static constexpr int kNumParameters = NumComponents * kParametersPerComponent;

  using ParameterVector = Eigen::Matrix<Scalar, kNumParameters, 1>;

  // The batch shapes of value_function_approx_concepts.hpp, named locally for brevity.
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

  // Builds the model from a parameter vector and the two metaparameters.
  //
  // `sharpness` is the `beta` above: larger values make the soft minimum a closer approximation
  // of the hard one. `diagonal_offset` is the `eps` above: the floor under every diagonal entry
  // of every `L_p`, and so the guarantee that each component's metric stays positive definite.
  //
  // Fails if either metaparameter is not finite and strictly positive, or if any parameter is not
  // finite.
  static auto Make(const ParameterVector& parameters, Scalar sharpness, Scalar diagonal_offset) noexcept
      -> Result<SoftMinQuadraticValueFunctionApprox> {
    RESULT_ASSERT(std::isfinite(sharpness) && sharpness > Scalar{0},
                  "SoftMinQuadraticValueFunctionApprox::Make: the sharpness beta must be finite and positive.");
    RESULT_ASSERT(std::isfinite(diagonal_offset) && diagonal_offset > Scalar{0},
                  "SoftMinQuadraticValueFunctionApprox::Make: the diagonal offset eps must be finite and positive.");
    RESULT_ASSERT(parameters.allFinite(), "SoftMinQuadraticValueFunctionApprox::Make: the parameters must be finite.");

    SoftMinQuadraticValueFunctionApprox approximation(sharpness, diagonal_offset);
    approximation.parameters_ = parameters;
    approximation.RefreshCaches();
    return SuccessResult(approximation);
  }

  // Returns the parameters for which this model reproduces the single quadratic
  //
  //   q(x) = (x - center)' metric (x - center) + offset
  //
  // exactly, whatever `NumComponents` is. Every component is given that same quadratic, and the
  // constant is raised by `log(NumComponents) / sharpness` to cancel the amount by which a soft
  // minimum over that many identical values falls below their common value.
  //
  // This is how an LQR value function -- which is exactly quadratic -- is installed in the model,
  // whether to seed a backward pass at the terminal stage or to check an estimator against a case
  // whose answer is known in closed form.
  //
  // Fails if `metric` is not positive definite, since it is stored through its Cholesky factor
  // `L`; if any diagonal entry of that factor does not exceed `diagonal_offset`, since the
  // diagonal is parameterized as `softplus(theta) + diagonal_offset` and so can never fall to or
  // below the offset; or if either metaparameter is not finite and positive.
  static auto QuadraticParameters(const StateMat& metric,
                                  const State& center,
                                  Scalar offset,
                                  Scalar sharpness,
                                  Scalar diagonal_offset) noexcept -> Result<ParameterVector> {
    RESULT_ASSERT(
        std::isfinite(sharpness) && sharpness > Scalar{0},
        "SoftMinQuadraticValueFunctionApprox::QuadraticParameters: the sharpness beta must be finite and positive.");
    RESULT_ASSERT(std::isfinite(diagonal_offset) && diagonal_offset > Scalar{0},
                  "SoftMinQuadraticValueFunctionApprox::QuadraticParameters: the diagonal offset eps must be finite "
                  "and positive.");

    const Eigen::LLT<StateMat> factorization(metric);
    RESULT_ASSERT(factorization.info() == Eigen::Success,
                  "SoftMinQuadraticValueFunctionApprox::QuadraticParameters: the metric is not positive definite, so "
                  "it has no Cholesky factor to store.");
    const StateMat factor = factorization.matrixL();

    // A soft minimum over `NumComponents` copies of the same quadratic sits log(NumComponents)/beta
    // below it; raising every component's constant by that much puts the result back on target.
    const Scalar component_offset = offset + (std::log(static_cast<Scalar>(NumComponents)) / sharpness);

    ParameterVector parameters = ParameterVector::Zero();
    for (std::size_t component = 0; component < kNumComponents; ++component) {
      const auto base = static_cast<Eigen::Index>(component * static_cast<std::size_t>(kParametersPerComponent));
      for (std::size_t col = 0; col < kStateDim; ++col) {
        for (std::size_t row = col; row < kStateDim; ++row) {
          const Scalar entry = factor(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(col));
          const Eigen::Index index = base + static_cast<Eigen::Index>(LowerTriangleIndex(row, col));
          if (row == col) {
            RESULT_ASSERT(entry > diagonal_offset,
                          "SoftMinQuadraticValueFunctionApprox::QuadraticParameters: a diagonal entry of the metric's "
                          "Cholesky factor does not exceed the diagonal offset eps, which is a floor this "
                          "parameterization cannot represent a value below.");
            parameters[index] = InverseSoftplus(entry - diagonal_offset);
          } else {
            parameters[index] = entry;
          }
        }
      }
      parameters.template segment<N>(base + kLowerTriangleSize) = center;
      parameters[base + kLowerTriangleSize + N] = component_offset;
    }
    return SuccessResult(parameters);
  }

  // Returns V~(state).
  auto operator()(const State& state) const noexcept -> Scalar {
    ValueBatch<1> values;
    Values<1>(state, values);
    return values[0];
  }

  // Returns the gradient of V~ in the state, at `state`.
  [[nodiscard]] auto Gradient(const State& state) const noexcept -> State {
    ValueBatch<1> values;
    State gradient;
    ValuesAndGradients<1>(state, values, gradient);
    return gradient;
  }

  // Returns the Hessian of V~ in the state, at `state`. Symmetric, and in general indefinite.
  [[nodiscard]] auto Hessian(const State& state) const noexcept -> StateMat {
    ValueBatch<1> values;
    State gradient;
    StateMat hessian;
    ValuesGradientsAndHessians<1>(state, values, gradient, hessian);
    return hessian;
  }

  // Returns the gradient of V~(state) with respect to the parameters, at the current parameters.
  [[nodiscard]] auto ParameterGradient(const State& state) const noexcept -> ParameterVector {
    ValueBatch<1> values;
    ParameterVector parameter_gradient;
    ValuesAndParameterGradients<1>(state, values, parameter_gradient);
    return parameter_gradient;
  }

  [[nodiscard]] auto Parameters() const noexcept -> const ParameterVector& { return parameters_; }

  // Replaces the parameters, unpacking them into the cached per-component quantities that every
  // evaluation below reads. Fails, leaving the model untouched, if any parameter is not finite.
  auto SetParameters(const ParameterVector& parameters) noexcept -> Result<> {
    RESULT_ASSERT(parameters.allFinite(),
                  "SoftMinQuadraticValueFunctionApprox::SetParameters: the parameters must be finite.");
    parameters_ = parameters;
    RefreshCaches();
    return SuccessResult();
  }

  // The `beta` of the soft minimum.
  [[nodiscard]] auto Sharpness() const noexcept -> Scalar { return sharpness_; }

  // The `eps` floor under every diagonal entry of every component's `L_p`.
  [[nodiscard]] auto DiagonalOffset() const noexcept -> Scalar { return diagonal_offset_; }

  // Writes V~ at each column of `states` into the corresponding column of `values_out`.
  template <int BatchSize>
  auto Values(const StateBatch<BatchSize>& states, ValueBatch<BatchSize>& values_out) const noexcept -> void {
    ComponentBatch<BatchSize> component_values;
    std::array<StateBatch<BatchSize>, kNumComponents> whitened;
    ComponentQuadratics<BatchSize>(states, component_values, whitened);

    ComponentBatch<BatchSize> weights;
    SoftMin<BatchSize>(component_values, values_out, weights);
  }

  // Writes V~ and its state gradient at each column of `states`.
  template <int BatchSize>
  auto ValuesAndGradients(const StateBatch<BatchSize>& states,
                          ValueBatch<BatchSize>& values_out,
                          GradientBatch<BatchSize>& gradients_out) const noexcept -> void {
    ComponentBatch<BatchSize> component_values;
    std::array<StateBatch<BatchSize>, kNumComponents> whitened;
    ComponentQuadratics<BatchSize>(states, component_values, whitened);

    ComponentBatch<BatchSize> weights;
    SoftMin<BatchSize>(component_values, values_out, weights);

    std::array<StateBatch<BatchSize>, kNumComponents> component_gradients;
    ComponentGradients<BatchSize>(whitened, component_gradients);
    Combine<BatchSize>(component_gradients, weights, gradients_out);
  }

  // Writes V~, its state gradient, and its state Hessian at each column of `states`. Sample `s`'s
  // Hessian lands in `hessians_out.block<N, N>(0, s * N)`.
  template <int BatchSize>
  auto ValuesGradientsAndHessians(const StateBatch<BatchSize>& states,
                                  ValueBatch<BatchSize>& values_out,
                                  GradientBatch<BatchSize>& gradients_out,
                                  HessianBatch<BatchSize>& hessians_out) const noexcept -> void {
    ComponentBatch<BatchSize> component_values;
    std::array<StateBatch<BatchSize>, kNumComponents> whitened;
    ComponentQuadratics<BatchSize>(states, component_values, whitened);

    ComponentBatch<BatchSize> weights;
    SoftMin<BatchSize>(component_values, values_out, weights);

    std::array<StateBatch<BatchSize>, kNumComponents> component_gradients;
    ComponentGradients<BatchSize>(whitened, component_gradients);
    Combine<BatchSize>(component_gradients, weights, gradients_out);

    constexpr auto kSampleCount = static_cast<std::size_t>(BatchSize);
    for (std::size_t sample = 0; sample < kSampleCount; ++sample) {
      const auto column = static_cast<Eigen::Index>(sample);

      // hess = 2 sum_p w_p A_p - beta ( sum_p w_p g_p g_p' - g g' ). Every term is exactly
      // symmetric -- the metrics were symmetrized when they were cached, and an outer product of
      // a vector with itself is symmetric coefficient by coefficient -- so the sum is too, and no
      // resymmetrization is needed here.
      StateMat curvature = StateMat::Zero();
      for (std::size_t component = 0; component < kNumComponents; ++component) {
        const Scalar weight = weights(static_cast<Eigen::Index>(component), column);
        const State component_gradient = component_gradients[component].col(column);
        curvature += (Scalar{2} * weight) * metrics_[component];
        curvature.noalias() -= (sharpness_ * weight) * (component_gradient * component_gradient.transpose());
      }
      const State gradient = gradients_out.col(column);
      curvature.noalias() += sharpness_ * (gradient * gradient.transpose());

      hessians_out.template block<N, N>(0, column * N) = curvature;
    }
  }

  // Writes V~ and its parameter gradient at each column of `states`.
  template <int BatchSize>
  auto ValuesAndParameterGradients(const StateBatch<BatchSize>& states,
                                   ValueBatch<BatchSize>& values_out,
                                   ParameterGradientBatch<BatchSize>& parameter_gradients_out) const noexcept -> void {
    ComponentBatch<BatchSize> component_values;
    std::array<StateBatch<BatchSize>, kNumComponents> whitened;
    ComponentQuadratics<BatchSize>(states, component_values, whitened);

    ComponentBatch<BatchSize> weights;
    SoftMin<BatchSize>(component_values, values_out, weights);

    std::array<StateBatch<BatchSize>, kNumComponents> component_gradients;
    ComponentGradients<BatchSize>(whitened, component_gradients);

    // Sample-major, so that each column of the output -- which is where one sample's whole
    // parameter gradient lives -- is written in one contiguous sweep.
    constexpr auto kSampleCount = static_cast<std::size_t>(BatchSize);
    for (std::size_t sample = 0; sample < kSampleCount; ++sample) {
      const auto column = static_cast<Eigen::Index>(sample);
      for (std::size_t component = 0; component < kNumComponents; ++component) {
        const auto base = static_cast<Eigen::Index>(component * static_cast<std::size_t>(kParametersPerComponent));
        const Scalar weight = weights(static_cast<Eigen::Index>(component), column);
        const State centered = states.col(column) - centers_[component];
        const State whitened_column = whitened[component].col(column);

        for (std::size_t col = 0; col < kStateDim; ++col) {
          for (std::size_t row = col; row < kStateDim; ++row) {
            // d q_p / d L[row, col] = 2 d[row] v[col], and on the diagonal the softplus
            // reparameterization contributes its own derivative on top.
            Scalar derivative =
                Scalar{2} * centered[static_cast<Eigen::Index>(row)] * whitened_column[static_cast<Eigen::Index>(col)];
            if (row == col) {
              derivative *= diagonal_slopes_[component][static_cast<Eigen::Index>(row)];
            }
            parameter_gradients_out(base + static_cast<Eigen::Index>(LowerTriangleIndex(row, col)), column) =
                weight * derivative;
          }
        }

        // d q_p / d b_p = -2 L_p v_p, which is the component gradient already in hand, negated.
        parameter_gradients_out.template block<N, 1>(base + kLowerTriangleSize, column) =
            -weight * component_gradients[component].col(column);

        // d q_p / d c_p = 1.
        parameter_gradients_out(base + kLowerTriangleSize + N, column) = weight;
      }
    }
  }

 private:
  // One row per component, one column per sample: the per-component quadratics and, later, the
  // softmax weights over them.
  template <int BatchSize>
  using ComponentBatch = Eigen::Matrix<Scalar, NumComponents, BatchSize>;

  SoftMinQuadraticValueFunctionApprox(Scalar sharpness, Scalar diagonal_offset) noexcept
      : sharpness_(sharpness), diagonal_offset_(diagonal_offset) {}

  // The position of `L(row, col)`, for `row >= col`, within a component's lower-triangle block.
  // The triangle is stored column by column -- (0,0), (1,0), ..., (N-1,0), (1,1), ... -- so that
  // the ordering mirrors Eigen's column-major storage of the matrix it fills.
  [[nodiscard]] static constexpr auto LowerTriangleIndex(std::size_t row, std::size_t col) noexcept -> std::size_t {
    // The first index of column `col` is sum_{k<col} (N - k), written without a subtraction that
    // would underflow an unsigned type at col == 0.
    return ((col * ((2 * kStateDim) - col + 1)) / 2) + (row - col);
  }

  // log(1 + e^v). Evaluated in two branches so that neither overflows: for large positive `value`
  // the exponential would, and for large negative it is the log's argument that loses every digit.
  [[nodiscard]] static auto Softplus(Scalar value) noexcept -> Scalar {
    return value > Scalar{0} ? value + std::log1p(std::exp(-value)) : std::log1p(std::exp(value));
  }

  // The derivative of Softplus, the logistic sigmoid. Same two branches, same reason.
  [[nodiscard]] static auto Logistic(Scalar value) noexcept -> Scalar {
    if (value >= Scalar{0}) {
      return Scalar{1} / (Scalar{1} + std::exp(-value));
    }
    const Scalar exponential = std::exp(value);
    return exponential / (Scalar{1} + exponential);
  }

  // The inverse of Softplus, for a strictly positive `value`. Algebraically log(e^v - 1), written
  // as `v + log(-expm1(-v))` so that it stays accurate both for small `value`, where expm1 keeps
  // the subtraction exact, and for large `value`, where the correction vanishes instead of the
  // exponential overflowing.
  [[nodiscard]] static auto InverseSoftplus(Scalar value) noexcept -> Scalar {
    return value + std::log(-std::expm1(-value));
  }

  // Unpacks `parameters_` into the per-component quantities every evaluation reads. Called once
  // per parameter assignment rather than once per evaluation, which is the whole reason
  // SetParameters exists as a function instead of the parameters being a public field.
  auto RefreshCaches() noexcept -> void {
    for (std::size_t component = 0; component < kNumComponents; ++component) {
      const auto base = static_cast<Eigen::Index>(component * static_cast<std::size_t>(kParametersPerComponent));

      StateMat& factor = factors_[component];
      factor.setZero();
      State& slope = diagonal_slopes_[component];
      for (std::size_t col = 0; col < kStateDim; ++col) {
        for (std::size_t row = col; row < kStateDim; ++row) {
          const Scalar raw = parameters_[base + static_cast<Eigen::Index>(LowerTriangleIndex(row, col))];
          const auto eigen_row = static_cast<Eigen::Index>(row);
          const auto eigen_col = static_cast<Eigen::Index>(col);
          if (row == col) {
            factor(eigen_row, eigen_col) = Softplus(raw) + diagonal_offset_;
            slope[eigen_row] = Logistic(raw);
          } else {
            factor(eigen_row, eigen_col) = raw;
          }
        }
      }

      // Symmetrized once here rather than after every Hessian: `L L'` is symmetric in exact
      // arithmetic and drifts out of it only by rounding, and doing it at this one point is what
      // lets Hessian() promise a symmetric result without paying for it per sample.
      const StateMat metric = factor * factor.transpose();
      metrics_[component] = Scalar{0.5} * (metric + metric.transpose());

      centers_[component] = parameters_.template segment<N>(base + kLowerTriangleSize);
      offsets_[component] = parameters_[base + kLowerTriangleSize + N];
    }
  }

  // Fills `component_values_out` with q_p(x_s) and `whitened_out[p]` with v_p(x_s) = L_p'(x_s - b_p),
  // the intermediate every derivative below is rebuilt from.
  template <int BatchSize>
  auto ComponentQuadratics(const StateBatch<BatchSize>& states,
                           ComponentBatch<BatchSize>& component_values_out,
                           std::array<StateBatch<BatchSize>, kNumComponents>& whitened_out) const noexcept -> void {
    for (std::size_t component = 0; component < kNumComponents; ++component) {
      // One N x N by N x BatchSize product per component: the whole batch's whitened offsets in a
      // single GEMM, which is where the batched interface earns its keep over a per-sample loop.
      whitened_out[component].noalias() = factors_[component].transpose() * (states.colwise() - centers_[component]);
      component_values_out.row(static_cast<Eigen::Index>(component)).array() =
          whitened_out[component].colwise().squaredNorm().array() + offsets_[component];
    }
  }

  // Turns the per-component quadratics into the soft minimum and the softmax weights over them.
  template <int BatchSize>
  auto SoftMin(const ComponentBatch<BatchSize>& component_values,
               ValueBatch<BatchSize>& values_out,
               ComponentBatch<BatchSize>& weights_out) const noexcept -> void {
    // Shift each sample by its smallest component before exponentiating. Without the shift a
    // component a few hundred above the smallest sends exp(-beta q) to zero and the sum with it;
    // with it the largest exponential is exactly one and the sum is at least one.
    const ValueBatch<BatchSize> smallest = component_values.colwise().minCoeff();
    weights_out = (-sharpness_ * (component_values.rowwise() - smallest).array()).exp();

    const ValueBatch<BatchSize> totals = weights_out.colwise().sum();
    values_out.array() = smallest.array() - (totals.array().log() / sharpness_);
    weights_out.array() /= totals.template replicate<NumComponents, 1>().array();
  }

  // Fills `component_gradients_out[p]` with grad q_p = 2 L_p v_p for every sample.
  template <int BatchSize>
  auto ComponentGradients(const std::array<StateBatch<BatchSize>, kNumComponents>& whitened,
                          std::array<StateBatch<BatchSize>, kNumComponents>& component_gradients_out) const noexcept
      -> void {
    for (std::size_t component = 0; component < kNumComponents; ++component) {
      component_gradients_out[component].noalias() = Scalar{2} * (factors_[component] * whitened[component]);
    }
  }

  // Forms sum_p w_p g_p, the softmax-weighted combination that turns per-component gradients into
  // the gradient of the soft minimum.
  template <int BatchSize>
  static auto Combine(const std::array<StateBatch<BatchSize>, kNumComponents>& component_gradients,
                      const ComponentBatch<BatchSize>& weights,
                      GradientBatch<BatchSize>& gradients_out) noexcept -> void {
    gradients_out.setZero();
    for (std::size_t component = 0; component < kNumComponents; ++component) {
      // Scaling each column by that sample's weight, expressed as a diagonal product so Eigen
      // keeps it a single pass rather than materializing the broadcast.
      gradients_out.noalias() +=
          component_gradients[component] * weights.row(static_cast<Eigen::Index>(component)).asDiagonal();
    }
  }

  Scalar sharpness_;
  Scalar diagonal_offset_;

  ParameterVector parameters_ = ParameterVector::Zero();

  // Unpacked from `parameters_` by RefreshCaches().
  std::array<StateMat, kNumComponents> factors_{};       // L_p
  std::array<StateMat, kNumComponents> metrics_{};       // A_p = L_p L_p'
  std::array<State, kNumComponents> centers_{};          // b_p
  std::array<State, kNumComponents> diagonal_slopes_{};  // sigmoid(theta) on the diagonal of L_p
  std::array<Scalar, kNumComponents> offsets_{};         // c_p
};

}  // namespace fbsde_traj_opt

#endif  // FBSDE_TRAJ_OPT_SOFT_MIN_QUADRATIC_VALUE_FUNCTION_APPROX_HPP_
