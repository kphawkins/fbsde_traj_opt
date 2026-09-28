// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#ifndef FBSDE_TRAJ_OPT_POLYNOMIAL_VALUE_FUNCTION_APPROX_HPP_
#define FBSDE_TRAJ_OPT_POLYNOMIAL_VALUE_FUNCTION_APPROX_HPP_

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include "fbsde_traj_opt/utils/result.hpp"
#include "fbsde_traj_opt/value_function_approx_concepts.hpp"

namespace fbsde_traj_opt {

// A BatchedParameterizedValueFunctionApprox (see value_function_approx_concepts.hpp) that is a
// linear combination of every monomial of total degree at most `Degree` in a normalized state,
//
//   V~(x) = sum_j theta_j phi_j(z),    phi_j(z) = prod_k z_k^{e_jk},    z = (x - center) ./ scale,
//
// the model class of the least-squares Monte Carlo (LSMC) regression in Hawkins (2021) -- see
// Appendix B.6, which states that most of that work's results use exactly this, with degree two.
//
// Why a linear-in-parameters model, next to the soft minimum of quadratics. The fit becomes a
// weighted linear least-squares problem with a closed-form answer (FitWeightedLeastSquares), which
// is what the thesis's backward pass solves at every stage (5.12): no learning rate, no epochs,
// no stochastic convergence to wait on -- one small normal-equation solve per stage. And at degree
// two the family contains every quadratic, so the Taylor estimators and the Taylor Q-function
// (Section 4.4) are exact on it, exactly as on the soft-min model with one component.
//
// Why a *fixed* normalization. The center and scale are chosen once, for a region of interest,
// and never refitted -- the thesis normalizes "to the interval [-1, 1]^n based on a parameterized
// region of interest" (Section 5.4.4). That keeps the regression well conditioned, since the
// monomials of a coordinate of order one are themselves of order one; and, because every model
// built with the same normalization has the same basis, two fitted models can be blended simply by
// blending their coefficient vectors. An iterative method that damps its update between
// iterations relies on that.
//
// `N` is the compile-time state dimension and `Degree` the maximum total degree. The number of
// basis functions is `binomial(N + Degree, Degree)`: 15 for a quadratic in four dimensions, 35
// for a cubic, 70 for a quartic.
template <int N, int Degree, typename Scalar = double>
class PolynomialValueFunctionApprox {
  static_assert(N >= 1, "A value function needs at least a one-dimensional state.");
  static_assert(Degree >= 0, "A polynomial's degree cannot be negative.");

 public:
  using State = Eigen::Matrix<Scalar, N, 1>;
  using StateMat = Eigen::Matrix<Scalar, N, N>;

  static constexpr std::size_t kStateDim = static_cast<std::size_t>(N);
  static constexpr auto kDegree = static_cast<std::size_t>(Degree);

  // binomial(N + Degree, Degree), computed without overflow for any size this could ever be used
  // at: the running product of i+1 consecutive integers is always divisible by (i+1)!.
  static constexpr int kNumParameters = [] {
    std::int64_t count = 1;
    for (int term = 1; term <= Degree; ++term) {
      count = (count * (N + term)) / term;
    }
    return static_cast<int>(count);
  }();

  using ParameterVector = Eigen::Matrix<Scalar, kNumParameters, 1>;
  using ParameterMat = Eigen::Matrix<Scalar, kNumParameters, kNumParameters>;

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

  // Builds the model from its coefficients and its normalization.
  //
  // Fails if any coefficient or any entry of `center` is not finite, or if any entry of `scale`
  // is not finite and strictly positive.
  static auto Make(const ParameterVector& parameters, const State& center, const State& scale) noexcept
      -> Result<PolynomialValueFunctionApprox> {
    RESULT_ASSERT(parameters.allFinite(), "PolynomialValueFunctionApprox::Make: the coefficients must be finite.");
    RESULT_ASSERT(center.allFinite(), "PolynomialValueFunctionApprox::Make: the center must be finite.");
    RESULT_ASSERT(scale.allFinite() && (scale.array() > Scalar{0}).all(),
                  "PolynomialValueFunctionApprox::Make: every scale must be finite and positive.");
    return SuccessResult(PolynomialValueFunctionApprox(parameters, center, scale));
  }

  // Returns V~(state).
  auto operator()(const State& state) const noexcept -> Scalar { return Features(state).dot(parameters_); }

  // Returns the gradient of V~ in the state.
  [[nodiscard]] auto Gradient(const State& state) const noexcept -> State {
    State gradient;
    StateMat hessian;
    auto value = Scalar{0};
    Derivatives</*WithHessian=*/false>(state, value, gradient, hessian);
    return gradient;
  }

  // Returns the Hessian of V~ in the state. Symmetric by construction: entry (k, l) and (l, k) are
  // computed by the same expression.
  [[nodiscard]] auto Hessian(const State& state) const noexcept -> StateMat {
    State gradient;
    StateMat hessian;
    auto value = Scalar{0};
    Derivatives</*WithHessian=*/true>(state, value, gradient, hessian);
    return hessian;
  }

  // Returns the value, gradient, and Hessian together, sharing the monomial powers they are all
  // built from.
  auto ValueGradientAndHessian(const State& state,
                               Scalar& value_out,
                               State& gradient_out,
                               StateMat& hessian_out) const noexcept -> void {
    Derivatives</*WithHessian=*/true>(state, value_out, gradient_out, hessian_out);
  }

  // Returns the basis functions at `state`, which is also the gradient of V~ with respect to the
  // coefficients.
  [[nodiscard]] auto ParameterGradient(const State& state) const noexcept -> ParameterVector { return Features(state); }

  [[nodiscard]] auto Parameters() const noexcept -> const ParameterVector& { return parameters_; }

  // Replaces the coefficients. Fails, leaving the model untouched, if any is not finite.
  auto SetParameters(const ParameterVector& parameters) noexcept -> Result<> {
    RESULT_ASSERT(parameters.allFinite(),
                  "PolynomialValueFunctionApprox::SetParameters: the coefficients must be finite.");
    parameters_ = parameters;
    return SuccessResult();
  }

  [[nodiscard]] auto Center() const noexcept -> const State& { return center_; }

  [[nodiscard]] auto Scale() const noexcept -> const State& { return scale_; }

  // Returns the coefficients that make this basis reproduce `state' metric state + offset`
  // exactly -- a quadratic centered on the origin of the *unnormalized* state, such as a terminal
  // cost. Requires `Degree >= 2`.
  //
  // Found by expanding `x = center + scale .* z` symbolically rather than by fitting, so that it is
  // exact to rounding rather than to the conditioning of a regression.
  [[nodiscard]] auto QuadraticParameters(const StateMat& metric, Scalar offset) const noexcept -> ParameterVector
    requires(Degree >= 2)
  {
    // x' A x = (c + S z)' A (c + S z) = c' A c + 2 c' A S z + z' S A S z.
    const StateMat symmetric = Scalar{0.5} * (metric + metric.transpose());
    const StateMat scaled = scale_.asDiagonal() * symmetric * scale_.asDiagonal();
    const State linear = Scalar{2} * (scale_.asDiagonal() * (symmetric * center_));

    ParameterVector parameters = ParameterVector::Zero();
    for (std::size_t term = 0; term < kExponents.size(); ++term) {
      const std::array<int, N>& exponents = kExponents[term];
      int degree = 0;
      std::array<int, 2> axes{};
      std::size_t found = 0;
      for (std::size_t axis = 0; axis < kStateDim; ++axis) {
        degree += exponents[axis];
        for (int power = 0; power < exponents[axis] && found < 2; ++power) {
          axes[found++] = static_cast<int>(axis);
        }
      }
      const auto index = static_cast<Eigen::Index>(term);
      if (degree == 0) {
        parameters[index] = center_.dot(symmetric * center_) + offset;
      } else if (degree == 1) {
        parameters[index] = linear[axes[0]];
      } else if (degree == 2) {
        // z_k^2 carries A_kk; z_k z_l (k != l) carries both A_kl and A_lk.
        parameters[index] = axes[0] == axes[1] ? scaled(axes[0], axes[0]) : Scalar{2} * scaled(axes[0], axes[1]);
      }
    }
    return parameters;
  }

  // Replaces the coefficients with the weighted, ridge-regularized least-squares fit to `targets`
  // over `states`,
  //
  //   theta = argmin sum_s w_s (phi(x_s)' theta - y_s)^2 + ridge * S * || theta ||^2,
  //
  // where `S` is the sample count, so that the ridge weight means the same thing at any sample
  // count. This is the regression of the thesis's backward pass (5.12), with the sample weights
  // being its local-entropy weights -- or all ones.
  //
  // The ridge term is what keeps the solve defined when the samples do not excite every basis
  // function -- a batch that has collapsed onto a lower-dimensional set, say -- and, more
  // usefully, what keeps a high-degree basis from spending its freedom fitting the samples'
  // noise.
  //
  // Fails, leaving the model untouched, if the weights are negative or not finite, if the ridge
  // weight is negative or not finite, or if the normal equations cannot be solved.
  template <int NumSamples>
  auto FitWeightedLeastSquares(const StateBatch<NumSamples>& states,
                               const ValueBatch<NumSamples>& targets,
                               const ValueBatch<NumSamples>& weights,
                               Scalar ridge) noexcept -> Result<> {
    RESULT_ASSERT(weights.allFinite() && (weights.array() >= Scalar{0}).all(),
                  "PolynomialValueFunctionApprox::FitWeightedLeastSquares: every weight must be finite and "
                  "non-negative.");
    RESULT_ASSERT(std::isfinite(ridge) && ridge >= Scalar{0},
                  "PolynomialValueFunctionApprox::FitWeightedLeastSquares: the ridge weight must be finite and "
                  "non-negative.");
    RESULT_ASSERT(targets.allFinite(),
                  "PolynomialValueFunctionApprox::FitWeightedLeastSquares: every target must be finite.");

    // Accumulated one sample at a time rather than through a P x S design matrix, which for the
    // sample counts a backward pass uses would be megabytes on the stack.
    ParameterMat normal_matrix = ParameterMat::Zero();
    ParameterVector normal_vector = ParameterVector::Zero();
    for (std::size_t sample = 0; sample < kSampleCount<NumSamples>; ++sample) {
      const auto column = static_cast<Eigen::Index>(sample);
      const ParameterVector features = Features(states.col(column));
      normal_matrix.template selfadjointView<Eigen::Lower>().rankUpdate(features, weights[column]);
      normal_vector += (weights[column] * targets[column]) * features;
    }
    normal_matrix.diagonal().array() += ridge * static_cast<Scalar>(NumSamples);

    const Eigen::LDLT<ParameterMat> factorization(normal_matrix.template selfadjointView<Eigen::Lower>());
    RESULT_ASSERT(factorization.info() == Eigen::Success,
                  "PolynomialValueFunctionApprox::FitWeightedLeastSquares: the normal equations are singular; add "
                  "a ridge weight or more varied samples.");
    const ParameterVector solution = factorization.solve(normal_vector);
    RESULT_ASSERT(solution.allFinite(),
                  "PolynomialValueFunctionApprox::FitWeightedLeastSquares: the fit produced non-finite "
                  "coefficients.");
    parameters_ = solution;
    return SuccessResult();
  }

  template <int BatchSize>
  auto Values(const StateBatch<BatchSize>& states, ValueBatch<BatchSize>& values_out) const noexcept -> void {
    for (std::size_t sample = 0; sample < kSampleCount<BatchSize>; ++sample) {
      const auto column = static_cast<Eigen::Index>(sample);
      values_out[column] = (*this)(states.col(column));
    }
  }

  template <int BatchSize>
  auto ValuesAndGradients(const StateBatch<BatchSize>& states,
                          ValueBatch<BatchSize>& values_out,
                          GradientBatch<BatchSize>& gradients_out) const noexcept -> void {
    for (std::size_t sample = 0; sample < kSampleCount<BatchSize>; ++sample) {
      const auto column = static_cast<Eigen::Index>(sample);
      State gradient;
      StateMat hessian;
      Derivatives</*WithHessian=*/false>(states.col(column), values_out[column], gradient, hessian);
      gradients_out.col(column) = gradient;
    }
  }

  template <int BatchSize>
  auto ValuesGradientsAndHessians(const StateBatch<BatchSize>& states,
                                  ValueBatch<BatchSize>& values_out,
                                  GradientBatch<BatchSize>& gradients_out,
                                  HessianBatch<BatchSize>& hessians_out) const noexcept -> void {
    for (std::size_t sample = 0; sample < kSampleCount<BatchSize>; ++sample) {
      const auto column = static_cast<Eigen::Index>(sample);
      State gradient;
      StateMat hessian;
      Derivatives</*WithHessian=*/true>(states.col(column), values_out[column], gradient, hessian);
      gradients_out.col(column) = gradient;
      hessians_out.template block<N, N>(0, column * N) = hessian;
    }
  }

  template <int BatchSize>
  auto ValuesAndParameterGradients(const StateBatch<BatchSize>& states,
                                   ValueBatch<BatchSize>& values_out,
                                   ParameterGradientBatch<BatchSize>& parameter_gradients_out) const noexcept -> void {
    for (std::size_t sample = 0; sample < kSampleCount<BatchSize>; ++sample) {
      const auto column = static_cast<Eigen::Index>(sample);
      const ParameterVector features = Features(states.col(column));
      parameter_gradients_out.col(column) = features;
      values_out[column] = features.dot(parameters_);
    }
  }

 private:
  // The exponent vectors of every basis function, in order of increasing total degree: the
  // constant first, then the N linear terms, then the quadratics, and so on.
  using ExponentTable = std::array<std::array<int, N>, static_cast<std::size_t>(kNumParameters)>;

  static constexpr ExponentTable kExponents = [] {
    ExponentTable table{};
    std::size_t next = 0;
    // Enumerates exponent vectors of each total degree by treating them as odometer readings: the
    // last axis counts fastest, and a reading is kept when its digits sum to the degree sought.
    for (int degree = 0; degree <= Degree; ++degree) {
      std::array<int, N> exponents{};
      while (true) {
        int total = 0;
        for (const int exponent : exponents) {
          total += exponent;
        }
        if (total == degree) {
          table[next++] = exponents;
        }
        int axis = N - 1;
        while (axis >= 0 && exponents[static_cast<std::size_t>(axis)] == degree) {
          exponents[static_cast<std::size_t>(axis)] = 0;
          --axis;
        }
        if (axis < 0) {
          break;
        }
        ++exponents[static_cast<std::size_t>(axis)];
      }
    }
    return table;
  }();

  // A batch's column count as an index type.
  template <int Count>
  static constexpr auto kSampleCount = static_cast<std::size_t>(Count);

  // Powers z_k^p for every axis and every p in 0..Degree.
  using PowerTable = std::array<std::array<Scalar, static_cast<std::size_t>(Degree) + 1>, kStateDim>;

  PolynomialValueFunctionApprox(ParameterVector parameters, State center, const State& scale) noexcept
      : parameters_(std::move(parameters)),
        center_(std::move(center)),
        scale_(scale),
        inverse_scale_(scale.cwiseInverse()) {}

  [[nodiscard]] auto Powers(const State& state) const noexcept -> PowerTable {
    const State normalized = (state - center_).cwiseProduct(inverse_scale_);
    PowerTable powers{};
    for (std::size_t axis = 0; axis < kStateDim; ++axis) {
      powers[axis][0] = Scalar{1};
      for (std::size_t power = 1; power <= kDegree; ++power) {
        powers[axis][power] = powers[axis][power - 1] * normalized[static_cast<Eigen::Index>(axis)];
      }
    }
    return powers;
  }

  [[nodiscard]] auto Features(const State& state) const noexcept -> ParameterVector {
    const PowerTable powers = Powers(state);
    ParameterVector features;
    for (std::size_t term = 0; term < kExponents.size(); ++term) {
      auto product = Scalar{1};
      for (std::size_t axis = 0; axis < kStateDim; ++axis) {
        product *= powers[axis][static_cast<std::size_t>(kExponents[term][axis])];
      }
      features[static_cast<Eigen::Index>(term)] = product;
    }
    return features;
  }

  // The value, the gradient, and (when asked) the Hessian, in normalized coordinates first and
  // then carried back to the state by the chain rule: d/dx_k = (1 / scale_k) d/dz_k.
  //
  // For a monomial prod_k z_k^{e_k}, the derivative along axis k is e_k z_k^{e_k - 1} times the
  // other factors, and the second derivative along k and l is the same rule applied twice. Each
  // is computed as a product over axes with the differentiated axes' exponents lowered, which is
  // exact and needs no division by a coordinate that might be zero.
  template <bool WithHessian>
  auto Derivatives(const State& state, Scalar& value_out, State& gradient_out, StateMat& hessian_out) const noexcept
      -> void {
    const PowerTable powers = Powers(state);
    value_out = Scalar{0};
    gradient_out.setZero();
    if constexpr (WithHessian) {
      hessian_out.setZero();
    }

    const auto monomial = [&powers](const std::array<int, N>& exponents) noexcept -> Scalar {
      auto product = Scalar{1};
      for (std::size_t axis = 0; axis < kStateDim; ++axis) {
        product *= powers[axis][static_cast<std::size_t>(exponents[axis])];
      }
      return product;
    };

    for (std::size_t term = 0; term < kExponents.size(); ++term) {
      const Scalar coefficient = parameters_[static_cast<Eigen::Index>(term)];
      if (coefficient == Scalar{0}) {
        continue;
      }
      const std::array<int, N>& exponents = kExponents[term];
      value_out += coefficient * monomial(exponents);

      for (std::size_t first = 0; first < kStateDim; ++first) {
        if (exponents[first] == 0) {
          continue;
        }
        std::array<int, N> lowered = exponents;
        --lowered[first];
        gradient_out[static_cast<Eigen::Index>(first)] +=
            coefficient * static_cast<Scalar>(exponents[first]) * monomial(lowered);

        if constexpr (WithHessian) {
          for (std::size_t second = 0; second <= first; ++second) {
            if (lowered[second] == 0) {
              continue;
            }
            std::array<int, N> twice_lowered = lowered;
            --twice_lowered[second];
            const Scalar entry = coefficient * static_cast<Scalar>(exponents[first]) *
                                 static_cast<Scalar>(lowered[second]) * monomial(twice_lowered);
            hessian_out(static_cast<Eigen::Index>(first), static_cast<Eigen::Index>(second)) += entry;
          }
        }
      }
    }

    gradient_out = gradient_out.cwiseProduct(inverse_scale_);
    if constexpr (WithHessian) {
      // Only the lower triangle was accumulated; mirror it, then apply the chain rule on both
      // sides.
      hessian_out.template triangularView<Eigen::StrictlyUpper>() = hessian_out.transpose();
      hessian_out = inverse_scale_.asDiagonal() * hessian_out * inverse_scale_.asDiagonal();
    }
  }

  ParameterVector parameters_;
  State center_;
  State scale_;
  State inverse_scale_;
};

}  // namespace fbsde_traj_opt

#endif  // FBSDE_TRAJ_OPT_POLYNOMIAL_VALUE_FUNCTION_APPROX_HPP_
