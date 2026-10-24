// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#ifndef FBSDE_TRAJ_OPT_VALUE_FUNCTION_LINEAR_BASIS_VALUE_FUNCTION_APPROX_HPP_
#define FBSDE_TRAJ_OPT_VALUE_FUNCTION_LINEAR_BASIS_VALUE_FUNCTION_APPROX_HPP_

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include "fbsde_traj_opt/utils/result.hpp"
#include "fbsde_traj_opt/value_function/value_function_approx_concepts.hpp"
#include "fbsde_traj_opt/value_function/value_function_slope_targets.hpp"

namespace fbsde_traj_opt {

// A value function that is a linear combination of fixed basis functions of a normalized state,
//
//   V~(x) = sum_j theta_j phi_j(z),    z = (x - center) ./ scale,
//
// the model class of least-squares Monte Carlo with any choice of basis. PolynomialValueFunction-
// Approx is the monomial instance; this class takes the basis as a parameter so that others --
// orthogonal polynomials, radial basis functions, splines -- can be compared on the same footing:
// the same normalization, the same closed-form weighted least-squares fit with optional slope rows,
// and the same coefficient blending that DtFbsdeIterativeSolver's line search relies on.
//
// A basis `BasisT` provides, in normalized coordinates,
//
//   static constexpr int kNumFeatures;
//   template <bool WithGradient, bool WithHessian>
//   void Jet(const Z& z, Features& phi, Gradients& dphi, Hessians& d2phi) const;
//
// with `Features` a K-vector, `Gradients` an N x K matrix whose column j is grad phi_j, and
// `Hessians` an (N*N) x K matrix whose column j is the column-major Hessian of phi_j; the flags say
// which of the derivatives the caller wants filled in.
//
// The ridge. The fit solves
//
//   theta = argmin sum_s w_s [ (phi(x_s)' theta - y_s)^2 + slope_weight sum_k (slope rows)^2 ]
//           + ridge * S * || theta - anchor ||^2,
//
// where the anchor is zero (RidgeAnchor::kZero, the classical ridge, which is what the polynomial
// model does) or the model's own coefficients when the fit is called (RidgeAnchor::kCurrent).
// DtFbsdeIterativeSolver fits a copy of the current table, so the second shrinks each update toward
// the current fit rather than toward zero: a basis function the batch does not excite keeps its
// coefficient instead of losing it. That matters for a local basis -- a spline or a radial basis
// function far from every sample -- fitted from the eight samples a half-batch of sixteen gives.
//
// The roughness penalty. With `roughness` positive the fit also adds
//
//   roughness * S * mean over the box [-1, 1]^N of || Hessian_z V~(z) ||_F^2,
//
// the thin-plate energy of the function in normalized coordinates. The ridge's preference depends on
// the basis -- under monomials it favors low degree, because a high power is small inside the box and
// needs a large coefficient, and under an orthonormal basis it favors nothing -- while this one is a
// property of the function alone: it charges curvature, wherever it is and whatever the basis, and
// leaves every affine function free. When the samples underdetermine the coefficients it is what
// decides which of the fits that match them is chosen.
enum class RidgeAnchor : std::uint8_t { kZero, kCurrent };

template <int N, typename BasisT, typename Scalar = double>
class LinearBasisValueFunctionApprox {
 public:
  using State = Eigen::Matrix<Scalar, N, 1>;
  using StateMat = Eigen::Matrix<Scalar, N, N>;
  using Basis = BasisT;

  static constexpr int kNumParameters = BasisT::kNumFeatures;
  static constexpr auto kStateDim = static_cast<std::size_t>(N);

  using ParameterVector = Eigen::Matrix<Scalar, kNumParameters, 1>;
  using Gradients = Eigen::Matrix<Scalar, N, kNumParameters>;
  using Hessians = Eigen::Matrix<Scalar, N * N, kNumParameters>;
  // The normal equations live on the heap: at a few hundred features they would be megabytes.
  using NormalMat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

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

  // Fails unless the coefficients and the center are finite and every scale finite and positive.
  static auto Make(const ParameterVector& parameters,
                   const State& center,
                   const State& scale,
                   BasisT basis,
                   RidgeAnchor anchor = RidgeAnchor::kZero,
                   Scalar roughness = Scalar{0}) noexcept -> Result<LinearBasisValueFunctionApprox> {
    RESULT_ASSERT(parameters.allFinite(), "LinearBasisValueFunctionApprox::Make: the coefficients must be finite.");
    RESULT_ASSERT(center.allFinite(), "LinearBasisValueFunctionApprox::Make: the center must be finite.");
    RESULT_ASSERT(scale.allFinite() && (scale.array() > Scalar{0}).all(),
                  "LinearBasisValueFunctionApprox::Make: every scale must be finite and positive.");
    RESULT_ASSERT(std::isfinite(roughness) && roughness >= Scalar{0},
                  "LinearBasisValueFunctionApprox::Make: the roughness weight must be finite and non-negative.");
    LinearBasisValueFunctionApprox model(parameters, center, scale, std::move(basis), anchor);
    if (roughness > Scalar{0}) {
      model.roughness_ = roughness;
      model.roughness_matrix_ = std::make_shared<const NormalMat>(RoughnessMatrix(model.basis_));
    }
    return SuccessResult(std::move(model));
  }

  auto operator()(const State& state) const noexcept -> Scalar { return Features(state).dot(parameters_); }

  [[nodiscard]] auto Gradient(const State& state) const noexcept -> State {
    Scalar value;
    State gradient;
    StateMat hessian;
    Derivatives<false>(state, value, gradient, hessian);
    return gradient;
  }

  [[nodiscard]] auto Hessian(const State& state) const noexcept -> StateMat {
    Scalar value;
    State gradient;
    StateMat hessian;
    Derivatives<true>(state, value, gradient, hessian);
    return hessian;
  }

  [[nodiscard]] auto ParameterGradient(const State& state) const noexcept -> ParameterVector { return Features(state); }
  [[nodiscard]] auto Parameters() const noexcept -> const ParameterVector& { return parameters_; }

  auto SetParameters(const ParameterVector& parameters) noexcept -> Result<> {
    RESULT_ASSERT(parameters.allFinite(),
                  "LinearBasisValueFunctionApprox::SetParameters: the coefficients must be finite.");
    parameters_ = parameters;
    return SuccessResult();
  }

  [[nodiscard]] auto Center() const noexcept -> const State& { return center_; }
  [[nodiscard]] auto Scale() const noexcept -> const State& { return scale_; }
  [[nodiscard]] auto Anchor() const noexcept -> RidgeAnchor { return anchor_; }

  // The basis functions and their slopes along `direction` at `state`, for a caller assembling its
  // own regression.
  [[nodiscard]] auto Features(const State& state) const noexcept -> ParameterVector {
    ParameterVector phi;
    Gradients unused_gradients;
    Hessians unused_hessians;
    basis_.template Jet<false, false>(Normalize(state), phi, unused_gradients, unused_hessians);
    return phi;
  }

  [[nodiscard]] auto FeatureSlopes(const State& state, const State& direction) const noexcept -> ParameterVector {
    ParameterVector phi;
    Gradients gradients;
    Hessians unused_hessians;
    basis_.template Jet<true, false>(Normalize(state), phi, gradients, unused_hessians);
    const State along = direction.cwiseProduct(inverse_scale_);
    return gradients.transpose() * along;
  }

  // The normal equations of a weighted least-squares fit, held apart from any one solve so that a
  // caller can carry them across fits: `matrix` (lower triangle) and `vector` are the sums
  // sum_s w_s phi phi' and sum_s w_s phi y, slope rows included, and `samples` the number of samples
  // they hold, which the ridge and roughness weights are scaled by. Decay() multiplies all three by
  // a factor, which is how a caller forgets old batches geometrically -- recursive least squares
  // with a forgetting factor, in information form.
  struct NormalEquations {
    NormalMat matrix = NormalMat::Zero(kNumParameters, kNumParameters);
    ParameterVector vector = ParameterVector::Zero();
    Scalar samples = Scalar{0};

    auto Decay(Scalar factor) noexcept -> void {
      matrix *= factor;
      vector *= factor;
      samples *= factor;
    }
  };

  // Adds a batch's rows to `equations`: the value rows, and with slopes the slope rows weighted by
  // `slope_weight`. Fails, leaving `equations` untouched, on a negative or non-finite weight or a
  // non-finite target.
  template <int NumSamples>
  auto AccumulateNormalEquations(const StateBatch<NumSamples>& states,
                                 const ValueBatch<NumSamples>& targets,
                                 const ValueBatch<NumSamples>& weights,
                                 NormalEquations& equations) const noexcept -> Result<> {
    RESULT_RETURN_IF_ERROR(CheckFitInputs(targets, weights, Scalar{0}, Scalar{0}));
    for (std::size_t sample = 0; sample < static_cast<std::size_t>(NumSamples); ++sample) {
      const auto column = static_cast<Eigen::Index>(sample);
      const ParameterVector phi = Features(states.col(column));
      equations.matrix.template selfadjointView<Eigen::Lower>().rankUpdate(phi, weights[column]);
      equations.vector += (weights[column] * targets[column]) * phi;
    }
    equations.samples += static_cast<Scalar>(NumSamples);
    return SuccessResult();
  }

  template <int NumSamples, int D>
  auto AccumulateNormalEquations(const StateBatch<NumSamples>& states,
                                 const ValueBatch<NumSamples>& targets,
                                 const ValueBatch<NumSamples>& weights,
                                 const ValueFunctionSlopeTargets<State, NumSamples, D>& slopes,
                                 Scalar slope_weight,
                                 NormalEquations& equations) const noexcept -> Result<> {
    RESULT_RETURN_IF_ERROR(CheckFitInputs(targets, weights, Scalar{0}, slope_weight));
    for (std::size_t direction = 0; direction < static_cast<std::size_t>(D); ++direction) {
      RESULT_ASSERT(slopes.slopes[direction].allFinite() && slopes.directions[direction].allFinite(),
                    "LinearBasisValueFunctionApprox::FitWeightedLeastSquares: every slope target and direction must "
                    "be finite.");
    }
    for (std::size_t sample = 0; sample < static_cast<std::size_t>(NumSamples); ++sample) {
      const auto column = static_cast<Eigen::Index>(sample);
      ParameterVector phi;
      Gradients gradients;
      Hessians unused_hessians;
      basis_.template Jet<true, false>(Normalize(states.col(column)), phi, gradients, unused_hessians);
      equations.matrix.template selfadjointView<Eigen::Lower>().rankUpdate(phi, weights[column]);
      equations.vector += (weights[column] * targets[column]) * phi;
      for (std::size_t direction = 0; direction < static_cast<std::size_t>(D); ++direction) {
        const State along = State(slopes.directions[direction].col(column)).cwiseProduct(inverse_scale_);
        const ParameterVector slope_features = gradients.transpose() * along;
        const Scalar weight = slope_weight * weights[column];
        equations.matrix.template selfadjointView<Eigen::Lower>().rankUpdate(slope_features, weight);
        equations.vector += (weight * slopes.slopes[direction][column]) * slope_features;
      }
    }
    equations.samples += static_cast<Scalar>(NumSamples);
    return SuccessResult();
  }

  // Replaces the coefficients with the solution of `equations` under a per-sample ridge weight
  // `ridge` (and the model's roughness penalty, if it has one), both scaled by the equations' sample
  // count. Fails, leaving the model untouched, if the ridge weight is negative or the solve fails.
  auto FitNormalEquations(const NormalEquations& equations, Scalar ridge) noexcept -> Result<> {
    RESULT_ASSERT(std::isfinite(ridge) && ridge >= Scalar{0},
                  "LinearBasisValueFunctionApprox::FitNormalEquations: the ridge weight must be finite and "
                  "non-negative.");
    ridge_samples_ = equations.samples;
    return Solve(equations.matrix, equations.vector, ridge * equations.samples);
  }

  template <int NumSamples>
  auto FitWeightedLeastSquares(const StateBatch<NumSamples>& states,
                               const ValueBatch<NumSamples>& targets,
                               const ValueBatch<NumSamples>& weights,
                               Scalar ridge) noexcept -> Result<> {
    RESULT_RETURN_IF_ERROR(CheckFitInputs(targets, weights, ridge, Scalar{0}));
    NormalEquations equations;
    RESULT_RETURN_IF_ERROR(AccumulateNormalEquations<NumSamples>(states, targets, weights, equations));
    return FitNormalEquations(equations, ridge);
  }

  template <int NumSamples, int D>
  auto FitWeightedLeastSquares(const StateBatch<NumSamples>& states,
                               const ValueBatch<NumSamples>& targets,
                               const ValueBatch<NumSamples>& weights,
                               const ValueFunctionSlopeTargets<State, NumSamples, D>& slopes,
                               Scalar slope_weight,
                               Scalar ridge) noexcept -> Result<> {
    RESULT_RETURN_IF_ERROR(CheckFitInputs(targets, weights, ridge, slope_weight));
    NormalEquations equations;
    RESULT_RETURN_IF_ERROR(
        (AccumulateNormalEquations<NumSamples, D>(states, targets, weights, slopes, slope_weight, equations)));
    return FitNormalEquations(equations, ridge);
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
      Derivatives<false>(states.col(column), values_out[column], gradient, hessian);
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
      Derivatives<true>(states.col(column), values_out[column], gradient, hessian);
      gradients_out.col(column) = gradient;
      hessians_out.template block<N, N>(0, column * N) = hessian;
    }
  }

  template <int BatchSize>
  auto ValuesAndParameterGradients(const StateBatch<BatchSize>& states,
                                   ValueBatch<BatchSize>& values_out,
                                   ParameterGradientBatch<BatchSize>& parameter_gradients_out) const noexcept -> void {
    for (Eigen::Index column = 0; column < BatchSize; ++column) {
      const ParameterVector phi = Features(states.col(column));
      parameter_gradients_out.col(column) = phi;
      values_out[column] = phi.dot(parameters_);
    }
  }

 private:
  LinearBasisValueFunctionApprox(
      const ParameterVector& parameters, const State& center, const State& scale, BasisT basis, RidgeAnchor anchor)
      : parameters_(parameters),
        center_(center),
        scale_(scale),
        inverse_scale_(scale.cwiseInverse()),
        basis_(std::move(basis)),
        anchor_(anchor) {}

  [[nodiscard]] auto Normalize(const State& state) const noexcept -> State {
    return (state - center_).cwiseProduct(inverse_scale_);
  }

  template <bool WithHessian>
  auto Derivatives(const State& state, Scalar& value_out, State& gradient_out, StateMat& hessian_out) const noexcept
      -> void {
    ParameterVector phi;
    Gradients gradients;
    Hessians hessians;
    basis_.template Jet<true, WithHessian>(Normalize(state), phi, gradients, hessians);
    value_out = phi.dot(parameters_);
    gradient_out = (gradients * parameters_).cwiseProduct(inverse_scale_);
    if constexpr (WithHessian) {
      const Eigen::Matrix<Scalar, N * N, 1> flat = hessians * parameters_;
      for (int column = 0; column < N; ++column) {
        for (int row = 0; row < N; ++row) {
          hessian_out(row, column) = flat[(column * N) + row] * inverse_scale_[row] * inverse_scale_[column];
        }
      }
    }
  }

  template <int NumSamples>
  [[nodiscard]] static auto CheckFitInputs(const ValueBatch<NumSamples>& targets,
                                           const ValueBatch<NumSamples>& weights,
                                           Scalar ridge,
                                           Scalar slope_weight) noexcept -> Result<> {
    RESULT_ASSERT(weights.allFinite() && (weights.array() >= Scalar{0}).all(),
                  "LinearBasisValueFunctionApprox::FitWeightedLeastSquares: every weight must be finite and "
                  "non-negative.");
    RESULT_ASSERT(std::isfinite(ridge) && ridge >= Scalar{0},
                  "LinearBasisValueFunctionApprox::FitWeightedLeastSquares: the ridge weight must be finite and "
                  "non-negative.");
    RESULT_ASSERT(std::isfinite(slope_weight) && slope_weight >= Scalar{0},
                  "LinearBasisValueFunctionApprox::FitWeightedLeastSquares: the slope weight must be finite and "
                  "non-negative.");
    RESULT_ASSERT(targets.allFinite(),
                  "LinearBasisValueFunctionApprox::FitWeightedLeastSquares: every target must be finite.");
    return SuccessResult();
  }

  // The mean over a midpoint grid of [-1, 1]^N of the Hessian features' Gram matrix: theta' R theta
  // is the mean squared Frobenius norm of the Hessian of theta' phi.
  static auto RoughnessMatrix(const BasisT& basis) -> NormalMat {
    const int per_axis = N <= 2 ? 24 : (N == 3 ? 12 : 8);
    int total = 1;
    for (int axis = 0; axis < N; ++axis) {
      total *= per_axis;
    }
    NormalMat gram = NormalMat::Zero(kNumParameters, kNumParameters);
    for (int point = 0; point < total; ++point) {
      State z;
      int rest = point;
      for (int axis = 0; axis < N; ++axis) {
        z[axis] = Scalar{-1} + ((Scalar{2} * static_cast<Scalar>(rest % per_axis)) + Scalar{1}) /
                                   static_cast<Scalar>(per_axis);
        rest /= per_axis;
      }
      ParameterVector phi;
      Gradients gradients;
      Hessians hessians;
      basis.template Jet<true, true>(z, phi, gradients, hessians);
      gram.template selfadjointView<Eigen::Lower>().rankUpdate(hessians.transpose(), Scalar{1});
    }
    gram /= static_cast<Scalar>(total);
    return NormalMat(gram.template selfadjointView<Eigen::Lower>());
  }

  auto Solve(NormalMat normal_matrix, ParameterVector normal_vector, Scalar ridge) noexcept -> Result<> {
    if (roughness_ > Scalar{0}) {
      // `ridge` is the per-sample ridge times the sample count; the roughness weight scales the same way.
      const Scalar samples = ridge_samples_;
      normal_matrix.template triangularView<Eigen::Lower>() +=
          (roughness_ * samples) * roughness_matrix_->template triangularView<Eigen::Lower>().toDenseMatrix();
    }
    normal_matrix.diagonal().array() += ridge;
    if (anchor_ == RidgeAnchor::kCurrent) {
      normal_vector += ridge * parameters_;
    }
    const Eigen::LDLT<NormalMat> factorization(normal_matrix.template selfadjointView<Eigen::Lower>());
    RESULT_ASSERT(factorization.info() == Eigen::Success,
                  "LinearBasisValueFunctionApprox::FitWeightedLeastSquares: the normal equations are singular; add a "
                  "ridge weight or more varied samples.");
    const ParameterVector solution = factorization.solve(normal_vector);
    RESULT_ASSERT(solution.allFinite(),
                  "LinearBasisValueFunctionApprox::FitWeightedLeastSquares: the fit produced non-finite "
                  "coefficients.");
    parameters_ = solution;
    return SuccessResult();
  }

  ParameterVector parameters_;
  State center_;
  State scale_;
  State inverse_scale_;
  BasisT basis_;
  RidgeAnchor anchor_;
  Scalar roughness_ = Scalar{0};
  Scalar ridge_samples_ = Scalar{1};
  std::shared_ptr<const NormalMat> roughness_matrix_{};
};

// ---------------------------------------------------------------------------------------------
// Bases
// ---------------------------------------------------------------------------------------------

namespace linear_basis_detail {

// The number of exponent vectors of total degree at most `degree` in `n` variables.
constexpr auto TotalDegreeCount(int n, int degree) -> int {
  std::int64_t count = 1;
  for (int term = 1; term <= degree; ++term) {
    count = (count * (n + term)) / term;
  }
  return static_cast<int>(count);
}

// Every exponent vector of total degree at most `Degree`, in order of increasing total degree.
template <int N, int Degree>
constexpr auto TotalDegreeExponents() {
  std::array<std::array<int, N>, static_cast<std::size_t>(TotalDegreeCount(N, Degree))> table{};
  std::size_t next = 0;
  for (int total = 0; total <= Degree; ++total) {
    std::array<int, N> exponents{};
    // Enumerate compositions of `total` into N parts in lexicographic order.
    const auto recurse = [&](auto&& self, int axis, int remaining) -> void {
      if (axis == N - 1) {
        exponents[static_cast<std::size_t>(axis)] = remaining;
        table[next++] = exponents;
        return;
      }
      for (int power = remaining; power >= 0; --power) {
        exponents[static_cast<std::size_t>(axis)] = power;
        self(self, axis + 1, remaining - power);
      }
    };
    recurse(recurse, 0, total);
  }
  return table;
}

// A one-dimensional family's values and first two derivatives at one coordinate, orders 0..Degree.
template <int Degree, typename Scalar>
struct Jet1d {
  std::array<Scalar, static_cast<std::size_t>(Degree) + 1> value{};
  std::array<Scalar, static_cast<std::size_t>(Degree) + 1> first{};
  std::array<Scalar, static_cast<std::size_t>(Degree) + 1> second{};
};

// A tensor-product total-degree basis from one-dimensional families: phi_e(z) = prod_k p_{e_k}(z_k).
template <int N, int Degree, typename Scalar, typename OneDimT>
struct TotalDegreeProductBasis {
  static constexpr int kNumFeatures = TotalDegreeCount(N, Degree);
  static constexpr auto kExponents = TotalDegreeExponents<N, Degree>();

  template <bool WithGradient, bool WithHessian>
  auto Jet(const Eigen::Matrix<Scalar, N, 1>& z,
           Eigen::Matrix<Scalar, kNumFeatures, 1>& phi,
           Eigen::Matrix<Scalar, N, kNumFeatures>& gradients,
           Eigen::Matrix<Scalar, N * N, kNumFeatures>& hessians) const noexcept -> void {
    std::array<Jet1d<Degree, Scalar>, static_cast<std::size_t>(N)> jets;
    for (std::size_t axis = 0; axis < static_cast<std::size_t>(N); ++axis) {
      jets[axis] = OneDimT::template Evaluate<Degree>(z[static_cast<Eigen::Index>(axis)]);
    }
    for (std::size_t term = 0; term < kExponents.size(); ++term) {
      const auto& e = kExponents[term];
      const auto j = static_cast<Eigen::Index>(term);
      // Value, and for each axis the product with that axis differentiated.
      Scalar value = Scalar{1};
      for (std::size_t axis = 0; axis < static_cast<std::size_t>(N); ++axis) {
        value *= jets[axis].value[static_cast<std::size_t>(e[axis])];
      }
      phi[j] = value;
      if constexpr (WithGradient || WithHessian) {
        for (int a = 0; a < N; ++a) {
          Scalar product = Scalar{1};
          for (int k = 0; k < N; ++k) {
            const auto& jet = jets[static_cast<std::size_t>(k)];
            const auto power = static_cast<std::size_t>(e[static_cast<std::size_t>(k)]);
            product *= k == a ? jet.first[power] : jet.value[power];
          }
          gradients(a, j) = product;
        }
      }
      if constexpr (WithHessian) {
        for (int a = 0; a < N; ++a) {
          for (int b = a; b < N; ++b) {
            Scalar product = Scalar{1};
            for (int k = 0; k < N; ++k) {
              const auto& jet = jets[static_cast<std::size_t>(k)];
              const auto power = static_cast<std::size_t>(e[static_cast<std::size_t>(k)]);
              if (k == a && k == b) {
                product *= jet.second[power];
              } else if (k == a || k == b) {
                product *= jet.first[power];
              } else {
                product *= jet.value[power];
              }
            }
            hessians((b * N) + a, j) = product;
            hessians((a * N) + b, j) = product;
          }
        }
      }
    }
  }
};

struct Monomials1d {
  template <int Degree, typename Scalar>
  static auto Evaluate(Scalar z) noexcept -> Jet1d<Degree, Scalar> {
    Jet1d<Degree, Scalar> jet;
    jet.value[0] = Scalar{1};
    for (std::size_t n = 1; n <= static_cast<std::size_t>(Degree); ++n) {
      jet.value[n] = jet.value[n - 1] * z;
    }
    for (std::size_t n = 1; n <= static_cast<std::size_t>(Degree); ++n) {
      jet.first[n] = static_cast<Scalar>(n) * jet.value[n - 1];
      if (n >= 2) {
        jet.second[n] = static_cast<Scalar>(n * (n - 1)) * jet.value[n - 2];
      }
    }
    return jet;
  }
};

// Legendre polynomials scaled to unit mean square on [-1, 1]: sqrt(2n + 1) P_n.
struct Legendre1d {
  template <int Degree, typename Scalar>
  static auto Evaluate(Scalar z) noexcept -> Jet1d<Degree, Scalar> {
    Jet1d<Degree, Scalar> jet;
    std::array<Scalar, static_cast<std::size_t>(Degree) + 2> p{};
    std::array<Scalar, static_cast<std::size_t>(Degree) + 2> dp{};
    std::array<Scalar, static_cast<std::size_t>(Degree) + 2> ddp{};
    p[0] = Scalar{1};
    if constexpr (Degree >= 1) {
      p[1] = z;
      dp[1] = Scalar{1};
    }
    for (std::size_t n = 1; n < static_cast<std::size_t>(Degree); ++n) {
      const auto order = static_cast<Scalar>(n);
      p[n + 1] = (((2 * order) + 1) * z * p[n] - order * p[n - 1]) / (order + 1);
      dp[n + 1] = dp[n - 1] + ((2 * order) + 1) * p[n];
      ddp[n + 1] = ddp[n - 1] + ((2 * order) + 1) * dp[n];
    }
    for (std::size_t n = 0; n <= static_cast<std::size_t>(Degree); ++n) {
      const Scalar normalization = std::sqrt((2 * static_cast<Scalar>(n)) + 1);
      jet.value[n] = normalization * p[n];
      jet.first[n] = normalization * dp[n];
      jet.second[n] = normalization * ddp[n];
    }
    return jet;
  }
};

}  // namespace linear_basis_detail

// Every monomial of total degree at most `Degree`: the span of PolynomialValueFunctionApprox.
template <int N, int Degree, typename Scalar = double>
using MonomialBasis =
    linear_basis_detail::TotalDegreeProductBasis<N, Degree, Scalar, linear_basis_detail::Monomials1d>;

// Products of normalized Legendre polynomials of total degree at most `Degree`: the same span as
// the monomials, orthonormal under the uniform measure on the box. The span is the same; what
// differs is the conditioning of the normal equations, and -- where the samples do not determine
// every coefficient -- which of the many fits the ridge picks.
template <int N, int Degree, typename Scalar = double>
using LegendreBasis =
    linear_basis_detail::TotalDegreeProductBasis<N, Degree, Scalar, linear_basis_detail::Legendre1d>;

// Gaussian radial basis functions on a regular `PerAxis`^N grid of centers spanning [-1, 1]^N,
// each of width `width_factor` times the grid spacing, followed by the Legendre basis of total
// degree two. The quadratic part carries the bowl the terminal cost puts everywhere and the growth
// outside the box, where the bumps vanish; the bumps carry what a quadratic cannot.
template <int N, int PerAxis, typename Scalar = double>
struct GaussianRbfBasis {
  static_assert(PerAxis >= 2, "A grid of centers needs at least two per axis.");
  static constexpr int kNumCenters = [] {
    int count = 1;
    for (int axis = 0; axis < N; ++axis) {
      count *= PerAxis;
    }
    return count;
  }();
  using Quadratic = LegendreBasis<N, 2, Scalar>;
  static constexpr int kNumFeatures = kNumCenters + Quadratic::kNumFeatures;

  Scalar width_factor = Scalar{1};

  template <bool WithGradient, bool WithHessian>
  auto Jet(const Eigen::Matrix<Scalar, N, 1>& z,
           Eigen::Matrix<Scalar, kNumFeatures, 1>& phi,
           Eigen::Matrix<Scalar, N, kNumFeatures>& gradients,
           Eigen::Matrix<Scalar, N * N, kNumFeatures>& hessians) const noexcept -> void {
    const Scalar spacing = Scalar{2} / static_cast<Scalar>(PerAxis - 1);
    const Scalar width = width_factor * spacing;
    const Scalar inverse_variance = Scalar{1} / (width * width);
    for (int center = 0; center < kNumCenters; ++center) {
      Eigen::Matrix<Scalar, N, 1> offset;
      int rest = center;
      for (int axis = 0; axis < N; ++axis) {
        offset[axis] = z[axis] - (Scalar{-1} + spacing * static_cast<Scalar>(rest % PerAxis));
        rest /= PerAxis;
      }
      const Scalar value = std::exp(Scalar{-0.5} * inverse_variance * offset.squaredNorm());
      phi[center] = value;
      if constexpr (WithGradient || WithHessian) {
        gradients.col(center) = -inverse_variance * value * offset;
      }
      if constexpr (WithHessian) {
        for (int b = 0; b < N; ++b) {
          for (int a = 0; a < N; ++a) {
            const Scalar delta = a == b ? Scalar{1} : Scalar{0};
            hessians((b * N) + a, center) =
                value * inverse_variance * ((inverse_variance * offset[a] * offset[b]) - delta);
          }
        }
      }
    }
    Eigen::Matrix<Scalar, Quadratic::kNumFeatures, 1> quadratic_phi;
    Eigen::Matrix<Scalar, N, Quadratic::kNumFeatures> quadratic_gradients;
    Eigen::Matrix<Scalar, N * N, Quadratic::kNumFeatures> quadratic_hessians;
    Quadratic{}.template Jet<WithGradient, WithHessian>(z, quadratic_phi, quadratic_gradients, quadratic_hessians);
    phi.template tail<Quadratic::kNumFeatures>() = quadratic_phi;
    if constexpr (WithGradient || WithHessian) {
      gradients.template rightCols<Quadratic::kNumFeatures>() = quadratic_gradients;
    }
    if constexpr (WithHessian) {
      hessians.template rightCols<Quadratic::kNumFeatures>() = quadratic_hessians;
    }
  }
};

// Tensor-product uniform cubic B-splines with `Intervals` knot intervals per axis over
// [-extent, extent]^N, followed by the Legendre quadratic. Cubic splines reproduce every quadratic
// inside their domain, so the quadratic part only decides the value outside it, where the splines
// vanish. Twice continuously differentiable, as the Taylor estimators need, with local support: a
// sample moves only the 4^N coefficients around it.
template <int N, int Intervals, typename Scalar = double>
struct CubicBSplineBasis {
  static_assert(Intervals >= 1, "A spline needs at least one knot interval.");
  static constexpr int kPerAxis = Intervals + 3;
  static constexpr int kNumSplines = [] {
    int count = 1;
    for (int axis = 0; axis < N; ++axis) {
      count *= kPerAxis;
    }
    return count;
  }();
  using Quadratic = LegendreBasis<N, 2, Scalar>;
  static constexpr int kNumFeatures = kNumSplines + Quadratic::kNumFeatures;

  Scalar extent = Scalar{1.25};

  template <bool WithGradient, bool WithHessian>
  auto Jet(const Eigen::Matrix<Scalar, N, 1>& z,
           Eigen::Matrix<Scalar, kNumFeatures, 1>& phi,
           Eigen::Matrix<Scalar, N, kNumFeatures>& gradients,
           Eigen::Matrix<Scalar, N * N, kNumFeatures>& hessians) const noexcept -> void {
    // Per axis: the values and derivatives of every cubic B-spline at z.
    const Scalar h = (Scalar{2} * extent) / static_cast<Scalar>(Intervals);
    std::array<std::array<std::array<Scalar, 3>, kPerAxis>, static_cast<std::size_t>(N)> axis_jets{};
    for (int axis = 0; axis < N; ++axis) {
      for (int index = 0; index < kPerAxis; ++index) {
        // Spline `index` is centered at knot index - 1, so the first is centered one interval below
        // the domain and the last one above it.
        const Scalar t = ((z[axis] + extent) / h) - static_cast<Scalar>(index - 1);
        axis_jets[static_cast<std::size_t>(axis)][static_cast<std::size_t>(index)] = CubicBSpline(t, h);
      }
    }
    phi.setZero();
    if constexpr (WithGradient || WithHessian) {
      gradients.setZero();
    }
    if constexpr (WithHessian) {
      hessians.setZero();
    }
    for (int spline = 0; spline < kNumSplines; ++spline) {
      std::array<int, static_cast<std::size_t>(N)> index{};
      int rest = spline;
      bool zero = false;
      const auto at = [&](int axis, int order) {
        const auto slot = static_cast<std::size_t>(axis);
        return axis_jets[slot][static_cast<std::size_t>(index[slot])][static_cast<std::size_t>(order)];
      };
      for (int axis = 0; axis < N; ++axis) {
        index[static_cast<std::size_t>(axis)] = rest % kPerAxis;
        rest /= kPerAxis;
        // A spline's value is zero exactly where it and its derivatives are: outside its support.
        zero = zero || at(axis, 0) == Scalar{0};
      }
      if (zero) {
        continue;
      }
      Scalar value = Scalar{1};
      for (int axis = 0; axis < N; ++axis) {
        value *= at(axis, 0);
      }
      phi[spline] = value;
      if constexpr (WithGradient || WithHessian) {
        for (int a = 0; a < N; ++a) {
          Scalar product = Scalar{1};
          for (int k = 0; k < N; ++k) {
            product *= at(k, k == a ? 1 : 0);
          }
          gradients(a, spline) = product;
        }
      }
      if constexpr (WithHessian) {
        for (int a = 0; a < N; ++a) {
          for (int b = a; b < N; ++b) {
            Scalar product = Scalar{1};
            for (int k = 0; k < N; ++k) {
              product *= at(k, (k == a ? 1 : 0) + (k == b ? 1 : 0));
            }
            hessians((b * N) + a, spline) = product;
            hessians((a * N) + b, spline) = product;
          }
        }
      }
    }
    Eigen::Matrix<Scalar, Quadratic::kNumFeatures, 1> quadratic_phi;
    Eigen::Matrix<Scalar, N, Quadratic::kNumFeatures> quadratic_gradients;
    Eigen::Matrix<Scalar, N * N, Quadratic::kNumFeatures> quadratic_hessians;
    Quadratic{}.template Jet<WithGradient, WithHessian>(z, quadratic_phi, quadratic_gradients, quadratic_hessians);
    phi.template tail<Quadratic::kNumFeatures>() = quadratic_phi;
    if constexpr (WithGradient || WithHessian) {
      gradients.template rightCols<Quadratic::kNumFeatures>() = quadratic_gradients;
    }
    if constexpr (WithHessian) {
      hessians.template rightCols<Quadratic::kNumFeatures>() = quadratic_hessians;
    }
  }

 private:
  // The uniform cubic B-spline with support [0, 4] at `t` (in knot units), and its first and second
  // derivatives in z, for knot spacing `h`.
  static auto CubicBSpline(Scalar t, Scalar h) noexcept -> std::array<Scalar, 3> {
    if (t <= Scalar{0} || t >= Scalar{4}) {
      return {Scalar{0}, Scalar{0}, Scalar{0}};
    }
    Scalar value;
    Scalar first;
    Scalar second;
    if (t < Scalar{1}) {
      value = t * t * t / 6;
      first = t * t / 2;
      second = t;
    } else if (t < Scalar{2}) {
      const Scalar u = t - 1;
      value = (1 + 3 * u + 3 * u * u - 3 * u * u * u) / 6;
      first = (3 + 6 * u - 9 * u * u) / 6;
      second = (6 - 18 * u) / 6;
    } else if (t < Scalar{3}) {
      const Scalar u = t - 2;
      value = (4 - 6 * u * u + 3 * u * u * u) / 6;
      first = (-12 * u + 9 * u * u) / 6;
      second = (-12 + 18 * u) / 6;
    } else {
      const Scalar u = 4 - t;
      value = u * u * u / 6;
      first = -u * u / 2;
      second = u;
    }
    return {value, first / h, second / (h * h)};
  }
};

}  // namespace fbsde_traj_opt

#endif  // FBSDE_TRAJ_OPT_VALUE_FUNCTION_LINEAR_BASIS_VALUE_FUNCTION_APPROX_HPP_
