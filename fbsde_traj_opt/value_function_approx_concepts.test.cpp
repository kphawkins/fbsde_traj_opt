// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#include "fbsde_traj_opt/value_function_approx_concepts.hpp"

#include <string>
#include <utility>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "fbsde_traj_opt/utils/result.hpp"

namespace fbsde_traj_opt {
namespace {

using State3 = Eigen::Vector3d;

constexpr int kBatch = 4;

// ---------------------------------------------------------------------------------------------
// Batch type aliases
// ---------------------------------------------------------------------------------------------

static_assert(ValueFunctionStateBatch<State3, kBatch>::RowsAtCompileTime == 3);
static_assert(ValueFunctionStateBatch<State3, kBatch>::ColsAtCompileTime == kBatch);

// One value per sample, laid out as a row so the columns line up with the state batch.
static_assert(ValueFunctionValueBatch<State3, kBatch>::RowsAtCompileTime == 1);
static_assert(ValueFunctionValueBatch<State3, kBatch>::ColsAtCompileTime == kBatch);

static_assert(ValueFunctionGradientBatch<State3, kBatch>::RowsAtCompileTime == 3);
static_assert(ValueFunctionGradientBatch<State3, kBatch>::ColsAtCompileTime == kBatch);

// `BatchSize` side-by-side 3 x 3 blocks.
static_assert(ValueFunctionHessianBatch<State3, kBatch>::RowsAtCompileTime == 3);
static_assert(ValueFunctionHessianBatch<State3, kBatch>::ColsAtCompileTime == 3 * kBatch);

static_assert(ValueFunctionParameterGradientBatch<State3, 7, kBatch>::RowsAtCompileTime == 7);
static_assert(ValueFunctionParameterGradientBatch<State3, 7, kBatch>::ColsAtCompileTime == kBatch);

// ---------------------------------------------------------------------------------------------
// A conforming model, satisfying all four concepts
// ---------------------------------------------------------------------------------------------

// V(x) = sum_j p_j^2 x_j^2, whose free parameters are the three weights p. Small enough to read,
// and squared rather than linear in p so that every required member -- the parameter gradient
// included -- genuinely depends on the parameters.
class DiagonalQuadraticValueFunction {
 public:
  static constexpr int kNumParameters = 3;

  using ParameterVector = Eigen::Matrix<double, kNumParameters, 1>;

  explicit DiagonalQuadraticValueFunction(ParameterVector parameters) noexcept : parameters_(std::move(parameters)) {}

  auto operator()(const State3& state) const noexcept -> double {
    return state.cwiseAbs2().dot(parameters_.cwiseAbs2());
  }

  [[nodiscard]] auto Gradient(const State3& state) const noexcept -> State3 {
    return 2.0 * parameters_.cwiseAbs2().cwiseProduct(state);
  }

  [[nodiscard]] auto Hessian([[maybe_unused]] const State3& state) const noexcept -> Eigen::Matrix3d {
    Eigen::Matrix3d hessian = Eigen::Matrix3d::Zero();
    hessian.diagonal() = 2.0 * parameters_.cwiseAbs2();
    return hessian;
  }

  [[nodiscard]] auto ParameterGradient(const State3& state) const noexcept -> ParameterVector {
    return 2.0 * parameters_.cwiseProduct(state.cwiseAbs2());
  }

  [[nodiscard]] auto Parameters() const noexcept -> const ParameterVector& { return parameters_; }

  auto SetParameters(const ParameterVector& parameters) noexcept -> Result<> {
    RESULT_ASSERT(parameters.allFinite(), "DiagonalQuadraticValueFunction: the parameters must all be finite.");
    parameters_ = parameters;
    return SuccessResult();
  }

  template <int BatchSize>
  auto Values(const ValueFunctionStateBatch<State3, BatchSize>& states,
              ValueFunctionValueBatch<State3, BatchSize>& values_out) const noexcept -> void {
    values_out = parameters_.cwiseAbs2().transpose() * states.cwiseAbs2();
  }

  template <int BatchSize>
  auto ValuesAndGradients(const ValueFunctionStateBatch<State3, BatchSize>& states,
                          ValueFunctionValueBatch<State3, BatchSize>& values_out,
                          ValueFunctionGradientBatch<State3, BatchSize>& gradients_out) const noexcept -> void {
    Values<BatchSize>(states, values_out);
    gradients_out = 2.0 * parameters_.cwiseAbs2().replicate(1, BatchSize).cwiseProduct(states);
  }

  template <int BatchSize>
  auto ValuesGradientsAndHessians(const ValueFunctionStateBatch<State3, BatchSize>& states,
                                  ValueFunctionValueBatch<State3, BatchSize>& values_out,
                                  ValueFunctionGradientBatch<State3, BatchSize>& gradients_out,
                                  ValueFunctionHessianBatch<State3, BatchSize>& hessians_out) const noexcept -> void {
    ValuesAndGradients<BatchSize>(states, values_out, gradients_out);
    hessians_out = Hessian(State3::Zero()).replicate(1, BatchSize);
  }

  template <int BatchSize>
  auto ValuesAndParameterGradients(
      const ValueFunctionStateBatch<State3, BatchSize>& states,
      ValueFunctionValueBatch<State3, BatchSize>& values_out,
      ValueFunctionParameterGradientBatch<State3, kNumParameters, BatchSize>& parameter_gradients_out) const noexcept
      -> void {
    Values<BatchSize>(states, values_out);
    parameter_gradients_out = 2.0 * parameters_.replicate(1, BatchSize).cwiseProduct(states.cwiseAbs2());
  }

 private:
  ParameterVector parameters_;
};

static_assert(ValueFunctionApprox<DiagonalQuadraticValueFunction, State3>);
static_assert(BatchedValueFunctionApprox<DiagonalQuadraticValueFunction, State3, 1>);
static_assert(BatchedValueFunctionApprox<DiagonalQuadraticValueFunction, State3, kBatch>);
static_assert(ParameterizedValueFunctionApprox<DiagonalQuadraticValueFunction, State3>);
static_assert(BatchedParameterizedValueFunctionApprox<DiagonalQuadraticValueFunction, State3, kBatch>);

// Rejected: the state type itself is dynamically sized, however well the model conforms.
static_assert(!ValueFunctionApprox<DiagonalQuadraticValueFunction, Eigen::VectorXd>);

// Rejected: the state type has the wrong fixed dimension, so no member has the right signature.
static_assert(!ValueFunctionApprox<DiagonalQuadraticValueFunction, Eigen::Vector2d>);

// ---------------------------------------------------------------------------------------------
// ValueFunctionApprox: negative cases
// ---------------------------------------------------------------------------------------------

// V(x) = w ||x||^2: a model conforming pointwise and nothing more. Most negative cases below
// derive from it and hide exactly one member, so that each differs from a conforming model in
// precisely the way its comment claims.
struct ScaledNormValueFunction {
  double weight = 1.0;

  auto operator()(const State3& state) const noexcept -> double { return weight * state.squaredNorm(); }

  [[nodiscard]] auto Gradient(const State3& state) const noexcept -> State3 { return 2.0 * weight * state; }

  [[nodiscard]] auto Hessian(const State3& /*state*/) const noexcept -> Eigen::Matrix3d {
    return 2.0 * weight * Eigen::Matrix3d::Identity();
  }
};

static_assert(ValueFunctionApprox<ScaledNormValueFunction, State3>);

// Rejected: no Hessian. A representation that cannot report curvature cannot be used with a
// second-order Taylor estimator at all, which is why this is a failure and not a lesser concept.
struct NoHessianValueFunction {
  double weight = 1.0;

  auto operator()(const State3& state) const noexcept -> double { return weight * state.squaredNorm(); }

  [[nodiscard]] auto Gradient(const State3& state) const noexcept -> State3 { return 2.0 * weight * state; }
};

static_assert(!ValueFunctionApprox<NoHessianValueFunction, State3>);

// Rejected: no Gradient.
struct NoGradientValueFunction {
  double weight = 1.0;

  auto operator()(const State3& state) const noexcept -> double { return weight * state.squaredNorm(); }

  [[nodiscard]] auto Hessian(const State3& /*state*/) const noexcept -> Eigen::Matrix3d {
    return 2.0 * weight * Eigen::Matrix3d::Identity();
  }
};

static_assert(!ValueFunctionApprox<NoGradientValueFunction, State3>);

// Rejected: the gradient's fixed size doesn't match the state's dimension.
struct WrongSizeGradientValueFunction : ScaledNormValueFunction {
  [[nodiscard]] auto Gradient(const State3& /*state*/) const noexcept -> Eigen::Vector2d {
    return Eigen::Vector2d::Constant(weight);
  }
};

static_assert(!ValueFunctionApprox<WrongSizeGradientValueFunction, State3>);

// Rejected: the gradient is dynamically sized, even though it happens to be size 3 at runtime.
struct DynamicGradientValueFunction : ScaledNormValueFunction {
  [[nodiscard]] auto Gradient(const State3& /*state*/) const noexcept -> Eigen::VectorXd {
    return Eigen::VectorXd::Constant(3, weight);
  }
};

static_assert(!ValueFunctionApprox<DynamicGradientValueFunction, State3>);

// Rejected: the Hessian isn't square.
struct NonSquareHessianValueFunction : ScaledNormValueFunction {
  [[nodiscard]] auto Hessian(const State3& /*state*/) const noexcept -> Eigen::Matrix<double, 3, 2> {
    return Eigen::Matrix<double, 3, 2>::Constant(weight);
  }
};

static_assert(!ValueFunctionApprox<NonSquareHessianValueFunction, State3>);

// Rejected: the value's scalar type differs from the state's.
struct WrongScalarValueFunction : ScaledNormValueFunction {
  auto operator()(const State3& state) const noexcept -> float {
    return static_cast<float>(weight * state.squaredNorm());
  }
};

static_assert(!ValueFunctionApprox<WrongScalarValueFunction, State3>);

// Rejected: evaluation mutates the model -- here by tallying its own calls -- so none of the
// members are callable on a const object.
struct NonConstValueFunction {
  double weight = 1.0;
  int call_count = 0;

  auto operator()(const State3& state) noexcept -> double {
    ++call_count;
    return weight * state.squaredNorm();
  }

  auto Gradient(const State3& state) noexcept -> State3 {
    ++call_count;
    return 2.0 * weight * state;
  }

  auto Hessian(const State3& /*state*/) noexcept -> Eigen::Matrix3d {
    ++call_count;
    return 2.0 * weight * Eigen::Matrix3d::Identity();
  }
};

static_assert(!ValueFunctionApprox<NonConstValueFunction, State3>);

// Rejected: not callable with a state at all.
struct WrongArgumentValueFunction : ScaledNormValueFunction {
  auto operator()(const std::string& /*state*/) const noexcept -> double { return weight; }
};

static_assert(!ValueFunctionApprox<WrongArgumentValueFunction, State3>);

// ---------------------------------------------------------------------------------------------
// BatchedValueFunctionApprox: negative cases
// ---------------------------------------------------------------------------------------------

// Rejected: no batch evaluation at all. Perfectly usable as a frozen next-stage representation in
// a pointwise estimator; simply not batched.
static_assert(!BatchedValueFunctionApprox<ScaledNormValueFunction, State3, kBatch>);

// Rejected: Values() writes into a column vector rather than the row-shaped value batch, so the
// output a caller declared with the documented type would not bind.
struct ColumnValuesValueFunction : ScaledNormValueFunction {
  template <int BatchSize>
  auto Values(const ValueFunctionStateBatch<State3, BatchSize>& states,
              Eigen::Matrix<double, BatchSize, 1>& values_out) const noexcept -> void {
    values_out = weight * states.cwiseAbs2().colwise().sum().transpose();
  }

  template <int BatchSize>
  auto ValuesAndGradients(const ValueFunctionStateBatch<State3, BatchSize>& states,
                          ValueFunctionValueBatch<State3, BatchSize>& values_out,
                          ValueFunctionGradientBatch<State3, BatchSize>& gradients_out) const noexcept -> void {
    values_out = weight * states.cwiseAbs2().colwise().sum();
    gradients_out = 2.0 * weight * states;
  }

  template <int BatchSize>
  auto ValuesGradientsAndHessians(const ValueFunctionStateBatch<State3, BatchSize>& states,
                                  ValueFunctionValueBatch<State3, BatchSize>& values_out,
                                  ValueFunctionGradientBatch<State3, BatchSize>& gradients_out,
                                  ValueFunctionHessianBatch<State3, BatchSize>& hessians_out) const noexcept -> void {
    ValuesAndGradients<BatchSize>(states, values_out, gradients_out);
    hessians_out = Hessian(State3::Zero()).replicate(1, BatchSize);
  }
};

static_assert(!BatchedValueFunctionApprox<ColumnValuesValueFunction, State3, kBatch>);

// Rejected: the Hessian batch is one N x N block rather than `BatchSize` of them, so only the
// first sample's curvature could ever be written.
struct SingleHessianBlockValueFunction : ScaledNormValueFunction {
  template <int BatchSize>
  auto Values(const ValueFunctionStateBatch<State3, BatchSize>& states,
              ValueFunctionValueBatch<State3, BatchSize>& values_out) const noexcept -> void {
    values_out = weight * states.cwiseAbs2().colwise().sum();
  }

  template <int BatchSize>
  auto ValuesAndGradients(const ValueFunctionStateBatch<State3, BatchSize>& states,
                          ValueFunctionValueBatch<State3, BatchSize>& values_out,
                          ValueFunctionGradientBatch<State3, BatchSize>& gradients_out) const noexcept -> void {
    Values<BatchSize>(states, values_out);
    gradients_out = 2.0 * weight * states;
  }

  template <int BatchSize>
  auto ValuesGradientsAndHessians(const ValueFunctionStateBatch<State3, BatchSize>& states,
                                  ValueFunctionValueBatch<State3, BatchSize>& values_out,
                                  ValueFunctionGradientBatch<State3, BatchSize>& gradients_out,
                                  Eigen::Matrix3d& hessians_out) const noexcept -> void {
    ValuesAndGradients<BatchSize>(states, values_out, gradients_out);
    hessians_out = Hessian(State3::Zero());
  }
};

static_assert(!BatchedValueFunctionApprox<SingleHessianBlockValueFunction, State3, kBatch>);

// ---------------------------------------------------------------------------------------------
// ParameterizedValueFunctionApprox: negative cases
// ---------------------------------------------------------------------------------------------

// Rejected: no parameters at all -- a fixed representation, usable as a frozen `V~_{i+1}` but not
// something a fitter can move.
static_assert(!ParameterizedValueFunctionApprox<ScaledNormValueFunction, State3>);

// Rejected: SetParameters returns void, so a caller has no way to learn that a non-finite
// parameter vector was rejected.
class VoidSetParametersValueFunction : public ScaledNormValueFunction {
 public:
  static constexpr int kNumParameters = 3;

  using ParameterVector = Eigen::Matrix<double, kNumParameters, 1>;

  [[nodiscard]] auto ParameterGradient(const State3& state) const noexcept -> ParameterVector {
    return parameters_.cwiseProduct(state.cwiseAbs2());
  }

  [[nodiscard]] auto Parameters() const noexcept -> const ParameterVector& { return parameters_; }

  auto SetParameters(const ParameterVector& parameters) noexcept -> void { parameters_ = parameters; }

 private:
  ParameterVector parameters_ = ParameterVector::Ones();
};

static_assert(!ParameterizedValueFunctionApprox<VoidSetParametersValueFunction, State3>);

// Rejected: the parameter gradient's dimension disagrees with kNumParameters.
class MismatchedParameterGradientValueFunction : public ScaledNormValueFunction {
 public:
  static constexpr int kNumParameters = 3;

  using ParameterVector = Eigen::Matrix<double, kNumParameters, 1>;

  [[nodiscard]] auto ParameterGradient(const State3& /*state*/) const noexcept -> Eigen::Vector2d {
    return Eigen::Vector2d::Constant(parameters_.sum());
  }

  [[nodiscard]] auto Parameters() const noexcept -> const ParameterVector& { return parameters_; }

  auto SetParameters(const ParameterVector& parameters) noexcept -> Result<> {
    parameters_ = parameters;
    return SuccessResult();
  }

 private:
  ParameterVector parameters_ = ParameterVector::Ones();
};

static_assert(!ParameterizedValueFunctionApprox<MismatchedParameterGradientValueFunction, State3>);

// Rejected: the ParameterVector type disagrees with kNumParameters, so a fitter would size its
// optimizer state from one of the two and its gradients from the other.
class MismatchedParameterVectorValueFunction : public ScaledNormValueFunction {
 public:
  static constexpr int kNumParameters = 3;

  using ParameterVector = Eigen::Matrix<double, 5, 1>;

  [[nodiscard]] auto ParameterGradient(const State3& /*state*/) const noexcept -> State3 {
    return State3::Constant(parameters_.sum());
  }

  [[nodiscard]] auto Parameters() const noexcept -> const ParameterVector& { return parameters_; }

  auto SetParameters(const ParameterVector& parameters) noexcept -> Result<> {
    parameters_ = parameters;
    return SuccessResult();
  }

 private:
  ParameterVector parameters_ = ParameterVector::Ones();
};

static_assert(!ParameterizedValueFunctionApprox<MismatchedParameterVectorValueFunction, State3>);

// Rejected: parameters can be read but never written.
class ReadOnlyParametersValueFunction : public ScaledNormValueFunction {
 public:
  static constexpr int kNumParameters = 3;

  using ParameterVector = Eigen::Matrix<double, kNumParameters, 1>;

  [[nodiscard]] auto ParameterGradient(const State3& state) const noexcept -> ParameterVector {
    return parameters_.cwiseProduct(state.cwiseAbs2());
  }

  [[nodiscard]] auto Parameters() const noexcept -> const ParameterVector& { return parameters_; }

 private:
  ParameterVector parameters_ = ParameterVector::Ones();
};

static_assert(!ParameterizedValueFunctionApprox<ReadOnlyParametersValueFunction, State3>);

// ---------------------------------------------------------------------------------------------
// BatchedParameterizedValueFunctionApprox: negative cases
// ---------------------------------------------------------------------------------------------

// Rejected: batched in values, gradients and Hessians, but its parameter gradients come one
// sample at a time, so a fitter would fall back to a per-sample loop for the one quantity it
// needs most. The derived member template hides the conforming one it inherits.
class UnbatchedParameterGradientValueFunction : public DiagonalQuadraticValueFunction {
 public:
  using DiagonalQuadraticValueFunction::DiagonalQuadraticValueFunction;

  template <int BatchSize>
  auto ValuesAndParameterGradients(
      const ValueFunctionStateBatch<State3, BatchSize>& states,
      ValueFunctionValueBatch<State3, BatchSize>& values_out,
      ValueFunctionParameterGradientBatch<State3, kNumParameters, 1>& parameter_gradients_out) const noexcept -> void {
    Values<BatchSize>(states, values_out);
    parameter_gradients_out = ParameterGradient(states.col(0));
  }
};

static_assert(BatchedValueFunctionApprox<UnbatchedParameterGradientValueFunction, State3, kBatch>);
static_assert(ParameterizedValueFunctionApprox<UnbatchedParameterGradientValueFunction, State3>);
static_assert(!BatchedParameterizedValueFunctionApprox<UnbatchedParameterGradientValueFunction, State3, kBatch>);

TEST(ValueFunctionApproxConceptsTest, CompileTimeChecksPassed) {
  // All the interesting checks for these concepts are the static_asserts above; this test exists
  // only so the target has a runnable case.
  SUCCEED();
}

}  // namespace
}  // namespace fbsde_traj_opt
