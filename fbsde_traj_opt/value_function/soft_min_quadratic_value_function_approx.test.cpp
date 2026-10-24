// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#include "fbsde_traj_opt/value_function/soft_min_quadratic_value_function_approx.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <gtest/gtest.h>

#include "fbsde_traj_opt/value_function/value_function_approx_concepts.hpp"

namespace fbsde_traj_opt {
namespace {

using Scalar1x1 = SoftMinQuadraticValueFunctionApprox<1, 1>;
using Scalar1x4 = SoftMinQuadraticValueFunctionApprox<1, 4>;
using Plane2x1 = SoftMinQuadraticValueFunctionApprox<2, 1>;
using Plane2x3 = SoftMinQuadraticValueFunctionApprox<2, 3>;
using Space3x3 = SoftMinQuadraticValueFunctionApprox<3, 3>;

constexpr double kSharpness = 1.5;
constexpr double kDiagonalOffset = 1e-3;

static_assert(ValueFunctionApprox<Scalar1x4, Eigen::Matrix<double, 1, 1>>);
static_assert(BatchedValueFunctionApprox<Scalar1x4, Eigen::Matrix<double, 1, 1>, 1>);
static_assert(BatchedValueFunctionApprox<Scalar1x4, Eigen::Matrix<double, 1, 1>, 8>);
static_assert(ParameterizedValueFunctionApprox<Scalar1x4, Eigen::Matrix<double, 1, 1>>);
static_assert(BatchedParameterizedValueFunctionApprox<Scalar1x4, Eigen::Matrix<double, 1, 1>, 8>);

static_assert(ValueFunctionApprox<Space3x3, Eigen::Vector3d>);
static_assert(BatchedValueFunctionApprox<Space3x3, Eigen::Vector3d, 1>);
static_assert(BatchedValueFunctionApprox<Space3x3, Eigen::Vector3d, 8>);
static_assert(ParameterizedValueFunctionApprox<Space3x3, Eigen::Vector3d>);
static_assert(BatchedParameterizedValueFunctionApprox<Space3x3, Eigen::Vector3d, 8>);

// Rejected: the state type's dimension disagrees with the model's.
static_assert(!ValueFunctionApprox<Space3x3, Eigen::Vector2d>);

// One component over a 3-dimensional state: 6 lower-triangle entries, 3 centers, 1 offset.
static_assert(SoftMinQuadraticValueFunctionApprox<3, 1>::kNumParameters == 10);
static_assert(Space3x3::kNumParameters == 30);
static_assert(Scalar1x1::kNumParameters == 3);

// A deterministic, well-spread source of parameter values. Not a statistical generator -- these
// tests want reproducible arbitrary numbers, and nothing more.
class Spread {
 public:
  explicit constexpr Spread(std::uint64_t seed) noexcept : state_(seed) {}

  // Returns a value in (-magnitude, magnitude).
  auto Next(double magnitude) noexcept -> double {
    state_ = (state_ * 6364136223846793005ULL) + 1442695040888963407ULL;
    const auto fraction = static_cast<double>(state_ >> 11U) / static_cast<double>(1ULL << 53U);
    return magnitude * ((2.0 * fraction) - 1.0);
  }

 private:
  std::uint64_t state_;
};

template <typename ModelT>
auto SpreadParameters(std::uint64_t seed, double magnitude) noexcept -> typename ModelT::ParameterVector {
  Spread spread(seed);
  typename ModelT::ParameterVector parameters;
  for (Eigen::Index index = 0; index < ModelT::kNumParameters; ++index) {
    parameters[index] = spread.Next(magnitude);
  }
  return parameters;
}

template <typename ModelT>
auto SpreadState(std::uint64_t seed, double magnitude) noexcept -> typename ModelT::State {
  Spread spread(seed);
  typename ModelT::State state;
  for (Eigen::Index index = 0; index < state.rows(); ++index) {
    state[index] = spread.Next(magnitude);
  }
  return state;
}

// ---------------------------------------------------------------------------------------------
// Make
// ---------------------------------------------------------------------------------------------

TEST(SoftMinQuadraticValueFunctionApproxTest, MakeSucceedsWithOrdinaryArguments) {
  const auto model = Space3x3::Make(SpreadParameters<Space3x3>(1, 1.0), kSharpness, kDiagonalOffset);

  EXPECT_TRUE(model.has_value());
}

TEST(SoftMinQuadraticValueFunctionApproxTest, MakeFailsWhenTheSharpnessIsNotPositive) {
  const auto parameters = SpreadParameters<Space3x3>(1, 1.0);

  EXPECT_FALSE(Space3x3::Make(parameters, 0.0, kDiagonalOffset).has_value());
  EXPECT_FALSE(Space3x3::Make(parameters, -1.0, kDiagonalOffset).has_value());
  EXPECT_FALSE(Space3x3::Make(parameters, std::numeric_limits<double>::infinity(), kDiagonalOffset).has_value());
}

TEST(SoftMinQuadraticValueFunctionApproxTest, MakeFailsWhenTheDiagonalOffsetIsNotPositive) {
  const auto parameters = SpreadParameters<Space3x3>(1, 1.0);

  EXPECT_FALSE(Space3x3::Make(parameters, kSharpness, 0.0).has_value());
  EXPECT_FALSE(Space3x3::Make(parameters, kSharpness, -1e-6).has_value());
}

TEST(SoftMinQuadraticValueFunctionApproxTest, MakeFailsWhenAParameterIsNotFinite) {
  auto parameters = SpreadParameters<Space3x3>(1, 1.0);
  parameters[4] = std::numeric_limits<double>::quiet_NaN();

  EXPECT_FALSE(Space3x3::Make(parameters, kSharpness, kDiagonalOffset).has_value());
}

TEST(SoftMinQuadraticValueFunctionApproxTest, SetParametersRejectsNonFiniteParametersAndLeavesTheModelUntouched) {
  const auto original = SpreadParameters<Space3x3>(2, 1.0);
  auto model = Space3x3::Make(original, kSharpness, kDiagonalOffset);
  ASSERT_TRUE(model.has_value());
  const Eigen::Vector3d state = SpreadState<Space3x3>(3, 2.0);
  const double before = (*model)(state);

  auto poisoned = original;
  poisoned[7] = std::numeric_limits<double>::infinity();

  EXPECT_FALSE(model->SetParameters(poisoned).has_value());
  EXPECT_EQ(model->Parameters(), original);
  EXPECT_EQ((*model)(state), before);
}

// ---------------------------------------------------------------------------------------------
// Exactness on a single quadratic
// ---------------------------------------------------------------------------------------------

TEST(SoftMinQuadraticValueFunctionApproxTest, OneComponentReproducesItsQuadraticExactly) {
  Eigen::Matrix3d metric;
  metric << 4.0, 1.0, 0.5, 1.0, 3.0, -0.25, 0.5, -0.25, 2.0;
  const Eigen::Vector3d center(0.5, -1.0, 2.0);
  constexpr double kOffset = -3.25;

  using Model = SoftMinQuadraticValueFunctionApprox<3, 1>;
  const auto parameters = Model::QuadraticParameters(metric, center, kOffset, kSharpness, kDiagonalOffset);
  ASSERT_TRUE(parameters.has_value()) << parameters.error();
  const auto model = Model::Make(*parameters, kSharpness, kDiagonalOffset);
  ASSERT_TRUE(model.has_value()) << model.error();

  for (std::uint64_t seed = 0; seed < 8; ++seed) {
    const Eigen::Vector3d state = SpreadState<Model>(seed + 10, 3.0);
    const Eigen::Vector3d centered = state - center;

    EXPECT_NEAR((*model)(state), centered.dot(metric * centered) + kOffset, 1e-9);
    EXPECT_LT(((*model).Gradient(state) - (2.0 * metric * centered)).norm(), 1e-9);
    EXPECT_LT(((*model).Hessian(state) - (2.0 * metric)).norm(), 1e-9);
  }
}

TEST(SoftMinQuadraticValueFunctionApproxTest, IdenticalComponentsReproduceTheirSharedQuadraticExactly) {
  // Three components carrying the same quadratic: the log(NumComponents)/beta correction baked
  // into QuadraticParameters is what keeps the soft minimum from sitting below it.
  Eigen::Matrix2d metric;
  metric << 2.0, 0.75, 0.75, 1.5;
  const Eigen::Vector2d center(-0.25, 1.75);
  constexpr double kOffset = 4.0;

  const auto parameters = Plane2x3::QuadraticParameters(metric, center, kOffset, kSharpness, kDiagonalOffset);
  ASSERT_TRUE(parameters.has_value()) << parameters.error();
  const auto model = Plane2x3::Make(*parameters, kSharpness, kDiagonalOffset);
  ASSERT_TRUE(model.has_value()) << model.error();

  for (std::uint64_t seed = 0; seed < 8; ++seed) {
    const Eigen::Vector2d state = SpreadState<Plane2x3>(seed + 20, 3.0);
    const Eigen::Vector2d centered = state - center;

    EXPECT_NEAR((*model)(state), centered.dot(metric * centered) + kOffset, 1e-9);
    EXPECT_LT(((*model).Gradient(state) - (2.0 * metric * centered)).norm(), 1e-9);
    EXPECT_LT(((*model).Hessian(state) - (2.0 * metric)).norm(), 1e-9);
  }
}

TEST(SoftMinQuadraticValueFunctionApproxTest, QuadraticParametersFailsOnAnIndefiniteMetric) {
  Eigen::Matrix2d metric;
  metric << 1.0, 0.0, 0.0, -1.0;

  const auto parameters = Plane2x1::QuadraticParameters(metric, Eigen::Vector2d::Zero(), 0.0, kSharpness, 1e-6);

  EXPECT_FALSE(parameters.has_value());
}

TEST(SoftMinQuadraticValueFunctionApproxTest, QuadraticParametersFailsWhenAFactorDiagonalIsBelowTheOffset) {
  // The Cholesky factor of this metric has diagonal (1e-4, ...), under an offset of 1e-3, and the
  // parameterization has no way to represent a diagonal entry that small.
  Eigen::Matrix2d metric;
  metric << 1e-8, 0.0, 0.0, 1.0;

  const auto parameters =
      Plane2x1::QuadraticParameters(metric, Eigen::Vector2d::Zero(), 0.0, kSharpness, kDiagonalOffset);

  EXPECT_FALSE(parameters.has_value());
}

// ---------------------------------------------------------------------------------------------
// Derivatives against central differences
// ---------------------------------------------------------------------------------------------

template <typename ModelT>
auto ExpectGradientMatchesCentralDifferences(const ModelT& model, const typename ModelT::State& state) -> void {
  constexpr double kStep = 1e-6;
  const typename ModelT::State gradient = model.Gradient(state);

  for (Eigen::Index index = 0; index < state.rows(); ++index) {
    typename ModelT::State forward = state;
    typename ModelT::State backward = state;
    forward[index] += kStep;
    backward[index] -= kStep;

    const double difference = (model(forward) - model(backward)) / (2.0 * kStep);
    EXPECT_NEAR(gradient[index], difference, 1e-6) << "component " << index;
  }
}

template <typename ModelT>
auto ExpectHessianMatchesCentralDifferences(const ModelT& model, const typename ModelT::State& state) -> void {
  constexpr double kStep = 1e-5;
  const typename ModelT::StateMat hessian = model.Hessian(state);

  for (Eigen::Index index = 0; index < state.rows(); ++index) {
    typename ModelT::State forward = state;
    typename ModelT::State backward = state;
    forward[index] += kStep;
    backward[index] -= kStep;

    const typename ModelT::State difference = (model.Gradient(forward) - model.Gradient(backward)) / (2.0 * kStep);
    EXPECT_LT((hessian.col(index) - difference).norm(), 1e-5) << "column " << index;
  }
}

template <typename ModelT>
auto ExpectParameterGradientMatchesCentralDifferences(ModelT& model, const typename ModelT::State& state) -> void {
  constexpr double kStep = 1e-6;
  const typename ModelT::ParameterVector original = model.Parameters();
  const typename ModelT::ParameterVector gradient = model.ParameterGradient(state);

  for (Eigen::Index index = 0; index < ModelT::kNumParameters; ++index) {
    typename ModelT::ParameterVector perturbed = original;

    perturbed[index] = original[index] + kStep;
    ASSERT_TRUE(model.SetParameters(perturbed).has_value());
    const double forward = model(state);

    perturbed[index] = original[index] - kStep;
    ASSERT_TRUE(model.SetParameters(perturbed).has_value());
    const double backward = model(state);

    EXPECT_NEAR(gradient[index], (forward - backward) / (2.0 * kStep), 1e-6) << "parameter " << index;
  }

  ASSERT_TRUE(model.SetParameters(original).has_value());
}

template <typename ModelT>
auto ExpectAllDerivativesMatchCentralDifferences(std::uint64_t seed) -> void {
  auto model = ModelT::Make(SpreadParameters<ModelT>(seed, 1.2), kSharpness, kDiagonalOffset);
  ASSERT_TRUE(model.has_value()) << model.error();

  for (std::uint64_t offset = 0; offset < 4; ++offset) {
    const typename ModelT::State state = SpreadState<ModelT>(seed + (100 * offset) + 1, 2.5);
    ExpectGradientMatchesCentralDifferences(*model, state);
    ExpectHessianMatchesCentralDifferences(*model, state);
    ExpectParameterGradientMatchesCentralDifferences(*model, state);
  }
}

TEST(SoftMinQuadraticValueFunctionApproxTest, DerivativesMatchCentralDifferencesInOneDimension) {
  ExpectAllDerivativesMatchCentralDifferences<SoftMinQuadraticValueFunctionApprox<1, 1>>(11);
  ExpectAllDerivativesMatchCentralDifferences<SoftMinQuadraticValueFunctionApprox<1, 3>>(12);
}

TEST(SoftMinQuadraticValueFunctionApproxTest, DerivativesMatchCentralDifferencesInTwoDimensions) {
  ExpectAllDerivativesMatchCentralDifferences<SoftMinQuadraticValueFunctionApprox<2, 1>>(21);
  ExpectAllDerivativesMatchCentralDifferences<SoftMinQuadraticValueFunctionApprox<2, 3>>(22);
}

TEST(SoftMinQuadraticValueFunctionApproxTest, DerivativesMatchCentralDifferencesInThreeDimensions) {
  ExpectAllDerivativesMatchCentralDifferences<SoftMinQuadraticValueFunctionApprox<3, 1>>(31);
  ExpectAllDerivativesMatchCentralDifferences<SoftMinQuadraticValueFunctionApprox<3, 3>>(32);
}

// ---------------------------------------------------------------------------------------------
// Batched evaluation
// ---------------------------------------------------------------------------------------------

TEST(SoftMinQuadraticValueFunctionApproxTest, BatchedEvaluationAgreesWithThePointwiseOne) {
  constexpr int kBatch = 8;
  const auto model = Space3x3::Make(SpreadParameters<Space3x3>(41, 1.0), kSharpness, kDiagonalOffset);
  ASSERT_TRUE(model.has_value()) << model.error();

  ValueFunctionStateBatch<Eigen::Vector3d, kBatch> states;
  for (Eigen::Index column = 0; column < kBatch; ++column) {
    states.col(column) = SpreadState<Space3x3>(static_cast<std::uint64_t>(column) + 50, 2.5);
  }

  ValueFunctionValueBatch<Eigen::Vector3d, kBatch> values;
  ValueFunctionGradientBatch<Eigen::Vector3d, kBatch> gradients;
  ValueFunctionHessianBatch<Eigen::Vector3d, kBatch> hessians;
  model->ValuesGradientsAndHessians<kBatch>(states, values, gradients, hessians);

  ValueFunctionValueBatch<Eigen::Vector3d, kBatch> parameter_gradient_values;
  ValueFunctionParameterGradientBatch<Eigen::Vector3d, Space3x3::kNumParameters, kBatch> parameter_gradients;
  model->ValuesAndParameterGradients<kBatch>(states, parameter_gradient_values, parameter_gradients);

  for (Eigen::Index column = 0; column < kBatch; ++column) {
    const Eigen::Vector3d state = states.col(column);

    EXPECT_NEAR(values[column], (*model)(state), 1e-12) << "sample " << column;
    EXPECT_EQ(values[column], parameter_gradient_values[column]);
    EXPECT_LT((gradients.col(column) - model->Gradient(state)).norm(), 1e-12) << "sample " << column;
    EXPECT_LT((hessians.block<3, 3>(0, column * 3) - model->Hessian(state)).norm(), 1e-12) << "sample " << column;
    EXPECT_LT((parameter_gradients.col(column) - model->ParameterGradient(state)).norm(), 1e-12) << "sample " << column;
  }
}

// ---------------------------------------------------------------------------------------------
// The soft minimum itself
// ---------------------------------------------------------------------------------------------

TEST(SoftMinQuadraticValueFunctionApproxTest, LargeSharpnessApproachesTheHardMinimum) {
  // Two wells in one dimension, at -2 and +2, of different depths. Between them the soft minimum
  // is smooth; as beta grows it collapses onto the lower of the two.
  using Model = SoftMinQuadraticValueFunctionApprox<1, 2>;
  using State = Model::State;

  // Two components, each (L, b, c): L = softplus(0) + eps = log(2) + eps, centers at -2 and +2,
  // and the right-hand well one unit deeper.
  Model::ParameterVector parameters;
  parameters << 0.0, -2.0, 0.0, 0.0, 2.0, -1.0;

  const auto hard_minimum = [](double position) noexcept -> double {
    const double slope = std::numbers::ln2 + kDiagonalOffset;
    const double left = (slope * (position + 2.0)) * (slope * (position + 2.0));
    const double right = ((slope * (position - 2.0)) * (slope * (position - 2.0))) - 1.0;
    return std::min(left, right);
  };

  for (const double sharpness : {1.0, 10.0, 100.0}) {
    const auto model = Model::Make(parameters, sharpness, kDiagonalOffset);
    ASSERT_TRUE(model.has_value()) << model.error();

    // Away from the ridge the soft minimum is within log(2)/beta of the hard one, and never above
    // it: a soft minimum is a lower bound on the hard minimum.
    for (const double position : {-4.0, -3.0, 3.0, 4.0}) {
      const double value = (*model)(State::Constant(position));
      const double reference = hard_minimum(position);
      EXPECT_LE(value, reference + 1e-12) << "beta " << sharpness << " at " << position;
      EXPECT_GT(value, reference - (std::numbers::ln2 / sharpness) - 1e-12)
          << "beta " << sharpness << " at " << position;
    }
  }
}

TEST(SoftMinQuadraticValueFunctionApproxTest, WidelySeparatedComponentsStayFinite) {
  // A component whose quadratic is hundreds above the smallest would overflow a softmax written
  // without the max shift. The value should simply be the smallest component, to within the
  // log(1 + tiny)/beta that the others contribute.
  using Model = SoftMinQuadraticValueFunctionApprox<1, 3>;
  using State = Model::State;

  Model::ParameterVector parameters;
  parameters << 0.0, 0.0, 0.0, 0.0, 0.0, 500.0, 0.0, 0.0, 1000.0;
  const auto model = Model::Make(parameters, 10.0, kDiagonalOffset);
  ASSERT_TRUE(model.has_value()) << model.error();

  const State state = State::Constant(0.5);
  const double slope = std::numbers::ln2 + kDiagonalOffset;
  const double smallest = (slope * 0.5) * (slope * 0.5);

  EXPECT_TRUE(std::isfinite((*model)(state)));
  EXPECT_TRUE(model->Gradient(state).allFinite());
  EXPECT_TRUE(model->Hessian(state).allFinite());
  EXPECT_TRUE(model->ParameterGradient(state).allFinite());
  EXPECT_NEAR((*model)(state), smallest, 1e-9);
}

TEST(SoftMinQuadraticValueFunctionApproxTest, TheHessianIsSymmetricAndCanBeIndefinite) {
  // On the ridge between two separated wells the softmax-covariance term dominates, and the
  // curvature of a soft minimum of convex quadratics goes negative along the line joining them.
  // A model that dropped the -beta covariance term would report a positive definite Hessian here.
  using Model = SoftMinQuadraticValueFunctionApprox<2, 2>;

  Eigen::Matrix2d metric = Eigen::Matrix2d::Identity();
  const auto first = Model::QuadraticParameters(metric, Eigen::Vector2d(-3.0, 0.0), 0.0, 4.0, kDiagonalOffset);
  ASSERT_TRUE(first.has_value()) << first.error();
  const auto second = Model::QuadraticParameters(metric, Eigen::Vector2d(3.0, 0.0), 0.0, 4.0, kDiagonalOffset);
  ASSERT_TRUE(second.has_value()) << second.error();

  // QuadraticParameters fills every component alike, so splice the two halves together.
  Model::ParameterVector parameters = *first;
  parameters.tail<Model::kParametersPerComponent>() = second->tail<Model::kParametersPerComponent>();

  const auto model = Model::Make(parameters, 4.0, kDiagonalOffset);
  ASSERT_TRUE(model.has_value()) << model.error();

  const Eigen::Vector2d ridge(0.0, 0.0);
  const Eigen::Matrix2d hessian = model->Hessian(ridge);

  EXPECT_LT((hessian - hessian.transpose()).norm(), 1e-15);

  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(hessian);
  ASSERT_EQ(solver.info(), Eigen::Success);
  EXPECT_LT(solver.eigenvalues().minCoeff(), 0.0);
}

}  // namespace
}  // namespace fbsde_traj_opt
