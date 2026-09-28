// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#include "fbsde_traj_opt/polynomial_value_function_approx.hpp"

#include <cmath>
#include <cstddef>
#include <limits>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "fbsde_traj_opt/utils/result.hpp"
#include "fbsde_traj_opt/value_function_approx_concepts.hpp"

namespace fbsde_traj_opt {
namespace {

using Cubic3 = PolynomialValueFunctionApprox<3, 3>;
using Quadratic2 = PolynomialValueFunctionApprox<2, 2>;

static_assert(BatchedParameterizedValueFunctionApprox<Cubic3, Eigen::Vector3d, 8>);
static_assert(Cubic3::kNumParameters == 20);
static_assert(Quadratic2::kNumParameters == 6);
static_assert(PolynomialValueFunctionApprox<4, 2>::kNumParameters == 15);
static_assert(PolynomialValueFunctionApprox<4, 4>::kNumParameters == 70);

// A cubic with every coefficient nonzero and distinct, on a normalization that is neither
// centered nor unit, so that a chain-rule mistake on either cannot hide.
auto MakeCubic() -> Cubic3 {
  Cubic3::ParameterVector parameters;
  for (Eigen::Index index = 0; index < parameters.size(); ++index) {
    parameters[index] = 0.3 * std::sin(1.7 * static_cast<double>(index) + 0.4);
  }
  const Result<Cubic3> model =
      Cubic3::Make(parameters, Eigen::Vector3d(0.5, -1.0, 2.0), Eigen::Vector3d(1.5, 0.7, 3.0));
  EXPECT_TRUE(model.has_value());
  return *model;
}

TEST(PolynomialValueFunctionApproxTest, GradientMatchesCentralDifferences) {
  const Cubic3 model = MakeCubic();
  const Eigen::Vector3d state(0.9, -0.4, 1.1);
  constexpr double kStep = 1e-6;

  const Eigen::Vector3d gradient = model.Gradient(state);
  for (Eigen::Index axis = 0; axis < 3; ++axis) {
    const Eigen::Vector3d offset = kStep * Eigen::Vector3d::Unit(axis);
    const double difference = (model(state + offset) - model(state - offset)) / (2.0 * kStep);
    EXPECT_NEAR(gradient[axis], difference, 1e-7) << "axis " << axis;
  }
}

TEST(PolynomialValueFunctionApproxTest, HessianMatchesCentralDifferencesOfGradient) {
  const Cubic3 model = MakeCubic();
  const Eigen::Vector3d state(-0.2, 0.3, 2.5);
  constexpr double kStep = 1e-6;

  const Eigen::Matrix3d hessian = model.Hessian(state);
  for (Eigen::Index axis = 0; axis < 3; ++axis) {
    const Eigen::Vector3d offset = kStep * Eigen::Vector3d::Unit(axis);
    const Eigen::Vector3d difference =
        (model.Gradient(state + offset) - model.Gradient(state - offset)) / (2.0 * kStep);
    EXPECT_TRUE(hessian.col(axis).isApprox(difference, 1e-6)) << "axis " << axis;
  }
  EXPECT_TRUE(hessian.isApprox(hessian.transpose()));
}

TEST(PolynomialValueFunctionApproxTest, ValueGradientAndHessianAgreesWithSeparateCalls) {
  const Cubic3 model = MakeCubic();
  const Eigen::Vector3d state(1.3, 0.1, -0.6);

  double value = 0.0;
  Eigen::Vector3d gradient;
  Eigen::Matrix3d hessian;
  model.ValueGradientAndHessian(state, value, gradient, hessian);

  EXPECT_DOUBLE_EQ(value, model(state));
  EXPECT_TRUE(gradient.isApprox(model.Gradient(state)));
  EXPECT_TRUE(hessian.isApprox(model.Hessian(state)));
}

TEST(PolynomialValueFunctionApproxTest, BatchEvaluationsMatchPointwise) {
  const Cubic3 model = MakeCubic();
  Eigen::Matrix<double, 3, 4> states;
  states << 0.1, 0.5, -1.0, 2.0,  //
      -0.3, 0.7, 0.0, 1.0,        //
      1.0, 1.5, 2.5, -0.5;

  Eigen::Matrix<double, 1, 4> values;
  Eigen::Matrix<double, 3, 4> gradients;
  Eigen::Matrix<double, 3, 12> hessians;
  model.ValuesGradientsAndHessians<4>(states, values, gradients, hessians);

  Eigen::Matrix<double, 1, 4> parameter_values;
  Eigen::Matrix<double, Cubic3::kNumParameters, 4> parameter_gradients;
  model.ValuesAndParameterGradients<4>(states, parameter_values, parameter_gradients);

  for (std::size_t sample = 0; sample < 4; ++sample) {
    const auto column = static_cast<Eigen::Index>(sample);
    const Eigen::Vector3d state = states.col(column);
    EXPECT_DOUBLE_EQ(values[column], model(state));
    EXPECT_DOUBLE_EQ(parameter_values[column], model(state));
    EXPECT_TRUE(gradients.col(column).isApprox(model.Gradient(state)));
    const Eigen::Matrix3d batched_hessian = hessians.block<3, 3>(0, column * 3);
    EXPECT_TRUE(batched_hessian.isApprox(model.Hessian(state)));
    // The model is linear in its coefficients, so the parameter gradient reproduces the value.
    EXPECT_NEAR(parameter_gradients.col(column).dot(model.Parameters()), model(state), 1e-12);
  }
}

TEST(PolynomialValueFunctionApproxTest, QuadraticParametersReproduceTheQuadraticExactly) {
  const Result<Quadratic2> base =
      Quadratic2::Make(Quadratic2::ParameterVector::Zero(), Eigen::Vector2d(1.0, -2.0), Eigen::Vector2d(3.0, 0.5));
  ASSERT_TRUE(base.has_value()) << base.error();
  Eigen::Matrix2d metric;
  metric << 2.0, 0.6,  //
      0.6, 1.0;

  Quadratic2 model = *base;
  ASSERT_TRUE(model.SetParameters(model.QuadraticParameters(metric, 0.25)).has_value());

  for (const Eigen::Vector2d& state :
       {Eigen::Vector2d(0.0, 0.0), Eigen::Vector2d(1.5, -0.3), Eigen::Vector2d(-4.0, 2.0)}) {
    EXPECT_NEAR(model(state), state.dot(metric * state) + 0.25, 1e-12);
    EXPECT_TRUE(model.Gradient(state).isApprox(2.0 * metric * state, 1e-12) || state.isZero());
    EXPECT_TRUE(model.Hessian(state).isApprox(2.0 * metric, 1e-12));
  }
}

TEST(PolynomialValueFunctionApproxTest, LeastSquaresRecoversAPolynomialInTheBasis) {
  const Cubic3 truth = MakeCubic();
  constexpr int kSamples = 64;
  Eigen::Matrix<double, 3, kSamples> states;
  Eigen::Matrix<double, 1, kSamples> targets;
  for (int sample = 0; sample < kSamples; ++sample) {
    const auto value = static_cast<double>(sample);
    states.col(sample) = Eigen::Vector3d(std::sin(value), std::cos(1.3 * value) * 2.0, 1.0 + std::sin(0.7 * value));
    targets[sample] = truth(Eigen::Vector3d(states.col(sample)));
  }

  const Result<Cubic3> blank = Cubic3::Make(Cubic3::ParameterVector::Zero(), truth.Center(), truth.Scale());
  ASSERT_TRUE(blank.has_value()) << blank.error();
  Cubic3 fitted = *blank;
  const Result<> fit =
      fitted.FitWeightedLeastSquares<kSamples>(states, targets, Eigen::Matrix<double, 1, kSamples>::Ones(), 0.0);
  ASSERT_TRUE(fit.has_value()) << fit.error();

  EXPECT_TRUE(fitted.Parameters().isApprox(truth.Parameters(), 1e-8));
}

TEST(PolynomialValueFunctionApproxTest, ZeroWeightSamplesDoNotInfluenceTheFit) {
  constexpr int kSamples = 12;
  Eigen::Matrix<double, 2, kSamples> states;
  Eigen::Matrix<double, 1, kSamples> targets;
  Eigen::Matrix<double, 1, kSamples> weights;
  for (int sample = 0; sample < kSamples; ++sample) {
    const auto value = static_cast<double>(sample);
    states.col(sample) = Eigen::Vector2d(std::sin(value), std::cos(0.9 * value));
    const Eigen::Vector2d state = states.col(sample);
    // The last two samples lie wildly off the quadratic, and carry no weight.
    const bool outlier = sample >= kSamples - 2;
    targets[sample] = outlier ? 1e6 : state.squaredNorm();
    weights[sample] = outlier ? 0.0 : 1.0;
  }

  const Result<Quadratic2> blank =
      Quadratic2::Make(Quadratic2::ParameterVector::Zero(), Eigen::Vector2d::Zero(), Eigen::Vector2d::Ones());
  ASSERT_TRUE(blank.has_value()) << blank.error();
  Quadratic2 fitted = *blank;
  ASSERT_TRUE(fitted.FitWeightedLeastSquares<kSamples>(states, targets, weights, 0.0).has_value());

  EXPECT_NEAR(fitted(Eigen::Vector2d(0.5, -0.5)), 0.5, 1e-9);
}

TEST(PolynomialValueFunctionApproxTest, MakeRejectsNonPositiveScaleAndNonFiniteInput) {
  const Quadratic2::ParameterVector zero = Quadratic2::ParameterVector::Zero();
  EXPECT_FALSE(Quadratic2::Make(zero, Eigen::Vector2d::Zero(), Eigen::Vector2d(1.0, 0.0)).has_value());
  EXPECT_FALSE(
      Quadratic2::Make(zero, Eigen::Vector2d(std::numeric_limits<double>::quiet_NaN(), 0.0), Eigen::Vector2d::Ones())
          .has_value());
  Quadratic2::ParameterVector bad = zero;
  bad[0] = std::numeric_limits<double>::infinity();
  EXPECT_FALSE(Quadratic2::Make(bad, Eigen::Vector2d::Zero(), Eigen::Vector2d::Ones()).has_value());
}

TEST(PolynomialValueFunctionApproxTest, FitRejectsNegativeWeights) {
  const Result<Quadratic2> blank =
      Quadratic2::Make(Quadratic2::ParameterVector::Zero(), Eigen::Vector2d::Zero(), Eigen::Vector2d::Ones());
  ASSERT_TRUE(blank.has_value()) << blank.error();
  Quadratic2 fitted = *blank;
  Eigen::Matrix<double, 1, 8> weights = Eigen::Matrix<double, 1, 8>::Ones();
  weights[3] = -1.0;
  EXPECT_FALSE(fitted
                   .FitWeightedLeastSquares<8>(
                       Eigen::Matrix<double, 2, 8>::Random(), Eigen::Matrix<double, 1, 8>::Zero(), weights, 0.0)
                   .has_value());
}

}  // namespace
}  // namespace fbsde_traj_opt
