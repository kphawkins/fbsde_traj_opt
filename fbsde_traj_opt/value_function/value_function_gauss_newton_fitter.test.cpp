// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#include "fbsde_traj_opt/value_function/value_function_gauss_newton_fitter.hpp"

#include <array>
#include <cmath>
#include <cstddef>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "fbsde_traj_opt/utils/result.hpp"
#include "fbsde_traj_opt/value_function/polynomial_value_function_approx.hpp"
#include "fbsde_traj_opt/value_function/soft_min_quadratic_value_function_approx.hpp"
#include "fbsde_traj_opt/value_function/value_function_approx_concepts.hpp"

namespace fbsde_traj_opt {
namespace {

constexpr int kSamples = 256;

using Polynomial = PolynomialValueFunctionApprox<2, 2>;
using SoftMin = SoftMinQuadraticValueFunctionApprox<2, 3>;
using States = ValueFunctionStateBatch<Eigen::Vector2d, kSamples>;
using Values = ValueFunctionValueBatch<Eigen::Vector2d, kSamples>;

constexpr double kSharpness = 1.0;
constexpr double kDiagonalOffset = 1e-3;

// A value function no quadratic reproduces exactly, so every fit below has something to do.
auto Targets(const States& states) -> Values {
  Values targets;
  for (Eigen::Index column = 0; column < kSamples; ++column) {
    const Eigen::Vector2d x = states.col(column);
    targets[column] = (2.0 * x[0] * x[0]) + (0.5 * x[1] * x[1]) + std::abs(x[0] + x[1]) + 1.0;
  }
  return targets;
}

auto MakePolynomial() -> Polynomial {
  const Result<Polynomial> model =
      Polynomial::Make(Polynomial::ParameterVector::Zero(), Eigen::Vector2d::Zero(), Eigen::Vector2d(2.0, 2.0));
  EXPECT_TRUE(model.has_value());
  return *model;
}

// Three bowls around the origin, far from the targets' shape.
auto MakeSoftMin() -> SoftMin {
  const auto parameters =
      SoftMin::QuadraticParametersAt(Eigen::Matrix2d::Identity(),
                                     {Eigen::Vector2d(-1.0, 0.0), Eigen::Vector2d(1.0, 0.0), Eigen::Vector2d(0.0, 1.0)},
                                     0.0,
                                     kSharpness,
                                     kDiagonalOffset);
  EXPECT_TRUE(parameters.has_value());
  const auto model = SoftMin::Make(*parameters, kSharpness, kDiagonalOffset);
  EXPECT_TRUE(model.has_value());
  return *model;
}

TEST(ValueFunctionGaussNewtonFitterTest, OneUndampedStepOnALinearModelIsTheLeastSquaresFit) {
  const States states = 2.0 * States::Random();
  const Values targets = Targets(states);
  const Values weights = (Values::Random().array() + 1.5).matrix();

  Polynomial exact = MakePolynomial();
  ASSERT_TRUE(exact.FitWeightedLeastSquares<kSamples>(states, targets, weights, 0.0).has_value());

  ValueFunctionGaussNewtonOptions<double> options;
  options.max_steps = 1;
  options.initial_damping = 0.0;
  const auto fitter = ValueFunctionGaussNewtonFitter<Polynomial, 64>::Make(options);
  ASSERT_TRUE(fitter.has_value());
  Polynomial fitted = MakePolynomial();
  ValueFunctionGaussNewtonReport<double> report;
  ASSERT_TRUE(fitter->Fit<kSamples>(fitted, states, targets, weights, report).has_value());

  EXPECT_EQ(report.steps_accepted, 1U);
  EXPECT_LT((fitted.Parameters() - exact.Parameters()).norm(), 1e-8 * (1.0 + exact.Parameters().norm()));
  EXPECT_LT(report.final_rms_residual, report.initial_rms_residual);
}

TEST(ValueFunctionGaussNewtonFitterTest, TheTrustRegionBoundsEveryStep) {
  const States states = 2.0 * States::Random();
  const Values targets = Targets(states);
  ValueFunctionGaussNewtonOptions<double> options;
  options.max_steps = 3;
  options.max_rms_value_change = 0.25;
  const auto fitter = ValueFunctionGaussNewtonFitter<Polynomial, 64>::Make(options);
  ASSERT_TRUE(fitter.has_value());

  Polynomial model = MakePolynomial();
  const Polynomial before = model;
  ValueFunctionGaussNewtonReport<double> report;
  ASSERT_TRUE(fitter->Fit<kSamples>(model, states, targets, Values::Ones(), report).has_value());

  EXPECT_EQ(report.steps_accepted, 3U);
  EXPECT_LE(report.max_step_rms_value_change, 0.25 * (1.0 + 1e-9));
  EXPECT_GT(report.max_step_rms_value_change, 0.2);
  // Three steps of at most 0.25 each: the model is linear, so the change is exactly J delta.
  Values moved;
  Values original;
  model.Values<kSamples>(states, moved);
  before.Values<kSamples>(states, original);
  EXPECT_LE((moved - original).norm() / std::sqrt(static_cast<double>(kSamples)), 0.75 + 1e-9);
  EXPECT_LT(report.final_rms_residual, report.initial_rms_residual);
}

TEST(ValueFunctionGaussNewtonFitterTest, FitsANonlinearModelMonotonically) {
  const States states = 2.0 * States::Random();
  const Values targets = Targets(states);
  ValueFunctionGaussNewtonOptions<double> options;
  options.max_steps = 3;
  const auto fitter = ValueFunctionGaussNewtonFitter<SoftMin, 64>::Make(options);
  ASSERT_TRUE(fitter.has_value());

  SoftMin model = MakeSoftMin();
  double previous = 0.0;
  for (int fit = 0; fit < 5; ++fit) {
    ValueFunctionGaussNewtonReport<double> report;
    ASSERT_TRUE(fitter->Fit<kSamples>(model, states, targets, Values::Ones(), report).has_value());
    EXPECT_LE(report.final_rms_residual, report.initial_rms_residual);
    if (fit == 0) {
      previous = report.initial_rms_residual;
    }
    EXPECT_LE(report.final_rms_residual, previous + 1e-12);
    previous = report.final_rms_residual;
  }
  // Starting from bowls of the wrong shape, fifteen steps get the fit well inside the targets' range.
  EXPECT_LT(previous, 0.5);
}

TEST(ValueFunctionGaussNewtonFitterTest, ZeroWeightSamplesDoNotMatter) {
  const States states = 2.0 * States::Random();
  Values targets = Targets(states);
  Values weights = Values::Ones();
  weights.rightCols<64>().setZero();
  Values corrupted = targets;
  corrupted.rightCols<64>().setConstant(1e6);

  ValueFunctionGaussNewtonOptions<double> options;
  options.initial_damping = 0.0;
  options.max_steps = 1;
  const auto fitter = ValueFunctionGaussNewtonFitter<Polynomial, 64>::Make(options);
  ASSERT_TRUE(fitter.has_value());
  Polynomial clean = MakePolynomial();
  Polynomial dirty = MakePolynomial();
  ValueFunctionGaussNewtonReport<double> report;
  ASSERT_TRUE(fitter->Fit<kSamples>(clean, states, targets, weights, report).has_value());
  ASSERT_TRUE(fitter->Fit<kSamples>(dirty, states, corrupted, weights, report).has_value());
  EXPECT_LT((clean.Parameters() - dirty.Parameters()).norm(), 1e-8);
}

TEST(ValueFunctionGaussNewtonFitterTest, RejectsBadOptionsAndWeights) {
  using Fitter = ValueFunctionGaussNewtonFitter<Polynomial, 64>;
  ValueFunctionGaussNewtonOptions<double> options;
  options.max_steps = 0;
  EXPECT_FALSE(Fitter::Make(options).has_value());
  options = {};
  options.initial_damping = -1.0;
  EXPECT_FALSE(Fitter::Make(options).has_value());
  options = {};
  options.damping_increase = 1.0;
  EXPECT_FALSE(Fitter::Make(options).has_value());
  options = {};
  options.damping_decrease = 1.0;
  EXPECT_FALSE(Fitter::Make(options).has_value());
  options = {};
  options.max_rms_value_change = 0.0;
  EXPECT_FALSE(Fitter::Make(options).has_value());

  const auto fitter = Fitter::Make({});
  ASSERT_TRUE(fitter.has_value());
  Polynomial model = MakePolynomial();
  const States states = States::Random();
  ValueFunctionGaussNewtonReport<double> report;
  EXPECT_FALSE(fitter->Fit<kSamples>(model, states, Targets(states), Values::Zero(), report).has_value());
  EXPECT_FALSE(fitter->Fit<kSamples>(model, states, Targets(states), -Values::Ones(), report).has_value());
}

}  // namespace
}  // namespace fbsde_traj_opt
