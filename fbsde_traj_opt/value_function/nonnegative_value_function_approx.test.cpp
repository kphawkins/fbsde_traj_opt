// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#include "fbsde_traj_opt/value_function/nonnegative_value_function_approx.hpp"

#include <array>
#include <cmath>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "fbsde_traj_opt/utils/result.hpp"
#include "fbsde_traj_opt/value_function/soft_min_quadratic_value_function_approx.hpp"
#include "fbsde_traj_opt/value_function/value_function_approx_concepts.hpp"

namespace fbsde_traj_opt {
namespace {

using Inner = SoftMinQuadraticValueFunctionApprox<2, 2>;
using Model = NonnegativeValueFunctionApprox<Inner>;
using State = Eigen::Vector2d;

static_assert(BatchedParameterizedValueFunctionApprox<Model, State, 8>);

constexpr double kSoftness = 0.5;

// Two bowls whose constants are negative, so the inner model goes below zero near their centers
// and is positive far from them: both branches of the softplus are exercised.
auto MakeModel() -> Model {
  const auto parameters =
      Inner::QuadraticParametersAt(Eigen::Matrix2d::Identity(), {State(-1.0, 0.0), State(1.0, 0.5)}, -2.0, 1.0, 1e-3);
  EXPECT_TRUE(parameters.has_value());
  const auto inner = Inner::Make(*parameters, 1.0, 1e-3);
  EXPECT_TRUE(inner.has_value());
  const auto model = Model::Make(*inner, kSoftness);
  EXPECT_TRUE(model.has_value());
  return *model;
}

const std::array<State, 5> kStates{
    State(-1.0, 0.0), State(0.0, 0.0), State(1.5, -0.5), State(3.0, 2.0), State(-2.5, 1.0)};

TEST(NonnegativeValueFunctionApproxTest, NeverGoesNegativeAndFollowsTheInnerModelWhereItIsPositive) {
  const Model model = MakeModel();
  for (const State& state : kStates) {
    EXPECT_GT(model(state), 0.0);
    const double inner = model.inner()(state);
    if (inner > 10.0 * kSoftness) {
      EXPECT_NEAR(model(state), inner, 1e-3 * inner);
    }
  }
  EXPECT_LT(model.inner()(State(-1.0, 0.0)), 0.0);
}

TEST(NonnegativeValueFunctionApproxTest, DerivativesMatchFiniteDifferences) {
  Model model = MakeModel();
  constexpr double kStep = 1e-6;
  for (const State& state : kStates) {
    const State gradient = model.Gradient(state);
    const Eigen::Matrix2d hessian = model.Hessian(state);
    for (int axis = 0; axis < 2; ++axis) {
      const State step = kStep * State::Unit(axis);
      EXPECT_NEAR(gradient[axis], (model(state + step) - model(state - step)) / (2.0 * kStep), 1e-5);
      const State difference = (model.Gradient(state + step) - model.Gradient(state - step)) / (2.0 * kStep);
      EXPECT_LT((hessian.col(axis) - difference).norm(), 1e-4);
    }
    const Model::ParameterVector parameter_gradient = model.ParameterGradient(state);
    const Model::ParameterVector parameters = model.Parameters();
    for (int index = 0; index < Model::kNumParameters; ++index) {
      const Model::ParameterVector step = kStep * Model::ParameterVector::Unit(index);
      ASSERT_TRUE(model.SetParameters(parameters + step).has_value());
      const double plus = model(state);
      ASSERT_TRUE(model.SetParameters(parameters - step).has_value());
      const double minus = model(state);
      EXPECT_NEAR(parameter_gradient[index], (plus - minus) / (2.0 * kStep), 1e-5);
    }
    ASSERT_TRUE(model.SetParameters(parameters).has_value());
  }
}

TEST(NonnegativeValueFunctionApproxTest, BatchedEvaluationMatchesPointwise) {
  const Model model = MakeModel();
  ValueFunctionStateBatch<State, 5> states;
  for (Eigen::Index column = 0; column < 5; ++column) {
    states.col(column) = kStates[static_cast<std::size_t>(column)];
  }
  ValueFunctionValueBatch<State, 5> values;
  ValueFunctionGradientBatch<State, 5> gradients;
  ValueFunctionHessianBatch<State, 5> hessians;
  ValueFunctionParameterGradientBatch<State, Model::kNumParameters, 5> parameter_gradients;
  model.ValuesGradientsAndHessians<5>(states, values, gradients, hessians);
  for (Eigen::Index column = 0; column < 5; ++column) {
    const State state = states.col(column);
    EXPECT_NEAR(values[column], model(state), 1e-12);
    EXPECT_LT((State(gradients.col(column)) - model.Gradient(state)).norm(), 1e-12);
    EXPECT_LT((Eigen::Matrix2d(hessians.block<2, 2>(0, column * 2)) - model.Hessian(state)).norm(), 1e-12);
  }
  model.ValuesAndParameterGradients<5>(states, values, parameter_gradients);
  for (Eigen::Index column = 0; column < 5; ++column) {
    EXPECT_LT(
        (Model::ParameterVector(parameter_gradients.col(column)) - model.ParameterGradient(states.col(column))).norm(),
        1e-12);
  }
}

TEST(NonnegativeValueFunctionApproxTest, MakeRejectsANonPositiveSoftness) {
  const Model model = MakeModel();
  EXPECT_FALSE(Model::Make(model.inner(), 0.0).has_value());
  EXPECT_FALSE(Model::Make(model.inner(), -1.0).has_value());
}

}  // namespace
}  // namespace fbsde_traj_opt
