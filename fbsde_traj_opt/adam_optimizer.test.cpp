// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#include "fbsde_traj_opt/adam_optimizer.hpp"

#include <cstddef>
#include <limits>

#include <Eigen/Core>
#include <gtest/gtest.h>

namespace fbsde_traj_opt {
namespace {

using Optimizer3 = AdamOptimizer<3>;
using Vector3 = Optimizer3::ParameterVector;

TEST(AdamOptimizerTest, MakeSucceedsWithTheDefaultHyperparameters) {
  EXPECT_TRUE(Optimizer3::Make(1e-2).has_value());
}

TEST(AdamOptimizerTest, MakeFailsWhenTheLearningRateIsNotPositive) {
  EXPECT_FALSE(Optimizer3::Make(0.0).has_value());
  EXPECT_FALSE(Optimizer3::Make(-1e-3).has_value());
  EXPECT_FALSE(Optimizer3::Make(std::numeric_limits<double>::quiet_NaN()).has_value());
}

TEST(AdamOptimizerTest, MakeFailsWhenEpsilonIsNotPositive) {
  EXPECT_FALSE(Optimizer3::Make(1e-2, 0.9, 0.999, 0.0).has_value());
  EXPECT_FALSE(Optimizer3::Make(1e-2, 0.9, 0.999, -1e-8).has_value());
}

TEST(AdamOptimizerTest, MakeFailsWhenADecayLiesOutsideTheUnitInterval) {
  // A decay of exactly one freezes its moment at zero forever, so it is rejected rather than
  // treated as "very slow".
  EXPECT_FALSE(Optimizer3::Make(1e-2, 1.0, 0.999).has_value());
  EXPECT_FALSE(Optimizer3::Make(1e-2, 0.9, 1.0).has_value());
  EXPECT_FALSE(Optimizer3::Make(1e-2, -0.1, 0.999).has_value());
  EXPECT_FALSE(Optimizer3::Make(1e-2, 0.9, -0.1).has_value());
}

TEST(AdamOptimizerTest, TheFirstStepIsTheLearningRateTimesTheSignOfTheGradient) {
  // Bias correction makes the first step's magnitude the learning rate in every coordinate,
  // whatever the gradient's scale, up to epsilon. Without the correction it would be roughly
  // (1 - beta1) / sqrt(1 - beta2) times that, which is an order of magnitude smaller.
  constexpr double kLearningRate = 1e-2;
  auto optimizer = Optimizer3::Make(kLearningRate);
  ASSERT_TRUE(optimizer.has_value()) << optimizer.error();

  const Vector3 gradient(1e-4, -7.0, 250.0);
  const Vector3 step = optimizer->Step(gradient);

  EXPECT_NEAR(step[0], -kLearningRate, 1e-6);
  EXPECT_NEAR(step[1], kLearningRate, 1e-9);
  EXPECT_NEAR(step[2], -kLearningRate, 1e-9);
  EXPECT_EQ(optimizer->StepCount(), 1U);
}

TEST(AdamOptimizerTest, AZeroGradientProducesAZeroStep) {
  auto optimizer = Optimizer3::Make(1e-2);
  ASSERT_TRUE(optimizer.has_value()) << optimizer.error();

  const Vector3 step = optimizer->Step(Vector3::Zero());

  EXPECT_EQ(step, Vector3::Zero());
}

TEST(AdamOptimizerTest, DescendsToTheMinimumOfAnIllConditionedQuadratic) {
  // f(p) = 1/2 (p - c)' A (p - c) with A = diag(1, 100, 1e-2): three orders of magnitude between
  // the stiffest and softest directions, which is the case a plain gradient step handles worst
  // and Adam's per-coordinate scaling handles well.
  const Vector3 curvature(1.0, 100.0, 1e-2);
  const Vector3 minimizer(2.0, -1.0, 5.0);

  auto optimizer = Optimizer3::Make(1e-2);
  ASSERT_TRUE(optimizer.has_value()) << optimizer.error();

  Vector3 parameters = Vector3::Zero();
  for (std::size_t iteration = 0; iteration < 20000; ++iteration) {
    const Vector3 gradient = curvature.cwiseProduct(parameters - minimizer);
    parameters += optimizer->Step(gradient);
  }

  EXPECT_LT((parameters - minimizer).norm(), 1e-3);
  EXPECT_EQ(optimizer->StepCount(), 20000U);
}

TEST(AdamOptimizerTest, ResetRestoresFirstStepBehavior) {
  auto optimizer = Optimizer3::Make(1e-2);
  ASSERT_TRUE(optimizer.has_value()) << optimizer.error();

  const Vector3 gradient(1.0, -2.0, 3.0);
  const Vector3 first = optimizer->Step(gradient);
  for (std::size_t iteration = 0; iteration < 10; ++iteration) {
    static_cast<void>(optimizer->Step(Vector3(-4.0, 5.0, -6.0)));
  }

  optimizer->Reset();
  EXPECT_EQ(optimizer->StepCount(), 0U);

  const Vector3 again = optimizer->Step(gradient);

  EXPECT_LT((again - first).norm(), 1e-15);
  EXPECT_EQ(optimizer->StepCount(), 1U);
}

}  // namespace
}  // namespace fbsde_traj_opt
