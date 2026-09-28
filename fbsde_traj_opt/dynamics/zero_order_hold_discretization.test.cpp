// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#include "fbsde_traj_opt/dynamics/zero_order_hold_discretization.hpp"

#include <cmath>
#include <limits>
#include <numbers>

#include <Eigen/Core>
#include <gtest/gtest.h>

namespace fbsde_traj_opt {
namespace {

TEST(ZeroOrderHoldDiscretizationTest, DoubleIntegratorMatchesItsClosedForm) {
  constexpr double kTimeStep = 0.05;

  Eigen::Matrix2d continuous_state_mat;
  continuous_state_mat << 0.0, 1.0,  //
      0.0, 0.0;
  const Eigen::Vector2d continuous_control_mat(0.0, 1.0);

  const auto discretized = ZeroOrderHoldDiscretization<2, 1>(continuous_state_mat, continuous_control_mat, kTimeStep);
  ASSERT_TRUE(discretized.has_value()) << discretized.error();

  // A_d = [[1, h], [0, 1]], so the increment A_d - I is [[0, h], [0, 0]]; B_d = [h^2/2, h].
  Eigen::Matrix2d expected_state_drift;
  expected_state_drift << 0.0, kTimeStep,  //
      0.0, 0.0;
  const Eigen::Vector2d expected_control_drift(0.5 * kTimeStep * kTimeStep, kTimeStep);

  EXPECT_TRUE(discretized->state_drift_mat.isApprox(expected_state_drift, 1e-12));
  EXPECT_TRUE(discretized->control_drift_mat.isApprox(expected_control_drift, 1e-12));
}

TEST(ZeroOrderHoldDiscretizationTest, ScalarDecayMatchesItsClosedForm) {
  constexpr double kRate = 2.5;
  constexpr double kGain = 3.0;
  constexpr double kTimeStep = 0.2;

  const Eigen::Matrix<double, 1, 1> continuous_state_mat = Eigen::Matrix<double, 1, 1>::Constant(-kRate);
  const Eigen::Matrix<double, 1, 1> continuous_control_mat = Eigen::Matrix<double, 1, 1>::Constant(kGain);

  const auto discretized = ZeroOrderHoldDiscretization<1, 1>(continuous_state_mat, continuous_control_mat, kTimeStep);
  ASSERT_TRUE(discretized.has_value()) << discretized.error();

  // A_d = exp(-a h); B_d = b (1 - exp(-a h)) / a.
  const double decay = std::exp(-kRate * kTimeStep);
  EXPECT_NEAR(discretized->state_drift_mat(0, 0), decay - 1.0, 1e-12);
  EXPECT_NEAR(discretized->control_drift_mat(0, 0), kGain * (1.0 - decay) / kRate, 1e-12);
}

// The reason this helper exists. An undamped oscillator's exact discrete map is a rotation, which
// conserves energy exactly; an explicit Euler step is a rotation scaled by sqrt(1 + (w h)^2) and
// adds energy every step.
TEST(ZeroOrderHoldDiscretizationTest, UndampedOscillatorConservesEnergyOverALongRollout) {
  // The natural frequency of the two-mass-spring benchmark's flexible mode.
  constexpr double kFrequency = std::numbers::sqrt2;
  constexpr double kTimeStep = 0.1;
  constexpr int kStepCount = 1000;

  Eigen::Matrix2d continuous_state_mat;
  continuous_state_mat << 0.0, 1.0,  //
      -kFrequency * kFrequency, 0.0;

  const Eigen::Vector2d continuous_control_mat = Eigen::Vector2d::Zero();

  const auto discretized = ZeroOrderHoldDiscretization<2, 1>(continuous_state_mat, continuous_control_mat, kTimeStep);
  ASSERT_TRUE(discretized.has_value()) << discretized.error();

  const Eigen::Matrix2d transition = Eigen::Matrix2d::Identity() + discretized->state_drift_mat;

  Eigen::Vector2d state(1.0, 0.0);
  const auto energy = [](const Eigen::Vector2d& value) noexcept {
    return (kFrequency * kFrequency * value.x() * value.x()) + (value.y() * value.y());
  };
  const double initial_energy = energy(state);

  for (int step = 0; step < kStepCount; ++step) {
    state = transition * state;
  }

  EXPECT_NEAR(energy(state), initial_energy, 1e-9 * initial_energy);

  // The same rollout under an explicit Euler step, for contrast: energy grows by more than half.
  const Eigen::Matrix2d euler_transition = Eigen::Matrix2d::Identity() + (continuous_state_mat * kTimeStep);
  Eigen::Vector2d euler_state(1.0, 0.0);
  for (int step = 0; step < kStepCount; ++step) {
    euler_state = euler_transition * euler_state;
  }
  EXPECT_GT(energy(euler_state), 1.5 * initial_energy);
}

TEST(ZeroOrderHoldDiscretizationTest, SingularSystemMatrixIsHandled) {
  // A pure integrator: A_c is singular, so the closed form B_d = A_c^-1 (A_d - I) B_c does not
  // exist and the augmented-matrix identity is what makes this case work at all.
  const Eigen::Matrix<double, 1, 1> continuous_state_mat = Eigen::Matrix<double, 1, 1>::Zero();
  const Eigen::Matrix<double, 1, 1> continuous_control_mat = Eigen::Matrix<double, 1, 1>::Constant(2.0);

  const auto discretized = ZeroOrderHoldDiscretization<1, 1>(continuous_state_mat, continuous_control_mat, 0.5);
  ASSERT_TRUE(discretized.has_value()) << discretized.error();

  EXPECT_NEAR(discretized->state_drift_mat(0, 0), 0.0, 1e-15);
  EXPECT_NEAR(discretized->control_drift_mat(0, 0), 1.0, 1e-12);
}

TEST(ZeroOrderHoldDiscretizationTest, ZeroAndNegativeTimeStepsAreRejected) {
  const Eigen::Matrix2d state_mat = Eigen::Matrix2d::Identity();
  const Eigen::Vector2d control_mat = Eigen::Vector2d::Ones();

  // The extra parentheses keep the preprocessor from reading the template argument list's comma
  // as a macro argument separator.
  EXPECT_FALSE((ZeroOrderHoldDiscretization<2, 1>(state_mat, control_mat, 0.0).has_value()));
  EXPECT_FALSE((ZeroOrderHoldDiscretization<2, 1>(state_mat, control_mat, -0.1).has_value()));
  EXPECT_FALSE((
      ZeroOrderHoldDiscretization<2, 1>(state_mat, control_mat, std::numeric_limits<double>::quiet_NaN()).has_value()));
}

TEST(ZeroOrderHoldDiscretizationTest, NonFiniteSystemMatricesAreRejected) {
  Eigen::Matrix2d state_mat = Eigen::Matrix2d::Identity();
  state_mat(0, 1) = std::numeric_limits<double>::infinity();
  const Eigen::Vector2d control_mat = Eigen::Vector2d::Ones();

  EXPECT_FALSE((ZeroOrderHoldDiscretization<2, 1>(state_mat, control_mat, 0.1).has_value()));
}

}  // namespace
}  // namespace fbsde_traj_opt
