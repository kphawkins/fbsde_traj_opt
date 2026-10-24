// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#include "examples/l1_benchmark_models.hpp"

#include <cmath>
#include <cstddef>
#include <numbers>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "fbsde_traj_opt/utils/result.hpp"

namespace fbsde_traj_opt::examples {
namespace {

using Control = Eigen::Matrix<double, 1, 1>;

TEST(L1BenchmarkModelsTest, EveryBenchmarkBuildsWithANonEmptyRegionAndBasis) {
  const Result<L1DoubleIntegrator> integrator = MakeL1DoubleIntegrator();
  const Result<L1Pendulum> pendulum = MakeL1Pendulum();
  const Result<L1DoubleInvertedPendulum> double_pendulum = MakeL1DoubleInvertedPendulum();
  ASSERT_TRUE(integrator.has_value()) << integrator.error();
  ASSERT_TRUE(pendulum.has_value()) << pendulum.error();
  ASSERT_TRUE(double_pendulum.has_value()) << double_pendulum.error();
  EXPECT_TRUE((integrator->region_upper.array() > integrator->region_lower.array()).all());
  EXPECT_TRUE((pendulum->region_upper.array() > pendulum->region_lower.array()).all());
  EXPECT_TRUE((double_pendulum->region_upper.array() > double_pendulum->region_lower.array()).all());
  EXPECT_TRUE((pendulum->normalization_scale.array() > 0.0).all());
}

TEST(L1BenchmarkModelsTest, TheDoubleIntegratorStepIsExactKinematics) {
  const Result<L1DoubleIntegrator> integrator = MakeL1DoubleIntegrator();
  ASSERT_TRUE(integrator.has_value());
  const double dt = integrator->time_step;
  EXPECT_DOUBLE_EQ(dt, 4.0 / 64.0);
  const Eigen::Vector2d state(0.7, -0.4);
  const Eigen::Vector2d next = integrator->forward_model(0, state, Control(1.0), Eigen::Vector2d::Zero());
  EXPECT_NEAR(next[0], 0.7 - (0.4 * dt) + (0.5 * dt * dt), 1e-12);
  EXPECT_NEAR(next[1], -0.4 + dt, 1e-12);
  // One unit of control for one step costs dt of fuel.
  EXPECT_DOUBLE_EQ(integrator->running_cost(0, state, Control(-1.0)), dt);
}

TEST(L1BenchmarkModelsTest, ThePendulumRestsHangingAndUprightAndTheMotorPushesForward) {
  const Result<L1Pendulum> pendulum = MakeL1Pendulum();
  ASSERT_TRUE(pendulum.has_value());
  const auto& model = pendulum->forward_model;
  const Eigen::Vector2d hanging(std::numbers::pi, 0.0);
  EXPECT_LT((model(0, hanging, Control::Zero(), Eigen::Vector2d::Zero()) - hanging).norm(), 1e-12);
  EXPECT_EQ(model(0, Eigen::Vector2d::Zero(), Control::Zero(), Eigen::Vector2d::Zero()), Eigen::Vector2d::Zero());

  // Full torque adds about a3 dt of rate in one step -- a percent less, since gravity pulls a
  // hanging pendulum back as it moves -- and the motor cannot hold the pendulum against gravity at
  // horizontal: the rate there still falls away from upright.
  const double dt = pendulum->time_step;
  EXPECT_NEAR(model.control_drift_mat()(0, hanging)[1], 4.0 * dt, 0.01 * 4.0 * dt);
  const Eigen::Vector2d horizontal(std::numbers::pi / 2.0, 0.0);
  EXPECT_GT(model(0, horizontal, Control(-1.0), Eigen::Vector2d::Zero())[1], 0.0);
}

TEST(L1BenchmarkModelsTest, ThePendulumStepMatchesAFineReference) {
  const Result<L1Pendulum> pendulum = MakeL1Pendulum();
  ASSERT_TRUE(pendulum.has_value());
  const double dt = pendulum->time_step;
  for (const double control : {-1.0, 0.0, 1.0}) {
    for (const Eigen::Vector2d state : {Eigen::Vector2d(2.5, 3.0), Eigen::Vector2d(0.3, -6.0)}) {
      const Eigen::Vector2d reference = IntegratePendulumStep(L1PendulumParameters{}, state, control, dt, 256);
      const Eigen::Vector2d step = pendulum->forward_model(0, state, Control(control), Eigen::Vector2d::Zero());
      EXPECT_LT((step - reference).norm(), 1e-4) << "control " << control;
    }
  }
}

TEST(L1BenchmarkModelsTest, DoingNothingCostsTheTerminalCostOfWhereTheSystemStarts) {
  const Result<L1Pendulum> pendulum = MakeL1Pendulum();
  ASSERT_TRUE(pendulum.has_value());
  EXPECT_NEAR(pendulum->terminal_cost(Eigen::Vector2d(std::numbers::pi, 0.0)),
              10.0 * std::numbers::pi * std::numbers::pi,
              1e-12);
  EXPECT_DOUBLE_EQ(pendulum->running_cost(0, Eigen::Vector2d::Zero(), Control::Zero()), 0.0);
}

}  // namespace
}  // namespace fbsde_traj_opt::examples
