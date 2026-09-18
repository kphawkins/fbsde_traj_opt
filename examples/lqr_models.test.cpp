// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#include "examples/lqr_models.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <gtest/gtest.h>

#include "examples/lqr_experiment.hpp"

namespace fbsde_traj_opt::examples {
namespace {

// Fewer trajectories than the demo runs with: enough for the margins asserted below, which are
// large, and fast enough to belong in a unit test.
constexpr std::size_t kTestTrajectories = 32;
constexpr std::uint64_t kSeed = 2026;

TEST(DoubleIntegratorSpecTest, ProportionalFeedbackIsMarginallyStable) {
  const LqrExperimentSpec<2, 1> spec = MakeDoubleIntegratorSpec();

  // The closed-loop continuous system under the baseline gain.
  const Eigen::Matrix2d closed_loop = spec.continuous_state_mat + (spec.continuous_control_mat * spec.baseline_gain);

  const Eigen::EigenSolver<Eigen::Matrix2d> solver(closed_loop);
  for (int index = 0; index < 2; ++index) {
    // Poles exactly on the imaginary axis: the mass oscillates and never settles. This is the
    // claim the model's description makes, checked rather than asserted in prose.
    EXPECT_NEAR(solver.eigenvalues()[index].real(), 0.0, 1e-12);
    EXPECT_GT(std::abs(solver.eigenvalues()[index].imag()), 0.1);
  }
}

TEST(CartPoleSpecTest, ThePlantIsOpenLoopUnstable) {
  const LqrExperimentSpec<4, 1> spec = MakeCartPoleSpec();

  const Eigen::EigenSolver<Eigen::Matrix4d> solver(spec.continuous_state_mat);

  double largest_real_part = 0.0;
  for (int index = 0; index < 4; ++index) {
    largest_real_part = std::max(largest_real_part, solver.eigenvalues()[index].real());
  }
  EXPECT_GT(largest_real_part, 1.0) << "the linearized pendulum should have a real unstable pole";
}

TEST(CartPoleSpecTest, TheBaselineStabilizesThePoleButNotTheCart) {
  const LqrExperimentSpec<4, 1> spec = MakeCartPoleSpec();

  const Eigen::Matrix4d closed_loop = spec.continuous_state_mat + (spec.continuous_control_mat * spec.baseline_gain);
  const Eigen::EigenSolver<Eigen::Matrix4d> solver(closed_loop);

  // Two poles are damped -- the pole angle subsystem -- and two sit at the origin, which is the
  // cart integrating an acceleration nothing is regulating.
  int damped_poles = 0;
  int integrator_poles = 0;
  for (int index = 0; index < 4; ++index) {
    const double real_part = solver.eigenvalues()[index].real();
    if (real_part < -0.1) {
      ++damped_poles;
    } else if (std::abs(real_part) < 1e-9) {
      ++integrator_poles;
    }
  }
  EXPECT_EQ(damped_poles, 2);
  EXPECT_EQ(integrator_poles, 2);
}

TEST(TwoMassSpringSpecTest, ThePlantIsUndamped) {
  const LqrExperimentSpec<4, 1> spec = MakeTwoMassSpringSpec();

  const Eigen::EigenSolver<Eigen::Matrix4d> solver(spec.continuous_state_mat);
  for (int index = 0; index < 4; ++index) {
    // Every pole on the imaginary axis: nothing removes energy from the plant on its own.
    EXPECT_NEAR(solver.eigenvalues()[index].real(), 0.0, 1e-12);
  }
}

// The point of each model: the optimal policy beats the controller an engineer would reach for,
// and by a margin large enough that no one has to squint at the figure.
TEST(LqrModelsTest, DoubleIntegratorLqrBeatsProportionalFeedback) {
  const auto report =
      RunLqrExperiment<2, 1, kTestTrajectories, kDoubleIntegratorNumStages>(MakeDoubleIntegratorSpec(), kSeed);
  ASSERT_TRUE(report.has_value()) << report.error();

  EXPECT_GT(report->baseline_initial_cost, 2.0 * report->optimal_initial_cost);
}

TEST(LqrModelsTest, CartPoleLqrBeatsPoleAnglePd) {
  const auto report = RunLqrExperiment<4, 1, kTestTrajectories, kCartPoleNumStages>(MakeCartPoleSpec(), kSeed);
  ASSERT_TRUE(report.has_value()) << report.error();

  EXPECT_GT(report->baseline_initial_cost, 2.0 * report->optimal_initial_cost);
}

TEST(LqrModelsTest, TwoMassSpringLqrBeatsColocatedPd) {
  const auto report =
      RunLqrExperiment<4, 1, kTestTrajectories, kTwoMassSpringNumStages>(MakeTwoMassSpringSpec(), kSeed);
  ASSERT_TRUE(report.has_value()) << report.error();

  EXPECT_GT(report->baseline_initial_cost, 2.0 * report->optimal_initial_cost);
}

TEST(LqrModelsTest, EveryModelNamesEveryStateAndPlotsValidOnes) {
  const LqrExperimentSpec<2, 1> double_integrator = MakeDoubleIntegratorSpec();
  for (const std::string& name : double_integrator.state_names) {
    EXPECT_FALSE(name.empty());
  }
  for (const std::size_t index : double_integrator.plotted_state_indices) {
    EXPECT_LT(index, 2U);
  }

  for (const auto& spec : {MakeCartPoleSpec(), MakeTwoMassSpringSpec()}) {
    for (const std::string& name : spec.state_names) {
      EXPECT_FALSE(name.empty());
    }
    for (const std::size_t index : spec.plotted_state_indices) {
      EXPECT_LT(index, 4U);
    }
    EXPECT_LT(spec.phase_horizontal_index, 4U);
    EXPECT_LT(spec.phase_vertical_index, 4U);
  }
}

}  // namespace
}  // namespace fbsde_traj_opt::examples
