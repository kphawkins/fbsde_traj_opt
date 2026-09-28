// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#include "examples/lqr_experiment.hpp"

#include <cstddef>
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

namespace fbsde_traj_opt::examples {
namespace {

constexpr std::size_t kNumTrajectories = 64;
constexpr std::size_t kNumStages = 21;

// A damped double integrator with a deliberately weak baseline gain, enough to exercise every
// branch of the runner without being a claim about any particular system.
auto MakeTestSpec() noexcept -> LqrExperimentSpec<2, 1> {
  LqrExperimentSpec<2, 1> spec;
  spec.name = "Test system";
  spec.description = "A double integrator.";
  spec.time_step = 0.1;

  spec.continuous_state_mat << 0.0, 1.0,  //
      0.0, 0.0;
  spec.continuous_control_mat << 0.0, 1.0;

  spec.state_cost_mat = Eigen::Vector2d(1.0, 0.1).asDiagonal();
  spec.control_cost_mat = Eigen::Matrix<double, 1, 1>::Constant(0.05);
  spec.terminal_cost_mat = Eigen::Vector2d(5.0, 1.0).asDiagonal();

  spec.initial_mean = Eigen::Vector2d(1.0, 0.0);
  spec.initial_covariance_diagonal = Eigen::Vector2d(0.02, 0.02);
  spec.diffusion_diagonal = Eigen::Vector2d(0.01, 0.02);

  spec.baseline_gain << -0.2, -0.05;
  spec.baseline_name = "weak feedback";

  spec.state_names = {"Position", "Velocity"};
  spec.plotted_state_indices = {0, 1};
  spec.plot_phase_portrait = true;
  return spec;
}

TEST(RunLqrExperimentTest, ProducesOneFigurePerPlottedStatePlusPhaseAndCost) {
  const auto report = RunLqrExperiment<2, 1, kNumTrajectories, kNumStages>(MakeTestSpec(), 7);
  ASSERT_TRUE(report.has_value()) << report.error();

  // Two state figures, one phase portrait, one cost-to-go.
  ASSERT_EQ(report->figures.size(), 4U);
  EXPECT_EQ(report->figures.at(0).title(), "Position over the horizon");
  EXPECT_EQ(report->figures.at(1).title(), "Velocity over the horizon");
  EXPECT_EQ(report->figures.at(2).title(), "Phase portrait: Velocity against Position");
  EXPECT_EQ(report->figures.at(3).title(), "Expected cost-to-go");
  EXPECT_EQ(report->name, "Test system");
}

TEST(RunLqrExperimentTest, PhasePortraitIsOmittedWhenNotRequested) {
  LqrExperimentSpec<2, 1> spec = MakeTestSpec();
  spec.plot_phase_portrait = false;

  const auto report = RunLqrExperiment<2, 1, kNumTrajectories, kNumStages>(spec, 7);
  ASSERT_TRUE(report.has_value()) << report.error();

  EXPECT_EQ(report->figures.size(), 3U);
}

// The claim the whole exercise rests on: the Riccati policy is optimal for this cost, so no
// fixed-gain feedback can do better on the same sampled noise.
TEST(RunLqrExperimentTest, TheOptimalPolicyCostsNoMoreThanTheBaseline) {
  const auto report = RunLqrExperiment<2, 1, kNumTrajectories, kNumStages>(MakeTestSpec(), 7);
  ASSERT_TRUE(report.has_value()) << report.error();

  EXPECT_LT(report->optimal_initial_cost, report->baseline_initial_cost);
}

TEST(RunLqrExperimentTest, TheOptimalPolicyLeadsAtEveryStage) {
  const auto report = RunLqrExperiment<2, 1, kNumTrajectories, kNumStages>(MakeTestSpec(), 7);
  ASSERT_TRUE(report.has_value()) << report.error();

  const nlohmann::json cost_figure = report->figures.back().ToJson();
  const std::vector<double> optimal = cost_figure.at("data").at(0).at("y").get<std::vector<double>>();
  const std::vector<double> baseline = cost_figure.at("data").at(1).at("y").get<std::vector<double>>();

  ASSERT_EQ(optimal.size(), kNumStages);
  ASSERT_EQ(baseline.size(), kNumStages);
  for (std::size_t stage = 0; stage + 1 < kNumStages; ++stage) {
    // Not at the terminal stage, where both policies are charged the same terminal cost on
    // whatever state they reached and nothing is left to optimize.
    EXPECT_LE(optimal[stage], baseline[stage]) << "at stage " << stage;
  }
}

TEST(RunLqrExperimentTest, TheStage0CostMatchesTheCostToGoFigure) {
  const auto report = RunLqrExperiment<2, 1, kNumTrajectories, kNumStages>(MakeTestSpec(), 7);
  ASSERT_TRUE(report.has_value()) << report.error();

  const nlohmann::json cost_figure = report->figures.back().ToJson();
  EXPECT_DOUBLE_EQ(report->optimal_initial_cost, cost_figure.at("data").at(0).at("y").at(0).get<double>());
  EXPECT_DOUBLE_EQ(report->baseline_initial_cost, cost_figure.at("data").at(1).at("y").at(0).get<double>());
}

// Both policies must start from the same states, or the comparison is between two draws rather
// than between two controllers.
TEST(RunLqrExperimentTest, BothPoliciesSeeTheSameInitialStates) {
  const auto report = RunLqrExperiment<2, 1, kNumTrajectories, kNumStages>(MakeTestSpec(), 7);
  ASSERT_TRUE(report.has_value()) << report.error();

  const nlohmann::json data = report->figures.at(0).ToJson().at("data");

  // The first trace of each batch is that batch's first sampled trajectory; they must agree at
  // stage 0 and, because the policies differ, diverge afterwards.
  const std::size_t traces_per_batch = data.size() / 2;
  const std::vector<double> optimal_first = data.at(0).at("y").get<std::vector<double>>();
  const std::vector<double> baseline_first = data.at(traces_per_batch).at("y").get<std::vector<double>>();

  EXPECT_DOUBLE_EQ(optimal_first.front(), baseline_first.front());
  EXPECT_NE(optimal_first.back(), baseline_first.back());
}

TEST(RunLqrExperimentTest, RejectsAPlottedStateIndexOutsideTheState) {
  LqrExperimentSpec<2, 1> spec = MakeTestSpec();
  spec.plotted_state_indices = {0, 5};

  EXPECT_FALSE((RunLqrExperiment<2, 1, kNumTrajectories, kNumStages>(spec, 7).has_value()));
}

TEST(RunLqrExperimentTest, RejectsANonPositiveTimeStep) {
  LqrExperimentSpec<2, 1> spec = MakeTestSpec();
  spec.time_step = 0.0;

  EXPECT_FALSE((RunLqrExperiment<2, 1, kNumTrajectories, kNumStages>(spec, 7).has_value()));
}

TEST(RunLqrExperimentTest, RejectsASingularControlCost) {
  LqrExperimentSpec<2, 1> spec = MakeTestSpec();
  spec.control_cost_mat = Eigen::Matrix<double, 1, 1>::Zero();
  spec.terminal_cost_mat = Eigen::Matrix2d::Zero();
  spec.state_cost_mat = Eigen::Matrix2d::Zero();

  EXPECT_FALSE((RunLqrExperiment<2, 1, kNumTrajectories, kNumStages>(spec, 7).has_value()));
}

}  // namespace
}  // namespace fbsde_traj_opt::examples
