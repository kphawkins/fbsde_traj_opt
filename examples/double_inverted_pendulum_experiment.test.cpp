// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#include "examples/double_inverted_pendulum_experiment.hpp"

#include <cmath>
#include <cstddef>

#include <gtest/gtest.h>

#include "fbsde_traj_opt/dt_fbsde_iterative_solver.hpp"
#include "fbsde_traj_opt/utils/result.hpp"
#include "fbsde_traj_opt/viz/plotly_figure.hpp"

namespace fbsde_traj_opt::examples {
namespace {

// Two iterations: enough to exercise every piece of the report, not to converge. The converged
// numbers are what the binary prints; this guards the machinery that produces them.
TEST(DoubleInvertedPendulumExperimentTest, ShortRunBuildsEveryFigureAndChecksTheSimulation) {
  std::size_t progress_calls = 0;
  const Result<DoubleInvertedPendulumReport> report = RunDoubleInvertedPendulumExperiment(
      DoubleInvertedPendulumExperimentOptions{.seed = 3, .iterations = 2},
      [&progress_calls](const DtFbsdeIterationReport<double>&) { ++progress_calls; });
  ASSERT_TRUE(report.has_value()) << report.error();

  EXPECT_EQ(progress_calls, 2U);
  EXPECT_EQ(report->iterations.size(), 2U);
  EXPECT_EQ(report->predicted_costs.size(), 2U);
  EXPECT_EQ(report->figures.size(), 14U);
  EXPECT_FALSE(report->summary.empty());

  // The linkage animation leads the report, and it moves.
  EXPECT_GT(report->figures.front().FrameCount(), 10U);

  // The per-iteration figures carry one frame per iteration.
  std::size_t per_iteration_animations = 0;
  for (const viz::PlotlyFigure& figure : report->figures) {
    per_iteration_animations += figure.FrameCount() == 2U ? 1U : 0U;
  }
  EXPECT_EQ(per_iteration_animations, 2U);

  EXPECT_TRUE(std::isfinite(report->final_mean_cost));
  EXPECT_NEAR(report->final_mean_cost, report->final_mean_fuel + report->final_mean_terminal_cost, 1e-9);
  EXPECT_GE(report->fraction_inverted, 0.0);
  EXPECT_LE(report->fraction_inverted, 1.0);

  // The simulation checks, which hold whatever the policy: the frictionless pendulum keeps its
  // energy, and every step the model takes agrees with a fine reference integration to well under
  // the noise the SDE injects per step (0.0047 rad and 0.028 rad/s).
  EXPECT_LT(report->max_relative_energy_error, 1e-5);
  EXPECT_LT(report->max_one_step_angle_error, 1e-3);
  EXPECT_LT(report->max_one_step_rate_error, 3e-2);
}

}  // namespace
}  // namespace fbsde_traj_opt::examples
