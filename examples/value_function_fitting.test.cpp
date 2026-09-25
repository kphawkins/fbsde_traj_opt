// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#include "examples/value_function_fitting.hpp"

#include <cstdint>

#include <gtest/gtest.h>

#include "fbsde_traj_opt/viz/plotly_figure.hpp"

namespace fbsde_traj_opt::examples {
namespace {

constexpr std::uint64_t kSeed = 20260924;

TEST(ValueFunctionFittingTest, TheGenericFitReachesTheTargetItIsShownReaching) {
  const auto experiment = RunGenericFunctionFittingExperiment(kSeed);
  ASSERT_TRUE(experiment.has_value()) << experiment.error();

  EXPECT_EQ(experiment->figures.size(), 4U);
  EXPECT_FALSE(experiment->summary.empty());

  // The figures claim a fit to a fraction of a percent of the target's range. If the model, the
  // fitter, or the tuning regresses, the claim in the report's prose becomes false and this
  // catches it before a reader does.
  EXPECT_LT(experiment->worst_relative_error, 0.01)
      << "the generic fit no longer reaches the accuracy the report claims";
  EXPECT_GT(experiment->worst_relative_error, 0.0);

  // Every figure in this experiment is a line figure.
  for (const viz::PlotlyFigure& figure : experiment->figures) {
    EXPECT_FALSE(figure.HoldsHeatmap()) << figure.title();
    EXPECT_GT(figure.LineCount(), 0U) << figure.title();
  }
}

TEST(ValueFunctionFittingTest, TheBackwardPassReproducesTheExactLqrValueFunction) {
  const auto experiment = RunLqrBackwardPassExperiment(kSeed);
  ASSERT_TRUE(experiment.has_value()) << experiment.error();

  EXPECT_EQ(experiment->figures.size(), 4U);

  // The sharp claim: on an LQR problem the Taylor Noiseless estimator is exact, so reading an
  // exact next-stage value function it must return the exact stage value to rounding. This is the
  // one number in the whole experiment that is not a matter of how long the optimizer ran.
  EXPECT_LT(experiment->estimator_relative_error, 1e-12)
      << "the estimator is no longer exact on a quadratic value function";

  // The softer claim: the regression tracks it closely.
  EXPECT_LT(experiment->worst_relative_error, 5e-3)
      << "the fitted representation no longer reaches the accuracy the report claims";
}

TEST(ValueFunctionFittingTest, TheTwoDimensionalFitDrawsItsSurfacesAsHeatmaps) {
  const auto experiment = RunTwoDimensionalFittingExperiment(kSeed);
  ASSERT_TRUE(experiment.has_value()) << experiment.error();

  ASSERT_EQ(experiment->figures.size(), 4U);
  EXPECT_LT(experiment->worst_relative_error, 0.05)
      << "the two-dimensional fit no longer reaches the accuracy the report claims";

  // The first three are the target, the approximation, and the signed error, over the plane.
  EXPECT_TRUE(experiment->figures[0].HoldsHeatmap());
  EXPECT_TRUE(experiment->figures[1].HoldsHeatmap());
  EXPECT_TRUE(experiment->figures[2].HoldsHeatmap());
  EXPECT_FALSE(experiment->figures[3].HoldsHeatmap());
}

TEST(ValueFunctionFittingTest, TheExperimentsAreReproducibleForAGivenSeed) {
  const auto first = RunGenericFunctionFittingExperiment(7);
  const auto second = RunGenericFunctionFittingExperiment(7);
  ASSERT_TRUE(first.has_value()) << first.error();
  ASSERT_TRUE(second.has_value()) << second.error();

  EXPECT_EQ(first->worst_relative_error, second->worst_relative_error);
  EXPECT_EQ(first->final_root_mean_squared_residual, second->final_root_mean_squared_residual);
}

TEST(ValueFunctionFittingTest, ADifferentSeedGivesADifferentRunThatStillFits) {
  const auto experiment = RunGenericFunctionFittingExperiment(1234);
  ASSERT_TRUE(experiment.has_value()) << experiment.error();

  // The seed steers only which samples land in which minibatch, so a different seed is a
  // different path to the same place, not a different answer.
  EXPECT_LT(experiment->worst_relative_error, 0.01);
}

}  // namespace
}  // namespace fbsde_traj_opt::examples
