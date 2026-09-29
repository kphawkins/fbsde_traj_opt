// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#include "examples/fbrrt_comparison_experiment.hpp"

#include <cstddef>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "fbsde_traj_opt/utils/result.hpp"

namespace fbsde_traj_opt::examples {
namespace {

TEST(FbrrtComparisonTest, AShortRunComparesBothMethodsOnTheSameStartingCost) {
  FbrrtComparisonOptions options;
  options.num_seeds = 1;
  options.iterations = 3;
  options.include_double_inverted_pendulum = false;

  std::size_t runs = 0;
  const Result<std::vector<BenchmarkComparison>> comparisons = RunFbrrtComparison(
      options, [&runs](const std::string&, const std::string&, std::size_t, const MethodRun&) { ++runs; });
  ASSERT_TRUE(comparisons.has_value()) << comparisons.error();
  ASSERT_EQ(comparisons->size(), 2U);
  EXPECT_EQ(runs, 4U);

  for (const BenchmarkComparison& comparison : *comparisons) {
    ASSERT_EQ(comparison.fbrrt.runs.size(), 1U);
    ASSERT_EQ(comparison.dt_fbsde.runs.size(), 1U);
    const MethodRun& fbrrt = comparison.fbrrt.runs.front();
    const MethodRun& dt_fbsde = comparison.dt_fbsde.runs.front();
    ASSERT_EQ(fbrrt.costs.size(), 4U);
    ASSERT_EQ(fbrrt.seconds.size(), 4U);

    // Both start from the same value functions and score on the same noise.
    EXPECT_DOUBLE_EQ(fbrrt.costs.front(), dt_fbsde.costs.front());
    EXPECT_LE(comparison.best_cost, comparison.initial_cost);
    EXPECT_DOUBLE_EQ(comparison.convergence_threshold, 1.05 * comparison.best_cost);
    // The best cost is somebody's: one of the methods converged by definition.
    EXPECT_GT(comparison.fbrrt.fraction_converged + comparison.dt_fbsde.fraction_converged, 0.0);
    EXPECT_NE(SummarizeFbrrtComparison(comparison).find(comparison.name), std::string::npos);
  }
}

}  // namespace
}  // namespace fbsde_traj_opt::examples
