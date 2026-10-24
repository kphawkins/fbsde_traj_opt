// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#include "fbsde_traj_opt/utils/phase_profiler.hpp"

#include <cstddef>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

namespace fbsde_traj_opt::profiling {
namespace {

// Restores the disabled, empty state every other test expects.
class PhaseProfilerTest : public ::testing::Test {
 protected:
  auto SetUp() -> void override {
    SetEnabled(false);
    Reset();
  }
  auto TearDown() -> void override {
    SetEnabled(false);
    Reset();
  }
};

TEST_F(PhaseProfilerTest, RecordsNothingWhileDisabled) {
  { const ScopedPhase phase(Phase::kForwardPass); }
  const PhaseTotals totals = Snapshot();
  EXPECT_EQ(totals.Calls(Phase::kForwardPass), 0U);
  EXPECT_EQ(totals.Seconds(Phase::kForwardPass), 0.0);
}

TEST_F(PhaseProfilerTest, CountsCallsAndTimeWhileEnabled) {
  SetEnabled(true);
  for (std::size_t call = 0; call < 3; ++call) {
    const ScopedPhase phase(Phase::kBackwardFit);
  }
  const PhaseTotals totals = Snapshot();
  EXPECT_EQ(totals.Calls(Phase::kBackwardFit), 3U);
  EXPECT_GE(totals.Seconds(Phase::kBackwardFit), 0.0);
  EXPECT_EQ(totals.Calls(Phase::kForwardPass), 0U);
}

TEST_F(PhaseProfilerTest, MergesWorkerThreadsWhenTheyExit) {
  SetEnabled(true);
  constexpr std::size_t kThreads = 4;
  constexpr std::size_t kCallsPerThread = 25;
  {
    std::vector<std::jthread> workers;
    for (std::size_t thread = 0; thread < kThreads; ++thread) {
      workers.emplace_back([] {
        for (std::size_t call = 0; call < kCallsPerThread; ++call) {
          const ScopedPhase phase(Phase::kRolloutPolicy);
        }
      });
    }
  }
  EXPECT_EQ(Snapshot().Calls(Phase::kRolloutPolicy), kThreads * kCallsPerThread);
}

TEST_F(PhaseProfilerTest, ResetClearsTheTotals) {
  SetEnabled(true);
  { const ScopedPhase phase(Phase::kLineSearch); }
  Reset();
  EXPECT_EQ(Snapshot().Calls(Phase::kLineSearch), 0U);
}

TEST_F(PhaseProfilerTest, EveryPhaseHasAName) {
  for (std::size_t index = 0; index < kPhaseCount; ++index) {
    EXPECT_FALSE(PhaseName(static_cast<Phase>(index)).empty()) << "phase " << index;
  }
}

}  // namespace
}  // namespace fbsde_traj_opt::profiling
