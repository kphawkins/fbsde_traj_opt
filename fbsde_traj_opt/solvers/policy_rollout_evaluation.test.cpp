// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#include "fbsde_traj_opt/solvers/policy_rollout_evaluation.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>

#include <Eigen/Core>
#include <gtest/gtest.h>

namespace fbsde_traj_opt {
namespace {

using State = Eigen::Matrix<double, 1, 1>;
using Control = Eigen::Matrix<double, 1, 1>;

// x' = x + u + noise_scale * w.
struct ScalarModel {
  double noise_scale = 0.0;

  auto operator()(std::size_t /*stage*/, const State& state, const Control& control, const State& noise) const noexcept
      -> State {
    return state + control + (noise_scale * noise);
  }
};

struct HalvingPolicy {
  auto operator()(std::size_t /*stage*/, const State& state) const noexcept -> Control { return -0.5 * state; }
};

struct ZeroPolicy {
  auto operator()(std::size_t /*stage*/, const State& /*state*/) const noexcept -> Control { return Control::Zero(); }
};

struct AbsoluteControlCost {
  auto operator()(std::size_t /*stage*/, const State& /*state*/, const Control& control) const noexcept -> double {
    return std::abs(control[0]);
  }
};

struct SquaredTerminalCost {
  auto operator()(const State& state) const noexcept -> double { return state[0] * state[0]; }
};

TEST(EvaluatePolicyByRolloutsTest, MatchesAHandComputedDeterministicRollout) {
  // 4 -> 2 -> 1 under u = -x/2, spending 2 + 1 of fuel and ending with a terminal cost of 1.
  const PolicyRolloutCosts<double> costs =
      EvaluatePolicyByRollouts(ScalarModel{},
                               HalvingPolicy{},
                               AbsoluteControlCost{},
                               SquaredTerminalCost{},
                               State(4.0),
                               Eigen::Matrix<double, 1, 1>(Eigen::Matrix<double, 1, 1>::Zero()),
                               8,
                               2,
                               3);
  EXPECT_DOUBLE_EQ(costs.running, 3.0);
  EXPECT_DOUBLE_EQ(costs.terminal, 1.0);
  EXPECT_DOUBLE_EQ(costs.total, 4.0);
}

TEST(EvaluatePolicyByRolloutsTest, AveragesOverNoiseAndRepeatsForTheSameSeed) {
  // A random walk of 4 unit steps from a unit-variance start: E[x_T^2] = 5.
  const ScalarModel model{.noise_scale = 1.0};
  const auto evaluate = [&](std::uint64_t seed) {
    return EvaluatePolicyByRollouts(model,
                                    ZeroPolicy{},
                                    AbsoluteControlCost{},
                                    SquaredTerminalCost{},
                                    State(0.0),
                                    Eigen::Matrix<double, 1, 1>(Eigen::Matrix<double, 1, 1>::Identity()),
                                    20000,
                                    4,
                                    seed);
  };
  const PolicyRolloutCosts<double> costs = evaluate(5);
  EXPECT_DOUBLE_EQ(costs.running, 0.0);
  EXPECT_NEAR(costs.terminal, 5.0, 0.2);
  EXPECT_DOUBLE_EQ(costs.total, costs.running + costs.terminal);
  EXPECT_DOUBLE_EQ(evaluate(5).total, costs.total);
  EXPECT_NE(evaluate(6).total, costs.total);
}

}  // namespace
}  // namespace fbsde_traj_opt
