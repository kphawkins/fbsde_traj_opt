// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#include "fbsde_traj_opt/fbrrt/epsilon_greedy_l1_control_sampler.hpp"

#include <cstddef>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "fbsde_traj_opt/sampling/counter_based_normal_sampler.hpp"

namespace fbsde_traj_opt {
namespace {

using Sampler = EpsilonGreedyL1ControlSampler<2, 1>;
using Control = Sampler::Control;

// A policy whose control is recognizably not one of the exploratory vertices.
struct ConstantPolicy {
  auto operator()(std::size_t /*stage*/, const Eigen::Vector2d& /*state*/) const noexcept -> Control {
    return Control(0.25);
  }
};

static_assert(FbrrtControlSampler<Sampler, ConstantPolicy, Eigen::Vector2d, Control, double>);

auto PolicyShare(double exploit_probability) -> double {
  const Result<Sampler> sampler = Sampler::Make(Control(-1.0), Control(1.0));
  EXPECT_TRUE(sampler.has_value());
  const CounterBasedNormalSampler stream(17);
  constexpr std::size_t kDraws = 20000;
  std::size_t from_policy = 0;
  for (std::size_t draw = 0; draw < kDraws; ++draw) {
    const Control control = (*sampler)(ConstantPolicy{}, 2, Eigen::Vector2d::Zero(), exploit_probability, stream, draw);
    if (control[0] == 0.25) {
      ++from_policy;
    } else {
      EXPECT_TRUE(control[0] == -1.0 || control[0] == 0.0 || control[0] == 1.0);
    }
  }
  return static_cast<double>(from_policy) / static_cast<double>(kDraws);
}

TEST(EpsilonGreedyL1ControlSamplerTest, ExploitsThePolicyWithTheGivenProbability) {
  EXPECT_DOUBLE_EQ(PolicyShare(0.0), 0.0);
  EXPECT_DOUBLE_EQ(PolicyShare(1.0), 1.0);
  EXPECT_NEAR(PolicyShare(0.3), 0.3, 0.015);
}

TEST(EpsilonGreedyL1ControlSamplerTest, MakeRejectsBoundsThatExcludeZero) {
  EXPECT_FALSE(Sampler::Make(Control(0.5), Control(1.0)).has_value());
  EXPECT_FALSE(Sampler::Make(Control(-1.0), Control(-0.5)).has_value());
}

}  // namespace
}  // namespace fbsde_traj_opt
