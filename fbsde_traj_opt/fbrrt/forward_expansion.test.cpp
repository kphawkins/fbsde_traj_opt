// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#include "fbsde_traj_opt/fbrrt/forward_expansion.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "fbsde_traj_opt/costs/l1_control_sde_running_cost_term.hpp"
#include "fbsde_traj_opt/dynamics/const_diagonal_sde_diffusion_term.hpp"
#include "fbsde_traj_opt/dynamics/const_linear_sde_control_drift_mat_term.hpp"
#include "fbsde_traj_opt/dynamics/const_linear_sde_state_drift_term.hpp"
#include "fbsde_traj_opt/fbrrt/epsilon_greedy_l1_control_sampler.hpp"
#include "fbsde_traj_opt/fbrrt/fbrrt_particle_set.hpp"
#include "fbsde_traj_opt/fbrrt/rrt_expansion_selector.hpp"
#include "fbsde_traj_opt/sde/composed_forward_sde_model.hpp"
#include "fbsde_traj_opt/utils/result.hpp"

namespace fbsde_traj_opt {
namespace {

constexpr double kTimeStep = 0.1;
constexpr std::size_t kCapacity = 32;
constexpr std::size_t kStages = 6;

using Model = ComposedForwardSdeModel<2,
                                      1,
                                      ConstLinearSdeStateDriftTerm<2>,
                                      ConstLinearSdeControlDriftMatTerm<2, 1>,
                                      ConstDiagonalSdeDiffusionTerm<2>>;
using Set = FbrrtParticleSet<2, 1, kCapacity, kStages>;
using Control = Eigen::Matrix<double, 1, 1>;

struct ZeroPolicy {
  auto operator()(std::size_t /*stage*/, const Eigen::Vector2d& /*state*/) const noexcept -> Control {
    return Control::Zero();
  }
};

// A double integrator with a large diffusion: if any noise leaked into the forward pass, the
// children would be visibly off the drift.
struct Fixture {
  Model model;
  L1ControlSdeRunningCostTerm<2, 1> running_cost;
  RrtExpansionSelector<2> selector;
  EpsilonGreedyL1ControlSampler<2, 1> control_sampler;
};

auto MakeFixture() -> Fixture {
  Eigen::Matrix2d state_matrix;
  state_matrix << 0.0, kTimeStep, 0.0, 0.0;
  const Result<ConstDiagonalSdeDiffusionTerm<2>> diffusion =
      ConstDiagonalSdeDiffusionTerm<2>::Make(Eigen::Vector2d(0.5, 0.5));
  const Result<L1ControlSdeRunningCostTerm<2, 1>> running_cost = L1ControlSdeRunningCostTerm<2, 1>::Make(0.3);
  const Result<RrtExpansionSelector<2>> selector =
      MakeBruteForceRrtExpansionSelector<2>(Eigen::Vector2d(-3.0, -3.0), Eigen::Vector2d(3.0, 3.0));
  const Result<EpsilonGreedyL1ControlSampler<2, 1>> control_sampler =
      EpsilonGreedyL1ControlSampler<2, 1>::Make(Control(-1.0), Control(1.0));
  EXPECT_TRUE(diffusion.has_value() && running_cost.has_value() && selector.has_value() && control_sampler.has_value());
  return Fixture{
      .model = Model(ConstLinearSdeStateDriftTerm<2>(state_matrix),
                     ConstLinearSdeControlDriftMatTerm<2, 1>(Eigen::Vector2d(0.0, kTimeStep)),
                     *diffusion),
      .running_cost = *running_cost,
      .selector = *selector,
      .control_sampler = *control_sampler,
  };
}

auto MakeSeededSet() -> Set {
  Set set;
  set.SetInitialStates(Set::StateBlock::Random());
  return set;
}

auto Expand(const Fixture& fixture, Set& set, std::uint64_t seed) -> Result<> {
  return ExpandParticles(set,
                         fixture.model,
                         fixture.running_cost,
                         fixture.selector,
                         fixture.control_sampler,
                         ZeroPolicy{},
                         0.5,
                         0.0,
                         seed);
}

TEST(ExpandParticlesTest, FillsEveryStageWithNoiselessEdges) {
  const Fixture fixture = MakeFixture();
  Set set = MakeSeededSet();
  ASSERT_TRUE(Expand(fixture, set, 1).has_value());

  for (std::size_t stage = 1; stage < kStages; ++stage) {
    ASSERT_EQ(set.count(stage), kCapacity);
    for (std::size_t slot = 0; slot < kCapacity; ++slot) {
      const auto column = static_cast<Eigen::Index>(slot);
      const Eigen::Vector2d source = set.source_states(stage).col(column);
      const double control = set.controls(stage)(0, column);
      const std::size_t parent = set.parents(stage)[slot];

      // The edge leaves from its parent's state, under an exploratory control.
      ASSERT_LT(parent, kCapacity);
      EXPECT_EQ(source, Eigen::Vector2d(set.states(stage - 1).col(static_cast<Eigen::Index>(parent))));
      EXPECT_TRUE(control == -1.0 || control == 0.0 || control == 1.0);

      // The drift is F(x, u), and the child sits exactly at x + K: no noise.
      const Eigen::Vector2d drift(kTimeStep * source[1], kTimeStep * control);
      EXPECT_NEAR((Eigen::Vector2d(set.drifts(stage).col(column)) - drift).norm(), 0.0, 1e-15);
      EXPECT_NEAR((Eigen::Vector2d(set.states(stage).col(column)) - (source + drift)).norm(), 0.0, 1e-15);

      // The accrued cost is the parent's plus this edge's fuel.
      const double parent_cost = set.accrued_costs(stage - 1)[static_cast<Eigen::Index>(parent)];
      EXPECT_NEAR(set.accrued_costs(stage)[column], parent_cost + (0.3 * std::abs(control)), 1e-12);
    }
  }
}

TEST(ExpandParticlesTest, WithNoiseTheChildrenScatterAroundTheDriftButTheEdgesAreUnchanged) {
  const Fixture fixture = MakeFixture();
  Set noiseless = MakeSeededSet();
  Set noisy = noiseless;
  ASSERT_TRUE(Expand(fixture, noiseless, 1).has_value());
  ASSERT_TRUE(ExpandParticles(noisy,
                              fixture.model,
                              fixture.running_cost,
                              fixture.selector,
                              fixture.control_sampler,
                              ZeroPolicy{},
                              0.5,
                              0.0,
                              1,
                              true)
                  .has_value());

  // The first transition draws the same parents and controls either way; only where the child
  // lands differs, by Sigma w with Sigma = diag(0.5, 0.5).
  EXPECT_EQ(noisy.source_states(1), noiseless.source_states(1));
  EXPECT_EQ(noisy.drifts(1), noiseless.drifts(1));
  const Set::StateBlock offsets = noisy.states(1) - (noisy.source_states(1) + noisy.drifts(1));
  const double variance = offsets.squaredNorm() / static_cast<double>(2 * kCapacity);
  EXPECT_GT(variance, 0.25 * 0.5);
  EXPECT_LT(variance, 0.25 * 1.5);
}

TEST(ExpandParticlesTest, IsDeterministicInTheSeedAndVariesWithIt) {
  const Fixture fixture = MakeFixture();
  Set first = MakeSeededSet();
  Set second = first;
  Set other = first;
  ASSERT_TRUE(Expand(fixture, first, 4).has_value());
  ASSERT_TRUE(Expand(fixture, second, 4).has_value());
  ASSERT_TRUE(Expand(fixture, other, 5).has_value());
  EXPECT_EQ(first.states(kStages - 1), second.states(kStages - 1));
  EXPECT_NE(first.states(kStages - 1), other.states(kStages - 1));
}

TEST(ExpandParticlesTest, RefillsOnlyTheSlotsErosionFreed) {
  const Fixture fixture = MakeFixture();
  Set set = MakeSeededSet();
  ASSERT_TRUE(Expand(fixture, set, 2).has_value());
  for (std::size_t stage = 1; stage < kStages; ++stage) {
    ASSERT_TRUE(set.KeepLowest(stage, set.accrued_costs(stage), kCapacity / 2).has_value());
  }
  const Set::StateBlock survivors = set.states(3);

  ASSERT_TRUE(Expand(fixture, set, 3).has_value());
  EXPECT_EQ(set.count(3), kCapacity);
  EXPECT_EQ(set.states(3).leftCols<kCapacity / 2>(), survivors.leftCols<kCapacity / 2>());
}

TEST(ExpandParticlesTest, FailsWithoutInitialParticles) {
  const Fixture fixture = MakeFixture();
  Set set;
  EXPECT_FALSE(Expand(fixture, set, 1).has_value());
}

}  // namespace
}  // namespace fbsde_traj_opt
