// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#include "fbsde_traj_opt/fbrrt/rrt_expansion_selector.hpp"

#include <cstddef>
#include <limits>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "fbsde_traj_opt/fbrrt/brute_force_nearest_neighbor.hpp"
#include "fbsde_traj_opt/sampling/counter_based_normal_sampler.hpp"

namespace fbsde_traj_opt {
namespace {

using Selector = RrtExpansionSelector<2>;
using Block = Eigen::Matrix<double, 2, 10>;

static_assert(FbrrtExpansionSelector<Selector, Block, double>);

// Nine particles packed near the corner (-0.9, -0.9) of [-1, 1]^2 and one alone at the center.
// The loner owns almost the whole region's Voronoi volume.
auto ClusterAndLoner() -> Block {
  Block states = (0.01 * Block::Random()).colwise() + Eigen::Vector2d(-0.9, -0.9);
  states.col(7) = Eigen::Vector2d::Zero();
  return states;
}

auto MakeUnitSelector() -> Selector {
  const Result<Selector> selector =
      MakeBruteForceRrtExpansionSelector<2>(Eigen::Vector2d(-1.0, -1.0), Eigen::Vector2d(1.0, 1.0));
  EXPECT_TRUE(selector.has_value());
  return *selector;
}

auto LonerShare(const Selector& selector, double rrt_probability) -> double {
  const Block states = ClusterAndLoner();
  const CounterBasedNormalSampler sampler(3);
  constexpr std::size_t kDraws = 4000;
  std::size_t loner = 0;
  for (std::size_t draw = 0; draw < kDraws; ++draw) {
    const std::size_t slot = selector(states, 10, rrt_probability, sampler, draw, 1);
    EXPECT_LT(slot, 10U);
    loner += slot == 7 ? 1 : 0;
  }
  return static_cast<double>(loner) / static_cast<double>(kDraws);
}

TEST(RrtExpansionSelectorTest, RrtDrawsFavorTheParticleWithTheLargestVoronoiCell) {
  // The loner is nearest to every target with x + y > -0.9: all of the square but a corner
  // triangle of area 1.1^2 / 2, so about 85% of it -- against the 10% a uniform draw would give.
  EXPECT_NEAR(LonerShare(MakeUnitSelector(), 1.0), 0.85, 0.03);
}

TEST(RrtExpansionSelectorTest, UniformDrawsTreatEveryParticleAlike) {
  EXPECT_NEAR(LonerShare(MakeUnitSelector(), 0.0), 0.1, 0.02);
}

TEST(RrtExpansionSelectorTest, IsAPureFunctionOfTheSamplerCoordinate) {
  const Selector selector = MakeUnitSelector();
  const Block states = ClusterAndLoner();
  const CounterBasedNormalSampler sampler(9);
  for (std::size_t draw = 0; draw < 32; ++draw) {
    EXPECT_EQ(selector(states, 10, 0.5, sampler, draw, 4), selector(states, 10, 0.5, sampler, draw, 4));
  }
}

TEST(RrtExpansionSelectorTest, MakeRejectsAnEmptyOrUnboundedRegion) {
  EXPECT_FALSE(
      MakeBruteForceRrtExpansionSelector<2>(Eigen::Vector2d(-1.0, 1.0), Eigen::Vector2d(1.0, 1.0)).has_value());
  const Result<BruteForceNearestNeighbor<2>> search = BruteForceNearestNeighbor<2>::Make(Eigen::Vector2d::Ones());
  ASSERT_TRUE(search.has_value());
  EXPECT_FALSE(Selector::Make(
                   Eigen::Vector2d(-1.0, -1.0), Eigen::Vector2d(1.0, std::numeric_limits<double>::infinity()), *search)
                   .has_value());
}

}  // namespace
}  // namespace fbsde_traj_opt
