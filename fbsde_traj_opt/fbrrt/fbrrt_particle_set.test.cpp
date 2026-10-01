// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#include "fbsde_traj_opt/fbrrt/fbrrt_particle_set.hpp"

#include <cstddef>

#include <Eigen/Core>
#include <gtest/gtest.h>

namespace fbsde_traj_opt {
namespace {

using Set = FbrrtParticleSet<2, 1, 6, 3>;

// A particle whose every field encodes `tag`, so that where it lands after thinning is checkable.
auto TaggedParticle(double tag, std::size_t parent) -> Set::Particle {
  Set::Particle particle;
  particle.state = Eigen::Vector2d(tag, -tag);
  particle.source_state = Eigen::Vector2d(tag + 0.5, 0.0);
  particle.drift = Eigen::Vector2d(0.0, tag);
  particle.control = Eigen::Matrix<double, 1, 1>(tag / 10.0);
  particle.accrued_cost = 2.0 * tag;
  particle.parent = parent;
  return particle;
}

auto FillStage(Set& set, std::size_t stage, std::size_t count) -> void {
  for (std::size_t slot = 0; slot < count; ++slot) {
    set.WriteParticle(stage, slot, TaggedParticle(static_cast<double>(slot), slot));
  }
  ASSERT_TRUE(set.SetCount(stage, count).has_value());
}

TEST(FbrrtParticleSetTest, StartsEmptyAndHoldsTheInitialStatesAtStageZero) {
  Set set;
  EXPECT_EQ(set.count(0), 0U);
  EXPECT_EQ(set.count(2), 0U);

  const Set::StateBlock initial = Set::StateBlock::Random();
  set.SetInitialStates(initial);
  EXPECT_EQ(set.count(0), 6U);
  EXPECT_EQ(set.states(0), initial);
  EXPECT_EQ(set.accrued_costs(0), Set::ValueBlock::Zero());
}

TEST(FbrrtParticleSetTest, WrittenParticlesReadBackFieldByField) {
  Set set;
  FillStage(set, 1, 4);
  EXPECT_EQ(set.count(1), 4U);
  EXPECT_EQ(set.states(1).col(3), Eigen::Vector2d(3.0, -3.0));
  EXPECT_EQ(set.source_states(1).col(3), Eigen::Vector2d(3.5, 0.0));
  EXPECT_EQ(set.drifts(1).col(3), Eigen::Vector2d(0.0, 3.0));
  EXPECT_DOUBLE_EQ(set.controls(1)(0, 3), 0.3);
  EXPECT_DOUBLE_EQ(set.accrued_costs(1)[3], 6.0);
  ASSERT_EQ(set.parents(1).size(), 4U);
  EXPECT_EQ(set.parents(1)[3], 3U);
}

TEST(FbrrtParticleSetTest, SetCountRejectsMoreThanTheCapacity) {
  Set set;
  EXPECT_FALSE(set.SetCount(1, 7).has_value());
  EXPECT_FALSE(set.SetCount(3, 1).has_value());
  EXPECT_TRUE(set.SetCount(1, 6).has_value());
}

TEST(FbrrtParticleSetTest, KeepLowestKeepsTheLowestKeysInTheirOriginalOrder) {
  Set set;
  FillStage(set, 1, 6);
  Set::ValueBlock keys;
  keys << 5.0, 0.0, 4.0, 1.0, 3.0, 2.0;
  set.SetHeuristics(1, keys);

  ASSERT_TRUE(set.KeepLowest(1, keys, 3).has_value());
  ASSERT_EQ(set.count(1), 3U);
  // Slots 1, 3 and 5 hold the three lowest keys; they move to the front in that order, every field
  // with them.
  EXPECT_EQ(set.states(1).leftCols<3>().row(0), Eigen::RowVector3d(1.0, 3.0, 5.0));
  EXPECT_EQ(set.source_states(1).leftCols<3>().row(0), Eigen::RowVector3d(1.5, 3.5, 5.5));
  EXPECT_EQ(set.accrued_costs(1).leftCols<3>(), Eigen::RowVector3d(2.0, 6.0, 10.0));
  EXPECT_EQ(set.heuristics(1).leftCols<3>(), Eigen::RowVector3d(0.0, 1.0, 2.0));
  EXPECT_EQ(set.parents(1)[1], 3U);
}

TEST(FbrrtParticleSetTest, KeepLowestRemapsTheNextStagesParentsAndDetachesOrphans) {
  Set set;
  FillStage(set, 1, 6);
  FillStage(set, 2, 6);
  Set::ValueBlock keys;
  keys << 5.0, 0.0, 4.0, 1.0, 3.0, 2.0;

  ASSERT_TRUE(set.KeepLowest(1, keys, 3).has_value());
  // Stage 2's particle in slot s had parent s. Survivors 1, 3, 5 are now at 0, 1, 2.
  const auto parents = set.parents(2);
  EXPECT_EQ(parents[0], Set::kDetachedParent);
  EXPECT_EQ(parents[1], 0U);
  EXPECT_EQ(parents[2], Set::kDetachedParent);
  EXPECT_EQ(parents[3], 1U);
  EXPECT_EQ(parents[4], Set::kDetachedParent);
  EXPECT_EQ(parents[5], 2U);
  // The orphans themselves are untouched: they still carry their own edge.
  EXPECT_EQ(set.count(2), 6U);
  EXPECT_EQ(set.source_states(2).col(4), Eigen::Vector2d(4.5, 0.0));
}

TEST(FbrrtParticleSetTest, AgesStartAtZeroIncrementAndMoveWithTheirParticles) {
  Set set;
  FillStage(set, 1, 6);
  EXPECT_EQ(set.ages(1)[4], 0U);
  set.IncrementAges();
  set.IncrementAges();
  set.WriteParticle(1, 5, TaggedParticle(5.0, 5));
  Set::ValueBlock keys;
  keys << 5.0, 0.0, 4.0, 1.0, 3.0, 2.0;
  ASSERT_TRUE(set.KeepLowest(1, keys, 3).has_value());
  // Survivors 1, 3, 5: two aged twice, and the freshly rewritten slot 5 at zero.
  EXPECT_EQ(set.ages(1)[0], 2U);
  EXPECT_EQ(set.ages(1)[1], 2U);
  EXPECT_EQ(set.ages(1)[2], 0U);
}

TEST(FbrrtParticleSetTest, KeepLowestLeavesASmallEnoughStageAloneAndNeverThinsStageZero) {
  Set set;
  FillStage(set, 1, 3);
  ASSERT_TRUE(set.KeepLowest(1, Set::ValueBlock::Zero(), 4).has_value());
  EXPECT_EQ(set.count(1), 3U);
  EXPECT_FALSE(set.KeepLowest(0, Set::ValueBlock::Zero(), 1).has_value());
}

}  // namespace
}  // namespace fbsde_traj_opt
