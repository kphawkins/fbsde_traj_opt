// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#include "fbsde_traj_opt/fbrrt/path_integral_erode.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "fbsde_traj_opt/fbrrt/fbrrt_particle_set.hpp"
#include "fbsde_traj_opt/utils/result.hpp"
#include "fbsde_traj_opt/value_function/polynomial_value_function_approx.hpp"

namespace fbsde_traj_opt {
namespace {

constexpr std::size_t kCapacity = 16;
constexpr std::size_t kStages = 4;

using Set = FbrrtParticleSet<2, 1, kCapacity, kStages>;
using ValueFunction = PolynomialValueFunctionApprox<2, 2>;

static_assert(FbrrtEroder<PathIntegralErode, Set, ValueFunction>);

// V~_i(x) = (i + 1) |x|^2 at every stage, so the heuristic is easy to compute by hand.
auto MakeTable() -> std::vector<ValueFunction> {
  const Result<ValueFunction> blank =
      ValueFunction::Make(ValueFunction::ParameterVector::Zero(), Eigen::Vector2d::Zero(), Eigen::Vector2d::Ones());
  EXPECT_TRUE(blank.has_value());
  std::vector<ValueFunction> table(kStages, *blank);
  for (std::size_t stage = 0; stage < kStages; ++stage) {
    const Eigen::Matrix2d metric = static_cast<double>(stage + 1) * Eigen::Matrix2d::Identity();
    EXPECT_TRUE(table[stage].SetParameters(blank->QuadraticParameters(metric, 0.0)).has_value());
  }
  return table;
}

// Every stage full, particle `s` at every stage branching from slot `s` of the one before.
auto MakeFullSet() -> Set {
  Set set;
  set.SetInitialStates(Set::StateBlock::Zero());
  for (std::size_t stage = 1; stage < kStages; ++stage) {
    const Set::StateBlock states = Set::StateBlock::Random();
    const Set::ValueBlock accrued = Set::ValueBlock::Random().cwiseAbs();
    for (std::size_t slot = 0; slot < kCapacity; ++slot) {
      Set::Particle particle;
      particle.state = states.col(static_cast<Eigen::Index>(slot));
      particle.accrued_cost = accrued[static_cast<Eigen::Index>(slot)];
      particle.parent = slot;
      set.WriteParticle(stage, slot, particle);
    }
    EXPECT_TRUE(set.SetCount(stage, kCapacity).has_value());
  }
  return set;
}

TEST(PathIntegralErodeTest, KeepsTheLowestHeuristicParticlesAtEveryStage) {
  const std::vector<ValueFunction> table = MakeTable();
  const Set before = MakeFullSet();
  Set after = before;
  const Result<PathIntegralErode> erode = PathIntegralErode::Make(5);
  ASSERT_TRUE(erode.has_value());
  ASSERT_TRUE((*erode)(after, table, 0).has_value());

  EXPECT_EQ(after.count(0), kCapacity);
  for (std::size_t stage = 1; stage < kStages; ++stage) {
    ASSERT_EQ(after.count(stage), 5U);

    // rho = accrued + (stage + 1) |x|^2, by hand.
    std::vector<double> heuristics(kCapacity);
    for (std::size_t slot = 0; slot < kCapacity; ++slot) {
      const auto column = static_cast<Eigen::Index>(slot);
      heuristics[slot] = before.accrued_costs(stage)[column] +
                         (static_cast<double>(stage + 1) * before.states(stage).col(column).squaredNorm());
    }
    std::vector<double> sorted = heuristics;
    std::ranges::sort(sorted);
    for (std::size_t slot = 0; slot < 5; ++slot) {
      const double kept = after.heuristics(stage)[static_cast<Eigen::Index>(slot)];
      EXPECT_LE(kept, sorted[4] + 1e-12) << "stage " << stage;
      EXPECT_NEAR(
          kept,
          after.accrued_costs(stage)[static_cast<Eigen::Index>(slot)] +
              (static_cast<double>(stage + 1) * after.states(stage).col(static_cast<Eigen::Index>(slot)).squaredNorm()),
          1e-9);
    }
  }
}

TEST(PathIntegralErodeTest, ThinsEachStageOnItsOwnEvenWhenThatOrphansSurvivors) {
  // With no children check, a stage keeps its best particles whether or not their parents
  // survived. With independent random heuristics at each stage, some must have been orphaned.
  const std::vector<ValueFunction> table = MakeTable();
  Set set = MakeFullSet();
  ASSERT_TRUE((*PathIntegralErode::Make(4))(set, table, 0).has_value());
  std::size_t orphans = 0;
  for (std::size_t stage = 2; stage < kStages; ++stage) {
    for (const std::size_t parent : set.parents(stage)) {
      orphans += parent == Set::kDetachedParent ? 1 : 0;
    }
  }
  EXPECT_GT(orphans, 0U);
}

// A table whose value is zero everywhere, so the heuristic is the accrued cost alone.
auto MakeZeroTable() -> std::vector<ValueFunction> {
  const Result<ValueFunction> blank =
      ValueFunction::Make(ValueFunction::ParameterVector::Zero(), Eigen::Vector2d::Zero(), Eigen::Vector2d::Ones());
  EXPECT_TRUE(blank.has_value());
  return std::vector<ValueFunction>(kStages, *blank);
}

// Every stage full, particle `s` at state `states[s]`, with accrued cost `s` and an edge from `source`.
auto MakeSetAt(const std::array<Eigen::Vector2d, kCapacity>& states, const Eigen::Vector2d& source) -> Set {
  Set set;
  set.SetInitialStates(Set::StateBlock::Zero());
  for (std::size_t stage = 1; stage < kStages; ++stage) {
    for (std::size_t slot = 0; slot < kCapacity; ++slot) {
      Set::Particle particle;
      particle.state = states[slot];
      particle.source_state = source;
      particle.accrued_cost = static_cast<double>(slot);
      set.WriteParticle(stage, slot, particle);
    }
    EXPECT_TRUE(set.SetCount(stage, kCapacity).has_value());
  }
  return set;
}

auto Spread() -> std::array<Eigen::Vector2d, kCapacity> {
  std::array<Eigen::Vector2d, kCapacity> states{};
  for (std::size_t slot = 0; slot < kCapacity; ++slot) {
    states[slot] = Eigen::Vector2d(static_cast<double>(slot), 0.0);
  }
  return states;
}

TEST(PathIntegralErodeTest, ResamplingKeepsSomeWorseParticlesAndAColdTemperatureIsTheCut) {
  const std::vector<ValueFunction> table = MakeZeroTable();
  PathIntegralErodeOptions<double> hot;
  hot.temperature = 100.0;
  PathIntegralErodeOptions<double> cold;
  cold.temperature = 1e-6;

  // Hot: survivors are drawn almost uniformly, so over many seeds some worse half survive.
  std::size_t worse_kept = 0;
  for (std::uint64_t seed = 0; seed < 20; ++seed) {
    Set set = MakeSetAt(Spread(), Eigen::Vector2d::Zero());
    ASSERT_TRUE((*PathIntegralErode::Make(8, hot))(set, table, seed).has_value());
    ASSERT_EQ(set.count(1), 8U);
    for (std::size_t slot = 0; slot < 8; ++slot) {
      worse_kept += set.accrued_costs(1)[static_cast<Eigen::Index>(slot)] >= 8.0 ? 1 : 0;
    }
  }
  EXPECT_GT(worse_kept, 40U);

  // Cold: exactly the eight cheapest, whatever the seed.
  Set set = MakeSetAt(Spread(), Eigen::Vector2d::Zero());
  ASSERT_TRUE((*PathIntegralErode::Make(8, cold))(set, table, 5).has_value());
  EXPECT_LT(set.accrued_costs(1).leftCols<8>().maxCoeff(), 8.0);
}

TEST(PathIntegralErodeTest, OrphanPruningRemovesParticlesNothingAtThePreviousStageReaches) {
  const std::vector<ValueFunction> table = MakeZeroTable();
  // Every edge leaves from (100, 0), far from every particle, so every particle past stage 1 is an
  // orphan however cheap it is. Stage 1's edges leave from stage 0, which is never eroded.
  Set set = MakeSetAt(Spread(), Eigen::Vector2d(100.0, 0.0));
  PathIntegralErodeOptions<double> options;
  options.orphan_radius = 1.0;
  ASSERT_TRUE((*PathIntegralErode::Make(8, options))(set, table, 0).has_value());
  EXPECT_EQ(set.count(1), 8U);
  EXPECT_EQ(set.count(2), 0U);
  EXPECT_EQ(set.count(3), 0U);

  // Edges from the origin are within reach of the particle surviving there.
  Set reachable = MakeSetAt(Spread(), Eigen::Vector2d::Zero());
  ASSERT_TRUE((*PathIntegralErode::Make(8, options))(reachable, table, 0).has_value());
  EXPECT_EQ(reachable.count(3), 8U);
}

TEST(PathIntegralErodeTest, DensityWeightingThinsAClusterBeforeAnIsolatedParticle) {
  const std::vector<ValueFunction> table = MakeZeroTable();
  // The fifteen cheapest particles sit nearly on top of each other; the most expensive is alone.
  std::array<Eigen::Vector2d, kCapacity> states{};
  for (std::size_t slot = 0; slot + 1 < kCapacity; ++slot) {
    states[slot] = Eigen::Vector2d(1e-3 * static_cast<double>(slot), 0.0);
  }
  states[kCapacity - 1] = Eigen::Vector2d(50.0, 0.0);
  PathIntegralErodeOptions<double> options;
  options.density_weight = 2.0;

  Set plain = MakeSetAt(states, Eigen::Vector2d::Zero());
  ASSERT_TRUE((*PathIntegralErode::Make(8))(plain, table, 0).has_value());
  EXPECT_LT(plain.states(1).leftCols<8>().row(0).maxCoeff(), 1.0);

  Set dense = MakeSetAt(states, Eigen::Vector2d::Zero());
  ASSERT_TRUE((*PathIntegralErode::Make(8, options))(dense, table, 0).has_value());
  EXPECT_DOUBLE_EQ(dense.states(1).leftCols<8>().row(0).maxCoeff(), 50.0);
}

TEST(PathIntegralErodeTest, ParticlesPastTheAgeLimitAreRemovedFirst) {
  const std::vector<ValueFunction> table = MakeZeroTable();
  PathIntegralErodeOptions<double> options;
  options.max_age = 1;
  Set set = MakeSetAt(Spread(), Eigen::Vector2d::Zero());
  const PathIntegralErode erode = *PathIntegralErode::Make(8, options);
  ASSERT_TRUE(erode(set, table, 0).has_value());  // Survivors now age 1.
  EXPECT_EQ(set.count(1), 8U);
  ASSERT_TRUE(erode(set, table, 1).has_value());  // Age 1 is not past the limit: they stay, aging to 2.
  EXPECT_EQ(set.count(1), 8U);
  ASSERT_TRUE(erode(set, table, 2).has_value());  // Now every one is past it.
  EXPECT_EQ(set.count(1), 0U);
}

TEST(PathIntegralErodeTest, MakeRejectsBadOptions) {
  PathIntegralErodeOptions<double> options;
  options.temperature = -1.0;
  EXPECT_FALSE(PathIntegralErode::Make(4, options).has_value());
  options = {};
  options.density_weight = -1.0;
  EXPECT_FALSE(PathIntegralErode::Make(4, options).has_value());
  options = {};
  options.orphan_radius = 0.0;
  EXPECT_FALSE(PathIntegralErode::Make(4, options).has_value());
  EXPECT_FALSE(PathIntegralErode::Make(4, {}, {1.0, -1.0}).has_value());
}

TEST(PathIntegralErodeTest, RejectsAKeepCountOfZeroAndAMismatchedTable) {
  EXPECT_FALSE(PathIntegralErode::Make(0).has_value());
  Set set = MakeFullSet();
  std::vector<ValueFunction> table = MakeTable();
  table.pop_back();
  EXPECT_FALSE((*PathIntegralErode::Make(4))(set, table, 0).has_value());
}

}  // namespace
}  // namespace fbsde_traj_opt
