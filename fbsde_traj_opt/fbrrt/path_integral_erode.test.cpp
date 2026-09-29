// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#include "fbsde_traj_opt/fbrrt/path_integral_erode.hpp"

#include <algorithm>
#include <cstddef>
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
  ASSERT_TRUE((*erode)(after, table).has_value());

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
  ASSERT_TRUE((*PathIntegralErode::Make(4))(set, table).has_value());
  std::size_t orphans = 0;
  for (std::size_t stage = 2; stage < kStages; ++stage) {
    for (const std::size_t parent : set.parents(stage)) {
      orphans += parent == Set::kDetachedParent ? 1 : 0;
    }
  }
  EXPECT_GT(orphans, 0U);
}

TEST(PathIntegralErodeTest, RejectsAKeepCountOfZeroAndAMismatchedTable) {
  EXPECT_FALSE(PathIntegralErode::Make(0).has_value());
  Set set = MakeFullSet();
  std::vector<ValueFunction> table = MakeTable();
  table.pop_back();
  EXPECT_FALSE((*PathIntegralErode::Make(4))(set, table).has_value());
}

}  // namespace
}  // namespace fbsde_traj_opt
