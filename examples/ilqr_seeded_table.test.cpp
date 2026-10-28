// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#include "examples/ilqr_seeded_table.hpp"

#include <cstddef>
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "fbsde_traj_opt/solvers/box_ilqr_solver.hpp"
#include "fbsde_traj_opt/utils/result.hpp"
#include "fbsde_traj_opt/value_function/polynomial_value_function_approx.hpp"
#include "fbsde_traj_opt/value_function/soft_min_quadratic_value_function_approx.hpp"

namespace fbsde_traj_opt::examples {
namespace {

// Two stages and a terminal one; the terminal expansion is never used.
auto Expansions() -> std::vector<BoxIlqrValueExpansion<2>> {
  std::vector<BoxIlqrValueExpansion<2>> expansions(3);
  expansions[0] = {.state = Eigen::Vector2d(1.0, -0.5),
                   .value = 3.0,
                   .gradient = Eigen::Vector2d(0.4, -1.2),
                   .hessian = (Eigen::Matrix2d() << 4.0, 1.0, 1.0, 2.0).finished()};
  // A Hessian flat along one direction, as on a coasting arc.
  expansions[1] = {.state = Eigen::Vector2d(0.2, 0.3),
                   .value = 1.5,
                   .gradient = Eigen::Vector2d(0.5, 0.5),
                   .hessian = (Eigen::Matrix2d() << 1.0, 1.0, 1.0, 1.0).finished()};
  return expansions;
}

auto Quadratic(const BoxIlqrValueExpansion<2>& expansion, const Eigen::Vector2d& state) -> double {
  const Eigen::Vector2d offset = state - expansion.state;
  return expansion.value + expansion.gradient.dot(offset) + (0.5 * offset.dot(expansion.hessian * offset));
}

TEST(IlqrSeededTableTest, APolynomialHoldsEveryExpansionExactly) {
  using Polynomial = PolynomialValueFunctionApprox<2, 4>;
  const Result<Polynomial> basis =
      Polynomial::Make(Polynomial::ParameterVector::Zero(), Eigen::Vector2d(0.5, 0.0), Eigen::Vector2d(2.0, 3.0));
  ASSERT_TRUE(basis.has_value());
  const std::vector<BoxIlqrValueExpansion<2>> expansions = Expansions();
  const Result<std::vector<Polynomial>> table = IlqrSeededTable<2, 4>(*basis, expansions, IlqrSeedOptions{});
  ASSERT_TRUE(table.has_value()) << table.error();
  ASSERT_EQ(table->size(), 2U);
  for (std::size_t stage = 0; stage < 2; ++stage) {
    for (const Eigen::Vector2d state :
         {Eigen::Vector2d(0.0, 0.0), Eigen::Vector2d(-1.3, 2.1), Eigen::Vector2d(2.4, -2.0)}) {
      EXPECT_NEAR((*table)[stage](state), Quadratic(expansions[stage], state), 1e-8);
    }
    EXPECT_TRUE((*table)[stage].Gradient(expansions[stage].state).isApprox(expansions[stage].gradient, 1e-8));
  }
}

// Unspread, the soft-min model is the floored bowl: the value and the slope at the nominal state are
// iLQR's, and so is the curvature wherever it is above the floor.
TEST(IlqrSeededTableTest, ASoftMinBowlKeepsTheValueAndSlopeAtTheNominalState) {
  using SoftMin = SoftMinQuadraticValueFunctionApprox<2, 3>;
  const Result<SoftMin::ParameterVector> parameters =
      SoftMin::QuadraticParameters(Eigen::Matrix2d::Identity(), Eigen::Vector2d::Zero(), 0.0, 1.0, 1e-3);
  ASSERT_TRUE(parameters.has_value());
  const Result<SoftMin> basis = SoftMin::Make(*parameters, 1.0, 1e-3);
  ASSERT_TRUE(basis.has_value());
  const std::vector<BoxIlqrValueExpansion<2>> expansions = Expansions();
  const Result<std::vector<SoftMin>> table =
      IlqrSeededTable<2, 3>(*basis, expansions, IlqrSeedOptions{.metric_floor = 0.05, .center_spread = 0.0});
  ASSERT_TRUE(table.has_value()) << table.error();
  for (std::size_t stage = 0; stage < 2; ++stage) {
    EXPECT_NEAR((*table)[stage](expansions[stage].state), expansions[stage].value, 1e-9);
    EXPECT_TRUE((*table)[stage].Gradient(expansions[stage].state).isApprox(expansions[stage].gradient, 1e-9));
  }
  EXPECT_TRUE((*table)[0].Hessian(expansions[0].state).isApprox(expansions[0].hessian, 1e-9));
  // The flat direction (1, -1) gets the floor's curvature, 2 * 0.05, instead of none.
  const Eigen::Vector2d flat = Eigen::Vector2d(1.0, -1.0).normalized();
  EXPECT_NEAR(flat.dot((*table)[1].Hessian(expansions[1].state) * flat), 0.1, 1e-9);

  // Spread, the components separate but the bowl near the nominal state barely moves.
  const Result<std::vector<SoftMin>> spread = IlqrSeededTable<2, 3>(*basis, expansions, IlqrSeedOptions{});
  ASSERT_TRUE(spread.has_value()) << spread.error();
  EXPECT_NE((*spread)[0].Parameters(), (*table)[0].Parameters());
  EXPECT_NEAR((*spread)[0](expansions[0].state), expansions[0].value, 0.5);
}

}  // namespace
}  // namespace fbsde_traj_opt::examples
