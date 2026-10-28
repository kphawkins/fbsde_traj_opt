// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#include "fbsde_traj_opt/solvers/nearest_neighbor_dt_fbsde_solver.hpp"

#include <algorithm>
#include <cstddef>
#include <memory>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <gtest/gtest.h>

#include "fbsde_traj_opt/costs/l1_control_sde_running_cost_term.hpp"
#include "fbsde_traj_opt/costs/quadratic_regulator_sde_terminal_cost_term.hpp"
#include "fbsde_traj_opt/dynamics/const_diagonal_sde_diffusion_term.hpp"
#include "fbsde_traj_opt/dynamics/const_linear_sde_control_drift_mat_term.hpp"
#include "fbsde_traj_opt/dynamics/const_linear_sde_state_drift_term.hpp"
#include "fbsde_traj_opt/sampling/diagonal_covariance_normal_distribution.hpp"
#include "fbsde_traj_opt/sde/composed_forward_sde_model.hpp"
#include "fbsde_traj_opt/solvers/policy_rollout_evaluation.hpp"
#include "fbsde_traj_opt/utils/result.hpp"

namespace fbsde_traj_opt {
namespace {

// The L1 double integrator of dt_fbsde_iterative_solver.test.cpp.
constexpr double kTimeStep = 0.1;
constexpr std::size_t kStages = 31;

using Model = ComposedForwardSdeModel<2,
                                      1,
                                      ConstLinearSdeStateDriftTerm<2>,
                                      ConstLinearSdeControlDriftMatTerm<2, 1>,
                                      ConstDiagonalSdeDiffusionTerm<2>>;
using TerminalCost = QuadraticRegulatorSdeTerminalCostTerm<2>;
using Solver = NearestNeighborDtFbsdeSolver<2, 1, Model, TerminalCost>;
using Control = Eigen::Matrix<double, 1, 1>;

struct Fixture {
  Model model;
  L1ControlSdeRunningCostTerm<2, 1> running_cost;
  TerminalCost terminal_cost;
  DiagonalCovarianceNormalDistribution<2> initial;
};

auto MakeFixture() -> Fixture {
  Eigen::Matrix2d state_matrix;
  state_matrix << 0.0, kTimeStep,  //
      0.0, 0.0;
  const Result<ConstDiagonalSdeDiffusionTerm<2>> diffusion =
      ConstDiagonalSdeDiffusionTerm<2>::Make(Eigen::Vector2d(0.01, 0.05));
  const Result<L1ControlSdeRunningCostTerm<2, 1>> running_cost =
      L1ControlSdeRunningCostTerm<2, 1>::Make(0.1 * kTimeStep);
  const Result<DiagonalCovarianceNormalDistribution<2>> initial =
      DiagonalCovarianceNormalDistribution<2>::Make(Eigen::Vector2d(2.0, 0.5), Eigen::Vector2d(1e-3, 1e-3));
  EXPECT_TRUE(diffusion.has_value() && running_cost.has_value() && initial.has_value());
  return Fixture{
      .model = Model(ConstLinearSdeStateDriftTerm<2>(state_matrix),
                     ConstLinearSdeControlDriftMatTerm<2, 1>(Eigen::Vector2d(0.5 * kTimeStep * kTimeStep, kTimeStep)),
                     *diffusion),
      .running_cost = *running_cost,
      .terminal_cost = TerminalCost(Eigen::Vector2d(5.0, 2.0).asDiagonal()),
      .initial = *initial,
  };
}

auto MakeSolver(const Fixture& fixture, const Solver::Options& options, std::uint64_t seed = 7) -> Result<Solver> {
  return Solver::Make(fixture.model,
                      fixture.running_cost,
                      fixture.terminal_cost,
                      fixture.initial,
                      Control(-1.0),
                      Control(1.0),
                      kStages,
                      options,
                      seed);
}

auto CostOnNoise(const Fixture& fixture, const Solver::Policy& policy, std::uint64_t seed) -> double {
  const Eigen::Matrix2d factor = Eigen::Matrix2d(fixture.initial.Covariance()).llt().matrixL();
  return EvaluatePolicyByRollouts(fixture.model,
                                  policy,
                                  fixture.running_cost,
                                  fixture.terminal_cost,
                                  Eigen::Vector2d(fixture.initial.Mean()),
                                  factor,
                                  256,
                                  kStages - 1,
                                  seed)
      .total;
}

TEST(NearestNeighborValueTableTest, ReadsTheNearestSampleOrTheMeanOfTheNearest) {
  NearestNeighborValueTable<2> table;
  table.states.emplace_back(2, 3);
  table.states[0] << 0.0, 1.0, 5.0,  //
      0.0, 0.0, 0.0;
  table.values.emplace_back(1, 3);
  table.values[0] << 10.0, 20.0, 30.0;
  const Eigen::Vector2d ones = Eigen::Vector2d::Ones();
  EXPECT_EQ(table.Value(0, Eigen::Vector2d(0.4, 3.0), ones, 1), 10.0);
  EXPECT_EQ(table.Value(0, Eigen::Vector2d(0.6, -3.0), ones, 1), 20.0);
  EXPECT_EQ(table.Value(0, Eigen::Vector2d(4.0, 0.0), ones, 2), 25.0);
  // Scaling the first coordinate down makes the second one decide.
  NearestNeighborValueTable<2> tall = table;
  tall.states[0](1, 2) = 0.5;
  EXPECT_EQ(tall.Value(0, Eigen::Vector2d(4.0, 0.5), Eigen::Vector2d(1e-3, 1.0), 1), 30.0);
}

TEST(NearestNeighborValueTableTest, TheIndexAgreesWithBruteForce) {
  NearestNeighborValueTable<2> table;
  table.states.emplace_back(Eigen::Matrix<double, 2, Eigen::Dynamic>::Random(2, 500));
  table.values.emplace_back(Eigen::Matrix<double, 1, Eigen::Dynamic>::Random(1, 500));
  const Eigen::Vector2d inverse_scale(1.0, 3.0);
  NearestNeighborValueTable<2> indexed = table;
  indexed.BuildIndex(inverse_scale);
  const Eigen::Matrix<double, 2, Eigen::Dynamic> queries = 1.5 * Eigen::Matrix<double, 2, Eigen::Dynamic>::Random(2, 200);
  for (Eigen::Index query = 0; query < queries.cols(); ++query) {
    for (const std::size_t neighbors : {std::size_t{1}, std::size_t{5}}) {
      EXPECT_NEAR(indexed.Value(0, queries.col(query), inverse_scale, neighbors),
                  table.Value(0, queries.col(query), inverse_scale, neighbors),
                  1e-12);
    }
  }
}

TEST(NearestNeighborDtFbsdeSolverTest, RejectsInvalidOptions) {
  const Fixture fixture = MakeFixture();
  Solver::Options options;
  options.num_samples = 0;
  EXPECT_FALSE(MakeSolver(fixture, options).has_value());
  options = Solver::Options{};
  options.exploration_probability = 1.5;
  EXPECT_FALSE(MakeSolver(fixture, options).has_value());
  options = Solver::Options{};
  options.distance_scale = Eigen::Vector2d(1.0, 0.0);
  EXPECT_FALSE(MakeSolver(fixture, options).has_value());
  options = Solver::Options{};
  options.num_neighbors = 0;
  EXPECT_FALSE(MakeSolver(fixture, options).has_value());
}

TEST(NearestNeighborDtFbsdeSolverTest, BatchCostIsThePolicysCostOnTheFixedNoise) {
  const Fixture fixture = MakeFixture();
  Solver::Options options;
  options.num_samples = 128;
  Result<Solver> solver = MakeSolver(fixture, options);
  ASSERT_TRUE(solver.has_value()) << solver.error();
  (void)solver->Iterate();
  // The next batch is drawn under the latest policy with the same noise the batch used: sampler
  // coordinate (path, stage), which is also how EvaluatePolicyByRollouts draws.
  const Solver::Policy policy = solver->LatestPolicy();
  const auto report = solver->Iterate();
  const Eigen::Matrix2d factor = Eigen::Matrix2d(fixture.initial.Covariance()).llt().matrixL();
  const double replayed = EvaluatePolicyByRollouts(fixture.model,
                                                   policy,
                                                   fixture.running_cost,
                                                   fixture.terminal_cost,
                                                   Eigen::Vector2d(fixture.initial.Mean()),
                                                   factor,
                                                   options.num_samples,
                                                   kStages - 1,
                                                   7)
                              .total;
  EXPECT_NEAR(report.batch_cost, replayed, 1e-9);
  // The pathwise Y at stage 0 averages to the batch cost.
  EXPECT_NEAR(solver->LatestTable()->values[0].mean(), report.batch_cost, 1e-9);
}

class NearestNeighborDtFbsdeSolverEstimateTest : public ::testing::TestWithParam<NearestNeighborYEstimate> {};

TEST_P(NearestNeighborDtFbsdeSolverEstimateTest, ImprovesFarOnDoingNothing) {
  const Fixture fixture = MakeFixture();
  Solver::Options options;
  options.num_samples = 256;
  options.num_neighbors = 4;
  options.y_estimate = GetParam();
  Result<Solver> solver = MakeSolver(fixture, options);
  ASSERT_TRUE(solver.has_value()) << solver.error();
  const double coasting_cost = CostOnNoise(fixture, solver->CurrentPolicy(), 99);
  for (std::size_t iteration = 0; iteration < 20; ++iteration) {
    const auto report = solver->Iterate();
    EXPECT_EQ(report.iteration, iteration);
    EXPECT_GE(report.changed_control_fraction, 0.0);
    EXPECT_LE(report.changed_control_fraction, 1.0);
    if (iteration > 0) {
      EXPECT_LE(report.best_batch_cost, report.batch_cost);
      EXPECT_EQ(report.best_batch_cost, solver->BestBatchCost());
    }
  }
  // The kept policy also does far better than coasting on noise it never saw.
  EXPECT_LT(CostOnNoise(fixture, solver->CurrentPolicy(), 99), 0.15 * coasting_cost);
}

TEST(NearestNeighborDtFbsdeSolverTest, ReportsAFixedPointOnceTheBatchRepeats) {
  // A single path cannot chatter between neighbors, so its iteration settles within a few steps.
  const Fixture fixture = MakeFixture();
  Solver::Options options;
  options.num_samples = 1;
  options.initial_exploration_probability = 0.0;
  Result<Solver> solver = MakeSolver(fixture, options);
  ASSERT_TRUE(solver.has_value());
  bool fixed_point = false;
  for (std::size_t iteration = 0; iteration < 50 && !fixed_point; ++iteration) {
    fixed_point = solver->Iterate().fixed_point;
  }
  EXPECT_TRUE(fixed_point);
  const auto repeat = solver->Iterate();
  EXPECT_TRUE(repeat.fixed_point);
  EXPECT_EQ(repeat.changed_control_fraction, 0.0);
}

INSTANTIATE_TEST_SUITE_P(Estimates,
                         NearestNeighborDtFbsdeSolverEstimateTest,
                         ::testing::Values(NearestNeighborYEstimate::kPathwise, NearestNeighborYEstimate::kGreedyBackup));

TEST(NearestNeighborDtFbsdeSolverTest, RepeatsExactlyForTheSameSeed) {
  const Fixture fixture = MakeFixture();
  Solver::Options options;
  options.num_samples = 64;
  options.exploration_probability = 0.05;
  options.exploring_path_fraction = 0.5;
  Result<Solver> first = MakeSolver(fixture, options);
  Result<Solver> second = MakeSolver(fixture, options);
  ASSERT_TRUE(first.has_value() && second.has_value());
  for (std::size_t iteration = 0; iteration < 5; ++iteration) {
    EXPECT_EQ(first->Iterate().batch_cost, second->Iterate().batch_cost);
  }
}

TEST(NearestNeighborDtFbsdeSolverTest, MemoryKeepsEveryArchivedSampleInTheTable) {
  const Fixture fixture = MakeFixture();
  Solver::Options options;
  options.num_samples = 32;
  options.memory_batches = 3;
  options.y_estimate = NearestNeighborYEstimate::kGreedyBackup;
  Result<Solver> solver = MakeSolver(fixture, options);
  ASSERT_TRUE(solver.has_value());
  for (std::size_t iteration = 0; iteration < 5; ++iteration) {
    const auto report = solver->Iterate();
    EXPECT_FALSE(report.fixed_point);
    const Eigen::Index kept = static_cast<Eigen::Index>(32 * std::min<std::size_t>(iteration + 1, 3));
    EXPECT_EQ(solver->LatestTable()->states[0].cols(), kept);
    EXPECT_TRUE(solver->LatestTable()->values[0].allFinite());
  }
}

TEST(NearestNeighborDtFbsdeSolverTest, ControlsAreVerticesOrZero) {
  const Fixture fixture = MakeFixture();
  Solver::Options options;
  options.num_samples = 64;
  Result<Solver> solver = MakeSolver(fixture, options);
  ASSERT_TRUE(solver.has_value());
  for (std::size_t iteration = 0; iteration < 4; ++iteration) {
    (void)solver->Iterate();
  }
  const Solver::Policy policy = solver->CurrentPolicy();
  Eigen::Vector2d state = fixture.initial.Mean();
  for (std::size_t stage = 0; stage + 1 < kStages; ++stage) {
    const Control control = policy(stage, state);
    EXPECT_TRUE(control[0] == -1.0 || control[0] == 0.0 || control[0] == 1.0);
    state = fixture.model(stage, state, control, Eigen::Vector2d::Zero());
  }
}

}  // namespace
}  // namespace fbsde_traj_opt
