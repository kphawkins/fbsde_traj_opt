// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#include "fbsde_traj_opt/solvers/fbrrt_solver.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "fbsde_traj_opt/costs/l1_control_sde_running_cost_term.hpp"
#include "fbsde_traj_opt/costs/quadratic_regulator_sde_terminal_cost_term.hpp"
#include "fbsde_traj_opt/dynamics/const_diagonal_sde_diffusion_term.hpp"
#include "fbsde_traj_opt/dynamics/const_linear_sde_control_drift_mat_term.hpp"
#include "fbsde_traj_opt/dynamics/const_linear_sde_state_drift_term.hpp"
#include "fbsde_traj_opt/sampling/diagonal_covariance_normal_distribution.hpp"
#include "fbsde_traj_opt/sde/composed_forward_sde_model.hpp"
#include "fbsde_traj_opt/utils/result.hpp"
#include "fbsde_traj_opt/value_function/polynomial_value_function_approx.hpp"

namespace fbsde_traj_opt {
namespace {

// The L1 double integrator the DT-FBSDE solver is tested on: bring a unit mass to rest at the
// origin with a bounded force, paying for fuel.
constexpr double kTimeStep = 0.1;
constexpr std::size_t kStages = 31;
constexpr std::size_t kCapacity = 256;

using Model = ComposedForwardSdeModel<2,
                                      1,
                                      ConstLinearSdeStateDriftTerm<2>,
                                      ConstLinearSdeControlDriftMatTerm<2, 1>,
                                      ConstDiagonalSdeDiffusionTerm<2>>;
using ValueFunction = PolynomialValueFunctionApprox<2, 2>;
using TerminalCost = QuadraticRegulatorSdeTerminalCostTerm<2>;
using Components = DefaultFbrrtComponents<2, 1, kCapacity, Model, ValueFunction>;
using Solver = FbrrtSolver<2, 1, kCapacity, kStages, 64, Model, TerminalCost, ValueFunction, Components>;
using Control = Eigen::Matrix<double, 1, 1>;

struct Fixture {
  Model model;
  L1ControlSdeRunningCostTerm<2, 1> running_cost;
  TerminalCost terminal_cost;
  DiagonalCovarianceNormalDistribution<2> initial;
  ValueFunction terminal_value_function;
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
  const Eigen::Matrix2d terminal_matrix = Eigen::Vector2d(5.0, 2.0).asDiagonal();
  const Result<ValueFunction> blank =
      ValueFunction::Make(ValueFunction::ParameterVector::Zero(), Eigen::Vector2d(1.0, 0.0), Eigen::Vector2d(2.0, 2.0));
  EXPECT_TRUE(diffusion.has_value() && running_cost.has_value() && initial.has_value() && blank.has_value());
  ValueFunction terminal_value_function = *blank;
  EXPECT_TRUE(terminal_value_function.SetParameters(blank->QuadraticParameters(terminal_matrix, 0.0)).has_value());
  return Fixture{
      .model = Model(ConstLinearSdeStateDriftTerm<2>(state_matrix),
                     ConstLinearSdeControlDriftMatTerm<2, 1>(Eigen::Vector2d(0.5 * kTimeStep * kTimeStep, kTimeStep)),
                     *diffusion),
      .running_cost = *running_cost,
      .terminal_cost = TerminalCost(terminal_matrix),
      .initial = *initial,
      .terminal_value_function = terminal_value_function,
  };
}

auto MakeSolver(const Fixture& fixture, const FbrrtOptions<double>& options, std::size_t keep_count) -> Result<Solver> {
  TaylorBackwardPassOptions<double> backward;
  backward.sgd.learning_rate = 0.05;
  backward.epochs_per_stage = 2;
  RESULT_ASSIGN_OR_RETURN(Components components,
                          (MakeDefaultFbrrtComponents<kCapacity, kStages>(fixture.model,
                                                                          fixture.running_cost,
                                                                          Eigen::Vector2d(-1.0, -2.5),
                                                                          Eigen::Vector2d(3.0, 1.5),
                                                                          Control(-1.0),
                                                                          Control(1.0),
                                                                          backward,
                                                                          keep_count,
                                                                          fixture.terminal_value_function)));
  return Solver::Make(fixture.model,
                      fixture.running_cost,
                      fixture.terminal_cost,
                      fixture.initial,
                      Control(-1.0),
                      Control(1.0),
                      fixture.terminal_value_function,
                      fixture.terminal_value_function,
                      std::move(components),
                      options,
                      7);
}

TEST(FbrrtSolverTest, EndsFarBelowDoingNothingAndReportsEveryPhase) {
  const Fixture fixture = MakeFixture();
  Result<Solver> solver = MakeSolver(fixture, {}, kCapacity / 2);
  ASSERT_TRUE(solver.has_value()) << solver.error();

  // Doing nothing at all costs 5 (2 + 1.5 * 3)^2 + 2 * 0.5^2.
  const double coasting_cost = (5.0 * 6.5 * 6.5) + (2.0 * 0.25);
  double best = coasting_cost;
  for (std::size_t iteration = 0; iteration < 15; ++iteration) {
    const Result<FbrrtIterationReport<double>> report = solver->Iterate();
    ASSERT_TRUE(report.has_value()) << report.error();
    EXPECT_EQ(report->iteration, iteration);
    EXPECT_NEAR(report->mean_cost, report->mean_running_cost + report->mean_terminal_cost, 1e-9);
    EXPECT_TRUE(std::isfinite(report->rms_residual));
    EXPECT_GE(report->expand_seconds, 0.0);
    // The first iteration explores fully; the rest mix in the policy.
    EXPECT_DOUBLE_EQ(report->rrt_probability, iteration == 0 ? 1.0 : 0.5);
    EXPECT_DOUBLE_EQ(report->exploit_probability, iteration == 0 ? 0.0 : 0.5);
    best = std::min(best, report->mean_cost);
  }
  EXPECT_LT(best, 0.02 * coasting_cost);
  EXPECT_EQ(solver->iteration_count(), 15U);
}

TEST(FbrrtSolverTest, WithTheLineSearchTheCostNeverIncreases) {
  const Fixture fixture = MakeFixture();
  FbrrtOptions<double> options;
  options.line_search = true;
  Result<Solver> solver = MakeSolver(fixture, options, kCapacity / 2);
  ASSERT_TRUE(solver.has_value()) << solver.error();

  double previous = solver->Evaluate().total;
  for (std::size_t iteration = 0; iteration < 10; ++iteration) {
    const Result<FbrrtIterationReport<double>> report = solver->Iterate();
    ASSERT_TRUE(report.has_value()) << report.error();
    EXPECT_LE(report->mean_cost, previous);
    EXPECT_GE(report->candidates_tried, 1U);
    EXPECT_GE(report->accepted_step, 0.0);
    EXPECT_LE(report->accepted_step, 1.0);
    // An accepted step strictly improved; a rejected one left the policy exactly as it was.
    if (report->accepted_step > 0.0) {
      EXPECT_LT(report->mean_cost, previous);
    } else {
      EXPECT_EQ(report->mean_cost, previous);
    }
    EXPECT_DOUBLE_EQ(report->mean_cost, solver->Evaluate().total);
    previous = report->mean_cost;
  }
}

TEST(FbrrtSolverTest, WithAWorkingCopyTheCostNeverIncreasesAndLearningNeverStops) {
  const Fixture fixture = MakeFixture();
  FbrrtOptions<double> options;
  options.line_search = true;
  options.keep_working_copy = true;
  Result<Solver> solver = MakeSolver(fixture, options, kCapacity / 2);
  ASSERT_TRUE(solver.has_value()) << solver.error();

  double previous = solver->Evaluate().total;
  bool diverged = false;
  for (std::size_t iteration = 0; iteration < 10; ++iteration) {
    const Result<FbrrtIterationReport<double>> report = solver->Iterate();
    ASSERT_TRUE(report.has_value()) << report.error();
    EXPECT_LE(report->mean_cost, previous);
    EXPECT_DOUBLE_EQ(report->mean_cost, solver->Evaluate().total);
    previous = report->mean_cost;
    // The working table keeps its own update whether or not the deployed one took it.
    for (std::size_t stage = 0; stage + 1 < kStages; ++stage) {
      diverged = diverged ||
                 solver->working_value_functions()[stage].Parameters() != solver->value_functions()[stage].Parameters();
    }
  }
  EXPECT_TRUE(diverged);
  EXPECT_EQ(solver->working_value_functions().back().Parameters(), solver->value_functions().back().Parameters());
}

TEST(FbrrtSolverTest, MakeRejectsAWorkingCopyWithoutTheLineSearch) {
  const Fixture fixture = MakeFixture();
  FbrrtOptions<double> options;
  options.keep_working_copy = true;
  EXPECT_FALSE(MakeSolver(fixture, options, kCapacity / 2).has_value());
}

TEST(FbrrtSolverTest, ErodesToTheKeepCountAndRefillsToCapacity) {
  const Fixture fixture = MakeFixture();
  FbrrtOptions<double> options;
  options.keep_pre_erosion_snapshot = true;
  Result<Solver> solver = MakeSolver(fixture, options, 100);
  ASSERT_TRUE(solver.has_value()) << solver.error();
  ASSERT_TRUE(solver->Iterate().has_value());
  for (std::size_t stage = 1; stage < kStages; ++stage) {
    EXPECT_EQ(solver->pre_erosion_particles().count(stage), kCapacity);
    EXPECT_EQ(solver->particles().count(stage), 100U);
  }
  EXPECT_EQ(solver->particles().count(0), kCapacity);
}

TEST(FbrrtSolverTest, IsDeterministicInTheSeed) {
  const Fixture fixture = MakeFixture();
  Result<Solver> first = MakeSolver(fixture, {}, kCapacity / 2);
  Result<Solver> second = MakeSolver(fixture, {}, kCapacity / 2);
  ASSERT_TRUE(first.has_value() && second.has_value());
  for (std::size_t iteration = 0; iteration < 3; ++iteration) {
    const Result<FbrrtIterationReport<double>> a = first->Iterate();
    const Result<FbrrtIterationReport<double>> b = second->Iterate();
    ASSERT_TRUE(a.has_value() && b.has_value());
    EXPECT_EQ(a->mean_cost, b->mean_cost);
  }
}

TEST(FbrrtSolverTest, MakeRejectsOutOfRangeOptions) {
  const Fixture fixture = MakeFixture();
  FbrrtOptions<double> options;
  options.rrt_probability = 1.5;
  EXPECT_FALSE(MakeSolver(fixture, options, kCapacity / 2).has_value());
  options = {};
  options.exploit_probability = -0.1;
  EXPECT_FALSE(MakeSolver(fixture, options, kCapacity / 2).has_value());
  EXPECT_FALSE(MakeSolver(fixture, {}, kCapacity).has_value());
}

}  // namespace
}  // namespace fbsde_traj_opt
