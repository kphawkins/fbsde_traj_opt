// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#include "fbsde_traj_opt/solvers/dt_fbsde_iterative_solver.hpp"

#include <algorithm>
#include <array>
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
#include "fbsde_traj_opt/value_function/soft_min_quadratic_value_function_approx.hpp"

namespace fbsde_traj_opt {
namespace {

// The L1 double integrator of Hawkins (2021), Section 5.5.1: bring a unit mass to rest at the
// origin with a bounded force, paying for fuel.
constexpr double kTimeStep = 0.1;
constexpr std::size_t kStages = 31;

using Model = ComposedForwardSdeModel<2,
                                      1,
                                      ConstLinearSdeStateDriftTerm<2>,
                                      ConstLinearSdeControlDriftMatTerm<2, 1>,
                                      ConstDiagonalSdeDiffusionTerm<2>>;
using ValueFunction = PolynomialValueFunctionApprox<2, 2>;
using TerminalCost = QuadraticRegulatorSdeTerminalCostTerm<2>;
using Solver = DtFbsdeIterativeSolver<2, 1, 256, kStages, 64, Model, TerminalCost, ValueFunction>;

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

auto MakeSolver(const Fixture& fixture, const DtFbsdeIterativeOptions<double>& options) -> Result<Solver> {
  return Solver::Make(fixture.model,
                      fixture.running_cost,
                      fixture.terminal_cost,
                      fixture.initial,
                      Eigen::Matrix<double, 1, 1>(-1.0),
                      Eigen::Matrix<double, 1, 1>(1.0),
                      fixture.terminal_value_function,
                      fixture.terminal_value_function,
                      options,
                      7);
}

class DtFbsdeIterativeSolverTest : public ::testing::TestWithParam<DtFbsdeBackwardTarget> {};

TEST_P(DtFbsdeIterativeSolverTest, CostNeverIncreasesAndEndsFarBelowDoingNothing) {
  const Fixture fixture = MakeFixture();
  DtFbsdeIterativeOptions<double> options;
  options.backward_target = GetParam();
  options.target_floor = 0.0;
  options.exploration_decay = 0.5;
  Result<Solver> solver = MakeSolver(fixture, options);
  ASSERT_TRUE(solver.has_value()) << solver.error();

  // The table the solver starts from is the terminal cost at every stage, whose greedy policy
  // already pushes toward the origin; doing nothing at all costs 5 (2 + 1.5 * 3)^2 + 2 * 0.5^2.
  const double coasting_cost = (5.0 * 6.5 * 6.5) + (2.0 * 0.25);
  double previous = solver->CurrentCost();
  for (std::size_t iteration = 0; iteration < 12; ++iteration) {
    const Result<DtFbsdeIterationReport<double>> report = solver->Iterate();
    ASSERT_TRUE(report.has_value()) << report.error();
    EXPECT_EQ(report->iteration, iteration);
    // The improvement step only ever accepts a blend that lowers the evaluated cost.
    EXPECT_LE(report->mean_cost, previous + 1e-12);
    EXPECT_NEAR(report->mean_cost, report->mean_running_cost + report->mean_terminal_cost, 1e-9);
    previous = report->mean_cost;
  }
  EXPECT_LT(solver->CurrentCost(), 0.02 * coasting_cost);
  EXPECT_EQ(solver->IterationCount(), 12U);
}

TEST_P(DtFbsdeIterativeSolverTest, ControlsStayWithinTheirBoundsAndCoastSomewhere) {
  const Fixture fixture = MakeFixture();
  DtFbsdeIterativeOptions<double> options;
  options.backward_target = GetParam();
  options.target_floor = 0.0;
  Result<Solver> solver = MakeSolver(fixture, options);
  ASSERT_TRUE(solver.has_value()) << solver.error();
  for (std::size_t iteration = 0; iteration < 6; ++iteration) {
    ASSERT_TRUE(solver->Iterate().has_value());
  }

  const Solver::Policy policy = solver->CurrentPolicy();
  Eigen::Vector2d state = fixture.initial.Mean();
  std::size_t coasting = 0;
  for (std::size_t stage = 0; stage + 1 < kStages; ++stage) {
    const Eigen::Matrix<double, 1, 1> control = policy(stage, state);
    EXPECT_LE(std::abs(control[0]), 1.0);
    coasting += control[0] == 0.0 ? 1 : 0;
    state = fixture.model(stage, state, control, Eigen::Vector2d::Zero());
  }
  // The minimum-fuel solution of a double integrator decelerates, coasts, and brakes: some stage
  // must spend no fuel at all.
  EXPECT_GT(coasting, 0U);
}

class DtFbsdeIterativeSolverSoftMinTest : public ::testing::TestWithParam<ValueFunctionFitMethod> {};

TEST_P(DtFbsdeIterativeSolverSoftMinTest, FitsANonlinearRepresentation) {
  // The soft-min quadratic model has no closed-form fit; the solver falls back to SGD or
  // Gauss-Newton.
  using SoftMin = SoftMinQuadraticValueFunctionApprox<2, 3>;
  using SoftMinSolver = DtFbsdeIterativeSolver<2, 1, 256, kStages, 64, Model, TerminalCost, SoftMin>;
  constexpr double kSharpness = 1.0;
  constexpr double kDiagonalOffset = 1e-3;
  const Fixture fixture = MakeFixture();
  const Eigen::Matrix2d terminal_matrix = Eigen::Vector2d(5.0, 2.0).asDiagonal();

  const auto terminal_parameters =
      SoftMin::QuadraticParameters(terminal_matrix, Eigen::Vector2d::Zero(), 0.0, kSharpness, kDiagonalOffset);
  const auto initial_parameters =
      SoftMin::QuadraticParametersAt(terminal_matrix,
                                     {Eigen::Vector2d(0.0, 0.0), Eigen::Vector2d(1.0, 0.0), Eigen::Vector2d(-0.5, 0.5)},
                                     0.0,
                                     kSharpness,
                                     kDiagonalOffset);
  ASSERT_TRUE(terminal_parameters.has_value() && initial_parameters.has_value());
  const auto terminal = SoftMin::Make(*terminal_parameters, kSharpness, kDiagonalOffset);
  const auto initial = SoftMin::Make(*initial_parameters, kSharpness, kDiagonalOffset);
  ASSERT_TRUE(terminal.has_value() && initial.has_value());

  DtFbsdeIterativeOptions<double> options;
  options.target_floor = 0.0;
  options.exploration_decay = 0.5;
  options.sgd.learning_rate = 0.05;
  options.sgd_epochs = 20;
  options.nonlinear_fit_method = GetParam();
  Result<SoftMinSolver> solver = SoftMinSolver::Make(fixture.model,
                                                     fixture.running_cost,
                                                     fixture.terminal_cost,
                                                     fixture.initial,
                                                     Eigen::Matrix<double, 1, 1>(-1.0),
                                                     Eigen::Matrix<double, 1, 1>(1.0),
                                                     *initial,
                                                     *terminal,
                                                     options,
                                                     7);
  ASSERT_TRUE(solver.has_value()) << solver.error();

  const double coasting_cost = (5.0 * 6.5 * 6.5) + (2.0 * 0.25);
  double previous = solver->CurrentCost();
  for (std::size_t iteration = 0; iteration < 8; ++iteration) {
    const Result<DtFbsdeIterationReport<double>> report = solver->Iterate();
    ASSERT_TRUE(report.has_value()) << report.error();
    EXPECT_LE(report->mean_cost, previous + 1e-12);
    previous = report->mean_cost;
  }
  EXPECT_LT(solver->CurrentCost(), 0.05 * coasting_cost);
}

INSTANTIATE_TEST_SUITE_P(FitMethods,
                         DtFbsdeIterativeSolverSoftMinTest,
                         ::testing::Values(ValueFunctionFitMethod::kSgd, ValueFunctionFitMethod::kGaussNewton));

TEST(DtFbsdeIterativeSolverOptionsSoftMinTest, MakeRejectsZeroSgdEpochs) {
  const Fixture fixture = MakeFixture();
  DtFbsdeIterativeOptions<double> options;
  options.sgd_epochs = 0;
  EXPECT_FALSE(MakeSolver(fixture, options).has_value());
}

INSTANTIATE_TEST_SUITE_P(BackwardTargets,
                         DtFbsdeIterativeSolverTest,
                         ::testing::Values(DtFbsdeBackwardTarget::kCurrentPolicy,
                                           DtFbsdeBackwardTarget::kDoubleGreedy));

TEST(DtFbsdeIterativeSolverOptionsTest, MakeRejectsOutOfRangeOptions) {
  const Fixture fixture = MakeFixture();
  const auto rejected = [&fixture](auto mutate) {
    DtFbsdeIterativeOptions<double> options;
    mutate(options);
    return !MakeSolver(fixture, options).has_value();
  };
  EXPECT_TRUE(rejected([](auto& options) { options.exploration_probability = 1.5; }));
  EXPECT_TRUE(rejected([](auto& options) { options.initial_exploration_probability = -0.1; }));
  EXPECT_TRUE(rejected([](auto& options) { options.exploration_decay = 1.0; }));
  EXPECT_TRUE(rejected([](auto& options) { options.probe_fraction = 1.0; }));
  EXPECT_TRUE(rejected([](auto& options) { options.ridge = -1.0; }));
  EXPECT_TRUE(rejected([](auto& options) { options.entropy_temperature = -1.0; }));
  EXPECT_TRUE(rejected([](auto& options) { options.screening_rollouts = 65; }));
  EXPECT_TRUE(rejected([](auto& options) { options.diffusion_inflation_growth = 0.5; }));
  EXPECT_TRUE(rejected([](auto& options) { options.max_diffusion_inflation = 0.5; }));
  EXPECT_TRUE(rejected([](auto& options) { options.base_diffusion_inflation = 0.5; }));
  EXPECT_TRUE(rejected([](auto& options) {
    options.base_diffusion_inflation = 4.0;
    options.max_diffusion_inflation = 2.0;
  }));
  EXPECT_TRUE(rejected([](auto& options) { options.stall_relative_improvement = -0.1; }));
  EXPECT_TRUE(rejected([](auto& options) { options.stall_patience = 0; }));
  EXPECT_TRUE(rejected([](auto& options) { options.uninflated_path_fraction = 1.5; }));
  EXPECT_TRUE(rejected([](auto& options) { options.slope_weight = -1.0; }));
  EXPECT_FALSE(rejected([](auto& options) { options.screening_rollouts = 64; }));
  EXPECT_FALSE(rejected([](auto&) {}));
}

TEST(DtFbsdeIterativeSolverLineSearchTest, ScreeningAndStoppingUphillStillOnlyAcceptImprovements) {
  const Fixture fixture = MakeFixture();
  DtFbsdeIterativeOptions<double> options;
  options.backward_target = DtFbsdeBackwardTarget::kDoubleGreedy;
  options.target_floor = 0.0;
  options.exploration_decay = 0.5;
  options.stop_uphill_line_search = true;
  options.screening_rollouts = 16;
  Result<Solver> solver = MakeSolver(fixture, options);
  ASSERT_TRUE(solver.has_value()) << solver.error();

  const double coasting_cost = (5.0 * 6.5 * 6.5) + (2.0 * 0.25);
  double previous = solver->CurrentCost();
  for (std::size_t iteration = 0; iteration < 12; ++iteration) {
    const Result<DtFbsdeIterationReport<double>> report = solver->Iterate();
    ASSERT_TRUE(report.has_value()) << report.error();
    // A screen can only reject; acceptance is still decided on every evaluation rollout.
    EXPECT_LE(report->mean_cost, previous + 1e-12);
    EXPECT_EQ(report->accepted_step > 0.0, report->mean_cost < previous);
    previous = report->mean_cost;
  }
  EXPECT_LT(solver->CurrentCost(), 0.02 * coasting_cost);
}

TEST(DtFbsdeIterativeSolverInflationTest, InflatesTheNoiseWhileStalledAndResetsOnImprovement) {
  const Fixture fixture = MakeFixture();
  DtFbsdeIterativeOptions<double> options;
  options.target_floor = 0.0;
  options.exploration_decay = 0.5;
  // Only the full step is tried, so that some iterations stall.
  options.max_step_halvings = 0;
  options.diffusion_inflation_growth = 2.0;
  options.max_diffusion_inflation = 4.0;
  Result<Solver> solver = MakeSolver(fixture, options);
  ASSERT_TRUE(solver.has_value()) << solver.error();

  double expected_inflation = 1.0;
  std::size_t inflated = 0;
  for (std::size_t iteration = 0; iteration < 20; ++iteration) {
    const Result<DtFbsdeIterationReport<double>> report = solver->Iterate();
    ASSERT_TRUE(report.has_value()) << report.error();
    EXPECT_DOUBLE_EQ(report->diffusion_inflation, expected_inflation);
    inflated += report->diffusion_inflation > 1.0 ? 1 : 0;
    expected_inflation = report->accepted_step > 0.0 ? 1.0 : std::min(2.0 * expected_inflation, 4.0);
  }
  EXPECT_GT(inflated, 0U);
}

TEST(DtFbsdeIterativeSolverContinuationTest, ADiffusionScaleRescoresTheProblemButNotTheDeployedPolicy) {
  const Fixture fixture = MakeFixture();
  DtFbsdeIterativeOptions<double> options;
  options.target_floor = 0.0;
  options.screening_rollouts = 16;
  Result<Solver> solver = MakeSolver(fixture, options);
  ASSERT_TRUE(solver.has_value()) << solver.error();
  for (std::size_t iteration = 0; iteration < 3; ++iteration) {
    ASSERT_TRUE(solver->Iterate().has_value());
  }
  const double unscaled_cost = solver->CurrentCost();
  const Eigen::Vector2d state = fixture.initial.Mean();
  const Eigen::Matrix<double, 1, 1> deployed = solver->CurrentPolicy()(0, state);

  EXPECT_FALSE(solver->SetDiffusionScale(0.0).has_value());
  ASSERT_TRUE(solver->SetDiffusionScale(4.0).has_value());
  EXPECT_DOUBLE_EQ(solver->DiffusionScale(), 4.0);
  // Four times the noise on a quadratic terminal cost costs more on the same rollouts.
  EXPECT_GT(solver->CurrentCost(), unscaled_cost);
  // The deployed policy acts on the problem as posed, whatever the solver is working on.
  EXPECT_EQ(solver->CurrentPolicy()(0, state), deployed);

  const Result<DtFbsdeIterationReport<double>> report = solver->Iterate();
  ASSERT_TRUE(report.has_value()) << report.error();
  EXPECT_DOUBLE_EQ(report->diffusion_scale, 4.0);

  ASSERT_TRUE(solver->SetDiffusionScale(1.0).has_value());
  // Back at the problem as posed, the cost is re-scored there: below what four times the noise cost.
  EXPECT_LT(solver->CurrentCost(), report->mean_cost);
}

TEST(DtFbsdeIterativeSolverInflationTest, AnExplorationBaseSetBetweenIterationsIsWhatTheNextOneUses) {
  const Fixture fixture = MakeFixture();
  DtFbsdeIterativeOptions<double> options;
  options.target_floor = 0.0;
  options.uninflated_path_fraction = 0.5;
  Result<Solver> solver = MakeSolver(fixture, options);
  ASSERT_TRUE(solver.has_value()) << solver.error();
  EXPECT_FALSE(solver->SetExplorationInflation(0.5).has_value());
  ASSERT_TRUE(solver->SetExplorationInflation(12.0).has_value());
  const Result<DtFbsdeIterationReport<double>> report = solver->Iterate();
  ASSERT_TRUE(report.has_value()) << report.error();
  // A base above the cap is used as it is: the cap bounds the stalls' growth, not the base.
  EXPECT_DOUBLE_EQ(report->diffusion_inflation, 12.0);
  EXPECT_DOUBLE_EQ(report->diffusion_scale, 1.0);
}

TEST(DtFbsdeIterativeSolverSlopeTest, MatchingSlopesInClosedFormStillConverges) {
  const Fixture fixture = MakeFixture();
  DtFbsdeIterativeOptions<double> options;
  options.backward_target = DtFbsdeBackwardTarget::kDoubleGreedy;
  options.target_floor = 0.0;
  options.exploration_decay = 0.5;
  options.slope_weight = 1.0;
  Result<Solver> solver = MakeSolver(fixture, options);
  ASSERT_TRUE(solver.has_value()) << solver.error();
  const double coasting_cost = (5.0 * 6.5 * 6.5) + (2.0 * 0.25);
  double previous = solver->CurrentCost();
  for (std::size_t iteration = 0; iteration < 12; ++iteration) {
    const Result<DtFbsdeIterationReport<double>> report = solver->Iterate();
    ASSERT_TRUE(report.has_value()) << report.error();
    EXPECT_LE(report->mean_cost, previous + 1e-12);
    previous = report->mean_cost;
  }
  EXPECT_LT(solver->CurrentCost(), 0.02 * coasting_cost);
}

TEST(DtFbsdeIterativeSolverSlopeTest, MatchingSlopesBySgdStillConverges) {
  using SoftMin = SoftMinQuadraticValueFunctionApprox<2, 3>;
  using SoftMinSolver = DtFbsdeIterativeSolver<2, 1, 256, kStages, 64, Model, TerminalCost, SoftMin>;
  const Fixture fixture = MakeFixture();
  const Eigen::Matrix2d terminal_matrix = Eigen::Vector2d(5.0, 2.0).asDiagonal();
  const auto terminal_parameters =
      SoftMin::QuadraticParameters(terminal_matrix, Eigen::Vector2d::Zero(), 0.0, 1.0, 1e-3);
  const auto initial_parameters =
      SoftMin::QuadraticParametersAt(terminal_matrix,
                                     {Eigen::Vector2d(0.0, 0.0), Eigen::Vector2d(1.0, 0.0), Eigen::Vector2d(-0.5, 0.5)},
                                     0.0,
                                     1.0,
                                     1e-3);
  ASSERT_TRUE(terminal_parameters.has_value() && initial_parameters.has_value());
  const auto terminal = SoftMin::Make(*terminal_parameters, 1.0, 1e-3);
  const auto initial = SoftMin::Make(*initial_parameters, 1.0, 1e-3);
  ASSERT_TRUE(terminal.has_value() && initial.has_value());

  DtFbsdeIterativeOptions<double> options;
  options.backward_target = DtFbsdeBackwardTarget::kDoubleGreedy;
  options.target_floor = 0.0;
  options.exploration_decay = 0.5;
  options.sgd.learning_rate = 0.05;
  options.sgd_epochs = 20;
  options.slope_weight = 1.0;
  Result<SoftMinSolver> solver = SoftMinSolver::Make(fixture.model,
                                                     fixture.running_cost,
                                                     fixture.terminal_cost,
                                                     fixture.initial,
                                                     Eigen::Matrix<double, 1, 1>(-1.0),
                                                     Eigen::Matrix<double, 1, 1>(1.0),
                                                     *initial,
                                                     *terminal,
                                                     options,
                                                     7);
  ASSERT_TRUE(solver.has_value()) << solver.error();
  const double coasting_cost = (5.0 * 6.5 * 6.5) + (2.0 * 0.25);
  double previous = solver->CurrentCost();
  for (std::size_t iteration = 0; iteration < 8; ++iteration) {
    const Result<DtFbsdeIterationReport<double>> report = solver->Iterate();
    ASSERT_TRUE(report.has_value()) << report.error();
    EXPECT_LE(report->mean_cost, previous + 1e-12);
    previous = report->mean_cost;
  }
  EXPECT_LT(solver->CurrentCost(), 0.05 * coasting_cost);
}

}  // namespace
}  // namespace fbsde_traj_opt
