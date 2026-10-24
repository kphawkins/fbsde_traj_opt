// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#include "fbsde_traj_opt/solvers/dt_fbsde_iterative_solver.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>

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
#include "fbsde_traj_opt/value_function/linear_basis_value_function_approx.hpp"
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

// A seeded solver runs the policy of the table it was given: seeding a fresh solver with another's
// converged table reproduces that solver's cost, on the same evaluation rollouts.
TEST(DtFbsdeIterativeSolverSeedTest, ASeededTableIsTheOneThePolicyRuns) {
  const Fixture fixture = MakeFixture();
  DtFbsdeIterativeOptions<double> options;
  options.target_floor = 0.0;
  options.exploration_decay = 0.5;
  Result<Solver> trained = MakeSolver(fixture, options);
  ASSERT_TRUE(trained.has_value()) << trained.error();
  for (std::size_t iteration = 0; iteration < 8; ++iteration) {
    ASSERT_TRUE(trained->Iterate().has_value());
  }
  Result<Solver> seeded = MakeSolver(fixture, options);
  ASSERT_TRUE(seeded.has_value()) << seeded.error();
  EXPECT_GT(seeded->CurrentCost(), trained->CurrentCost());

  const std::span<const ValueFunction> table = trained->ValueFunctions();
  EXPECT_FALSE(seeded->SeedValueFunctions(table).has_value());
  const Result<double> predicted = seeded->SeededCost(table.first(kStages - 1));
  ASSERT_TRUE(predicted.has_value()) << predicted.error();
  ASSERT_TRUE(seeded->SeedValueFunctions(table.first(kStages - 1)).has_value());
  EXPECT_DOUBLE_EQ(*predicted, seeded->CurrentCost());
  EXPECT_DOUBLE_EQ(seeded->CurrentCost(), trained->CurrentCost());
  for (std::size_t stage = 0; stage < kStages; ++stage) {
    EXPECT_EQ(seeded->ValueFunctions()[stage].Parameters(), table[stage].Parameters());
  }
  // Iterating from the seed only ever improves on it.
  const double seeded_cost = seeded->CurrentCost();
  ASSERT_TRUE(seeded->Iterate().has_value());
  EXPECT_LE(seeded->CurrentCost(), seeded_cost);
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

namespace fbsde_traj_opt {
namespace {

// With sigma points the forward pass draws nothing: two solvers seeded differently sample the same
// states. The line search's rollouts still depend on the seed, so only the forward pass is compared.
TEST(DtFbsdeIterativeSolverSigmaPointTest, TheForwardPassIsDeterministic) {
  const Fixture fixture = MakeFixture();
  DtFbsdeIterativeOptions<double> options;
  options.backward_target = DtFbsdeBackwardTarget::kDoubleGreedy;
  options.sigma_point_rule = SigmaPointRule::kUnscented;
  Result<Solver> first = MakeSolver(fixture, options);
  Result<Solver> second = Solver::Make(fixture.model,
                                       fixture.running_cost,
                                       fixture.terminal_cost,
                                       fixture.initial,
                                       Eigen::Matrix<double, 1, 1>(-1.0),
                                       Eigen::Matrix<double, 1, 1>(1.0),
                                       fixture.terminal_value_function,
                                       fixture.terminal_value_function,
                                       options,
                                       12345);
  ASSERT_TRUE(first.has_value() && second.has_value());
  ASSERT_TRUE(first->Iterate().has_value());
  ASSERT_TRUE(second->Iterate().has_value());
  for (std::size_t stage = 0; stage < kStages; ++stage) {
    EXPECT_EQ(first->SampledStates()[stage], second->SampledStates()[stage]);
  }
}

// Every rule, with mixed exploration and probes, drives the cost far below coasting, as the random
// forward pass does, and keeps every sampled state finite.
TEST(DtFbsdeIterativeSolverSigmaPointTest, EveryRuleConvergesOnALinearProblem) {
  const Fixture fixture = MakeFixture();
  const double coasting_cost = (5.0 * 6.5 * 6.5) + (2.0 * 0.25);
  for (const SigmaPointRule rule : {SigmaPointRule::kUnscented,
                                    SigmaPointRule::kCubature5,
                                    SigmaPointRule::kConjugate,
                                    SigmaPointRule::kGaussHermite3}) {
    DtFbsdeIterativeOptions<double> options;
    options.backward_target = DtFbsdeBackwardTarget::kDoubleGreedy;
    options.target_floor = 0.0;
    options.sigma_point_rule = rule;
    options.base_diffusion_inflation = 6.0;
    options.max_diffusion_inflation = 6.0;
    options.uninflated_path_fraction = 0.5;
    Result<Solver> solver = MakeSolver(fixture, options);
    ASSERT_TRUE(solver.has_value()) << solver.error();
    double previous = solver->CurrentCost();
    for (std::size_t iteration = 0; iteration < 12; ++iteration) {
      const Result<DtFbsdeIterationReport<double>> report = solver->Iterate();
      ASSERT_TRUE(report.has_value()) << report.error();
      EXPECT_LE(report->mean_cost, previous + 1e-12);
      previous = report->mean_cost;
      EXPECT_TRUE(solver->SampledStates()[kStages - 1].allFinite());
    }
    EXPECT_LT(solver->CurrentCost(), 0.02 * coasting_cost) << static_cast<int>(rule);
  }
}

// Randomly rotated points still match the covariance: the first stage's points, all drawn from the
// initial distribution, have its mean, whatever the rotation.
TEST(DtFbsdeIterativeSolverSigmaPointTest, RandomlyRotatedPointsKeepTheInitialMean) {
  const Fixture fixture = MakeFixture();
  DtFbsdeIterativeOptions<double> options;
  options.backward_target = DtFbsdeBackwardTarget::kDoubleGreedy;
  options.sigma_point_rule = SigmaPointRule::kUnscented;
  options.sigma_point_probes = false;
  options.randomize_sigma_point_rotation = true;
  Result<Solver> solver = MakeSolver(fixture, options);
  ASSERT_TRUE(solver.has_value());
  ASSERT_TRUE(solver->Iterate().has_value());
  // Five points per half and 256 columns: the first ten columns are each half's set once.
  const SigmaPointSet<2> set = MakeSigmaPoints<2>(SigmaPointRule::kUnscented);
  Eigen::Vector2d mean = Eigen::Vector2d::Zero();
  for (std::size_t point = 0; point < set.points.size(); ++point) {
    mean += set.weights[point] * solver->SampledStates()[0].col(static_cast<Eigen::Index>(2 * point)).eval();
  }
  EXPECT_NEAR(mean[0], 2.0, 1e-9);
  EXPECT_NEAR(mean[1], 0.5, 1e-9);
}

}  // namespace
}  // namespace fbsde_traj_opt

namespace fbsde_traj_opt {
namespace {

using LinearPoly = LinearBasisValueFunctionApprox<2, MonomialBasis<2, 2>>;
using LinearSolver = DtFbsdeIterativeSolver<2, 1, 256, kStages, 64, Model, TerminalCost, LinearPoly>;

auto MakeLinearSolver(const Fixture& fixture, const DtFbsdeIterativeOptions<double>& options, std::uint64_t seed)
    -> Result<LinearSolver> {
  auto blank = LinearPoly::Make(LinearPoly::ParameterVector::Zero(),
                                Eigen::Vector2d(1.0, 0.0),
                                Eigen::Vector2d(2.0, 2.0),
                                MonomialBasis<2, 2>{});
  EXPECT_TRUE(blank.has_value());
  // The terminal cost 5 x0^2 + 2 x1^2 in the normalized monomials (1, z0, z1, z0^2, z0 z1, z1^2) is
  // fitted on a grid rather than written out, so the test does not depend on the basis order.
  constexpr int kGrid = 121;
  Eigen::Matrix<double, 2, kGrid> grid;
  Eigen::Matrix<double, 1, kGrid> values;
  for (int index = 0; index < kGrid; ++index) {
    grid.col(index) = Eigen::Vector2d(-1.0 + (0.4 * (index % 11)), -2.0 + (0.4 * (index / 11)));
    values[index] = (5.0 * grid(0, index) * grid(0, index)) + (2.0 * grid(1, index) * grid(1, index));
  }
  EXPECT_TRUE(blank->FitWeightedLeastSquares<kGrid>(grid, values, Eigen::Matrix<double, 1, kGrid>::Ones(), 0.0)
                  .has_value());
  return LinearSolver::Make(fixture.model,
                            fixture.running_cost,
                            fixture.terminal_cost,
                            fixture.initial,
                            Eigen::Matrix<double, 1, 1>(-1.0),
                            Eigen::Matrix<double, 1, 1>(1.0),
                            *blank,
                            *blank,
                            options,
                            seed);
}

// The backward pass's targets, checked against exact Bellman backups on a linear-quadratic problem,
// for both backward targets, with and without slope rows. Two cases, each one where the backup has a
// closed form:
//
//   * Fuel so expensive the control is zero everywhere: every stage's fit must be the quadratic of
//     P_k = A' P_{k+1} A, P_K = F, down the whole horizon. This checks that the control is chosen
//     and scored by the right tables at the right stage, that the noise term only shifts the level,
//     and that the halves are averaged.
//   * Free fuel and bounds too wide to bind: the last control stage's fit must be the one-step
//     Riccati backup P_{K-1} = A' F A - A' F b (b' F b)^-1 b' F A, which checks the interior
//     minimizer's target. (Further back the problem is deadbeat -- one control zeroes both states in
//     two steps -- so every control ties and the backup is degenerate.)
TEST(DtFbsdeIterativeSolverTargetTest, FitsAreExactBellmanBackupsOnALinearQuadraticProblem) {
  Eigen::Matrix2d transition = Eigen::Matrix2d::Identity();
  transition(0, 1) = kTimeStep;
  const Eigen::Vector2d column(0.5 * kTimeStep * kTimeStep, kTimeStep);
  const Eigen::Matrix2d terminal = Eigen::Vector2d(5.0, 2.0).asDiagonal();
  std::vector<Eigen::Matrix2d> coasting(kStages);
  coasting[kStages - 1] = terminal;
  for (std::size_t stage = kStages - 1; stage-- > 0;) {
    coasting[stage] = transition.transpose() * coasting[stage + 1] * transition;
  }
  const Eigen::RowVector2d gain = (column.transpose() * terminal * transition) / column.dot(terminal * column);
  const Eigen::Matrix2d one_step =
      transition.transpose() * terminal * transition - (transition.transpose() * terminal * column) * gain;

  for (const bool coast : {true, false}) {
    Fixture fixture = MakeFixture();
    const Result<L1ControlSdeRunningCostTerm<2, 1>> fuel = L1ControlSdeRunningCostTerm<2, 1>::Make(coast ? 1e6 : 0.0);
    ASSERT_TRUE(fuel.has_value());
    fixture.running_cost = *fuel;
    // A wide initial spread, so that without exploration the samples still span the state space and
    // the six-coefficient fits are well conditioned; the fixture's tight start leaves them on a thin
    // tube around one trajectory, where a quadratic is fitted only to a few digits.
    const Result<DiagonalCovarianceNormalDistribution<2>> wide =
        DiagonalCovarianceNormalDistribution<2>::Make(Eigen::Vector2d(1.0, 0.0), Eigen::Vector2d(1.0, 1.0));
    ASSERT_TRUE(wide.has_value());
    fixture.initial = *wide;
    const double bound = coast ? 1.0 : 1e4;
    for (const DtFbsdeBackwardTarget target : {DtFbsdeBackwardTarget::kDoubleGreedy,
                                               DtFbsdeBackwardTarget::kCurrentPolicy,
                                               DtFbsdeBackwardTarget::kTaylorNoiseless}) {
      for (const double slope_weight : {0.0, 1.0}) {
        DtFbsdeIterativeOptions<double> options;
        options.backward_target = target;
        options.initial_exploration_probability = 0.0;
        options.exploration_probability = 0.0;
        options.probe_fraction = 0.0;
        options.ridge = 1e-12;
        options.slope_weight = slope_weight;
        Result<LinearSolver> tables = MakeLinearSolver(fixture, DtFbsdeIterativeOptions<double>{}, 5);
        ASSERT_TRUE(tables.has_value()) << tables.error();
        Result<LinearSolver> solver = LinearSolver::Make(fixture.model,
                                                         fixture.running_cost,
                                                         fixture.terminal_cost,
                                                         fixture.initial,
                                                         Eigen::Matrix<double, 1, 1>(-bound),
                                                         Eigen::Matrix<double, 1, 1>(bound),
                                                         tables->ValueFunctions()[0],
                                                         tables->ValueFunctions()[kStages - 1],
                                                         options,
                                                         5);
        ASSERT_TRUE(solver.has_value()) << solver.error();
        ASSERT_TRUE(solver->Iterate().has_value());
        const auto check = [&](std::size_t stage, const Eigen::Matrix2d& expected) {
          const Eigen::Matrix2d hessian = solver->FittedValueFunctions()[stage].Hessian(Eigen::Vector2d(0.3, -0.2));
          EXPECT_LT((hessian - (2.0 * expected)).norm(), 1e-6 * (1.0 + expected.norm()))
              << (coast ? "coasting" : "free fuel") << ", stage " << stage << ", target " << static_cast<int>(target)
              << ", slope weight " << slope_weight << "\nfitted\n" << hessian << "\nexpected\n" << 2.0 * expected;
        };
        if (coast) {
          for (std::size_t stage = 0; stage + 1 < kStages; ++stage) {
            check(stage, coasting[stage]);
          }
        } else {
          check(kStages - 2, one_step);
        }
      }
    }
  }
}

// The Taylor noiseless estimator is exact for a quadratic next-stage value function whatever drift
// the sample was taken with, so even with every path exploring -- every sampled control a random
// one of {-1, 0, 1}, far from the coasting policy's zero -- its targets are the coasting policy's
// exact values, and each stage's fit follows P_k = A' P_{k+1} A. The Girsanov drift it reports is
// then well above zero, because the sampling is off-policy.
TEST(DtFbsdeIterativeSolverTargetTest, TaylorNoiselessTargetsAreExactOffPolicy) {
  Fixture fixture = MakeFixture();
  const Result<L1ControlSdeRunningCostTerm<2, 1>> fuel = L1ControlSdeRunningCostTerm<2, 1>::Make(1e6);
  ASSERT_TRUE(fuel.has_value());
  fixture.running_cost = *fuel;
  const Result<DiagonalCovarianceNormalDistribution<2>> wide =
      DiagonalCovarianceNormalDistribution<2>::Make(Eigen::Vector2d(1.0, 0.0), Eigen::Vector2d(1.0, 1.0));
  ASSERT_TRUE(wide.has_value());
  fixture.initial = *wide;
  Eigen::Matrix2d transition = Eigen::Matrix2d::Identity();
  transition(0, 1) = kTimeStep;
  std::vector<Eigen::Matrix2d> coasting(kStages);
  coasting[kStages - 1] = Eigen::Vector2d(5.0, 2.0).asDiagonal();
  for (std::size_t stage = kStages - 1; stage-- > 0;) {
    coasting[stage] = transition.transpose() * coasting[stage + 1] * transition;
  }
  for (const double slope_weight : {0.0, 1.0}) {
    DtFbsdeIterativeOptions<double> options;
    options.backward_target = DtFbsdeBackwardTarget::kTaylorNoiseless;
    options.initial_exploration_probability = 1.0;
    options.ridge = 1e-12;
    options.slope_weight = slope_weight;
    Result<LinearSolver> solver = MakeLinearSolver(fixture, options, 5);
    ASSERT_TRUE(solver.has_value()) << solver.error();
    const Result<DtFbsdeIterationReport<double>> report = solver->Iterate();
    ASSERT_TRUE(report.has_value()) << report.error();
    EXPECT_GT(report->mean_girsanov_drift, 1.0);
    for (std::size_t stage = 0; stage + 1 < kStages; ++stage) {
      const Eigen::Matrix2d hessian = solver->FittedValueFunctions()[stage].Hessian(Eigen::Vector2d(0.3, -0.2));
      EXPECT_LT((hessian - (2.0 * coasting[stage])).norm(), 1e-6 * (1.0 + coasting[stage].norm()))
          << "stage " << stage << ", slope weight " << slope_weight;
    }
  }
}

// With exploration kept on, the Taylor noiseless target's first-order correction g' (F^mu - K) is
// nonzero, and every variant -- greedy target policy, held exploratory arcs, Girsanov weighting --
// still only ever lowers the cost and converges far below coasting.
TEST(DtFbsdeIterativeSolverTargetTest, TaylorNoiselessVariantsConvergeWithExploration) {
  const Fixture fixture = MakeFixture();
  const double coasting_cost = (5.0 * 6.5 * 6.5) + (2.0 * 0.25);
  struct Variant {
    DtFbsdeTaylorTargetPolicy target_policy;
    std::size_t hold;
    double girsanov_weight;
  };
  for (const Variant variant : {Variant{DtFbsdeTaylorTargetPolicy::kCurrent, 1, 0.0},
                                Variant{DtFbsdeTaylorTargetPolicy::kGreedy, 1, 0.0},
                                Variant{DtFbsdeTaylorTargetPolicy::kCurrent, 4, 0.0},
                                Variant{DtFbsdeTaylorTargetPolicy::kCurrent, 1, 1.0}}) {
    DtFbsdeIterativeOptions<double> options;
    options.backward_target = DtFbsdeBackwardTarget::kTaylorNoiseless;
    options.target_floor = 0.0;
    options.exploration_probability = 0.3;
    options.initial_exploration_probability = 0.3;
    options.taylor_target_policy = variant.target_policy;
    options.exploration_hold_stages = variant.hold;
    options.girsanov_weight = variant.girsanov_weight;
    Result<LinearSolver> solver = MakeLinearSolver(fixture, options, 7);
    ASSERT_TRUE(solver.has_value()) << solver.error();
    double previous = solver->CurrentCost();
    double correction = 0.0;
    for (std::size_t iteration = 0; iteration < 12; ++iteration) {
      const Result<DtFbsdeIterationReport<double>> report = solver->Iterate();
      ASSERT_TRUE(report.has_value()) << report.error();
      EXPECT_LE(report->mean_cost, previous + 1e-12);
      previous = report->mean_cost;
      correction += report->mean_first_order_correction;
    }
    EXPECT_GT(correction, 0.0);
    EXPECT_LT(solver->CurrentCost(), 0.05 * coasting_cost) << "hold " << variant.hold;
  }
}

// With a quadratic value function the second-order expansion is exact, so the full estimator's
// Taylor residual is zero to rounding however far off-policy the samples are, while dropping the
// first-order term leaves the targets wrong by about the size of that term.
TEST(DtFbsdeIterativeSolverTargetTest, TaylorResidualVanishesForAQuadraticAndTheAblationsDoNot) {
  const Fixture fixture = MakeFixture();
  for (const std::size_t order : {std::size_t{2}, std::size_t{0}}) {
    DtFbsdeIterativeOptions<double> options;
    options.backward_target = DtFbsdeBackwardTarget::kTaylorNoiseless;
    options.target_floor = -1e9;
    options.exploration_probability = 0.3;
    options.initial_exploration_probability = 0.3;
    options.taylor_correction_order = order;
    options.measure_taylor_residual = true;
    Result<LinearSolver> solver = MakeLinearSolver(fixture, options, 11);
    ASSERT_TRUE(solver.has_value()) << solver.error();
    for (std::size_t iteration = 0; iteration < 3; ++iteration) {
      const Result<DtFbsdeIterationReport<double>> report = solver->Iterate();
      ASSERT_TRUE(report.has_value()) << report.error();
      EXPECT_GT(report->off_policy_fraction, 0.1);
      EXPECT_GT(report->mean_off_policy_correction, report->mean_first_order_correction);
      if (iteration == 0) {
        continue;  // The first pass fits the blank tables' targets, which have no gradient.
      }
      if (order == 2) {
        EXPECT_LT(report->mean_taylor_residual, 1e-8 * (1.0 + report->mean_target_magnitude));
      } else {
        EXPECT_GT(report->mean_taylor_residual, 0.1 * report->mean_off_policy_correction);
      }
    }
  }
}

// Only the exploring fraction of paths ever leaves the policy: with none, every path follows it even
// at an exploration probability of one, and with half, about half of what full exploration gives.
TEST(DtFbsdeIterativeSolverTargetTest, OnlyExploringPathsLeaveThePolicy) {
  const Fixture fixture = MakeFixture();
  double previous_fraction = -1.0;
  for (const double explorers : {0.0, 0.5, 1.0}) {
    DtFbsdeIterativeOptions<double> options;
    options.backward_target = DtFbsdeBackwardTarget::kTaylorNoiseless;
    options.target_floor = 0.0;
    options.exploration_probability = 1.0;
    options.initial_exploration_probability = 1.0;
    options.exploring_path_fraction = explorers;
    Result<LinearSolver> solver = MakeLinearSolver(fixture, options, 5);
    ASSERT_TRUE(solver.has_value()) << solver.error();
    ASSERT_TRUE(solver->Iterate().has_value());
    const Result<DtFbsdeIterationReport<double>> report = solver->Iterate();
    ASSERT_TRUE(report.has_value()) << report.error();
    if (explorers == 0.0) {
      EXPECT_EQ(report->off_policy_fraction, 0.0);
    } else {
      EXPECT_GT(report->off_policy_fraction, previous_fraction + 0.1);
    }
    previous_fraction = report->off_policy_fraction;
  }
}

// A memory of zero is the plain fit: the solver takes the same steps either way.
TEST(DtFbsdeIterativeSolverNormalEquationTest, ZeroMemoryIsThePlainFit) {
  const Fixture fixture = MakeFixture();
  DtFbsdeIterativeOptions<double> options;
  options.backward_target = DtFbsdeBackwardTarget::kDoubleGreedy;
  options.target_floor = 0.0;
  Result<LinearSolver> plain = MakeLinearSolver(fixture, options, 3);
  options.reset_normal_equations_on_acceptance = true;
  Result<LinearSolver> zero = MakeLinearSolver(fixture, options, 3);
  ASSERT_TRUE(plain.has_value() && zero.has_value());
  for (std::size_t iteration = 0; iteration < 5; ++iteration) {
    ASSERT_TRUE(plain->Iterate().has_value());
    ASSERT_TRUE(zero->Iterate().has_value());
    EXPECT_DOUBLE_EQ(plain->CurrentCost(), zero->CurrentCost());
  }
}

// Carried normal equations, with and without the reset on acceptance, still only ever lower the
// cost and drive it far below coasting. On this easy problem, with a large batch, every update is
// accepted, so remembering old batches only adds stale targets and slows the descent; and the reset
// then clears the equations every iteration, which makes the method exactly the plain fit.
TEST(DtFbsdeIterativeSolverNormalEquationTest, CarriedNormalEquationsStillConverge) {
  const Fixture fixture = MakeFixture();
  const double coasting_cost = (5.0 * 6.5 * 6.5) + (2.0 * 0.25);
  double plain_cost = 0.0;
  for (const auto& [memory, reset] : {std::pair{0.0, false}, std::pair{0.9, false}, std::pair{1.0, true}}) {
    DtFbsdeIterativeOptions<double> options;
    options.backward_target = DtFbsdeBackwardTarget::kDoubleGreedy;
    options.target_floor = 0.0;
    options.exploration_decay = 0.5;
    options.normal_equation_memory = memory;
    options.reset_normal_equations_on_acceptance = reset;
    Result<LinearSolver> solver = MakeLinearSolver(fixture, options, 7);
    ASSERT_TRUE(solver.has_value()) << solver.error();
    double previous = solver->CurrentCost();
    for (std::size_t iteration = 0; iteration < 12; ++iteration) {
      const Result<DtFbsdeIterationReport<double>> report = solver->Iterate();
      ASSERT_TRUE(report.has_value()) << report.error();
      EXPECT_LE(report->mean_cost, previous + 1e-12);
      previous = report->mean_cost;
    }
    EXPECT_LT(solver->CurrentCost(), 0.05 * coasting_cost) << "memory " << memory << ", reset " << reset;
    if (memory == 0.0) {
      plain_cost = solver->CurrentCost();
    }
    if (reset) {
      EXPECT_DOUBLE_EQ(solver->CurrentCost(), plain_cost);
    }
  }
}

// Replaying earlier batches, their targets recomputed, still only ever lowers the cost and converges;
// and replay cannot be combined with a memory or the single-table backward target.
TEST(DtFbsdeIterativeSolverNormalEquationTest, ReplayedBatchesStillConverge) {
  const Fixture fixture = MakeFixture();
  const double coasting_cost = (5.0 * 6.5 * 6.5) + (2.0 * 0.25);
  DtFbsdeIterativeOptions<double> options;
  options.backward_target = DtFbsdeBackwardTarget::kDoubleGreedy;
  options.target_floor = 0.0;
  options.exploration_decay = 0.5;
  options.replay_batches = 3;
  Result<LinearSolver> solver = MakeLinearSolver(fixture, options, 7);
  ASSERT_TRUE(solver.has_value()) << solver.error();
  double previous = solver->CurrentCost();
  for (std::size_t iteration = 0; iteration < 12; ++iteration) {
    const Result<DtFbsdeIterationReport<double>> report = solver->Iterate();
    ASSERT_TRUE(report.has_value()) << report.error();
    EXPECT_LE(report->mean_cost, previous + 1e-12);
    previous = report->mean_cost;
  }
  EXPECT_LT(solver->CurrentCost(), 0.05 * coasting_cost);

  options.normal_equation_memory = 0.5;
  EXPECT_FALSE(MakeLinearSolver(fixture, options, 7).has_value());
  options.normal_equation_memory = 0.0;
  options.backward_target = DtFbsdeBackwardTarget::kCurrentPolicy;
  EXPECT_FALSE(MakeLinearSolver(fixture, options, 7).has_value());
}

// A representation without normal equations cannot carry them.
TEST(DtFbsdeIterativeSolverNormalEquationTest, MakeRejectsMemoryForASoftMinModel) {
  using SoftMin = SoftMinQuadraticValueFunctionApprox<2, 3>;
  using SoftMinSolver = DtFbsdeIterativeSolver<2, 1, 256, kStages, 64, Model, TerminalCost, SoftMin>;
  const Fixture fixture = MakeFixture();
  const Eigen::Matrix2d terminal_matrix = Eigen::Vector2d(5.0, 2.0).asDiagonal();
  auto parameters = SoftMin::QuadraticParameters(terminal_matrix, Eigen::Vector2d::Zero(), 0.0, 1.0, 1e-3);
  ASSERT_TRUE(parameters.has_value());
  auto model = SoftMin::Make(*parameters, 1.0, 1e-3);
  ASSERT_TRUE(model.has_value());
  DtFbsdeIterativeOptions<double> options;
  options.normal_equation_memory = 0.9;
  EXPECT_FALSE(SoftMinSolver::Make(fixture.model,
                                   fixture.running_cost,
                                   fixture.terminal_cost,
                                   fixture.initial,
                                   Eigen::Matrix<double, 1, 1>(-1.0),
                                   Eigen::Matrix<double, 1, 1>(1.0),
                                   *model,
                                   *model,
                                   options,
                                   1)
                   .has_value());
}

}  // namespace
}  // namespace fbsde_traj_opt
