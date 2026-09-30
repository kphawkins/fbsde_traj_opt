// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#include "fbsde_traj_opt/fbrrt/sgd_taylor_backward_pass.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "fbsde_traj_opt/costs/quadratic_regulator_sde_running_cost_term.hpp"
#include "fbsde_traj_opt/costs/quadratic_regulator_sde_terminal_cost_term.hpp"
#include "fbsde_traj_opt/dynamics/const_diagonal_sde_diffusion_term.hpp"
#include "fbsde_traj_opt/dynamics/const_linear_sde_control_drift_mat_term.hpp"
#include "fbsde_traj_opt/dynamics/const_linear_sde_state_drift_term.hpp"
#include "fbsde_traj_opt/fbrrt/fbrrt_particle_set.hpp"
#include "fbsde_traj_opt/sde/composed_cost_sde_model.hpp"
#include "fbsde_traj_opt/sde/composed_forward_sde_model.hpp"
#include "fbsde_traj_opt/solvers/finite_horizon_lqr.hpp"
#include "fbsde_traj_opt/utils/result.hpp"
#include "fbsde_traj_opt/value_function/polynomial_value_function_approx.hpp"

namespace fbsde_traj_opt {
namespace {

constexpr std::size_t kNumStages = 6;
constexpr std::size_t kCapacity = 256;

using State = Eigen::Vector2d;
using StateMat = Eigen::Matrix2d;
using ForwardModel = ComposedForwardSdeModel<2,
                                             1,
                                             ConstLinearSdeStateDriftTerm<2>,
                                             ConstLinearSdeControlDriftMatTerm<2, 1>,
                                             ConstDiagonalSdeDiffusionTerm<2>>;
using RunningCost = QuadraticRegulatorSdeRunningCostTerm<2, 1>;
using CostModel = ComposedCostSdeModel<2, 1, RunningCost, QuadraticRegulatorSdeTerminalCostTerm<2>>;
using ValueFunction = PolynomialValueFunctionApprox<2, 2>;
using Set = FbrrtParticleSet<2, 1, kCapacity, kNumStages>;
using BackwardPass =
    SgdTaylorBackwardPass<ValueFunction, ForwardModel, RunningCost, static_cast<int>(kCapacity), 32, 64>;

// An LQR problem, unstable in the first coordinate and actuated only through the second, whose
// value function is exactly x' P_i x + s_i: something the backward pass can be held to.
const StateMat kStateDrift = (StateMat() << 0.1, 0.4, -0.2, 0.05).finished();
const Eigen::Vector2d kControlDrift(0.0, 0.7);
const Eigen::Vector2d kDiffusionDiagonal(0.3, 0.2);
const StateMat kStateCost = (StateMat() << 2.0, 0.3, 0.3, 1.0).finished();
const Eigen::Matrix<double, 1, 1> kControlCost = Eigen::Matrix<double, 1, 1>::Constant(0.5);
const StateMat kTerminalCost = 3.0 * StateMat::Identity();

auto MakeForwardModel() -> ForwardModel {
  const Result<ConstDiagonalSdeDiffusionTerm<2>> diffusion = ConstDiagonalSdeDiffusionTerm<2>::Make(kDiffusionDiagonal);
  EXPECT_TRUE(diffusion.has_value());
  return {
      ConstLinearSdeStateDriftTerm<2>(kStateDrift), ConstLinearSdeControlDriftMatTerm<2, 1>(kControlDrift), *diffusion};
}

auto MakeRunningCost() -> RunningCost {
  return {kStateCost, kControlCost, Eigen::Vector2d::Zero()};
}

auto MakeCostModel() -> CostModel {
  return {MakeRunningCost(), QuadraticRegulatorSdeTerminalCostTerm<2>(kTerminalCost)};
}

// Edges into every stage from states spread over [-2, 2]^2, each drawn under a drift that has
// nothing to do with the policy's: the estimator is off-policy, so the fit must not care.
auto MakeEdges() -> Set {
  Set set;
  set.SetInitialStates(Set::StateBlock::Zero());
  for (std::size_t stage = 1; stage < kNumStages; ++stage) {
    const Set::StateBlock sources = 2.0 * Set::StateBlock::Random();
    for (std::size_t slot = 0; slot < kCapacity; ++slot) {
      Set::Particle particle;
      particle.source_state = sources.col(static_cast<Eigen::Index>(slot));
      particle.drift = -0.2 * particle.source_state;
      particle.state = particle.source_state + particle.drift;
      set.WriteParticle(stage, slot, particle);
    }
    EXPECT_TRUE(set.SetCount(stage, kCapacity).has_value());
  }
  return set;
}

auto MakeTable(const StateMat& terminal_metric) -> std::vector<ValueFunction> {
  const Result<ValueFunction> blank =
      ValueFunction::Make(ValueFunction::ParameterVector::Zero(), State::Zero(), State(2.0, 2.0));
  EXPECT_TRUE(blank.has_value());
  ValueFunction terminal = *blank;
  EXPECT_TRUE(terminal.SetParameters(blank->QuadraticParameters(terminal_metric, 0.0)).has_value());
  std::vector<ValueFunction> table(kNumStages, terminal);
  return table;
}

TEST(SgdTaylorBackwardPassTest, RepeatedPassesConvergeToTheRiccatiValueFunction) {
  const ForwardModel forward_model = MakeForwardModel();
  const auto solution = SolveFiniteHorizonLqrWithCostToGo<kNumStages>(forward_model, MakeCostModel());
  ASSERT_TRUE(solution.has_value()) << solution.error();

  // The noise's constant: s_K = 0, s_i = s_{i+1} + tr(Sigma' P_{i+1} Sigma).
  const StateMat diffusion = kDiffusionDiagonal.asDiagonal();
  std::array<double, kNumStages> offsets{};
  for (std::size_t stage = kNumStages - 1; stage-- > 0;) {
    offsets[stage] =
        offsets[stage + 1] + (diffusion.transpose() * solution->cost_to_go_hessians[stage + 1] * diffusion).trace();
  }

  SgdTaylorBackwardPassOptions<double> options;
  options.sgd.learning_rate = 0.05;
  options.epochs_per_stage = 4;
  std::vector<ValueFunction> table = MakeTable(kTerminalCost);
  Result<BackwardPass> pass = BackwardPass::Make(forward_model, MakeRunningCost(), options, table);
  ASSERT_TRUE(pass.has_value()) << pass.error();

  const Set edges = MakeEdges();
  std::vector<double> residuals;
  for (std::uint64_t iteration = 0; iteration < 60; ++iteration) {
    const Result<SgdTaylorBackwardPassReport<double>> report = (*pass)(table, edges, solution->policy, iteration);
    ASSERT_TRUE(report.has_value()) << report.error();
    ASSERT_EQ(report->stage_rms_residuals.size(), kNumStages - 1);
    residuals.push_back(report->rms_residual);
  }
  EXPECT_LT(residuals.back(), 0.02 * residuals.front());

  for (std::size_t stage = 0; stage < kNumStages; ++stage) {
    for (int trial = 0; trial < 8; ++trial) {
      const State state = 1.5 * State::Random();
      const double exact = state.dot(solution->cost_to_go_hessians[stage] * state) + offsets[stage];
      EXPECT_NEAR(table[stage](state), exact, 0.01 * (1.0 + std::abs(exact))) << "stage " << stage;
    }
  }
}

TEST(SgdTaylorBackwardPassTest, EachPassMovesTheValueFunctionsOnlyPartWay) {
  // One pass from a table that is badly wrong must leave it still visibly wrong: the pass is a
  // step, not a solve.
  const ForwardModel forward_model = MakeForwardModel();
  const auto solution = SolveFiniteHorizonLqrWithCostToGo<kNumStages>(forward_model, MakeCostModel());
  ASSERT_TRUE(solution.has_value());

  SgdTaylorBackwardPassOptions<double> options;
  options.sgd.learning_rate = 0.01;
  options.sgd.max_rms_value_change = 0.05;
  options.epochs_per_stage = 1;
  std::vector<ValueFunction> table = MakeTable(kTerminalCost);
  const std::vector<ValueFunction> before = table;
  Result<BackwardPass> pass = BackwardPass::Make(forward_model, MakeRunningCost(), options, table);
  ASSERT_TRUE(pass.has_value());
  ASSERT_TRUE((*pass)(table, MakeEdges(), solution->policy, 0).has_value());

  // Four minibatch steps of at most 0.05 each, to first order.
  const State probe(1.0, -0.5);
  for (std::size_t stage = 0; stage + 1 < kNumStages; ++stage) {
    EXPECT_LT(std::abs(table[stage](probe) - before[stage](probe)), 0.5) << "stage " << stage;
    EXPECT_NE(table[stage].Parameters(), before[stage].Parameters()) << "stage " << stage;
  }
  EXPECT_EQ(table.back().Parameters(), before.back().Parameters());
}

TEST(SgdTaylorBackwardPassTest, TheTargetFloorLiftsEveryTargetBelowIt) {
  // Every exact target here is well under 1000; with the floor there, every stage is pulled up.
  const ForwardModel forward_model = MakeForwardModel();
  const auto solution = SolveFiniteHorizonLqrWithCostToGo<kNumStages>(forward_model, MakeCostModel());
  ASSERT_TRUE(solution.has_value());
  SgdTaylorBackwardPassOptions<double> options;
  options.sgd.learning_rate = 0.5;
  options.target_floor = 1000.0;
  std::vector<ValueFunction> table = MakeTable(kTerminalCost);
  const std::vector<ValueFunction> before = table;
  Result<BackwardPass> pass = BackwardPass::Make(forward_model, MakeRunningCost(), options, table);
  ASSERT_TRUE(pass.has_value());
  ASSERT_TRUE((*pass)(table, MakeEdges(), solution->policy, 0).has_value());
  for (std::size_t stage = 0; stage + 1 < kNumStages; ++stage) {
    EXPECT_GT(table[stage](State::Zero()), before[stage](State::Zero()) + 1.0) << "stage " << stage;
  }
}

TEST(SgdTaylorBackwardPassTest, TheEntropyWeightingConcentratesTheFitAndReportsHowMuch) {
  const ForwardModel forward_model = MakeForwardModel();
  const auto solution = SolveFiniteHorizonLqrWithCostToGo<kNumStages>(forward_model, MakeCostModel());
  ASSERT_TRUE(solution.has_value());
  const Set edges = MakeEdges();

  const auto run = [&](double temperature) {
    SgdTaylorBackwardPassOptions<double> options;
    options.entropy_temperature = temperature;
    std::vector<ValueFunction> table = MakeTable(kTerminalCost);
    Result<BackwardPass> pass = BackwardPass::Make(forward_model, MakeRunningCost(), options, table);
    EXPECT_TRUE(pass.has_value());
    const Result<SgdTaylorBackwardPassReport<double>> report = (*pass)(table, edges, solution->policy, 0);
    EXPECT_TRUE(report.has_value());
    return std::make_pair(*report, table);
  };
  const auto [unweighted, unweighted_table] = run(0.0);
  const auto [weighted, weighted_table] = run(0.5);
  const auto [sharper, sharper_table] = run(0.1);

  EXPECT_DOUBLE_EQ(unweighted.effective_sample_fraction, 1.0);
  EXPECT_GT(weighted.effective_sample_fraction, 0.0);
  EXPECT_LT(weighted.effective_sample_fraction, 1.0);
  // A lower temperature concentrates the weights further.
  EXPECT_LT(sharper.effective_sample_fraction, weighted.effective_sample_fraction);
  EXPECT_NE(weighted_table[0].Parameters(), unweighted_table[0].Parameters());
}

TEST(SgdTaylorBackwardPassTest, RejectsAStageThatIsNotFull) {
  const ForwardModel forward_model = MakeForwardModel();
  const auto solution = SolveFiniteHorizonLqrWithCostToGo<kNumStages>(forward_model, MakeCostModel());
  ASSERT_TRUE(solution.has_value());
  std::vector<ValueFunction> table = MakeTable(kTerminalCost);
  Result<BackwardPass> pass = BackwardPass::Make(forward_model, MakeRunningCost(), {}, table);
  ASSERT_TRUE(pass.has_value());
  Set edges = MakeEdges();
  ASSERT_TRUE(edges.KeepLowest(3, Set::ValueBlock::Zero(), kCapacity / 2).has_value());
  EXPECT_FALSE((*pass)(table, edges, solution->policy, 0).has_value());
}

TEST(SgdTaylorBackwardPassTest, MakeRejectsZeroEpochsAndABadFitter) {
  const std::vector<ValueFunction> table = MakeTable(kTerminalCost);
  SgdTaylorBackwardPassOptions<double> options;
  options.epochs_per_stage = 0;
  EXPECT_FALSE(BackwardPass::Make(MakeForwardModel(), MakeRunningCost(), options, table).has_value());
  options.epochs_per_stage = 1;
  options.sgd.learning_rate = -1.0;
  EXPECT_FALSE(BackwardPass::Make(MakeForwardModel(), MakeRunningCost(), options, table).has_value());
  options.sgd.learning_rate = 0.01;
  options.entropy_temperature = -1.0;
  EXPECT_FALSE(BackwardPass::Make(MakeForwardModel(), MakeRunningCost(), options, table).has_value());
}

}  // namespace
}  // namespace fbsde_traj_opt
