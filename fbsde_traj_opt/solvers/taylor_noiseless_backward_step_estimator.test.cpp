// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#include "fbsde_traj_opt/solvers/taylor_noiseless_backward_step_estimator.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "fbsde_traj_opt/costs/quadratic_regulator_sde_running_cost_term.hpp"
#include "fbsde_traj_opt/costs/quadratic_regulator_sde_terminal_cost_term.hpp"
#include "fbsde_traj_opt/dynamics/const_diagonal_sde_diffusion_term.hpp"
#include "fbsde_traj_opt/dynamics/const_linear_sde_control_drift_mat_term.hpp"
#include "fbsde_traj_opt/dynamics/const_linear_sde_state_drift_term.hpp"
#include "fbsde_traj_opt/policies/const_linear_feedback_sde_control_policy_term.hpp"
#include "fbsde_traj_opt/policies/stage_varying_linear_feedback_sde_control_policy_term.hpp"
#include "fbsde_traj_opt/sde/composed_cost_sde_model.hpp"
#include "fbsde_traj_opt/sde/composed_forward_sde_model.hpp"
#include "fbsde_traj_opt/solvers/finite_horizon_lqr.hpp"
#include "fbsde_traj_opt/value_function/soft_min_quadratic_value_function_approx.hpp"
#include "fbsde_traj_opt/value_function/value_function_approx_concepts.hpp"

namespace fbsde_traj_opt {
namespace {

constexpr int kStateDim = 2;
constexpr int kControlDim = 1;
constexpr double kSharpness = 1.0;
constexpr double kDiagonalOffset = 1e-4;

using State = Eigen::Matrix<double, kStateDim, 1>;
using Control = Eigen::Matrix<double, kControlDim, 1>;
using StateMat = Eigen::Matrix<double, kStateDim, kStateDim>;

using ForwardModel = ComposedForwardSdeModel<kStateDim,
                                             kControlDim,
                                             ConstLinearSdeStateDriftTerm<kStateDim>,
                                             ConstLinearSdeControlDriftMatTerm<kStateDim, kControlDim>,
                                             ConstDiagonalSdeDiffusionTerm<kStateDim>>;
using CostModel = ComposedCostSdeModel<kStateDim,
                                       kControlDim,
                                       QuadraticRegulatorSdeRunningCostTerm<kStateDim, kControlDim>,
                                       QuadraticRegulatorSdeTerminalCostTerm<kStateDim>>;
using Policy = ConstLinearFeedbackSdeControlPolicyTerm<kStateDim, kControlDim>;
using RunningCost = QuadraticRegulatorSdeRunningCostTerm<kStateDim, kControlDim>;
using Estimator = TaylorNoiselessBackwardStepEstimator<kStateDim, kControlDim, ForwardModel, Policy, RunningCost>;

// A single-component soft minimum is exactly a quadratic, which is what makes it the right
// next-stage representation for a problem whose value function is known to be one.
using Quadratic = SoftMinQuadraticValueFunctionApprox<kStateDim, 1>;

// The problem's constants. The dynamics are unstable in the first coordinate and the control
// reaches only the second, so the optimal gains are not trivial.
const StateMat kStateDrift = (StateMat() << 0.1, 0.4, -0.2, 0.05).finished();
const Eigen::Matrix<double, kStateDim, kControlDim> kControlDrift =
    (Eigen::Matrix<double, kStateDim, kControlDim>() << 0.0, 0.7).finished();
const Eigen::Matrix<double, kStateDim, 1> kDiffusionDiagonal =
    (Eigen::Matrix<double, kStateDim, 1>() << 0.3, 0.2).finished();
const StateMat kStateCost = (StateMat() << 2.0, 0.3, 0.3, 1.0).finished();
const Eigen::Matrix<double, kControlDim, kControlDim> kControlCost =
    Eigen::Matrix<double, kControlDim, kControlDim>::Constant(0.5);
const Eigen::Matrix<double, kStateDim, kControlDim> kCrossCost = Eigen::Matrix<double, kStateDim, kControlDim>::Zero();
const StateMat kTerminalCost = 3.0 * StateMat::Identity();

auto MakeForwardModel() noexcept -> ForwardModel {
  const auto diffusion = ConstDiagonalSdeDiffusionTerm<kStateDim>::Make(kDiffusionDiagonal);
  EXPECT_TRUE(diffusion.has_value());
  return {ConstLinearSdeStateDriftTerm<kStateDim>(kStateDrift),
          ConstLinearSdeControlDriftMatTerm<kStateDim, kControlDim>(kControlDrift),
          *diffusion};
}

auto MakeCostModel() noexcept -> CostModel {
  return {RunningCost(kStateCost, kControlCost, kCrossCost),
          QuadraticRegulatorSdeTerminalCostTerm<kStateDim>(kTerminalCost)};
}

auto MakeEstimator(const Eigen::Matrix<double, kControlDim, kStateDim>& gain) noexcept -> Estimator {
  return {MakeForwardModel(), Policy(gain, Control::Zero()), RunningCost(kStateCost, kControlCost, kCrossCost)};
}

auto DiffusionMat() noexcept -> StateMat {
  return StateMat(kDiffusionDiagonal.asDiagonal());
}

// A deterministic, well-spread source of values.
class Spread {
 public:
  explicit constexpr Spread(std::uint64_t seed) noexcept : state_(seed) {}

  auto Next(double magnitude) noexcept -> double {
    state_ = (state_ * 6364136223846793005ULL) + 1442695040888963407ULL;
    const auto fraction = static_cast<double>(state_ >> 11U) / static_cast<double>(1ULL << 53U);
    return magnitude * ((2.0 * fraction) - 1.0);
  }

 private:
  std::uint64_t state_;
};

auto SpreadState(std::uint64_t seed, double magnitude) noexcept -> State {
  Spread spread(seed);
  return {spread.Next(magnitude), spread.Next(magnitude)};
}

// The quadratic V(x) = x' hessian x + offset, as a next-stage representation.
auto MakeQuadratic(const StateMat& hessian, double offset) noexcept -> Quadratic {
  const auto parameters = Quadratic::QuadraticParameters(hessian, State::Zero(), offset, kSharpness, kDiagonalOffset);
  EXPECT_TRUE(parameters.has_value());
  const auto model = Quadratic::Make(*parameters, kSharpness, kDiagonalOffset);
  EXPECT_TRUE(model.has_value());
  return *model;
}

// ---------------------------------------------------------------------------------------------
// The estimator on a quadratic next-stage value function, where it is exact
// ---------------------------------------------------------------------------------------------

// For V~_{i+1}(y) = y' P y + s the one-step Bellman value under the target policy is available in
// closed form: L^mu_i + (x + F^mu_i)' P (x + F^mu_i) + s + tr(Sigma' P Sigma). The estimator must
// return exactly that, for any drift K_i at all.
auto ExactOneStepValue(const Estimator& estimator,
                       const StateMat& next_hessian,
                       double next_offset,
                       const State& state) noexcept -> double {
  const Control control = estimator.policy()(0, state);
  const double running_cost = estimator.running_cost()(0, state, control);
  const State on_policy_mean = state + estimator.PolicyDrift(0, state);
  const StateMat diffusion = DiffusionMat();
  return running_cost + on_policy_mean.dot(next_hessian * on_policy_mean) + next_offset +
         (diffusion.transpose() * next_hessian * diffusion).trace();
}

TEST(TaylorNoiselessBackwardStepEstimatorTest, IsExactOnAQuadraticForEveryDrift) {
  const Eigen::Matrix<double, kControlDim, kStateDim> gain =
      (Eigen::Matrix<double, kControlDim, kStateDim>() << -0.6, -1.1).finished();
  const Estimator estimator = MakeEstimator(gain);

  const StateMat next_hessian = (StateMat() << 1.7, 0.4, 0.4, 2.3).finished();
  constexpr double kNextOffset = 1.25;
  const Quadratic next_value_function = MakeQuadratic(next_hessian, kNextOffset);

  for (std::uint64_t seed = 0; seed < 12; ++seed) {
    const State state = SpreadState(seed + 1, 3.0);
    const double expected = ExactOneStepValue(estimator, next_hessian, kNextOffset, state);

    // Every drift, from the on-policy one to drifts nowhere near it, must give the same answer.
    const State on_policy_drift = estimator.PolicyDrift(0, state);
    const std::array<State, 4> drifts = {
        on_policy_drift, State::Zero(), -0.2 * state, on_policy_drift + SpreadState(seed + 500, 2.0)};

    for (const State& drift : drifts) {
      EXPECT_NEAR(estimator(0, state, drift, next_value_function), expected, 1e-9)
          << "seed " << seed << ", drift " << drift.transpose();
    }
  }
}

TEST(TaylorNoiselessBackwardStepEstimatorTest, ReducesToTheOnPolicyFormWhenTheDriftIsThePolicyDrift) {
  // With K_i = F^mu_i the two Taylor terms in dF vanish and only L + Ybar + tr(Mbar)/2 remains.
  const Eigen::Matrix<double, kControlDim, kStateDim> gain =
      (Eigen::Matrix<double, kControlDim, kStateDim>() << -0.3, -0.9).finished();
  const Estimator estimator = MakeEstimator(gain);

  const StateMat next_hessian = (StateMat() << 2.0, -0.5, -0.5, 1.4).finished();
  const Quadratic next_value_function = MakeQuadratic(next_hessian, -0.75);

  for (std::uint64_t seed = 0; seed < 6; ++seed) {
    const State state = SpreadState(seed + 40, 2.0);
    const State drift = estimator.PolicyDrift(0, state);
    const State mean_next_state = state + drift;

    const Control control = estimator.policy()(0, state);
    const StateMat diffusion = DiffusionMat();
    const StateMat scaled_hessian = diffusion.transpose() * next_value_function.Hessian(mean_next_state) * diffusion;
    const double on_policy = estimator.running_cost()(0, state, control) + next_value_function(mean_next_state) +
                             (0.5 * scaled_hessian.trace());

    EXPECT_NEAR(estimator(0, state, drift, next_value_function), on_policy, 1e-12);
  }
}

TEST(TaylorNoiselessBackwardStepEstimatorTest, TheBatchedPathAgreesWithThePointwiseOne) {
  constexpr int kBatch = 8;
  const Eigen::Matrix<double, kControlDim, kStateDim> gain =
      (Eigen::Matrix<double, kControlDim, kStateDim>() << -0.45, -0.8).finished();
  const Estimator estimator = MakeEstimator(gain);

  // Three components, so the next-stage representation is genuinely non-quadratic and the batched
  // Hessian path has something to get wrong.
  using MultiWell = SoftMinQuadraticValueFunctionApprox<kStateDim, 3>;
  MultiWell::ParameterVector parameters;
  Spread spread(97);
  for (Eigen::Index index = 0; index < MultiWell::kNumParameters; ++index) {
    parameters[index] = spread.Next(1.0);
  }
  const auto next_value_function = MultiWell::Make(parameters, kSharpness, kDiagonalOffset);
  ASSERT_TRUE(next_value_function.has_value()) << next_value_function.error();

  ValueFunctionStateBatch<State, kBatch> states;
  ValueFunctionStateBatch<State, kBatch> drifts;
  for (Eigen::Index column = 0; column < kBatch; ++column) {
    states.col(column) = SpreadState(static_cast<std::uint64_t>(column) + 300, 2.0);
    drifts.col(column) = SpreadState(static_cast<std::uint64_t>(column) + 400, 0.5);
  }

  ValueFunctionValueBatch<State, kBatch> targets;
  estimator.EstimateTargets<kBatch>(3, states, drifts, *next_value_function, targets);

  for (Eigen::Index column = 0; column < kBatch; ++column) {
    EXPECT_NEAR(targets[column], estimator(3, states.col(column), drifts.col(column), *next_value_function), 1e-12)
        << "sample " << column;
  }
}

// ---------------------------------------------------------------------------------------------
// The backward recursion against the exact LQR value function
// ---------------------------------------------------------------------------------------------

TEST(TaylorNoiselessBackwardStepEstimatorTest, ReproducesTheRiccatiValueFunctionStageByStage) {
  // The sharpest check available: on an LQR problem the true value function is exactly
  // x' P_k x + s_k, the second-order model is therefore exact, and the estimator must reproduce
  // the Riccati recursion to rounding -- under the optimal policy and from any sampling drift.
  constexpr std::size_t kNumStages = 8;

  const ForwardModel forward_model = MakeForwardModel();
  const CostModel cost_model = MakeCostModel();

  const auto solution = SolveFiniteHorizonLqrWithCostToGo<kNumStages>(forward_model, cost_model);
  ASSERT_TRUE(solution.has_value()) << solution.error();

  // The constant the additive noise adds: s_K = 0, s_k = s_{k+1} + tr(Sigma' P_{k+1} Sigma).
  const StateMat diffusion = DiffusionMat();
  std::array<double, kNumStages> offsets{};
  for (std::size_t stage = kNumStages - 1; stage-- > 0;) {
    offsets[stage] =
        offsets[stage + 1] + (diffusion.transpose() * solution->cost_to_go_hessians[stage + 1] * diffusion).trace();
  }

  using OptimalPolicy = StageVaryingLinearFeedbackSdeControlPolicyTerm<kStateDim, kControlDim, kNumStages - 1>;
  using OptimalEstimator =
      TaylorNoiselessBackwardStepEstimator<kStateDim, kControlDim, ForwardModel, OptimalPolicy, RunningCost>;
  const OptimalEstimator estimator(forward_model, solution->policy, RunningCost(kStateCost, kControlCost, kCrossCost));

  for (std::size_t stage = 0; stage + 1 < kNumStages; ++stage) {
    const Quadratic next_value_function = MakeQuadratic(solution->cost_to_go_hessians[stage + 1], offsets[stage + 1]);

    for (std::uint64_t seed = 0; seed < 6; ++seed) {
      const State state = SpreadState((static_cast<std::uint64_t>(stage) * 13) + seed + 700, 2.0);
      const double exact = state.dot(solution->cost_to_go_hessians[stage] * state) + offsets[stage];

      // On-policy, and off-policy with a drift that has nothing to do with the policy's.
      const State on_policy_drift = estimator.PolicyDrift(stage, state);
      EXPECT_NEAR(estimator(stage, state, on_policy_drift, next_value_function), exact, 1e-9)
          << "stage " << stage << ", seed " << seed;
      EXPECT_NEAR(estimator(stage, state, -0.2 * state, next_value_function), exact, 1e-9)
          << "stage " << stage << ", seed " << seed;
    }
  }
}

// ---------------------------------------------------------------------------------------------
// Girsanov drift diagnostics
// ---------------------------------------------------------------------------------------------

TEST(TaylorNoiselessBackwardStepEstimatorTest, GirsanovDriftNormsAreZeroOnPolicyAndSigmaScaledOffIt) {
  constexpr int kBatch = 4;
  const Eigen::Matrix<double, kControlDim, kStateDim> gain =
      (Eigen::Matrix<double, kControlDim, kStateDim>() << -0.5, -1.0).finished();
  const Estimator estimator = MakeEstimator(gain);

  ValueFunctionStateBatch<State, kBatch> states;
  ValueFunctionStateBatch<State, kBatch> on_policy_drifts;
  for (Eigen::Index column = 0; column < kBatch; ++column) {
    const State state = SpreadState(static_cast<std::uint64_t>(column) + 900, 2.0);
    states.col(column) = state;
    on_policy_drifts.col(column) = estimator.PolicyDrift(0, state);
  }

  ValueFunctionValueBatch<State, kBatch> norms;
  ASSERT_TRUE(estimator.GirsanovDriftNorms<kBatch>(0, states, on_policy_drifts, norms).has_value());
  EXPECT_LT(norms.cwiseAbs().maxCoeff(), 1e-14);

  // Perturbing the drift by exactly one diffusion column makes ||D|| exactly one: this is the
  // scale the bias bound is stated against, and the reason the perturbation should be measured in
  // units of Sigma rather than of the drift.
  ValueFunctionStateBatch<State, kBatch> perturbed_drifts = on_policy_drifts;
  for (Eigen::Index column = 0; column < kBatch; ++column) {
    perturbed_drifts.col(column) += DiffusionMat().col(column % kStateDim);
  }
  ASSERT_TRUE(estimator.GirsanovDriftNorms<kBatch>(0, states, perturbed_drifts, norms).has_value());
  for (Eigen::Index column = 0; column < kBatch; ++column) {
    EXPECT_NEAR(norms[column], 1.0, 1e-12) << "sample " << column;
  }
}

}  // namespace
}  // namespace fbsde_traj_opt
