// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#include "fbsde_traj_opt/sde/trajectory_batch.hpp"

#include <cstddef>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "fbsde_traj_opt/costs/quadratic_regulator_sde_running_cost_term.hpp"
#include "fbsde_traj_opt/costs/quadratic_regulator_sde_terminal_cost_term.hpp"
#include "fbsde_traj_opt/dynamics/const_diagonal_sde_diffusion_term.hpp"
#include "fbsde_traj_opt/dynamics/const_linear_sde_control_drift_mat_term.hpp"
#include "fbsde_traj_opt/dynamics/const_linear_sde_state_drift_term.hpp"
#include "fbsde_traj_opt/policies/const_linear_feedback_sde_control_policy_term.hpp"
#include "fbsde_traj_opt/sampling/diagonal_covariance_normal_distribution.hpp"
#include "fbsde_traj_opt/sde/composed_cost_sde_model.hpp"
#include "fbsde_traj_opt/sde/composed_forward_sde_model.hpp"

namespace fbsde_traj_opt {
namespace {

constexpr std::size_t kNumTrajectories = 256;
constexpr std::size_t kNumStages = 6;

using StateDrift2 = ConstLinearSdeStateDriftTerm<2>;
using ControlDrift2x1 = ConstLinearSdeControlDriftMatTerm<2, 1>;
using Diffusion2 = ConstDiagonalSdeDiffusionTerm<2>;
using Model2x1 = ComposedForwardSdeModel<2, 1, StateDrift2, ControlDrift2x1, Diffusion2>;
using Policy2x1 = ConstLinearFeedbackSdeControlPolicyTerm<2, 1>;
using Distribution2 = DiagonalCovarianceNormalDistribution<2>;
using CostModel2x1 =
    ComposedCostSdeModel<2, 1, QuadraticRegulatorSdeRunningCostTerm<2, 1>, QuadraticRegulatorSdeTerminalCostTerm<2>>;
using Batch = TrajectoryBatch<2, 1, kNumTrajectories, kNumStages>;

using Control1 = Eigen::Matrix<double, 1, 1>;

// The smallest diffusion and initial covariance the term factories accept, used by the tests
// that want a batch deterministic enough to compare against an analytic rollout.
//
// These are not the same scale: a diffusion diagonal element multiplies a unit normal directly,
// so 1e-10 there means noise of size 1e-10, while a covariance diagonal element is a variance, so
// 1e-10 there means a standard deviation of 1e-5. The initial draw therefore dominates the
// residual randomness, and kNoiselessTolerance is set to sit an order of magnitude above it.
constexpr double kNegligible = 1e-10;
constexpr double kNoiselessTolerance = 1e-4;

auto MakeModel(const Eigen::Matrix2d& a, const Eigen::Vector2d& b, const Eigen::Vector2d& sigma) noexcept -> Model2x1 {
  const auto diffusion = Diffusion2::Make(sigma);
  EXPECT_TRUE(diffusion.has_value());
  return {StateDrift2(a), ControlDrift2x1(b), *diffusion};
}

auto MakeDistribution(const Eigen::Vector2d& mean, const Eigen::Vector2d& covariance_diagonal) noexcept
    -> Distribution2 {
  const auto distribution = Distribution2::Make(mean, covariance_diagonal);
  EXPECT_TRUE(distribution.has_value());
  return *distribution;
}

// A double integrator discretized at dt = 0.1: position integrates velocity, velocity integrates
// acceleration. Written as the drift A of `x_{k+1} = x_k + A x_k + B u_k + ...`, so A holds only
// the increment, not the identity.
auto MakeDoubleIntegratorDrift() noexcept -> Eigen::Matrix2d {
  Eigen::Matrix2d a;
  a << 0.0, 0.1,  //
      0.0, 0.0;
  return a;
}

// A NormalDistribution whose covariance is indefinite, which the concept cannot reject but
// TrajectoryBatch::Make() must.
class IndefiniteDistribution {
 public:
  [[nodiscard]] static auto Mean() noexcept -> Eigen::Vector2d { return Eigen::Vector2d::Zero(); }

  [[nodiscard]] static auto Covariance() noexcept -> Eigen::Matrix2d {
    Eigen::Matrix2d covariance;
    covariance << 1.0, 0.0,  //
        0.0, -1.0;
    return covariance;
  }
};

static_assert(NormalDistribution<IndefiniteDistribution, Eigen::Vector2d>);

TEST(TrajectoryBatchTest, MakeFillsEveryStageAndControlStage) {
  const Model2x1 model = MakeModel(MakeDoubleIntegratorDrift(), Eigen::Vector2d(0.0, 0.1), Eigen::Vector2d(0.05, 0.05));
  const Policy2x1 policy(Eigen::Matrix<double, 1, 2>::Zero(), Control1::Zero());
  const Distribution2 distribution = MakeDistribution(Eigen::Vector2d(1.0, 0.0), Eigen::Vector2d(0.01, 0.01));

  const auto batch = Batch::Make(model, policy, distribution, 7);
  ASSERT_TRUE(batch.has_value()) << batch.error();

  EXPECT_EQ(Batch::kNumStages, kNumStages);
  EXPECT_EQ(Batch::kNumControlStages, kNumStages - 1);
  for (std::size_t stage = 0; stage < Batch::kNumStages; ++stage) {
    EXPECT_TRUE(batch->StatesAtStage(stage).allFinite());
  }
  for (std::size_t stage = 0; stage < Batch::kNumControlStages; ++stage) {
    EXPECT_TRUE(batch->ControlsAtStage(stage).allFinite());
  }
}

TEST(TrajectoryBatchTest, MakeRejectsAnIndefiniteInitialCovariance) {
  const Model2x1 model = MakeModel(Eigen::Matrix2d::Zero(), Eigen::Vector2d::Zero(), Eigen::Vector2d(1.0, 1.0));
  const Policy2x1 policy(Eigen::Matrix<double, 1, 2>::Zero(), Control1::Zero());

  const auto batch = Batch::Make(model, policy, IndefiniteDistribution{}, 7);

  EXPECT_FALSE(batch.has_value());
}

TEST(TrajectoryBatchTest, NoiselessRolloutMatchesTheAnalyticTrajectory) {
  const Eigen::Matrix2d a = MakeDoubleIntegratorDrift();
  const Eigen::Vector2d b(0.0, 0.1);
  const Model2x1 model = MakeModel(a, b, Eigen::Vector2d(kNegligible, kNegligible));

  Eigen::Matrix<double, 1, 2> gain;
  gain << -1.0, -2.0;
  const Policy2x1 policy(gain, Control1::Zero());

  const Eigen::Vector2d initial_mean(1.0, 0.0);
  const Distribution2 distribution = MakeDistribution(initial_mean, Eigen::Vector2d(kNegligible, kNegligible));

  const auto batch = Batch::Make(model, policy, distribution, 11);
  ASSERT_TRUE(batch.has_value()) << batch.error();

  Eigen::Vector2d expected_state = initial_mean;
  for (std::size_t stage = 0; stage < Batch::kNumControlStages; ++stage) {
    EXPECT_TRUE(batch->StateAt(0, stage).isApprox(expected_state, kNoiselessTolerance)) << "at stage " << stage;

    const Control1 expected_control = gain * expected_state;
    EXPECT_TRUE(batch->ControlAt(0, stage).isApprox(expected_control, kNoiselessTolerance)) << "at stage " << stage;

    expected_state += (a * expected_state) + (b * expected_control);
  }
  EXPECT_TRUE(batch->StateAt(0, Batch::kNumStages - 1).isApprox(expected_state, kNoiselessTolerance));
}

TEST(TrajectoryBatchTest, ControlsAreThePolicyAppliedToTheStates) {
  const Model2x1 model = MakeModel(MakeDoubleIntegratorDrift(), Eigen::Vector2d(0.0, 0.1), Eigen::Vector2d(0.1, 0.1));

  Eigen::Matrix<double, 1, 2> gain;
  gain << -3.0, -1.0;
  const Control1 offset = Control1::Constant(0.25);
  const Policy2x1 policy(gain, offset);

  const Distribution2 distribution = MakeDistribution(Eigen::Vector2d(1.0, 0.0), Eigen::Vector2d(0.04, 0.04));

  const auto batch = Batch::Make(model, policy, distribution, 3);
  ASSERT_TRUE(batch.has_value()) << batch.error();

  for (std::size_t stage = 0; stage < Batch::kNumControlStages; ++stage) {
    for (std::size_t trajectory = 0; trajectory < Batch::kNumTrajectories; ++trajectory) {
      const Control1 expected = (gain * batch->StateAt(trajectory, stage)) + offset;
      EXPECT_TRUE(batch->ControlAt(trajectory, stage).isApprox(expected));
    }
  }
}

TEST(TrajectoryBatchTest, InitialStatesReproduceTheInitialDistribution) {
  const Model2x1 model = MakeModel(Eigen::Matrix2d::Zero(), Eigen::Vector2d::Zero(), Eigen::Vector2d(1.0, 1.0));
  const Policy2x1 policy(Eigen::Matrix<double, 1, 2>::Zero(), Control1::Zero());

  const Eigen::Vector2d mean(2.0, -1.0);
  const Eigen::Vector2d covariance_diagonal(0.25, 4.0);
  const Distribution2 distribution = MakeDistribution(mean, covariance_diagonal);

  const auto batch = Batch::Make(model, policy, distribution, 20260918);
  ASSERT_TRUE(batch.has_value()) << batch.error();

  const Eigen::Vector2d sample_mean = batch->MeanStateAtStage(0);
  EXPECT_NEAR(sample_mean.x(), mean.x(), 0.1);
  EXPECT_NEAR(sample_mean.y(), mean.y(), 0.3);

  const Batch::StateBlock centered = batch->StatesAtStage(0).colwise() - sample_mean;
  const Eigen::Matrix2d sample_covariance =
      (centered * centered.transpose()) / static_cast<double>(Batch::kNumTrajectories - 1);
  EXPECT_NEAR(sample_covariance(0, 0), covariance_diagonal.x(), 0.1);
  EXPECT_NEAR(sample_covariance(1, 1), covariance_diagonal.y(), 1.0);
  EXPECT_NEAR(sample_covariance(0, 1), 0.0, 0.2);
}

TEST(TrajectoryBatchTest, SameSeedGivesAnIdenticalBatch) {
  const Model2x1 model = MakeModel(MakeDoubleIntegratorDrift(), Eigen::Vector2d(0.0, 0.1), Eigen::Vector2d(0.1, 0.1));
  const Policy2x1 policy(Eigen::Matrix<double, 1, 2>::Zero(), Control1::Zero());
  const Distribution2 distribution = MakeDistribution(Eigen::Vector2d(1.0, 0.0), Eigen::Vector2d(0.01, 0.01));

  const auto first = Batch::Make(model, policy, distribution, 42);
  const auto second = Batch::Make(model, policy, distribution, 42);
  const auto other_seed = Batch::Make(model, policy, distribution, 43);
  ASSERT_TRUE(first.has_value() && second.has_value() && other_seed.has_value());

  for (std::size_t stage = 0; stage < Batch::kNumStages; ++stage) {
    EXPECT_TRUE(first->StatesAtStage(stage).isApprox(second->StatesAtStage(stage)));
  }
  EXPECT_FALSE(first->StatesAtStage(1).isApprox(other_seed->StatesAtStage(1)));
  EXPECT_EQ(first->seed(), 42U);
}

// The reproducibility guarantee that makes two batches comparable: with the same seed and the
// same batch shape, both see the same underlying noise even though their dynamics differ.
//
// Both models here have zero drift, zero control drift, and a zero policy, so every state
// increment is exactly `Sigma * z_k`. The second model's Sigma is twice the first's, so if and
// only if both drew the same z_k are the second's increments exactly twice the first's.
TEST(TrajectoryBatchTest, DifferentModelsOfTheSameShapeSeeTheSameNoise) {
  const Policy2x1 policy(Eigen::Matrix<double, 1, 2>::Zero(), Control1::Zero());
  const Distribution2 distribution = MakeDistribution(Eigen::Vector2d::Zero(), Eigen::Vector2d(1.0, 1.0));

  const Model2x1 unit_model = MakeModel(Eigen::Matrix2d::Zero(), Eigen::Vector2d::Zero(), Eigen::Vector2d(1.0, 1.0));
  const Model2x1 doubled_model = MakeModel(Eigen::Matrix2d::Zero(), Eigen::Vector2d::Zero(), Eigen::Vector2d(2.0, 2.0));

  const auto unit_batch = Batch::Make(unit_model, policy, distribution, 1234);
  const auto doubled_batch = Batch::Make(doubled_model, policy, distribution, 1234);
  ASSERT_TRUE(unit_batch.has_value() && doubled_batch.has_value());

  // The initial draw uses the same covariance in both batches, so it is identical, not doubled.
  EXPECT_TRUE(unit_batch->StatesAtStage(0).isApprox(doubled_batch->StatesAtStage(0)));

  for (std::size_t stage = 0; stage < Batch::kNumControlStages; ++stage) {
    const Eigen::Matrix<double, 2, kNumTrajectories> unit_increment =
        unit_batch->StatesAtStage(stage + 1) - unit_batch->StatesAtStage(stage);
    const Eigen::Matrix<double, 2, kNumTrajectories> doubled_increment =
        doubled_batch->StatesAtStage(stage + 1) - doubled_batch->StatesAtStage(stage);

    EXPECT_TRUE(doubled_increment.isApprox(2.0 * unit_increment)) << "at stage " << stage;
  }
}

TEST(TrajectoryBatchTest, MeanAccessorsAverageOverTrajectories) {
  const Model2x1 model = MakeModel(MakeDoubleIntegratorDrift(), Eigen::Vector2d(0.0, 0.1), Eigen::Vector2d(0.1, 0.1));

  Eigen::Matrix<double, 1, 2> gain;
  gain << -1.0, -2.0;
  const Policy2x1 policy(gain, Control1::Zero());
  const Distribution2 distribution = MakeDistribution(Eigen::Vector2d(1.0, 0.0), Eigen::Vector2d(0.01, 0.01));

  const auto batch = Batch::Make(model, policy, distribution, 5);
  ASSERT_TRUE(batch.has_value()) << batch.error();

  for (std::size_t stage = 0; stage < Batch::kNumControlStages; ++stage) {
    EXPECT_TRUE(batch->MeanStateAtStage(stage).isApprox(batch->StatesAtStage(stage).rowwise().mean()));
    EXPECT_TRUE(batch->MeanControlAtStage(stage).isApprox(batch->ControlsAtStage(stage).rowwise().mean()));
  }
}

TEST(TrajectoryBatchTest, ExpectedCostToGoAtTheTerminalStageIsTheMeanTerminalCost) {
  const Model2x1 model = MakeModel(MakeDoubleIntegratorDrift(), Eigen::Vector2d(0.0, 0.1), Eigen::Vector2d(0.1, 0.1));
  const Policy2x1 policy(Eigen::Matrix<double, 1, 2>::Zero(), Control1::Zero());
  const Distribution2 distribution = MakeDistribution(Eigen::Vector2d(1.0, 0.0), Eigen::Vector2d(0.01, 0.01));

  const auto batch = Batch::Make(model, policy, distribution, 17);
  ASSERT_TRUE(batch.has_value()) << batch.error();

  const CostModel2x1 cost_model(
      {Eigen::Matrix2d::Identity(), Eigen::Matrix<double, 1, 1>::Identity(), Eigen::Matrix<double, 2, 1>::Zero()},
      QuadraticRegulatorSdeTerminalCostTerm<2>(Eigen::Matrix2d::Identity() * 5.0));

  const auto cost_to_go = batch->ExpectedCostToGo(cost_model);

  double expected_terminal_mean = 0.0;
  for (std::size_t trajectory = 0; trajectory < Batch::kNumTrajectories; ++trajectory) {
    expected_terminal_mean += cost_model.terminal_cost()(batch->StateAt(trajectory, Batch::kNumStages - 1));
  }
  expected_terminal_mean /= static_cast<double>(Batch::kNumTrajectories);

  EXPECT_NEAR(cost_to_go[Batch::kNumStages - 1], expected_terminal_mean, 1e-9);
}

// E[c_k] = E[l_k] + E[c_{k+1}] holds exactly, sample by sample, because every expectation is
// taken over the same set of trajectories.
TEST(TrajectoryBatchTest, ExpectedCostToGoSatisfiesTheBackwardRecursion) {
  const Model2x1 model = MakeModel(MakeDoubleIntegratorDrift(), Eigen::Vector2d(0.0, 0.1), Eigen::Vector2d(0.1, 0.1));

  Eigen::Matrix<double, 1, 2> gain;
  gain << -1.0, -2.0;
  const Policy2x1 policy(gain, Control1::Zero());
  const Distribution2 distribution = MakeDistribution(Eigen::Vector2d(1.0, 0.0), Eigen::Vector2d(0.01, 0.01));

  const auto batch = Batch::Make(model, policy, distribution, 23);
  ASSERT_TRUE(batch.has_value()) << batch.error();

  const CostModel2x1 cost_model(
      {Eigen::Matrix2d::Identity(), Eigen::Matrix<double, 1, 1>::Identity() * 0.5, Eigen::Matrix<double, 2, 1>::Zero()},
      QuadraticRegulatorSdeTerminalCostTerm<2>(Eigen::Matrix2d::Identity() * 5.0));

  const auto cost_to_go = batch->ExpectedCostToGo(cost_model);

  for (std::size_t stage = 0; stage < Batch::kNumControlStages; ++stage) {
    double mean_running_cost = 0.0;
    for (std::size_t trajectory = 0; trajectory < Batch::kNumTrajectories; ++trajectory) {
      mean_running_cost +=
          cost_model.running_cost()(stage, batch->StateAt(trajectory, stage), batch->ControlAt(trajectory, stage));
    }
    mean_running_cost /= static_cast<double>(Batch::kNumTrajectories);

    EXPECT_NEAR(cost_to_go[stage], mean_running_cost + cost_to_go[stage + 1], 1e-9) << "at stage " << stage;
  }
}

TEST(TrajectoryBatchTest, ExpectedCostToGoMatchesAnAnalyticNoiselessRollout) {
  const Eigen::Matrix2d a = MakeDoubleIntegratorDrift();
  const Eigen::Vector2d b(0.0, 0.1);
  const Model2x1 model = MakeModel(a, b, Eigen::Vector2d(kNegligible, kNegligible));
  const Policy2x1 policy(Eigen::Matrix<double, 1, 2>::Zero(), Control1::Zero());

  const Eigen::Vector2d initial_mean(1.0, 0.0);
  const Distribution2 distribution = MakeDistribution(initial_mean, Eigen::Vector2d(kNegligible, kNegligible));

  const auto batch = Batch::Make(model, policy, distribution, 29);
  ASSERT_TRUE(batch.has_value()) << batch.error();

  const CostModel2x1 cost_model(
      {Eigen::Matrix2d::Identity(), Eigen::Matrix<double, 1, 1>::Identity(), Eigen::Matrix<double, 2, 1>::Zero()},
      QuadraticRegulatorSdeTerminalCostTerm<2>(Eigen::Matrix2d::Identity() * 2.0));

  // With a zero policy the state never changes: x stays at (1, 0), so every running cost is
  // x^T Q x = 1 and the terminal cost is 2. The cost-to-go from stage k is therefore
  // (K - k) * 1 + 2.
  const auto cost_to_go = batch->ExpectedCostToGo(cost_model);

  for (std::size_t stage = 0; stage < Batch::kNumStages; ++stage) {
    const double expected = static_cast<double>(Batch::kNumStages - 1 - stage) + 2.0;
    EXPECT_NEAR(cost_to_go[stage], expected, kNoiselessTolerance) << "at stage " << stage;
  }
}

}  // namespace
}  // namespace fbsde_traj_opt
