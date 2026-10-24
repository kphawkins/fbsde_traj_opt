// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#include "fbsde_traj_opt/value_function/time_shared_linear_fit.hpp"

#include <cmath>
#include <cstddef>
#include <random>
#include <span>
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "fbsde_traj_opt/value_function/linear_basis_value_function_approx.hpp"

namespace fbsde_traj_opt {
namespace {

using Model = LinearBasisValueFunctionApprox<2, MonomialBasis<2, 3>>;
using State = Model::State;
using Parameters = Model::ParameterVector;
constexpr int kSamples = 64;
constexpr std::size_t kStages = 12;

auto MakeModel() -> Model {
  return *Model::Make(Parameters::Zero(), State::Zero(), State::Ones(), MonomialBasis<2, 3>{}, RidgeAnchor::kZero, 0.0);
}

// Normal equations of noise-free samples of phi(x)' theta at random states.
auto Equations(const Model& model, const Parameters& theta, unsigned seed) -> Model::NormalEquations {
  std::mt19937 generator(seed);
  std::uniform_real_distribution<double> uniform(-1.0, 1.0);
  ValueFunctionStateBatch<State, kSamples> states;
  ValueFunctionValueBatch<State, kSamples> targets;
  for (int sample = 0; sample < kSamples; ++sample) {
    states.col(sample) = State(uniform(generator), uniform(generator));
    targets[sample] = model.Features(states.col(sample)).dot(theta);
  }
  Model::NormalEquations equations;
  EXPECT_TRUE((model.AccumulateNormalEquations<kSamples>(
                   states, targets, ValueFunctionValueBatch<State, kSamples>::Ones(), equations))
                  .has_value());
  return equations;
}

auto RandomParameters(unsigned seed) -> Parameters {
  std::mt19937 generator(seed);
  std::normal_distribution<double> normal;
  Parameters theta;
  for (Eigen::Index index = 0; index < theta.size(); ++index) {
    theta[index] = normal(generator);
  }
  return theta;
}

TEST(TimeBasisTest, RejectsBadKnotCounts) {
  EXPECT_FALSE(TimeBasis<>::Make(TimeBasisKind::kPiecewiseLinear, 10, 1).has_value());
  EXPECT_FALSE(TimeBasis<>::Make(TimeBasisKind::kPiecewiseConstant, 10, 0).has_value());
  EXPECT_FALSE(TimeBasis<>::Make(TimeBasisKind::kPiecewiseConstant, 10, 11).has_value());
  EXPECT_FALSE(TimeBasis<>::Make(TimeBasisKind::kPerStage, 0, 1).has_value());
}

TEST(TimeBasisTest, EveryKindIsAPartitionOfUnity) {
  for (const TimeBasisKind kind :
       {TimeBasisKind::kPerStage, TimeBasisKind::kPiecewiseConstant, TimeBasisKind::kPiecewiseLinear}) {
    const auto basis = TimeBasis<>::Make(kind, kStages, 4);
    ASSERT_TRUE(basis.has_value());
    for (std::size_t stage = 0; stage < kStages; ++stage) {
      double sum = 0.0;
      for (const auto& term : basis->Terms(stage)) {
        EXPECT_LT(term.knot, basis->NumKnots());
        EXPECT_GE(term.weight, 0.0);
        sum += term.weight;
      }
      EXPECT_NEAR(sum, 1.0, 1e-12);
    }
  }
  // The linear basis's end knots sit on the first and last stages.
  const auto linear = TimeBasis<>::Make(TimeBasisKind::kPiecewiseLinear, kStages, 4);
  EXPECT_NEAR(linear->Terms(0)[0].weight, 1.0, 1e-12);
  EXPECT_NEAR(linear->Terms(kStages - 1)[1].weight, 1.0, 1e-12);
}

TEST(TimeSharedLinearFitTest, PerStageMatchesTheModelsOwnFit) {
  const Model model = MakeModel();
  auto fit = TimeSharedLinearFit<Model>::Make({.kind = TimeBasisKind::kPerStage}, kStages);
  ASSERT_TRUE(fit.has_value());
  std::vector<Model> table(kStages, model);
  std::vector<Model> reference(kStages, model);
  for (std::size_t stage = 0; stage < kStages; ++stage) {
    const auto equations = Equations(model, RandomParameters(static_cast<unsigned>(stage)), 100U + stage);
    ASSERT_TRUE(fit->SetStage(stage, equations, 1e-3).has_value());
    ASSERT_TRUE(reference[stage].FitNormalEquations(equations, 1e-3).has_value());
  }
  ASSERT_TRUE(fit->Fit().has_value());
  ASSERT_TRUE(fit->WriteTable(std::span<Model>(table)).has_value());
  EXPECT_EQ(fit->NumParameters(), kStages * static_cast<std::size_t>(Model::kNumParameters));
  for (std::size_t stage = 0; stage < kStages; ++stage) {
    EXPECT_LT((table[stage].Parameters() - reference[stage].Parameters()).norm(), 1e-9);
  }
}

TEST(TimeSharedLinearFitTest, StageWeightsLeaveThePerStageFitAlone) {
  const Model model = MakeModel();
  auto plain = TimeSharedLinearFit<Model>::Make({.kind = TimeBasisKind::kPerStage}, kStages);
  auto weighted = TimeSharedLinearFit<Model>::Make({.kind = TimeBasisKind::kPerStage}, kStages);
  ASSERT_TRUE(plain.has_value() && weighted.has_value());
  for (std::size_t stage = 0; stage < kStages; ++stage) {
    const auto equations = Equations(model, RandomParameters(static_cast<unsigned>(stage)), 600U + stage);
    ASSERT_TRUE(plain->SetStage(stage, equations, 1e-3).has_value());
    ASSERT_TRUE(weighted->SetStage(stage, equations, 1e-3, nullptr, 1e-3 + static_cast<double>(stage)).has_value());
  }
  EXPECT_FALSE(weighted->SetStage(0, Equations(model, RandomParameters(0), 1), 1e-3, nullptr, 0.0).has_value());
  ASSERT_TRUE(plain->Fit().has_value() && weighted->Fit().has_value());
  for (std::size_t stage = 0; stage < kStages; ++stage) {
    EXPECT_LT((plain->StageParameters(stage) - weighted->StageParameters(stage)).norm(), 1e-6);
  }
}

TEST(TimeSharedLinearFitTest, PiecewiseLinearRecoversInterpolatedCoefficients) {
  const Model model = MakeModel();
  constexpr std::size_t kKnots = 4;
  const auto basis = TimeBasis<>::Make(TimeBasisKind::kPiecewiseLinear, kStages, kKnots);
  std::vector<Parameters> knots;
  for (std::size_t knot = 0; knot < kKnots; ++knot) {
    knots.push_back(RandomParameters(10U + static_cast<unsigned>(knot)));
  }
  auto fit = TimeSharedLinearFit<Model>::Make({.kind = TimeBasisKind::kPiecewiseLinear, .num_knots = kKnots}, kStages);
  ASSERT_TRUE(fit.has_value());
  std::vector<Parameters> truth;
  for (std::size_t stage = 0; stage < kStages; ++stage) {
    Parameters theta = Parameters::Zero();
    for (const auto& term : basis->Terms(stage)) {
      theta += term.weight * knots[term.knot];
    }
    truth.push_back(theta);
    ASSERT_TRUE(fit->SetStage(stage, Equations(model, theta, 200U + stage), 0.0).has_value());
  }
  ASSERT_TRUE(fit->Fit().has_value());
  EXPECT_EQ(fit->NumParameters(), kKnots * static_cast<std::size_t>(Model::kNumParameters));
  for (std::size_t stage = 0; stage < kStages; ++stage) {
    EXPECT_LT((fit->StageParameters(stage) - truth[stage]).norm(), 1e-7);
  }
}

TEST(TimeSharedLinearFitTest, LowRankRecoversASharedBasis) {
  const Model model = MakeModel();
  constexpr std::size_t kRank = 2;
  const Parameters first = RandomParameters(1);
  const Parameters second = RandomParameters(2);
  auto fit = TimeSharedLinearFit<Model>::Make({.kind = TimeBasisKind::kPerStage, .rank = kRank, .sweeps = 10}, kStages);
  ASSERT_TRUE(fit.has_value());
  std::vector<Parameters> truth;
  for (std::size_t stage = 0; stage < kStages; ++stage) {
    const double phase = 0.5 * static_cast<double>(stage);
    truth.emplace_back((std::cos(phase) * first) + (std::sin(phase) * second));
    ASSERT_TRUE(fit->SetStage(stage, Equations(model, truth.back(), 300U + stage), 0.0).has_value());
  }
  ASSERT_TRUE(fit->Fit().has_value());
  EXPECT_EQ(fit->NumParameters(), (static_cast<std::size_t>(Model::kNumParameters) * kRank) + (kStages * kRank));
  for (std::size_t stage = 0; stage < kStages; ++stage) {
    EXPECT_LT((fit->StageParameters(stage) - truth[stage]).norm(), 1e-6);
  }
}

TEST(TimeSharedLinearFitTest, LocalSweepsConvergeToTheJointSolve) {
  const Model model = MakeModel();
  const TimeSharedFitOptions options{.kind = TimeBasisKind::kPiecewiseLinear, .num_knots = 5};
  auto joint = TimeSharedLinearFit<Model>::Make(options, kStages);
  auto local = TimeSharedLinearFit<Model>::Make(options, kStages);
  ASSERT_TRUE(joint.has_value() && local.has_value());
  for (std::size_t stage = 0; stage < kStages; ++stage) {
    // Data no time basis fits exactly, so the joint solve is a compromise.
    const auto equations = Equations(model, RandomParameters(400U + static_cast<unsigned>(stage)), 500U + stage);
    ASSERT_TRUE(joint->SetStage(stage, equations, 1e-3).has_value());
    ASSERT_TRUE(local->SetStage(stage, equations, 1e-3).has_value());
  }
  ASSERT_TRUE(joint->Fit().has_value());
  for (std::size_t sweep = 0; sweep < 200; ++sweep) {
    for (std::size_t stage = kStages; stage-- > 0;) {
      ASSERT_TRUE(local->SolveAroundStage(stage).has_value());
    }
  }
  for (std::size_t stage = 0; stage < kStages; ++stage) {
    EXPECT_LT((local->StageParameters(stage) - joint->StageParameters(stage)).norm(), 1e-6);
  }
}

TEST(TimeSharedLinearFitTest, InitializeReproducesARepresentableTable) {
  const Model model = MakeModel();
  std::vector<Model> table(kStages, model);
  const Parameters theta = RandomParameters(7);
  for (Model& stage : table) {
    ASSERT_TRUE(stage.SetParameters(theta).has_value());
  }
  auto fit =
      TimeSharedLinearFit<Model>::Make({.kind = TimeBasisKind::kPiecewiseLinear, .num_knots = 3, .rank = 2}, kStages);
  ASSERT_TRUE(fit.has_value());
  ASSERT_TRUE(fit->Initialize(std::span<const Model>(table)).has_value());
  for (std::size_t stage = 0; stage < kStages; ++stage) {
    EXPECT_LT((fit->StageParameters(stage) - theta).norm(), 1e-6);
  }
}

}  // namespace
}  // namespace fbsde_traj_opt
