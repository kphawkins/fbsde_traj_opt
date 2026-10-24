// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#include "fbsde_traj_opt/value_function/value_function_sgd_fitter.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "fbsde_traj_opt/value_function/soft_min_quadratic_value_function_approx.hpp"
#include "fbsde_traj_opt/value_function/value_function_approx_concepts.hpp"

namespace fbsde_traj_opt {
namespace {

constexpr int kBatchSize = 8;
constexpr int kNumSamples = 60;
constexpr double kSharpness = 2.0;
constexpr double kDiagonalOffset = 1e-3;

using Model = SoftMinQuadraticValueFunctionApprox<2, 2>;
using State = Model::State;
using Fitter = ValueFunctionSgdFitter<Model, kBatchSize>;
using Options = ValueFunctionSgdOptions<double>;
using StepReport = ValueFunctionSgdStepReport<double>;

using SampleStates = ValueFunctionStateBatch<State, kNumSamples>;
using SampleValues = ValueFunctionValueBatch<State, kNumSamples>;

// A deterministic, well-spread source of values. Reproducible arbitrary numbers, nothing more.
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

auto SpreadParameters(std::uint64_t seed, double magnitude) noexcept -> Model::ParameterVector {
  Spread spread(seed);
  Model::ParameterVector parameters;
  for (Eigen::Index index = 0; index < Model::kNumParameters; ++index) {
    parameters[index] = spread.Next(magnitude);
  }
  return parameters;
}

auto SpreadStates(std::uint64_t seed, double magnitude) noexcept -> SampleStates {
  Spread spread(seed);
  SampleStates states;
  for (Eigen::Index column = 0; column < kNumSamples; ++column) {
    for (Eigen::Index row = 0; row < 2; ++row) {
      states(row, column) = spread.Next(magnitude);
    }
  }
  return states;
}

// A reachable regression problem: the targets are the values of another model of the same family,
// so a perfect fit exists and the residual has somewhere to go.
struct Problem {
  SampleStates states;
  SampleValues targets;
  Model model;
};

auto MakeProblem(std::uint64_t target_seed, std::uint64_t start_seed) noexcept -> Problem {
  const auto target_model = Model::Make(SpreadParameters(target_seed, 1.0), kSharpness, kDiagonalOffset);
  EXPECT_TRUE(target_model.has_value());
  const auto start_model = Model::Make(SpreadParameters(start_seed, 1.0), kSharpness, kDiagonalOffset);
  EXPECT_TRUE(start_model.has_value());

  const SampleStates states = SpreadStates(7, 2.0);
  SampleValues targets;
  target_model->Values<kNumSamples>(states, targets);

  return Problem{.states = states, .targets = targets, .model = *start_model};
}

auto MeanSquaredResidual(const Model& model, const SampleStates& states, const SampleValues& targets) noexcept
    -> double {
  SampleValues values;
  model.Values<kNumSamples>(states, values);
  return (values - targets).squaredNorm() / static_cast<double>(kNumSamples);
}

// ---------------------------------------------------------------------------------------------
// Make
// ---------------------------------------------------------------------------------------------

TEST(ValueFunctionSgdFitterTest, MakeSucceedsWithDefaultOptions) {
  EXPECT_TRUE(Fitter::Make(Options{}, Model::ParameterVector::Zero()).has_value());
}

TEST(ValueFunctionSgdFitterTest, MakeRejectsTheAdamHyperparametersItDelegates) {
  Options options;
  options.learning_rate = -1.0;

  EXPECT_FALSE(Fitter::Make(options, Model::ParameterVector::Zero()).has_value());
}

TEST(ValueFunctionSgdFitterTest, MakeRejectsOutOfRangeDampingOptions) {
  const Model::ParameterVector anchor = Model::ParameterVector::Zero();

  Options negative_proximal;
  negative_proximal.proximal_weight = -1e-3;
  EXPECT_FALSE(Fitter::Make(negative_proximal, anchor).has_value());

  Options negative_trust_region;
  negative_trust_region.max_rms_value_change = -0.1;
  EXPECT_FALSE(Fitter::Make(negative_trust_region, anchor).has_value());

  Options averaging_above_one;
  averaging_above_one.parameter_averaging_rate = 1.5;
  EXPECT_FALSE(Fitter::Make(averaging_above_one, anchor).has_value());

  Options non_finite_huber;
  non_finite_huber.huber_threshold = std::numeric_limits<double>::infinity();
  EXPECT_FALSE(Fitter::Make(non_finite_huber, anchor).has_value());
}

TEST(ValueFunctionSgdFitterTest, MakeFailsOnANonFiniteAnchor) {
  Model::ParameterVector anchor = Model::ParameterVector::Zero();
  anchor[0] = std::numeric_limits<double>::quiet_NaN();

  EXPECT_FALSE(Fitter::Make(Options{}, anchor).has_value());
}

// ---------------------------------------------------------------------------------------------
// Fitting
// ---------------------------------------------------------------------------------------------

TEST(ValueFunctionSgdFitterTest, FittingReducesTheResidualOnAReachableTarget) {
  Problem problem = MakeProblem(101, 202);
  const double before = MeanSquaredResidual(problem.model, problem.states, problem.targets);

  Options options;
  options.learning_rate = 2e-2;
  auto fitter = Fitter::Make(options, problem.model.Parameters());
  ASSERT_TRUE(fitter.has_value()) << fitter.error();

  std::vector<StepReport> history;
  ASSERT_TRUE(
      fitter->FitEpochs<kNumSamples>(problem.model, problem.states, problem.targets, 400, 5, history).has_value());

  const double after = MeanSquaredResidual(problem.model, problem.states, problem.targets);

  EXPECT_LT(after, before / 100.0) << "before " << before << ", after " << after;
  EXPECT_EQ(history.size(), 400U * ((kNumSamples + kBatchSize - 1) / kBatchSize));
}

TEST(ValueFunctionSgdFitterTest, AZeroLearningRateIsRejectedRatherThanSilentlyDoingNothing) {
  Options options;
  options.learning_rate = 0.0;

  EXPECT_FALSE(Fitter::Make(options, Model::ParameterVector::Zero()).has_value());
}

TEST(ValueFunctionSgdFitterTest, TheStepReportDescribesTheStepThatWasTaken) {
  Problem problem = MakeProblem(103, 204);

  auto fitter = Fitter::Make(Options{}, problem.model.Parameters());
  ASSERT_TRUE(fitter.has_value()) << fitter.error();

  ValueFunctionStateBatch<State, kBatchSize> states = problem.states.leftCols<kBatchSize>();
  ValueFunctionValueBatch<State, kBatchSize> targets = problem.targets.leftCols<kBatchSize>();

  ValueFunctionValueBatch<State, kBatchSize> values;
  problem.model.Values<kBatchSize>(states, values);
  const double expected_mean_squared = (values - targets).squaredNorm() / static_cast<double>(kBatchSize);

  StepReport report;
  ASSERT_TRUE(fitter->FitMinibatch(problem.model, states, targets, report).has_value());

  EXPECT_NEAR(report.mean_squared_residual, expected_mean_squared, 1e-12);
  EXPECT_NEAR(
      report.root_mean_squared_residual * report.root_mean_squared_residual, report.mean_squared_residual, 1e-12);
  EXPECT_GT(report.gradient_norm, 0.0);
  EXPECT_EQ(report.step_scale, 1.0);

  // With no trust region in force, the realized change should track the predicted one closely:
  // one default-rate Adam step is small enough that the model is still locally linear over it.
  EXPECT_NEAR(report.rms_value_change, report.predicted_rms_value_change, 0.1 * report.predicted_rms_value_change);
}

// ---------------------------------------------------------------------------------------------
// Damping
// ---------------------------------------------------------------------------------------------

TEST(ValueFunctionSgdFitterTest, TheTrustRegionBoundsThePredictedValueChangeOfEveryStep) {
  Problem problem = MakeProblem(105, 206);

  constexpr double kBound = 1e-3;
  Options options;
  // A learning rate far too large for this problem, so that the trust region is what holds the
  // steps down rather than the rate itself.
  options.learning_rate = 1.0;
  options.max_rms_value_change = kBound;

  auto fitter = Fitter::Make(options, problem.model.Parameters());
  ASSERT_TRUE(fitter.has_value()) << fitter.error();

  std::vector<StepReport> history;
  ASSERT_TRUE(
      fitter->FitEpochs<kNumSamples>(problem.model, problem.states, problem.targets, 30, 11, history).has_value());

  ASSERT_FALSE(history.empty());
  std::size_t bound_steps = 0;
  for (const StepReport& report : history) {
    const double requested = report.predicted_rms_value_change * report.step_scale;
    EXPECT_LE(requested, kBound * (1.0 + 1e-12));
    // The realized change is a nonlinear function of the step, so it is allowed to exceed the
    // first-order bound -- but only by a little, which is the whole claim the trust region makes.
    EXPECT_LT(report.rms_value_change, 4.0 * kBound);
    if (report.step_scale < 1.0) {
      ++bound_steps;
    }
  }
  EXPECT_GT(bound_steps, history.size() / 2) << "the trust region never bound, so the test proves nothing";
}

TEST(ValueFunctionSgdFitterTest, TheTrustRegionSlowsTheFitDownRatherThanStoppingIt) {
  // Same problem, same budget, one with the trust region and one without: the damped run must
  // still improve, and must move less far in parameter space.
  Problem free_problem = MakeProblem(107, 208);
  Problem damped_problem = MakeProblem(107, 208);
  const Model::ParameterVector start = free_problem.model.Parameters();
  const double before = MeanSquaredResidual(free_problem.model, free_problem.states, free_problem.targets);

  Options free_options;
  free_options.learning_rate = 2e-2;
  Options damped_options = free_options;
  damped_options.max_rms_value_change = 1e-3;

  auto free_fitter = Fitter::Make(free_options, start);
  ASSERT_TRUE(free_fitter.has_value()) << free_fitter.error();
  auto damped_fitter = Fitter::Make(damped_options, start);
  ASSERT_TRUE(damped_fitter.has_value()) << damped_fitter.error();

  std::vector<StepReport> free_history;
  std::vector<StepReport> damped_history;
  ASSERT_TRUE(
      free_fitter
          ->FitEpochs<kNumSamples>(free_problem.model, free_problem.states, free_problem.targets, 60, 13, free_history)
          .has_value());
  ASSERT_TRUE(damped_fitter
                  ->FitEpochs<kNumSamples>(
                      damped_problem.model, damped_problem.states, damped_problem.targets, 60, 13, damped_history)
                  .has_value());

  const double free_after = MeanSquaredResidual(free_problem.model, free_problem.states, free_problem.targets);
  const double damped_after = MeanSquaredResidual(damped_problem.model, damped_problem.states, damped_problem.targets);

  EXPECT_LT(damped_after, before);
  EXPECT_LT(free_after, damped_after);
  EXPECT_LT((damped_problem.model.Parameters() - start).norm(), (free_problem.model.Parameters() - start).norm());
}

TEST(ValueFunctionSgdFitterTest, TheProximalTermHoldsTheParametersNearerTheAnchor) {
  Problem free_problem = MakeProblem(109, 210);
  Problem proximal_problem = MakeProblem(109, 210);
  const Model::ParameterVector anchor = free_problem.model.Parameters();

  Options free_options;
  free_options.learning_rate = 2e-2;
  Options proximal_options = free_options;
  proximal_options.proximal_weight = 5.0;

  auto free_fitter = Fitter::Make(free_options, anchor);
  ASSERT_TRUE(free_fitter.has_value()) << free_fitter.error();
  auto proximal_fitter = Fitter::Make(proximal_options, anchor);
  ASSERT_TRUE(proximal_fitter.has_value()) << proximal_fitter.error();

  std::vector<StepReport> history;
  ASSERT_TRUE(
      free_fitter
          ->FitEpochs<kNumSamples>(free_problem.model, free_problem.states, free_problem.targets, 80, 17, history)
          .has_value());
  ASSERT_TRUE(proximal_fitter
                  ->FitEpochs<kNumSamples>(
                      proximal_problem.model, proximal_problem.states, proximal_problem.targets, 80, 17, history)
                  .has_value());

  EXPECT_LT((proximal_problem.model.Parameters() - anchor).norm(), (free_problem.model.Parameters() - anchor).norm());
}

TEST(ValueFunctionSgdFitterTest, PolyakAveragingTrailsTheLiveParametersAndIsOffByDefault) {
  Problem problem = MakeProblem(111, 212);
  const Model::ParameterVector anchor = problem.model.Parameters();

  Options options;
  options.learning_rate = 2e-2;
  options.parameter_averaging_rate = 0.05;

  auto fitter = Fitter::Make(options, anchor);
  ASSERT_TRUE(fitter.has_value()) << fitter.error();
  EXPECT_EQ(fitter->AveragedParameters(), anchor);

  std::vector<StepReport> history;
  ASSERT_TRUE(
      fitter->FitEpochs<kNumSamples>(problem.model, problem.states, problem.targets, 40, 19, history).has_value());

  const Model::ParameterVector live = problem.model.Parameters();
  const Model::ParameterVector averaged = fitter->AveragedParameters();

  EXPECT_NE(averaged, anchor);
  EXPECT_LT((averaged - anchor).norm(), (live - anchor).norm());

  Problem unaveraged_problem = MakeProblem(111, 212);
  Options unaveraged_options;
  unaveraged_options.learning_rate = 2e-2;
  auto unaveraged_fitter = Fitter::Make(unaveraged_options, anchor);
  ASSERT_TRUE(unaveraged_fitter.has_value()) << unaveraged_fitter.error();
  history.clear();
  ASSERT_TRUE(unaveraged_fitter
                  ->FitEpochs<kNumSamples>(
                      unaveraged_problem.model, unaveraged_problem.states, unaveraged_problem.targets, 40, 19, history)
                  .has_value());

  EXPECT_EQ(unaveraged_fitter->AveragedParameters(), anchor);
}

TEST(ValueFunctionSgdFitterTest, GradientClippingBoundsTheReportedGradientNorm) {
  Problem problem = MakeProblem(113, 214);

  constexpr double kBound = 1e-2;
  Options options;
  options.max_gradient_norm = kBound;

  auto fitter = Fitter::Make(options, problem.model.Parameters());
  ASSERT_TRUE(fitter.has_value()) << fitter.error();

  std::vector<StepReport> history;
  ASSERT_TRUE(
      fitter->FitEpochs<kNumSamples>(problem.model, problem.states, problem.targets, 10, 23, history).has_value());

  ASSERT_FALSE(history.empty());
  for (const StepReport& report : history) {
    EXPECT_LE(report.gradient_norm, kBound * (1.0 + 1e-12));
  }
}

TEST(ValueFunctionSgdFitterTest, TheHuberThresholdLimitsWhatOneOutlierCanDo) {
  // One target is moved a thousand units away. Under the squared loss its residual dominates the
  // minibatch gradient; under the Huber loss its contribution is capped at the threshold.
  Problem squared_problem = MakeProblem(115, 216);
  Problem huber_problem = MakeProblem(115, 216);
  squared_problem.targets[3] += 1000.0;
  huber_problem.targets[3] += 1000.0;

  ValueFunctionStateBatch<State, kBatchSize> states = squared_problem.states.leftCols<kBatchSize>();
  ValueFunctionValueBatch<State, kBatchSize> squared_targets = squared_problem.targets.leftCols<kBatchSize>();
  ValueFunctionValueBatch<State, kBatchSize> huber_targets = huber_problem.targets.leftCols<kBatchSize>();

  Options squared_options;
  Options huber_options;
  huber_options.huber_threshold = 1.0;

  auto squared_fitter = Fitter::Make(squared_options, squared_problem.model.Parameters());
  ASSERT_TRUE(squared_fitter.has_value()) << squared_fitter.error();
  auto huber_fitter = Fitter::Make(huber_options, huber_problem.model.Parameters());
  ASSERT_TRUE(huber_fitter.has_value()) << huber_fitter.error();

  StepReport squared_report;
  StepReport huber_report;
  ASSERT_TRUE(squared_fitter->FitMinibatch(squared_problem.model, states, squared_targets, squared_report).has_value());
  ASSERT_TRUE(huber_fitter->FitMinibatch(huber_problem.model, states, huber_targets, huber_report).has_value());

  EXPECT_LT(huber_report.gradient_norm, squared_report.gradient_norm / 10.0);
}

TEST(ValueFunctionSgdFitterTest, AZeroSampleWeightRemovesASampleFromTheStep) {
  Problem weighted_problem = MakeProblem(117, 218);
  Problem trimmed_problem = MakeProblem(117, 218);

  ValueFunctionStateBatch<State, kBatchSize> states = weighted_problem.states.leftCols<kBatchSize>();
  ValueFunctionValueBatch<State, kBatchSize> targets = weighted_problem.targets.leftCols<kBatchSize>();

  // Zero the last sample's weight, and separately make the last sample's residual zero by setting
  // its target to the model's own value. Both should produce the same step.
  ValueFunctionValueBatch<State, kBatchSize> weights = ValueFunctionValueBatch<State, kBatchSize>::Ones();
  weights[kBatchSize - 1] = 0.0;

  ValueFunctionValueBatch<State, kBatchSize> values;
  trimmed_problem.model.Values<kBatchSize>(states, values);
  ValueFunctionValueBatch<State, kBatchSize> trimmed_targets = targets;
  trimmed_targets[kBatchSize - 1] = values[kBatchSize - 1];

  auto weighted_fitter = Fitter::Make(Options{}, weighted_problem.model.Parameters());
  ASSERT_TRUE(weighted_fitter.has_value()) << weighted_fitter.error();
  auto trimmed_fitter = Fitter::Make(Options{}, trimmed_problem.model.Parameters());
  ASSERT_TRUE(trimmed_fitter.has_value()) << trimmed_fitter.error();

  StepReport weighted_report;
  StepReport trimmed_report;
  ASSERT_TRUE(
      weighted_fitter->FitMinibatch(weighted_problem.model, states, targets, weights, weighted_report).has_value());
  ASSERT_TRUE(trimmed_fitter->FitMinibatch(trimmed_problem.model, states, trimmed_targets, trimmed_report).has_value());

  EXPECT_LT((weighted_problem.model.Parameters() - trimmed_problem.model.Parameters()).norm(), 1e-14);
}

TEST(ValueFunctionSgdFitterTest, SetAnchorMovesTheAnchorAndRestartsTheAverage) {
  Problem problem = MakeProblem(119, 220);
  const Model::ParameterVector anchor = problem.model.Parameters();

  auto fitter = Fitter::Make(Options{}, anchor);
  ASSERT_TRUE(fitter.has_value()) << fitter.error();

  const Model::ParameterVector moved = anchor + Model::ParameterVector::Constant(0.5);
  ASSERT_TRUE(fitter->SetAnchor(moved).has_value());

  EXPECT_EQ(fitter->Anchor(), moved);
  EXPECT_EQ(fitter->AveragedParameters(), moved);

  Model::ParameterVector poisoned = anchor;
  poisoned[1] = std::numeric_limits<double>::infinity();
  EXPECT_FALSE(fitter->SetAnchor(poisoned).has_value());
  EXPECT_EQ(fitter->Anchor(), moved);
}

TEST(ValueFunctionSgdFitterTest, EveryEpochVisitsEverySample) {
  // The wrap-around permutation is what keeps every minibatch full, and the thing to check is
  // that it does not drop a sample on the way. Checked behaviorally, one sample at a time: give
  // every sample a target the model already matches except sample k, so that a minibatch without
  // k has an exactly zero gradient and therefore leaves both the parameters and Adam's moments
  // untouched. The parameters move after one epoch if and only if k was visited.
  constexpr std::size_t kBatchesPerEpoch =
      (static_cast<std::size_t>(kNumSamples) + kBatchSize - 1) / static_cast<std::size_t>(kBatchSize);

  const Problem reference = MakeProblem(121, 222);
  SampleValues self_targets;
  reference.model.Values<kNumSamples>(reference.states, self_targets);

  for (Eigen::Index sample = 0; sample < kNumSamples; ++sample) {
    Problem problem = MakeProblem(121, 222);
    SampleValues targets = self_targets;
    targets[sample] += 1.0;

    auto fitter = Fitter::Make(Options{}, problem.model.Parameters());
    ASSERT_TRUE(fitter.has_value()) << fitter.error();
    const Model::ParameterVector before = problem.model.Parameters();

    std::vector<StepReport> history;
    ASSERT_TRUE(fitter->FitEpochs<kNumSamples>(problem.model, problem.states, targets, 1, 29, history).has_value());

    EXPECT_EQ(history.size(), kBatchesPerEpoch);
    EXPECT_GT((problem.model.Parameters() - before).norm(), 0.0) << "sample " << sample << " was never visited";
  }
}

}  // namespace
}  // namespace fbsde_traj_opt
