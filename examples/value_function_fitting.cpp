// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#include "examples/value_function_fitting.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <numbers>
#include <ranges>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "fbsde_traj_opt/composed_cost_sde_model.hpp"
#include "fbsde_traj_opt/composed_forward_sde_model.hpp"
#include "fbsde_traj_opt/const_diagonal_sde_diffusion_term.hpp"
#include "fbsde_traj_opt/const_linear_feedback_sde_control_policy_term.hpp"
#include "fbsde_traj_opt/const_linear_sde_control_drift_mat_term.hpp"
#include "fbsde_traj_opt/const_linear_sde_state_drift_term.hpp"
#include "fbsde_traj_opt/diagonal_covariance_normal_distribution.hpp"
#include "fbsde_traj_opt/finite_horizon_lqr.hpp"
#include "fbsde_traj_opt/quadratic_regulator_sde_running_cost_term.hpp"
#include "fbsde_traj_opt/quadratic_regulator_sde_terminal_cost_term.hpp"
#include "fbsde_traj_opt/soft_min_quadratic_value_function_approx.hpp"
#include "fbsde_traj_opt/stage_varying_linear_feedback_sde_control_policy_term.hpp"
#include "fbsde_traj_opt/taylor_noiseless_backward_step_estimator.hpp"
#include "fbsde_traj_opt/trajectory_batch.hpp"
#include "fbsde_traj_opt/utils/result.hpp"
#include "fbsde_traj_opt/value_function_approx_concepts.hpp"
#include "fbsde_traj_opt/value_function_sgd_fitter.hpp"
#include "fbsde_traj_opt/viz/plotly_figure.hpp"

namespace fbsde_traj_opt::examples {
namespace {

using StepReport = ValueFunctionSgdStepReport<double>;
using SgdOptions = ValueFunctionSgdOptions<double>;

constexpr double kDiagonalOffset = 1e-4;

// Cumulative epoch counts on a geometric schedule, ending at `total`.
//
// Geometric rather than even because a fit moves fast at first and slowly later: evenly spaced
// frames would spend most of an animation on a curve that has already stopped moving, and would
// skip over the part where all the motion is.
auto GeometricEpochSchedule(std::size_t total, std::size_t count) noexcept -> std::vector<std::size_t> {
  std::vector<std::size_t> schedule;
  schedule.reserve(count);
  for (std::size_t index = 1; index <= count; ++index) {
    const double exponent = static_cast<double>(index) / static_cast<double>(count);
    const auto epochs = static_cast<std::size_t>(std::llround(std::pow(static_cast<double>(total), exponent)));
    // Strictly increasing: the low end of a geometric schedule rounds to the same integer twice.
    if (schedule.empty() || epochs > schedule.back()) {
      schedule.push_back(epochs);
    }
  }
  if (!schedule.empty()) {
    schedule.back() = total;
  }
  return schedule;
}

// The index of the frame whose epoch count sits closest to `epochs`.
auto NearestFrame(const std::vector<std::size_t>& frame_epochs, std::size_t epochs) noexcept -> std::size_t {
  std::size_t best = 0;
  for (std::size_t index = 1; index < frame_epochs.size(); ++index) {
    const auto current = static_cast<double>(frame_epochs[index]);
    const auto incumbent = static_cast<double>(frame_epochs[best]);
    const auto wanted = static_cast<double>(epochs);
    if (std::abs(current - wanted) < std::abs(incumbent - wanted)) {
      best = index;
    }
  }
  return best;
}

// Animation frames rounded to four decimals before they are serialized.
//
// A frame is a picture, not a record: it is read as a position on a curve or a step on a color
// ramp, neither of which can show more than a few significant digits. Full double precision costs
// roughly two and a half times the bytes for detail nothing on screen resolves, and an animated
// heatmap is thousands of values per frame. The static figures beside these, and the table twins
// under them, keep every digit.
auto RoundedForDisplay(std::vector<double> values) noexcept -> std::vector<double> {
  for (double& value : values) {
    value = std::round(value * 1e4) / 1e4;
  }
  return values;
}

// A relative error, written so that a value at the level of rounding does not print as zero.
auto FormatError(double value) noexcept -> std::string {
  return std::format("{:.2e}", value);
}

// At most this many points are plotted from a fitting history. A run of tens of thousands of
// steps says the same thing at six hundred points as at all of them, and the page stays light.
constexpr std::size_t kMaxPlottedSteps = 600;

// The categorical slots, assigned in order and never cycled.
constexpr std::array<viz::PlotColorRole, 5> kSeriesRoles{viz::PlotColorRole::kSeries1,
                                                         viz::PlotColorRole::kSeries2,
                                                         viz::PlotColorRole::kSeries3,
                                                         viz::PlotColorRole::kSeries4,
                                                         viz::PlotColorRole::kSeries5};

// Every `stride`-th entry of `history`, as (step index, value) ranges ready to plot.
struct SampledHistory {
  std::vector<double> steps;
  std::vector<double> values;
};

auto SampleHistory(const std::vector<StepReport>& history, double (*select)(const StepReport&)) noexcept
    -> SampledHistory {
  const std::size_t stride = std::max<std::size_t>(1, (history.size() + kMaxPlottedSteps - 1) / kMaxPlottedSteps);
  SampledHistory sampled;
  for (std::size_t index = 0; index < history.size(); index += stride) {
    sampled.steps.push_back(static_cast<double>(index));
    sampled.values.push_back(select(history[index]));
  }
  return sampled;
}

// A residual curve on a logarithmic axis. Values that reach exactly zero would be dropped by a
// log axis without a word, so they are floored at a value the caption can name.
auto AddLogSeries(viz::PlotlyFigure& figure,
                  const SampledHistory& sampled,
                  const std::string& name,
                  viz::PlotColorRole role,
                  double floor) noexcept -> Result<> {
  std::vector<double> floored;
  floored.reserve(sampled.values.size());
  for (const double value : sampled.values) {
    floored.push_back(std::max(value, floor));
  }
  return figure.AddLine(sampled.steps, floored, viz::LineStyle{.name = name, .color_role = role});
}

// ---------------------------------------------------------------------------------------------
// Experiment (a): a one-dimensional multi-basin target
// ---------------------------------------------------------------------------------------------

constexpr int kGenericComponents = 8;
constexpr int kGenericSamples = 121;
constexpr int kGenericBatch = 16;
constexpr std::size_t kGenericTotalEpochs = 4000;
constexpr std::size_t kGenericFrames = 44;
constexpr double kGenericLow = -4.0;
constexpr double kGenericHigh = 4.0;

using GenericModel = SoftMinQuadraticValueFunctionApprox<1, kGenericComponents>;
using GenericState = GenericModel::State;
using GenericFitter = ValueFunctionSgdFitter<GenericModel, kGenericBatch>;

// The target: a quadratic bowl with a ripple on it, so that it has several basins and is nowhere
// exactly a quadratic. Nothing about it was chosen to suit the model.
auto GenericTarget(double position) noexcept -> double {
  return (0.5 * position * position) + (3.0 * std::sin(1.5 * position)) + 2.0;
}

// The starting parameters: one broad bowl per component, their centers spread across the domain.
//
// Spreading them is not cosmetic. Components that start identical have identical parameter
// gradients and stay identical forever -- the softmax weights them equally, so nothing in the
// update can tell them apart. Breaking that symmetry at initialization is what lets the
// components find separate basins.
auto GenericInitialParameters() noexcept -> Result<GenericModel::ParameterVector> {
  const auto parameters = GenericModel::QuadraticParameters(
      Eigen::Matrix<double, 1, 1>::Constant(0.5), GenericState::Zero(), 4.0, 1.0, kDiagonalOffset);
  if (!parameters.has_value()) {
    return ErrorResult(parameters.error().message, parameters.error().location);
  }

  GenericModel::ParameterVector spread = *parameters;
  for (std::size_t component = 0; component < GenericModel::kNumComponents; ++component) {
    // The block layout is part of the model's documented interface: the lower triangle of L_p
    // first, then b_p, then c_p.
    const auto center_index =
        static_cast<Eigen::Index>(component * static_cast<std::size_t>(GenericModel::kParametersPerComponent)) +
        GenericModel::kLowerTriangleSize;
    spread[center_index] = kGenericLow + ((kGenericHigh - kGenericLow) * (static_cast<double>(component) + 0.5) /
                                          static_cast<double>(kGenericComponents));
  }
  return SuccessResult(spread);
}

}  // namespace

auto RunGenericFunctionFittingExperiment(std::uint64_t seed) noexcept -> Result<ValueFunctionFittingReport> {
  // The samples: a uniform grid over the domain. 121 of them into minibatches of 16, so the
  // fitter's wrap-around permutation is exercised rather than sidestepped.
  ValueFunctionStateBatch<GenericState, kGenericSamples> states;
  ValueFunctionValueBatch<GenericState, kGenericSamples> targets;
  for (Eigen::Index index = 0; index < kGenericSamples; ++index) {
    const double position =
        kGenericLow + ((kGenericHigh - kGenericLow) * static_cast<double>(index) / (kGenericSamples - 1));
    states(0, index) = position;
    targets[index] = GenericTarget(position);
  }

  const Result<GenericModel::ParameterVector> initial = GenericInitialParameters();
  if (!initial.has_value()) {
    return ErrorResult(initial.error().message, initial.error().location);
  }

  // The sharpness beta is the family's main dial, and it trades representation against
  // trainability in a way worth stating plainly. A large beta drives the soft minimum toward the
  // hard one, which is sharper at the ridges -- but the softmax then gives almost all the weight
  // to whichever component is currently lowest, so a component that never wins anywhere receives
  // almost no gradient and never moves. Measured on this target, beta = 4 leaves the fit stuck
  // around 8% of the target's range however long it runs, while beta = 1 reaches 0.4%. Sharpness
  // is not free.
  Result<GenericModel> model = GenericModel::Make(*initial, 1.0, kDiagonalOffset);
  if (!model.has_value()) {
    return ErrorResult(model.error().message, model.error().location);
  }

  SgdOptions options;
  options.learning_rate = 1.5e-2;
  // The damping the whole exercise is about: no step may move the fitted function by more than
  // 0.02 in root-mean-square over the samples, whatever the optimizer proposes.
  options.max_rms_value_change = 2e-2;

  Result<GenericFitter> fitter = GenericFitter::Make(options, model->Parameters());
  if (!fitter.has_value()) {
    return ErrorResult(fitter.error().message, fitter.error().location);
  }

  // A fine grid for drawing, independent of the grid that was fitted.
  constexpr int kPlotPoints = 401;
  std::vector<double> plot_positions;
  std::vector<double> plot_target;
  plot_positions.reserve(kPlotPoints);
  plot_target.reserve(kPlotPoints);
  for (int index = 0; index < kPlotPoints; ++index) {
    const double position = kGenericLow + ((kGenericHigh - kGenericLow) * static_cast<double>(index) /
                                           static_cast<double>(kPlotPoints - 1));
    plot_positions.push_back(position);
    plot_target.push_back(GenericTarget(position));
  }

  const auto sample_curve = [&plot_positions](const GenericModel& fitted) noexcept -> std::vector<double> {
    std::vector<double> curve;
    curve.reserve(plot_positions.size());
    for (const double position : plot_positions) {
      curve.push_back(fitted(GenericState::Constant(position)));
    }
    return curve;
  };

  // One pass, captured at every step of a geometric schedule. The animation gets every frame; the
  // snapshot figure picks four of them, so the two figures are two views of the same run rather
  // than two runs.
  const std::vector<std::size_t> schedule = GeometricEpochSchedule(kGenericTotalEpochs, kGenericFrames);
  std::vector<StepReport> history;
  std::vector<viz::AnimationFrame> frames;
  std::vector<std::size_t> frame_epochs;

  frames.push_back(viz::AnimationFrame{.label = "0 epochs", .trace_values = {RoundedForDisplay(sample_curve(*model))}});
  frame_epochs.push_back(0);

  std::size_t completed_epochs = 0;
  for (std::size_t step = 0; step < schedule.size(); ++step) {
    const std::size_t epochs = schedule[step] - completed_epochs;
    if (const Result<> fitted =
            fitter->FitEpochs<kGenericSamples>(*model, states, targets, epochs, seed + step, history);
        !fitted.has_value()) {
      return ErrorResult(fitted.error().message, fitted.error().location);
    }
    completed_epochs = schedule[step];

    frames.push_back(viz::AnimationFrame{.label = std::to_string(completed_epochs) + " epochs",
                                         .trace_values = {RoundedForDisplay(sample_curve(*model))}});
    frame_epochs.push_back(completed_epochs);
  }

  viz::PlotlyFigure animated("Approximation over training", "State x", "Value");
  RESULT_ASSERT(
      animated.AddLine(plot_positions, plot_target, viz::LineStyle{.name = "Target", .color_role = kSeriesRoles[0]})
          .has_value(),
      "RunGenericFunctionFittingExperiment: the animated figure's target curve could not be plotted.");
  RESULT_ASSERT(animated
                    .AddLine(plot_positions,
                             frames.back().trace_values[0],
                             viz::LineStyle{.name = "Approximation", .color_role = kSeriesRoles[1]})
                    .has_value(),
                "RunGenericFunctionFittingExperiment: the animated approximation curve could not be plotted.");
  constexpr std::array<std::size_t, 1> kAnimatedTrace{1};
  RESULT_ASSERT(
      animated
          .Animate(
              kAnimatedTrace, frames, viz::AnimationStyle{.frame_duration_ms = 70.0, .slider_prefix = "Fitted after "})
          .has_value(),
      "RunGenericFunctionFittingExperiment: the approximation animation could not be built.");

  // The same run, four frames of it, side by side for a reader who wants them all at once.
  viz::PlotlyFigure approach("Target and approximation", "State x", "Value");
  RESULT_ASSERT(
      approach.AddLine(plot_positions, plot_target, viz::LineStyle{.name = "Target", .color_role = kSeriesRoles[0]})
          .has_value(),
      "RunGenericFunctionFittingExperiment: the target curve could not be plotted.");
  constexpr std::array<std::size_t, 4> kSnapshotEpochs{5, 45, 450, kGenericTotalEpochs};
  for (std::size_t snapshot = 0; snapshot < kSnapshotEpochs.size(); ++snapshot) {
    const std::size_t frame = NearestFrame(frame_epochs, kSnapshotEpochs[snapshot]);
    const std::string name = "After " + std::to_string(frame_epochs[frame]) + " epochs";
    RESULT_ASSERT(approach
                      .AddLine(plot_positions,
                               frames[frame].trace_values[0],
                               viz::LineStyle{.name = name, .color_role = kSeriesRoles[snapshot + 1]})
                      .has_value(),
                  "RunGenericFunctionFittingExperiment: an approximation curve could not be plotted.");
  }

  viz::PlotlyFigure residual("Fit residual", "Minibatch step", "RMS residual");
  residual.UseLogarithmicYAxis();
  RESULT_ASSERT(
      AddLogSeries(
          residual,
          SampleHistory(history, [](const StepReport& report) noexcept { return report.root_mean_squared_residual; }),
          "RMS residual over the minibatch",
          kSeriesRoles[0],
          1e-12)
          .has_value(),
      "RunGenericFunctionFittingExperiment: the residual curve could not be plotted.");

  viz::PlotlyFigure change("Value change per step", "Minibatch step", "RMS change in the fitted value");
  change.UseLogarithmicYAxis();
  RESULT_ASSERT(
      AddLogSeries(
          change,
          SampleHistory(history, [](const StepReport& report) noexcept { return report.predicted_rms_value_change; }),
          "Proposed, before the trust region",
          kSeriesRoles[0],
          1e-12)
          .has_value(),
      "RunGenericFunctionFittingExperiment: the proposed-change curve could not be plotted.");
  RESULT_ASSERT(
      AddLogSeries(change,
                   SampleHistory(history, [](const StepReport& report) noexcept { return report.rms_value_change; }),
                   "Realized, after it",
                   kSeriesRoles[1],
                   1e-12)
          .has_value(),
      "RunGenericFunctionFittingExperiment: the realized-change curve could not be plotted.");

  viz::PlotlyFigure activity("Trust region activity", "Minibatch step", "Fraction of the proposed step taken");
  const SampledHistory scales =
      SampleHistory(history, [](const StepReport& report) noexcept { return report.step_scale; });
  RESULT_ASSERT(
      activity.AddLine(scales.steps, scales.values, viz::LineStyle{.name = "Step scale", .color_role = kSeriesRoles[0]})
          .has_value(),
      "RunGenericFunctionFittingExperiment: the step-scale curve could not be plotted.");
  const std::vector<double> unity(scales.steps.size(), 1.0);
  RESULT_ASSERT(activity
                    .AddLine(scales.steps,
                             unity,
                             viz::LineStyle{.name = "Unshortened",
                                            .color_role = viz::PlotColorRole::kMuted,
                                            .width = 1.5,
                                            .show_in_legend = true,
                                            .show_on_hover = false})
                    .has_value(),
                "RunGenericFunctionFittingExperiment: the reference line could not be plotted.");

  // The headline numbers, measured on the plotting grid rather than the fitted one, so the figure
  // and the number agree.
  const std::vector<double> final_curve = sample_curve(*model);
  double worst_relative = 0.0;
  double squared_total = 0.0;
  double target_span = 0.0;
  {
    const auto [low, high] = std::ranges::minmax_element(plot_target);
    target_span = *high - *low;
  }
  for (std::size_t index = 0; index < final_curve.size(); ++index) {
    const double error = std::abs(final_curve[index] - plot_target[index]);
    squared_total += error * error;
    worst_relative = std::max(worst_relative, error / target_span);
  }

  return SuccessResult(ValueFunctionFittingReport{
      .name = "Fitting a generic function",
      .description = "A soft minimum of eight quadratics, fitted by stochastic gradient descent to a target that is "
                     "nowhere a quadratic: a bowl with a ripple on it, sampled on a uniform grid. Every step is held "
                     "to a root-mean-square change of 0.02 in the fitted values, so the approximation walks toward "
                     "the target rather than jumping at it -- press play on the first figure to watch it do so, and "
                     "see the last figure for how often that bound binds. No FBSDE machinery is involved; this is "
                     "the model and the fitter alone.",
      .summary = "fitted a bowl-plus-ripple to within a fraction of a percent of its range, by SGD alone",
      .figures =
          {std::move(animated), std::move(approach), std::move(residual), std::move(change), std::move(activity)},
      .final_root_mean_squared_residual = std::sqrt(squared_total / static_cast<double>(final_curve.size())),
      .worst_relative_error = worst_relative});
}

namespace {

// ---------------------------------------------------------------------------------------------
// Experiment (b): a backward pass over an LQR problem, against the exact answer
// ---------------------------------------------------------------------------------------------

constexpr int kLqrStateDim = 2;
constexpr int kLqrControlDim = 1;
constexpr std::size_t kLqrStages = 12;
constexpr std::size_t kLqrTrajectories = 64;
constexpr auto kLqrTrajectoryCount = static_cast<Eigen::Index>(kLqrTrajectories);
constexpr int kLqrBatch = 16;

using LqrState = Eigen::Matrix<double, kLqrStateDim, 1>;
using LqrControl = Eigen::Matrix<double, kLqrControlDim, 1>;
using LqrStateMat = Eigen::Matrix<double, kLqrStateDim, kLqrStateDim>;
using LqrControlMat = Eigen::Matrix<double, kLqrStateDim, kLqrControlDim>;
using LqrGain = Eigen::Matrix<double, kLqrControlDim, kLqrStateDim>;

using LqrForwardModel = ComposedForwardSdeModel<kLqrStateDim,
                                                kLqrControlDim,
                                                ConstLinearSdeStateDriftTerm<kLqrStateDim>,
                                                ConstLinearSdeControlDriftMatTerm<kLqrStateDim, kLqrControlDim>,
                                                ConstDiagonalSdeDiffusionTerm<kLqrStateDim>>;
using LqrCostModel = ComposedCostSdeModel<kLqrStateDim,
                                          kLqrControlDim,
                                          QuadraticRegulatorSdeRunningCostTerm<kLqrStateDim, kLqrControlDim>,
                                          QuadraticRegulatorSdeTerminalCostTerm<kLqrStateDim>>;
using LqrRunningCost = QuadraticRegulatorSdeRunningCostTerm<kLqrStateDim, kLqrControlDim>;
using LqrOptimalPolicy = StageVaryingLinearFeedbackSdeControlPolicyTerm<kLqrStateDim, kLqrControlDim, kLqrStages - 1>;
using LqrBaselinePolicy = ConstLinearFeedbackSdeControlPolicyTerm<kLqrStateDim, kLqrControlDim>;
using LqrEstimator = TaylorNoiselessBackwardStepEstimator<kLqrStateDim,
                                                          kLqrControlDim,
                                                          LqrForwardModel,
                                                          LqrOptimalPolicy,
                                                          LqrRunningCost>;

// One component, because the value function of an LQR problem is exactly one quadratic: the model
// can represent the right answer with no error of its own, so whatever error the figures show
// belongs to the estimator or to the regression rather than to the family.
using LqrValueFunction = SoftMinQuadraticValueFunctionApprox<kLqrStateDim, 1>;
using LqrFitter = ValueFunctionSgdFitter<LqrValueFunction, kLqrBatch>;

// A lightly damped oscillator that the control reaches only through its second coordinate.
const LqrStateMat kLqrStateDrift = (LqrStateMat() << 0.0, 0.2, -0.3, -0.05).finished();
const LqrControlMat kLqrControlDrift = (LqrControlMat() << 0.0, 0.25).finished();
const LqrState kLqrDiffusionDiagonal = (LqrState() << 0.05, 0.08).finished();
const LqrStateMat kLqrStateCost = (LqrStateMat() << 1.0, 0.0, 0.0, 0.2).finished();
const Eigen::Matrix<double, kLqrControlDim, kLqrControlDim> kLqrControlCost =
    Eigen::Matrix<double, kLqrControlDim, kLqrControlDim>::Constant(0.1);
const LqrControlMat kLqrCrossCost = LqrControlMat::Zero();
const LqrStateMat kLqrTerminalCost = (LqrStateMat() << 2.0, 0.0, 0.0, 0.5).finished();

// The drift the forward pass uses. Deliberately far from optimal, so that D_i -- the Girsanov
// drift between the sampling measure and the policy's -- is genuinely nonzero and the off-policy
// terms of the estimator are doing work rather than vanishing.
const LqrGain kLqrBaselineGain = (LqrGain() << -0.4, -0.4).finished();

const LqrState kLqrInitialMean = (LqrState() << 1.5, -0.5).finished();
const LqrState kLqrInitialCovariance = (LqrState() << 0.16, 0.09).finished();

auto MakeLqrForwardModel() noexcept -> Result<LqrForwardModel> {
  const Result<ConstDiagonalSdeDiffusionTerm<kLqrStateDim>> diffusion =
      ConstDiagonalSdeDiffusionTerm<kLqrStateDim>::Make(kLqrDiffusionDiagonal);
  if (!diffusion.has_value()) {
    return ErrorResult(diffusion.error().message, diffusion.error().location);
  }
  return SuccessResult(
      LqrForwardModel(ConstLinearSdeStateDriftTerm<kLqrStateDim>(kLqrStateDrift),
                      ConstLinearSdeControlDriftMatTerm<kLqrStateDim, kLqrControlDim>(kLqrControlDrift),
                      *diffusion));
}

// The quadratic x' hessian x + offset, as a value function approximation.
auto MakeLqrQuadratic(const LqrStateMat& hessian, double offset) noexcept -> Result<LqrValueFunction> {
  const Result<LqrValueFunction::ParameterVector> parameters =
      LqrValueFunction::QuadraticParameters(hessian, LqrState::Zero(), offset, 1.0, kDiagonalOffset);
  if (!parameters.has_value()) {
    return ErrorResult(parameters.error().message, parameters.error().location);
  }
  return LqrValueFunction::Make(*parameters, 1.0, kDiagonalOffset);
}

}  // namespace

auto RunLqrBackwardPassExperiment(std::uint64_t seed) noexcept -> Result<ValueFunctionFittingReport> {
  const Result<LqrForwardModel> forward_model = MakeLqrForwardModel();
  if (!forward_model.has_value()) {
    return ErrorResult(forward_model.error().message, forward_model.error().location);
  }
  const LqrCostModel cost_model(LqrRunningCost(kLqrStateCost, kLqrControlCost, kLqrCrossCost),
                                QuadraticRegulatorSdeTerminalCostTerm<kLqrStateDim>(kLqrTerminalCost));

  const Result<FiniteHorizonLqrSolution<LqrForwardModel, kLqrStages>> solution =
      SolveFiniteHorizonLqrWithCostToGo<kLqrStages>(*forward_model, cost_model);
  if (!solution.has_value()) {
    return ErrorResult(solution.error().message, solution.error().location);
  }

  // The constant the additive noise adds to the value function: s_K = 0, and each step back picks
  // up the trace of the noise the next stage will inject.
  const LqrStateMat diffusion(kLqrDiffusionDiagonal.asDiagonal());
  std::array<double, kLqrStages> offsets{};
  for (std::size_t stage = kLqrStages - 1; stage-- > 0;) {
    offsets[stage] =
        offsets[stage + 1] + (diffusion.transpose() * solution->cost_to_go_hessians[stage + 1] * diffusion).trace();
  }
  const auto exact_value = [&solution, &offsets](std::size_t stage, const LqrState& state) noexcept -> double {
    return state.dot(solution->cost_to_go_hessians[stage] * state) + offsets[stage];
  };

  // The forward pass, rolled out under the baseline gain rather than the optimal one.
  const Result<DiagonalCovarianceNormalDistribution<kLqrStateDim>> initial_distribution =
      DiagonalCovarianceNormalDistribution<kLqrStateDim>::Make(kLqrInitialMean, kLqrInitialCovariance);
  if (!initial_distribution.has_value()) {
    return ErrorResult(initial_distribution.error().message, initial_distribution.error().location);
  }
  const LqrBaselinePolicy baseline_policy(kLqrBaselineGain, LqrControl::Zero());
  const Result<TrajectoryBatch<kLqrStateDim, kLqrControlDim, kLqrTrajectories, kLqrStages>> batch =
      TrajectoryBatch<kLqrStateDim, kLqrControlDim, kLqrTrajectories, kLqrStages>::Make(
          *forward_model, baseline_policy, *initial_distribution, seed);
  if (!batch.has_value()) {
    return ErrorResult(batch.error().message, batch.error().location);
  }

  const LqrEstimator estimator(
      *forward_model, solution->policy, LqrRunningCost(kLqrStateCost, kLqrControlCost, kLqrCrossCost));

  // The recursion starts from the terminal stage, where the value function is the terminal cost
  // exactly and nothing has to be estimated.
  Result<LqrValueFunction> next_value_function =
      MakeLqrQuadratic(solution->cost_to_go_hessians[kLqrStages - 1], offsets[kLqrStages - 1]);
  if (!next_value_function.has_value()) {
    return ErrorResult(next_value_function.error().message, next_value_function.error().location);
  }

  SgdOptions options;
  options.learning_rate = 1e-2;
  // Damped, but not so hard that a stage cannot reach its answer within its budget. Warm-starting
  // each stage from the one after it is what makes that possible: consecutive value functions are
  // close, so the distance a stage has to travel is small to begin with.
  options.max_rms_value_change = 5e-2;
  Result<LqrFitter> fitter = LqrFitter::Make(options, next_value_function->Parameters());
  if (!fitter.has_value()) {
    return ErrorResult(fitter.error().message, fitter.error().location);
  }

  std::vector<double> stage_axis;
  std::vector<double> isolated_estimator_errors;
  std::vector<double> propagated_estimator_errors;
  std::vector<double> fitted_errors;
  std::vector<double> mean_drift_norms;
  std::vector<double> max_drift_norms;
  std::vector<StepReport> history;
  std::array<LqrValueFunction::ParameterVector, kLqrStages> fitted_parameters{};
  fitted_parameters[kLqrStages - 1] = next_value_function->Parameters();

  for (std::size_t stage = kLqrStages - 1; stage-- > 0;) {
    const auto& states = batch->StatesAtStage(stage);
    const auto& controls = batch->ControlsAtStage(stage);

    // The drift the forward pass actually used to leave this stage: the baseline policy's, not
    // the target policy's. Reconstructed rather than stored, since the batch keeps the controls.
    ValueFunctionStateBatch<LqrState, kLqrTrajectories> drifts;
    for (Eigen::Index column = 0; column < kLqrTrajectoryCount; ++column) {
      const LqrState state = states.col(column);
      drifts.col(column) = (kLqrStateDrift * state) + (kLqrControlDrift * controls.col(column));
    }

    ValueFunctionValueBatch<LqrState, kLqrTrajectories> targets;
    estimator.EstimateTargets<kLqrTrajectories>(stage, states, drifts, *next_value_function, targets);

    // The same estimate, taken against the *exact* next-stage value function rather than the
    // fitted one. The difference between these two curves is what separates the estimator's own
    // error -- which on a quadratic problem is rounding and nothing else -- from the error the
    // regression at every later stage has already put into the representation it is reading.
    const Result<LqrValueFunction> exact_next_value_function =
        MakeLqrQuadratic(solution->cost_to_go_hessians[stage + 1], offsets[stage + 1]);
    if (!exact_next_value_function.has_value()) {
      return ErrorResult(exact_next_value_function.error().message, exact_next_value_function.error().location);
    }
    ValueFunctionValueBatch<LqrState, kLqrTrajectories> isolated_targets;
    estimator.EstimateTargets<kLqrTrajectories>(stage, states, drifts, *exact_next_value_function, isolated_targets);

    ValueFunctionValueBatch<LqrState, kLqrTrajectories> drift_norms;
    if (const Result<> measured = estimator.GirsanovDriftNorms<kLqrTrajectories>(stage, states, drifts, drift_norms);
        !measured.has_value()) {
      return ErrorResult(measured.error().message, measured.error().location);
    }

    double isolated_error = 0.0;
    double propagated_error = 0.0;
    double value_scale = 0.0;
    for (Eigen::Index column = 0; column < kLqrTrajectoryCount; ++column) {
      const double exact = exact_value(stage, states.col(column));
      isolated_error = std::max(isolated_error, std::abs(isolated_targets[column] - exact));
      propagated_error = std::max(propagated_error, std::abs(targets[column] - exact));
      value_scale = std::max(value_scale, std::abs(exact));
    }

    // Fit this stage, starting from the stage after it.
    Result<LqrValueFunction> current = *next_value_function;
    if (const Result<> anchored = fitter->SetAnchor(current->Parameters()); !anchored.has_value()) {
      return ErrorResult(anchored.error().message, anchored.error().location);
    }
    fitter->ResetOptimizer();
    if (const Result<> fitted =
            fitter->FitEpochs<kLqrTrajectories>(*current, states, targets, 1200, seed + stage, history);
        !fitted.has_value()) {
      return ErrorResult(fitted.error().message, fitted.error().location);
    }

    double fitted_error = 0.0;
    for (Eigen::Index column = 0; column < kLqrTrajectoryCount; ++column) {
      const LqrState state = states.col(column);
      fitted_error = std::max(fitted_error, std::abs((*current)(state)-exact_value(stage, state)));
    }

    stage_axis.push_back(static_cast<double>(stage));
    isolated_estimator_errors.push_back(isolated_error / value_scale);
    propagated_estimator_errors.push_back(propagated_error / value_scale);
    fitted_errors.push_back(fitted_error / value_scale);
    mean_drift_norms.push_back(drift_norms.mean());
    max_drift_norms.push_back(drift_norms.maxCoeff());

    fitted_parameters[stage] = current->Parameters();
    next_value_function = std::move(current);
  }

  std::ranges::reverse(stage_axis);
  std::ranges::reverse(isolated_estimator_errors);
  std::ranges::reverse(propagated_estimator_errors);
  std::ranges::reverse(fitted_errors);
  std::ranges::reverse(mean_drift_norms);
  std::ranges::reverse(max_drift_norms);

  // A slice through the state plane, at two stages, exact against fitted.
  constexpr int kSlicePoints = 161;
  std::vector<double> slice_positions;
  slice_positions.reserve(kSlicePoints);
  for (int index = 0; index < kSlicePoints; ++index) {
    slice_positions.push_back(-3.0 + (6.0 * static_cast<double>(index) / static_cast<double>(kSlicePoints - 1)));
  }

  // The exact and the fitted value function along that slice, at every stage. Ordered the way the
  // recursion ran -- from the last stage it estimated back to stage 0 -- so that playing the
  // animation is watching the backward pass happen, and the figure comes to rest on stage 0.
  const auto curves_at = [&slice_positions, &exact_value, &fitted_parameters](
                             std::size_t stage) noexcept -> Result<std::vector<std::vector<double>>> {
    const Result<LqrValueFunction> fitted = LqrValueFunction::Make(fitted_parameters[stage], 1.0, kDiagonalOffset);
    if (!fitted.has_value()) {
      return ErrorResult(fitted.error().message, fitted.error().location);
    }
    std::vector<double> exact_curve;
    std::vector<double> fitted_curve;
    exact_curve.reserve(slice_positions.size());
    fitted_curve.reserve(slice_positions.size());
    for (const double position : slice_positions) {
      const LqrState state(position, 0.0);
      exact_curve.push_back(exact_value(stage, state));
      fitted_curve.push_back((*fitted)(state));
    }
    return SuccessResult(std::vector<std::vector<double>>{RoundedForDisplay(std::move(exact_curve)),
                                                          RoundedForDisplay(std::move(fitted_curve))});
  };

  std::vector<viz::AnimationFrame> slice_frames;
  for (std::size_t stage = kLqrStages - 1; stage-- > 0;) {
    Result<std::vector<std::vector<double>>> curves = curves_at(stage);
    if (!curves.has_value()) {
      return ErrorResult(curves.error().message, curves.error().location);
    }
    slice_frames.push_back(
        viz::AnimationFrame{.label = "stage " + std::to_string(stage), .trace_values = std::move(*curves)});
  }

  viz::PlotlyFigure slice("Value function along the line x2 = 0", "State x1", "Value");
  RESULT_ASSERT(slice
                    .AddLine(slice_positions,
                             slice_frames.back().trace_values[0],
                             viz::LineStyle{.name = "Exact", .color_role = kSeriesRoles[0]})
                    .has_value(),
                "RunLqrBackwardPassExperiment: the exact value curve could not be plotted.");
  RESULT_ASSERT(slice
                    .AddLine(slice_positions,
                             slice_frames.back().trace_values[1],
                             viz::LineStyle{.name = "Fitted", .color_role = kSeriesRoles[1]})
                    .has_value(),
                "RunLqrBackwardPassExperiment: the fitted value curve could not be plotted.");
  constexpr std::array<std::size_t, 2> kAnimatedSliceTraces{0, 1};
  RESULT_ASSERT(slice
                    .Animate(kAnimatedSliceTraces,
                             slice_frames,
                             viz::AnimationStyle{.frame_duration_ms = 320.0, .slider_prefix = "Backward pass at "})
                    .has_value(),
                "RunLqrBackwardPassExperiment: the backward pass animation could not be built.");

  viz::PlotlyFigure accuracy("Accuracy by stage, relative to the exact value function", "Stage", "Relative error");
  accuracy.UseLogarithmicYAxis();
  // Floored so the log axis can draw the stages where the estimator's own error is genuinely zero.
  std::vector<double> floored_isolated;
  floored_isolated.reserve(isolated_estimator_errors.size());
  for (const double error : isolated_estimator_errors) {
    floored_isolated.push_back(std::max(error, 1e-17));
  }
  RESULT_ASSERT(
      accuracy
          .AddLine(stage_axis,
                   floored_isolated,
                   viz::LineStyle{.name = "Estimator, reading the exact next stage", .color_role = kSeriesRoles[0]})
          .has_value(),
      "RunLqrBackwardPassExperiment: the isolated estimator error curve could not be plotted.");
  RESULT_ASSERT(
      accuracy
          .AddLine(stage_axis,
                   propagated_estimator_errors,
                   viz::LineStyle{.name = "Estimator, reading the fitted next stage", .color_role = kSeriesRoles[1]})
          .has_value(),
      "RunLqrBackwardPassExperiment: the propagated estimator error curve could not be plotted.");
  RESULT_ASSERT(
      accuracy
          .AddLine(
              stage_axis, fitted_errors, viz::LineStyle{.name = "Fitted representation", .color_role = kSeriesRoles[2]})
          .has_value(),
      "RunLqrBackwardPassExperiment: the fitted error curve could not be plotted.");

  viz::PlotlyFigure drift("Girsanov drift magnitude by stage", "Stage", "||D_i||");
  RESULT_ASSERT(
      drift.AddLine(stage_axis, mean_drift_norms, viz::LineStyle{.name = "Mean", .color_role = kSeriesRoles[0]})
          .has_value(),
      "RunLqrBackwardPassExperiment: the mean drift curve could not be plotted.");
  RESULT_ASSERT(
      drift.AddLine(stage_axis, max_drift_norms, viz::LineStyle{.name = "Worst sample", .color_role = kSeriesRoles[1]})
          .has_value(),
      "RunLqrBackwardPassExperiment: the worst drift curve could not be plotted.");
  const std::vector<double> bound(stage_axis.size(), 1.0);
  RESULT_ASSERT(drift
                    .AddLine(stage_axis,
                             bound,
                             viz::LineStyle{.name = "The bound's comfortable regime",
                                            .color_role = viz::PlotColorRole::kMuted,
                                            .width = 1.5,
                                            .show_on_hover = false})
                    .has_value(),
                "RunLqrBackwardPassExperiment: the drift reference line could not be plotted.");

  viz::PlotlyFigure residual("Fit residual over the whole backward pass", "Minibatch step", "RMS residual");
  residual.UseLogarithmicYAxis();
  RESULT_ASSERT(
      AddLogSeries(
          residual,
          SampleHistory(history, [](const StepReport& report) noexcept { return report.root_mean_squared_residual; }),
          "RMS residual over the minibatch",
          kSeriesRoles[0],
          1e-14)
          .has_value(),
      "RunLqrBackwardPassExperiment: the residual curve could not be plotted.");

  const double worst_isolated = *std::ranges::max_element(isolated_estimator_errors);
  const double worst_fitted = *std::ranges::max_element(fitted_errors);

  return SuccessResult(ValueFunctionFittingReport{
      .name = "A backward pass over an LQR problem",
      .description =
          "The Taylor Noiseless estimator swept backward over a linear-quadratic problem, fitting a value function "
          "approximation at every stage to the targets it produces. Press play on the first figure to watch the "
          "recursion march back from the terminal stage to stage 0, the fitted curve tracking the exact one the "
          "whole way. The forward samples were rolled out under a "
          "deliberately suboptimal drift, so the estimator is genuinely off-policy; the third figure reports the "
          "Girsanov drift that costs. Because the true value function of this problem is exactly quadratic, the "
          "Riccati recursion gives the right answer at every stage, and the second figure takes the error apart "
          "into its three pieces: the estimator reading the exact next-stage value function, which is its own error "
          "and sits at the level of rounding; the estimator reading the fitted one, which adds whatever the "
          "regression has already put into the representation; and the fitted representation itself. Worst relative "
          "error over all stages: " +
          FormatError(worst_isolated) + " for the estimator itself, " + FormatError(worst_fitted) +
          " for the fitted representation.",
      .summary = "the estimator itself matched the exact Riccati value function to a relative " +
                 FormatError(worst_isolated) + "; the fitted representation to " + FormatError(worst_fitted),
      .figures = {std::move(slice), std::move(accuracy), std::move(drift), std::move(residual)},
      .final_root_mean_squared_residual = history.empty() ? 0.0 : history.back().root_mean_squared_residual,
      .worst_relative_error = worst_fitted,
      .estimator_relative_error = worst_isolated});
}

namespace {

// ---------------------------------------------------------------------------------------------
// Experiment (c): a two-dimensional multi-basin target
// ---------------------------------------------------------------------------------------------

constexpr int kPlaneComponents = 12;
constexpr int kPlaneAxis = 21;
constexpr int kPlaneSamples = kPlaneAxis * kPlaneAxis;
constexpr int kPlaneBatch = 32;
constexpr double kPlaneLow = -3.0;
constexpr double kPlaneHigh = 3.0;

// A finer grid for drawing than the one that was fitted, so the figures are not a picture of the
// training set.
constexpr int kPlaneDrawAxis = 61;

// The animated figures redraw their whole grid on every frame, so they use a coarser one: at the
// drawing grid's resolution a few dozen frames would be several megabytes of JSON, for detail an
// animation cannot be read at anyway.
constexpr int kPlaneAnimationAxis = 41;
constexpr std::size_t kPlaneTotalEpochs = 1500;
constexpr std::size_t kPlaneFrames = 22;

using PlaneModel = SoftMinQuadraticValueFunctionApprox<2, kPlaneComponents>;
using PlaneState = PlaneModel::State;
using PlaneFitter = ValueFunctionSgdFitter<PlaneModel, kPlaneBatch>;

// The target: a bowl with a two-dimensional ripple, so that it has several distinct basins in the
// plane rather than the one a quadratic would have.
auto PlaneTarget(const PlaneState& state) noexcept -> double {
  return (0.4 * state.squaredNorm()) + (2.5 * std::sin(1.2 * state[0]) * std::cos(1.2 * state[1])) + 4.0;
}

auto PlaneAxisValues(int count) noexcept -> std::vector<double> {
  std::vector<double> axis;
  axis.reserve(static_cast<std::size_t>(count));
  for (int index = 0; index < count; ++index) {
    axis.push_back(kPlaneLow +
                   ((kPlaneHigh - kPlaneLow) * static_cast<double>(index) / static_cast<double>(count - 1)));
  }
  return axis;
}

// The starting parameters: one bowl per component, their centers laid out on two concentric
// rings. As in the one-dimensional case, identical components would never separate; two radii
// rather than one because a single ring leaves the middle of the domain uncovered.
auto PlaneInitialParameters() noexcept -> Result<PlaneModel::ParameterVector> {
  const Result<PlaneModel::ParameterVector> parameters =
      PlaneModel::QuadraticParameters(0.5 * Eigen::Matrix2d::Identity(), PlaneState::Zero(), 6.0, 1.0, kDiagonalOffset);
  if (!parameters.has_value()) {
    return ErrorResult(parameters.error().message, parameters.error().location);
  }

  PlaneModel::ParameterVector spread = *parameters;
  for (std::size_t component = 0; component < PlaneModel::kNumComponents; ++component) {
    const auto center_index =
        static_cast<Eigen::Index>(component * static_cast<std::size_t>(PlaneModel::kParametersPerComponent)) +
        PlaneModel::kLowerTriangleSize;
    const double angle =
        2.0 * std::numbers::pi * static_cast<double>(component) / static_cast<double>(kPlaneComponents);
    const double radius = (component % 2 == 0) ? 2.0 : 1.0;
    spread[center_index] = radius * std::cos(angle);
    spread[center_index + 1] = radius * std::sin(angle);
  }
  return SuccessResult(spread);
}

}  // namespace

auto RunTwoDimensionalFittingExperiment(std::uint64_t seed) noexcept -> Result<ValueFunctionFittingReport> {
  const std::vector<double> fit_axis = PlaneAxisValues(kPlaneAxis);

  ValueFunctionStateBatch<PlaneState, kPlaneSamples> states;
  ValueFunctionValueBatch<PlaneState, kPlaneSamples> targets;
  for (Eigen::Index row = 0; row < kPlaneAxis; ++row) {
    for (Eigen::Index column = 0; column < kPlaneAxis; ++column) {
      const Eigen::Index index = (row * kPlaneAxis) + column;
      const PlaneState state(fit_axis[static_cast<std::size_t>(column)], fit_axis[static_cast<std::size_t>(row)]);
      states.col(index) = state;
      targets[index] = PlaneTarget(state);
    }
  }

  const Result<PlaneModel::ParameterVector> initial = PlaneInitialParameters();
  if (!initial.has_value()) {
    return ErrorResult(initial.error().message, initial.error().location);
  }
  // A softer sharpness than the one-dimensional case, for the same reason and more of it: with
  // twelve components in a plane, a sharp softmax would leave most of them without gradient. At
  // this beta the components blend rather than compete, which is both more expressive on a smooth
  // target and far easier to descend.
  Result<PlaneModel> model = PlaneModel::Make(*initial, 0.25, kDiagonalOffset);
  if (!model.has_value()) {
    return ErrorResult(model.error().message, model.error().location);
  }

  SgdOptions options;
  options.learning_rate = 1.5e-2;
  options.max_rms_value_change = 2e-2;
  Result<PlaneFitter> fitter = PlaneFitter::Make(options, model->Parameters());
  if (!fitter.has_value()) {
    return ErrorResult(fitter.error().message, fitter.error().location);
  }

  // The coarse grid the animation redraws, and the two frame sequences taken over it.
  const std::vector<double> animation_axis = PlaneAxisValues(kPlaneAnimationAxis);
  const auto animation_grids = [&animation_axis, &model]() noexcept -> std::vector<std::vector<double>> {
    std::vector<double> approximation;
    std::vector<double> error;
    const auto cells = animation_axis.size() * animation_axis.size();
    approximation.reserve(cells);
    error.reserve(cells);
    for (const double second : animation_axis) {
      for (const double first : animation_axis) {
        const PlaneState state(first, second);
        const double value = (*model)(state);
        approximation.push_back(value);
        error.push_back(value - PlaneTarget(state));
      }
    }
    return {std::move(approximation), std::move(error)};
  };

  std::vector<StepReport> history;
  std::vector<viz::AnimationFrame> approximation_frames;
  std::vector<viz::AnimationFrame> error_frames;
  {
    std::vector<std::vector<double>> grids = animation_grids();
    approximation_frames.push_back(
        viz::AnimationFrame{.label = "0 epochs", .trace_values = {RoundedForDisplay(grids[0])}});
    error_frames.push_back(viz::AnimationFrame{.label = "0 epochs", .trace_values = {RoundedForDisplay(grids[1])}});
  }

  const std::vector<std::size_t> schedule = GeometricEpochSchedule(kPlaneTotalEpochs, kPlaneFrames);
  std::size_t completed_epochs = 0;
  for (std::size_t step = 0; step < schedule.size(); ++step) {
    const std::size_t epochs = schedule[step] - completed_epochs;
    if (const Result<> fitted = fitter->FitEpochs<kPlaneSamples>(*model, states, targets, epochs, seed + step, history);
        !fitted.has_value()) {
      return ErrorResult(fitted.error().message, fitted.error().location);
    }
    completed_epochs = schedule[step];

    std::vector<std::vector<double>> grids = animation_grids();
    const std::string label = std::to_string(completed_epochs) + " epochs";
    approximation_frames.push_back(viz::AnimationFrame{.label = label, .trace_values = {RoundedForDisplay(grids[0])}});
    error_frames.push_back(viz::AnimationFrame{.label = label, .trace_values = {RoundedForDisplay(grids[1])}});
  }

  // The three grids the figures draw, all on the finer axis.
  const std::vector<double> draw_axis = PlaneAxisValues(kPlaneDrawAxis);
  std::vector<double> target_grid;
  std::vector<double> fitted_grid;
  std::vector<double> error_grid;
  const auto cell_count = static_cast<std::size_t>(kPlaneDrawAxis) * static_cast<std::size_t>(kPlaneDrawAxis);
  target_grid.reserve(cell_count);
  fitted_grid.reserve(cell_count);
  error_grid.reserve(cell_count);

  double worst_error = 0.0;
  double squared_total = 0.0;
  double lowest = 0.0;
  double highest = 0.0;
  bool seen = false;
  for (const double second : draw_axis) {
    for (const double first : draw_axis) {
      const PlaneState state(first, second);
      const double target = PlaneTarget(state);
      const double fitted = (*model)(state);
      target_grid.push_back(target);
      fitted_grid.push_back(fitted);
      error_grid.push_back(fitted - target);

      const double error = std::abs(fitted - target);
      worst_error = std::max(worst_error, error);
      squared_total += error * error;
      lowest = seen ? std::min(lowest, target) : target;
      highest = seen ? std::max(highest, target) : target;
      seen = true;
    }
  }
  const double target_span = highest - lowest;

  viz::PlotlyFigure target_figure("Target function", "State x1", "State x2");
  RESULT_ASSERT(
      target_figure
          .AddHeatmap(draw_axis, draw_axis, target_grid, viz::HeatmapStyle{.name = "Target", .value_label = "Value"})
          .has_value(),
      "RunTwoDimensionalFittingExperiment: the target heatmap could not be plotted.");

  viz::PlotlyFigure fitted_figure("Fitted approximation", "State x1", "State x2");
  RESULT_ASSERT(
      fitted_figure
          .AddHeatmap(
              draw_axis, draw_axis, fitted_grid, viz::HeatmapStyle{.name = "Approximation", .value_label = "Value"})
          .has_value(),
      "RunTwoDimensionalFittingExperiment: the approximation heatmap could not be plotted.");

  // Signed, so a diverging scale about a neutral midpoint: over-estimates and under-estimates are
  // opposite things, and a sequential ramp would hide which is which.
  viz::PlotlyFigure error_figure("Signed error, approximation minus target", "State x1", "State x2");
  RESULT_ASSERT(error_figure
                    .AddHeatmap(draw_axis,
                                draw_axis,
                                error_grid,
                                viz::HeatmapStyle{.name = "Error",
                                                  .colorscale_role = viz::PlotColorscaleRole::kDiverging,
                                                  .value_label = "Error",
                                                  .centered_on_zero = true})
                    .has_value(),
                "RunTwoDimensionalFittingExperiment: the error heatmap could not be plotted.");

  constexpr std::array<std::size_t, 1> kAnimatedGrid{0};

  viz::PlotlyFigure approximation_animation("Approximation over training", "State x1", "State x2");
  RESULT_ASSERT(approximation_animation
                    .AddHeatmap(animation_axis,
                                animation_axis,
                                approximation_frames.back().trace_values[0],
                                viz::HeatmapStyle{.name = "Approximation", .value_label = "Value"})
                    .has_value(),
                "RunTwoDimensionalFittingExperiment: the animated approximation heatmap could not be plotted.");
  RESULT_ASSERT(approximation_animation
                    .Animate(kAnimatedGrid,
                             approximation_frames,
                             viz::AnimationStyle{.frame_duration_ms = 140.0, .slider_prefix = "Fitted after "})
                    .has_value(),
                "RunTwoDimensionalFittingExperiment: the approximation animation could not be built.");

  // The one to watch. With the color range pinned across every frame -- and pinned symmetrically,
  // since the scale is centered on zero -- the error map literally drains of color as the fit
  // converges. An unpinned scale would rescale to whatever error remained at each frame and show
  // nothing happening at all.
  viz::PlotlyFigure error_animation("Signed error over training", "State x1", "State x2");
  RESULT_ASSERT(error_animation
                    .AddHeatmap(animation_axis,
                                animation_axis,
                                error_frames.back().trace_values[0],
                                viz::HeatmapStyle{.name = "Error",
                                                  .colorscale_role = viz::PlotColorscaleRole::kDiverging,
                                                  .value_label = "Error",
                                                  .centered_on_zero = true})
                    .has_value(),
                "RunTwoDimensionalFittingExperiment: the animated error heatmap could not be plotted.");
  RESULT_ASSERT(error_animation
                    .Animate(kAnimatedGrid,
                             error_frames,
                             viz::AnimationStyle{.frame_duration_ms = 140.0, .slider_prefix = "Fitted after "})
                    .has_value(),
                "RunTwoDimensionalFittingExperiment: the error animation could not be built.");

  viz::PlotlyFigure residual("Fit residual", "Minibatch step", "RMS residual");
  residual.UseLogarithmicYAxis();
  RESULT_ASSERT(
      AddLogSeries(
          residual,
          SampleHistory(history, [](const StepReport& report) noexcept { return report.root_mean_squared_residual; }),
          "RMS residual over the minibatch",
          kSeriesRoles[0],
          1e-12)
          .has_value(),
      "RunTwoDimensionalFittingExperiment: the residual curve could not be plotted.");

  return SuccessResult(ValueFunctionFittingReport{
      .name = "Fitting a function of two states",
      .description =
          "The same model and the same fitter, over a two-dimensional state: a bowl with a ripple on it, which has "
          "several distinct basins in the plane. This is the case that exercises the parts a one-dimensional problem "
          "cannot reach -- the full state Hessian, and twelve soft-minimum components with somewhere separate to go. "
          "The components are started on two concentric rings, because components that start identical have "
          "identical gradients and would stay identical forever. Press play on the first figure to watch the error "
          "drain out of the plane; its color range is pinned across every frame, so what the color shows is the "
          "error shrinking rather than the scale following it down. The error maps are drawn on a diverging scale "
          "about zero, so over-estimates and under-estimates read as the opposite things they are.",
      .summary = "fitted a two-dimensional multi-basin target, exercising the full state Hessian",
      .figures = {std::move(error_animation),
                  std::move(approximation_animation),
                  std::move(target_figure),
                  std::move(fitted_figure),
                  std::move(error_figure),
                  std::move(residual)},
      .final_root_mean_squared_residual = std::sqrt(squared_total / static_cast<double>(cell_count)),
      .worst_relative_error = worst_error / target_span});
}

}  // namespace fbsde_traj_opt::examples
