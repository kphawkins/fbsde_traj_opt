// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#include "examples/double_inverted_pendulum_experiment.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <memory>
#include <numbers>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "fbsde_traj_opt/composed_cost_sde_model.hpp"
#include "fbsde_traj_opt/composed_forward_sde_model.hpp"
#include "fbsde_traj_opt/const_diagonal_sde_diffusion_term.hpp"
#include "fbsde_traj_opt/diagonal_covariance_normal_distribution.hpp"
#include "fbsde_traj_opt/dt_fbsde_iterative_solver.hpp"
#include "fbsde_traj_opt/l1_control_sde_running_cost_term.hpp"
#include "fbsde_traj_opt/polynomial_value_function_approx.hpp"
#include "fbsde_traj_opt/quadratic_regulator_sde_terminal_cost_term.hpp"
#include "fbsde_traj_opt/taylor_q_function_l1_policy.hpp"
#include "fbsde_traj_opt/trajectory_batch.hpp"
#include "fbsde_traj_opt/utils/parallel_for.hpp"
#include "fbsde_traj_opt/utils/result.hpp"
#include "fbsde_traj_opt/viz/plotly_figure.hpp"
#include "examples/double_inverted_pendulum_model.hpp"

namespace fbsde_traj_opt::examples {
namespace {

// ---------------------------------------------------------------------------------------------
// The problem
// ---------------------------------------------------------------------------------------------

constexpr double kHorizon = 2.5;         // s
constexpr std::size_t kNumStages = 101;  // 100 control steps of 25 ms
constexpr double kTimeStep = kHorizon / static_cast<double>(kNumStages - 1);
constexpr double kFuelWeight = 1.0;        // c0
constexpr std::size_t kNumSamples = 4096;  // the forward batch, paths and probes together
constexpr std::size_t kNumEvaluationSamples = 512;
constexpr std::size_t kNumPlotTrajectories = 128;

// The polynomial degree of the value function representation: 70 coefficients per stage. A
// quadratic cannot bend with the swing; in the tuning runs the quartic ended with the lowest
// cost and the tightest spread over rollouts.
constexpr int kDegree = 4;

// The best open-loop, noise-free solution of the same discrete problem found by bounded
// quasi-Newton optimization of the 100 controls (best of eight random restarts; the restarts
// ranged from 1.97 to 3.40, so the problem has many local optima). Reported beside the method's
// result as a reference point, not a bound: it ignores the noise, which can only add cost.
constexpr double kOpenLoopReferenceCost = 1.97;

using State = DoubleInvertedPendulumDynamics::State;
using Control = DoubleInvertedPendulumDynamics::Control;
using Diffusion = ConstDiagonalSdeDiffusionTerm<4>;
using ForwardModel = ComposedForwardSdeModel<4,
                                             1,
                                             DoubleInvertedPendulumStateDriftTerm,
                                             DoubleInvertedPendulumControlDriftMatTerm,
                                             Diffusion>;
using RunningCost = L1ControlSdeRunningCostTerm<4, 1>;
using TerminalCost = QuadraticRegulatorSdeTerminalCostTerm<4>;
using CostModel = ComposedCostSdeModel<4, 1, RunningCost, TerminalCost>;
using ValueFunction = PolynomialValueFunctionApprox<4, kDegree>;
using Solver = DtFbsdeIterativeSolver<4,
                                      1,
                                      kNumSamples,
                                      kNumStages,
                                      kNumEvaluationSamples,
                                      ForwardModel,
                                      TerminalCost,
                                      ValueFunction>;
using Policy = Solver::Policy;
using PlotBatch = TrajectoryBatch<4, 1, kNumPlotTrajectories, kNumStages>;

// The thesis's diffusion (5.18), in continuous time; the discrete step scales it by sqrt(dt).
const State kContinuousDiffusion(0.03, 0.03, 0.18, 0.18);

// The terminal cost weights c1..c4 of (5.19): the angles matter ten times as much as the rates.
const State kTerminalWeights(10.0, 10.0, 1.0, 1.0);

// The region of interest the value functions are normalized over: the first link anywhere from
// just past upright on one side to well past hanging on the other, and rates up to a few times
// what the swing reaches.
const State kNormalizationCenter(1.5, 0.0, 0.0, 0.0);
const State kNormalizationScale(2.5, 2.0, 8.0, 12.0);

// The initial distribution: hanging at rest, with a spread of well under a degree.
const State kInitialVariance = State::Constant(1e-4);

// The fixed categorical order of the report's hues.
constexpr std::array<viz::PlotColorRole, 4> kSeries{viz::PlotColorRole::kSeries1,
                                                    viz::PlotColorRole::kSeries2,
                                                    viz::PlotColorRole::kSeries3,
                                                    viz::PlotColorRole::kSeries4};

struct Problem {
  DoubleInvertedPendulumDynamics dynamics;
  ForwardModel forward_model;
  CostModel cost_model;
};

auto MakeProblem(const DoubleInvertedPendulumPhysicalParameters& physical) noexcept -> Result<Problem> {
  const Result<DoubleInvertedPendulumDynamics> dynamics = DoubleInvertedPendulumDynamics::Make(physical);
  if (!dynamics.has_value()) {
    return ErrorResult(dynamics.error().message, dynamics.error().location);
  }
  const Result<DoubleInvertedPendulumStateDriftTerm> drift =
      DoubleInvertedPendulumStateDriftTerm::Make(*dynamics, kTimeStep);
  const Result<DoubleInvertedPendulumControlDriftMatTerm> control =
      DoubleInvertedPendulumControlDriftMatTerm::Make(*dynamics, kTimeStep);
  const Result<Diffusion> diffusion = Diffusion::Make(kContinuousDiffusion * std::sqrt(kTimeStep));
  const Result<RunningCost> running_cost = RunningCost::Make(kFuelWeight * kTimeStep);
  RESULT_ASSERT(drift.has_value() && control.has_value() && diffusion.has_value() && running_cost.has_value(),
                "MakeProblem: a term of the pendulum problem could not be built.");
  return SuccessResult(Problem{
      .dynamics = *dynamics,
      .forward_model = ForwardModel(*drift, *control, *diffusion),
      .cost_model = CostModel(*running_cost, TerminalCost(Eigen::Matrix4d(kTerminalWeights.asDiagonal()))),
  });
}

auto StageTime(std::size_t stage) noexcept -> double {
  return static_cast<double>(stage) * kTimeStep;
}

auto TimeAxis(std::size_t count) noexcept -> std::vector<double> {
  std::vector<double> times(count);
  for (std::size_t stage = 0; stage < count; ++stage) {
    times[stage] = StageTime(stage);
  }
  return times;
}

// ---------------------------------------------------------------------------------------------
// Did it invert?
// ---------------------------------------------------------------------------------------------

// The index of the plotted trajectory whose total cost is the median: the representative one to
// animate, neither the luckiest nor the unluckiest.
auto MedianTrajectory(const PlotBatch& batch, const CostModel& cost_model) noexcept -> std::size_t {
  std::vector<std::pair<double, std::size_t>> costs;
  costs.reserve(kNumPlotTrajectories);
  for (std::size_t trajectory = 0; trajectory < kNumPlotTrajectories; ++trajectory) {
    double cost = cost_model.terminal_cost()(batch.StateAt(trajectory, kNumStages - 1));
    for (std::size_t stage = 0; stage + 1 < kNumStages; ++stage) {
      cost += cost_model.running_cost()(stage, batch.StateAt(trajectory, stage), batch.ControlAt(trajectory, stage));
    }
    costs.emplace_back(cost, trajectory);
  }
  const auto middle = costs.begin() + static_cast<std::ptrdiff_t>(costs.size() / 2);
  std::ranges::nth_element(costs, middle);
  return middle->second;
}

// The polyline pivot -> elbow -> tip for one state.
auto LinkagePoints(const DoubleInvertedPendulumDynamics& dynamics, const State& state) noexcept
    -> std::pair<std::vector<double>, std::vector<double>> {
  const DoubleInvertedPendulumDynamics::JointPositions joints = dynamics.Joints(state);
  return {{0.0, joints.elbow.x(), joints.tip.x()}, {0.0, joints.elbow.y(), joints.tip.y()}};
}

auto MakeLinkageAnimation(const DoubleInvertedPendulumDynamics& dynamics,
                          const PlotBatch& batch,
                          std::size_t trajectory) noexcept -> Result<viz::PlotlyFigure> {
  viz::PlotlyFigure figure("The swing-up, animated", "Horizontal position (m)", "Height above the motor (m)");
  const double reach = dynamics.FirstLinkLength() + dynamics.SecondLinkLength();
  RESULT_ASSERT(figure.SetAxisRanges(-1.25 * reach, 1.25 * reach, -1.15 * reach, 1.15 * reach).has_value(),
                "MakeLinkageAnimation: the axis ranges were rejected.");
  figure.UseEqualAspect();

  // The start and the goal, faint, so the reader can see where the links are headed.
  const auto [hanging_x, hanging_y] = LinkagePoints(dynamics, DoubleInvertedPendulumHangingState());
  const auto [upright_x, upright_y] = LinkagePoints(dynamics, State::Zero());
  RESULT_ASSERT(figure
                    .AddLine(hanging_x,
                             hanging_y,
                             viz::LineStyle{.name = "Start: hanging at rest",
                                            .color_role = viz::PlotColorRole::kMuted,
                                            .width = 2.0,
                                            .opacity = 0.5,
                                            .show_on_hover = false})
                    .has_value(),
                "MakeLinkageAnimation: the start pose could not be drawn.");
  RESULT_ASSERT(figure
                    .AddLine(upright_x,
                             upright_y,
                             viz::LineStyle{.name = "Goal: inverted",
                                            .color_role = viz::PlotColorRole::kMuted,
                                            .width = 2.0,
                                            .opacity = 0.5,
                                            .show_on_hover = false})
                    .has_value(),
                "MakeLinkageAnimation: the goal pose could not be drawn.");

  // The path of the tip over the whole horizon, as a trail.
  std::vector<double> tip_x;
  std::vector<double> tip_y;
  for (std::size_t stage = 0; stage < kNumStages; ++stage) {
    const auto joints = dynamics.Joints(batch.StateAt(trajectory, stage));
    tip_x.push_back(joints.tip.x());
    tip_y.push_back(joints.tip.y());
  }
  RESULT_ASSERT(figure
                    .AddLine(tip_x,
                             tip_y,
                             viz::LineStyle{.name = "Path of the tip",
                                            .color_role = kSeries[1],
                                            .width = 1.0,
                                            .opacity = 0.45,
                                            .show_on_hover = false})
                    .has_value(),
                "MakeLinkageAnimation: the tip path could not be drawn.");

  // The linkage itself, one frame every other stage.
  std::vector<viz::AnimationFrame> frames;
  for (std::size_t stage = 0; stage < kNumStages; stage += 2) {
    auto [x_values, y_values] = LinkagePoints(dynamics, batch.StateAt(trajectory, stage));
    frames.push_back(viz::AnimationFrame{.label = std::format("t = {:.2f} s", StageTime(stage)),
                                         .trace_values = {std::move(y_values)},
                                         .trace_x_values = {std::move(x_values)}});
  }
  RESULT_ASSERT(figure
                    .AddLine(frames.back().trace_x_values[0],
                             frames.back().trace_values[0],
                             viz::LineStyle{
                                 .name = "The two links", .color_role = kSeries[0], .width = 5.0, .show_markers = true})
                    .has_value(),
                "MakeLinkageAnimation: the linkage could not be drawn.");
  constexpr std::array<std::size_t, 1> kAnimated{3};
  RESULT_ASSERT(figure.Animate(kAnimated, frames, viz::AnimationStyle{.frame_duration_ms = 50.0, .slider_prefix = ""})
                    .has_value(),
                "MakeLinkageAnimation: the animation could not be built.");
  return SuccessResult(std::move(figure));
}

// A faint cloud of `count` sampled paths of one quantity, with its mean over the top.
auto AddCloud(viz::PlotlyFigure& figure,
              std::span<const double> times,
              const std::vector<std::vector<double>>& paths,
              std::size_t count,
              const std::string& name,
              viz::PlotColorRole role) noexcept -> Result<> {
  std::vector<double> mean(times.size(), 0.0);
  for (const std::vector<double>& path : paths) {
    for (std::size_t index = 0; index < times.size(); ++index) {
      mean[index] += path[index] / static_cast<double>(paths.size());
    }
  }
  for (std::size_t index = 0; index < std::min(count, paths.size()); ++index) {
    RESULT_ASSERT(figure
                      .AddLine(times,
                               paths[index],
                               viz::LineStyle{.name = name + " (rollouts)",
                                              .color_role = role,
                                              .width = 1.0,
                                              .opacity = 0.14,
                                              .show_in_legend = index == 0,
                                              .legend_group = name,
                                              .show_on_hover = false})
                      .has_value(),
                  "AddCloud: a sampled path could not be plotted.");
  }
  RESULT_ASSERT(
      figure.AddLine(times, mean, viz::LineStyle{.name = name + " (mean)", .color_role = role, .legend_group = name})
          .has_value(),
      "AddCloud: the mean path could not be plotted.");
  return SuccessResult();
}

auto StatePaths(const PlotBatch& batch, Eigen::Index component) noexcept -> std::vector<std::vector<double>> {
  std::vector<std::vector<double>> paths(kNumPlotTrajectories, std::vector<double>(kNumStages));
  for (std::size_t trajectory = 0; trajectory < kNumPlotTrajectories; ++trajectory) {
    for (std::size_t stage = 0; stage < kNumStages; ++stage) {
      paths[trajectory][stage] = batch.StateAt(trajectory, stage)[component];
    }
  }
  return paths;
}

auto MakeAngleFigure(const PlotBatch& batch) noexcept -> Result<viz::PlotlyFigure> {
  viz::PlotlyFigure figure("Link angles, measured from inverted", "Time (s)", "Angle (rad)");
  const std::vector<double> times = TimeAxis(kNumStages);
  if (const Result<> added = AddCloud(figure,
                                      times,
                                      StatePaths(batch, DoubleInvertedPendulumDynamics::kAlpha),
                                      48,
                                      "First link, alpha",
                                      kSeries[0]);
      !added.has_value()) {
    return ErrorResult(added.error().message, added.error().location);
  }
  if (const Result<> added = AddCloud(figure,
                                      times,
                                      StatePaths(batch, DoubleInvertedPendulumDynamics::kBeta),
                                      48,
                                      "Second link relative to first, beta",
                                      kSeries[1]);
      !added.has_value()) {
    return ErrorResult(added.error().message, added.error().location);
  }
  const std::vector<double> zero(times.size(), 0.0);
  RESULT_ASSERT(
      figure
          .AddLine(
              times,
              zero,
              viz::LineStyle{
                  .name = "Inverted", .color_role = viz::PlotColorRole::kMuted, .width = 1.5, .show_on_hover = false})
          .has_value(),
      "MakeAngleFigure: the reference line could not be plotted.");
  return SuccessResult(std::move(figure));
}

// ---------------------------------------------------------------------------------------------
// Is the simulation right?
// ---------------------------------------------------------------------------------------------

struct ReferenceComparison {
  viz::PlotlyFigure open_loop;
  viz::PlotlyFigure one_step;
  double max_angle_error = 0.0;
  double max_one_step_angle_error = 0.0;
  double max_one_step_rate_error = 0.0;
};

// Two comparisons of the discrete model with a fine RK4 reference (64 substeps a step) along the
// representative rollout.
//
// The one-step comparison is the measure of the model itself: from every state the rollout
// actually visited, step once under the control it actually applied, and compare. The open-loop
// comparison replays the whole control sequence from the start, with no noise, through both; on a
// chaotic system that also measures how fast small differences grow, which is a property of the
// pendulum rather than of the model, so it is shown as a picture rather than taken as the test.
auto MakeReferenceComparison(const Problem& problem, const PlotBatch& batch, std::size_t trajectory) noexcept
    -> Result<ReferenceComparison> {
  constexpr std::size_t kReferenceSubsteps = 64;
  ReferenceComparison comparison{
      .open_loop = viz::PlotlyFigure(
          "Open-loop replay: the discrete model against a fine RK4 reference", "Time (s)", "Angle (rad)"),
      .one_step = viz::PlotlyFigure(
          "One-step error of the discrete model, from the states the rollout visited", "Time (s)", "Absolute error"),
  };

  State discrete = batch.StateAt(trajectory, 0);
  State reference = discrete;
  std::vector<std::vector<double>> curves(4, std::vector<double>(kNumStages));
  std::vector<double> angle_errors(kNumStages - 1);
  std::vector<double> rate_errors(kNumStages - 1);
  for (std::size_t stage = 0; stage < kNumStages; ++stage) {
    curves[0][stage] = discrete[DoubleInvertedPendulumDynamics::kAlpha];
    curves[1][stage] = reference[DoubleInvertedPendulumDynamics::kAlpha];
    curves[2][stage] = discrete[DoubleInvertedPendulumDynamics::kBeta];
    curves[3][stage] = reference[DoubleInvertedPendulumDynamics::kBeta];
    comparison.max_angle_error =
        std::max(comparison.max_angle_error, (discrete.head<2>() - reference.head<2>()).cwiseAbs().maxCoeff());
    if (stage + 1 == kNumStages) {
      break;
    }
    const Control control = batch.ControlAt(trajectory, stage);
    discrete = problem.forward_model(stage, discrete, control, State::Zero());
    reference = IntegrateReferenceStep(problem.dynamics, reference, control[0], kTimeStep, kReferenceSubsteps);

    const State visited = batch.StateAt(trajectory, stage);
    const State step_error =
        problem.forward_model(stage, visited, control, State::Zero()) -
        IntegrateReferenceStep(problem.dynamics, visited, control[0], kTimeStep, kReferenceSubsteps);
    // Floored for the log axis, which cannot draw an exact zero.
    angle_errors[stage] = std::max(step_error.head<2>().cwiseAbs().maxCoeff(), 1e-12);
    rate_errors[stage] = std::max(step_error.tail<2>().cwiseAbs().maxCoeff(), 1e-12);
  }
  comparison.max_one_step_angle_error = *std::ranges::max_element(angle_errors);
  comparison.max_one_step_rate_error = *std::ranges::max_element(rate_errors);

  const std::vector<double> times = TimeAxis(kNumStages);
  const std::array<std::string, 4> names{
      "alpha, discrete model", "alpha, RK4 reference", "beta, discrete model", "beta, RK4 reference"};
  for (std::size_t curve = 0; curve < curves.size(); ++curve) {
    const bool reference_curve = curve % 2 == 1;
    RESULT_ASSERT(comparison.open_loop
                      .AddLine(times,
                               curves[curve],
                               viz::LineStyle{.name = names[curve],
                                              .color_role = kSeries[curve / 2],
                                              .width = reference_curve ? 4.0 : 1.5,
                                              .opacity = reference_curve ? 0.35 : 1.0})
                      .has_value(),
                  "MakeReferenceComparison: a curve could not be plotted.");
  }

  // Against the yardstick that matters: the standard deviation of the noise the SDE injects at
  // every step anyway. A model error well under it is invisible to the method.
  const std::vector<double> step_times = TimeAxis(kNumStages - 1);
  comparison.one_step.UseLogarithmicYAxis();
  const double angle_noise = kContinuousDiffusion[0] * std::sqrt(kTimeStep);
  const double rate_noise = kContinuousDiffusion[2] * std::sqrt(kTimeStep);
  RESULT_ASSERT(
      comparison.one_step
              .AddLine(step_times, angle_errors, viz::LineStyle{.name = "Angles (rad)", .color_role = kSeries[0]})
              .has_value() &&
          comparison.one_step
              .AddLine(step_times, rate_errors, viz::LineStyle{.name = "Rates (rad/s)", .color_role = kSeries[1]})
              .has_value() &&
          comparison.one_step
              .AddLine(step_times,
                       std::vector<double>(step_times.size(), angle_noise),
                       viz::LineStyle{.name = "Per-step noise, angles (rad)",
                                      .color_role = kSeries[0],
                                      .width = 1.0,
                                      .opacity = 0.5,
                                      .show_on_hover = false})
              .has_value() &&
          comparison.one_step
              .AddLine(step_times,
                       std::vector<double>(step_times.size(), rate_noise),
                       viz::LineStyle{.name = "Per-step noise, rates (rad/s)",
                                      .color_role = kSeries[1],
                                      .width = 1.0,
                                      .opacity = 0.5,
                                      .show_on_hover = false})
              .has_value(),
      "MakeReferenceComparison: the one-step error could not be plotted.");
  return SuccessResult(std::move(comparison));
}

struct EnergyCheck {
  viz::PlotlyFigure energy;
  viz::PlotlyFigure convergence;
  double max_relative_energy_error = 0.0;
};

// Energy of the unforced, frictionless pendulum released from horizontal, under the discrete step
// and under the reference; and the discrete step's error against the reference as the step
// shrinks.
auto MakeEnergyCheck(const DoubleInvertedPendulumPhysicalParameters& physical) noexcept -> Result<EnergyCheck> {
  DoubleInvertedPendulumPhysicalParameters frictionless = physical;
  frictionless.first_joint_friction = 0.0;
  frictionless.second_joint_friction = 0.0;
  const Result<DoubleInvertedPendulumDynamics> dynamics = DoubleInvertedPendulumDynamics::Make(frictionless);
  if (!dynamics.has_value()) {
    return ErrorResult(dynamics.error().message, dynamics.error().location);
  }
  const State released(std::numbers::pi / 2.0, 0.3, 0.0, 0.0);

  const auto discrete_step = [&dynamics](const State& state, double time_step) noexcept -> State {
    const Result<DoubleInvertedPendulumStateDriftTerm> drift =
        DoubleInvertedPendulumStateDriftTerm::Make(*dynamics, time_step);
    return state + (*drift)(0, state);
  };

  // Energy over the experiment's horizon at the experiment's step.
  std::vector<std::vector<double>> energies(2, std::vector<double>(kNumStages));
  State discrete = released;
  State reference = released;
  const double initial_energy = dynamics->TotalEnergy(released);
  const double energy_scale = std::abs(initial_energy) + dynamics->Constants().f1 + dynamics->Constants().f2;
  double max_error = 0.0;
  for (std::size_t stage = 0; stage < kNumStages; ++stage) {
    energies[0][stage] = dynamics->TotalEnergy(discrete);
    energies[1][stage] = dynamics->TotalEnergy(reference);
    max_error = std::max(max_error, std::abs(energies[0][stage] - initial_energy) / energy_scale);
    discrete = discrete_step(discrete, kTimeStep);
    reference = IntegrateReferenceStep(*dynamics, reference, 0.0, kTimeStep, 64);
  }
  viz::PlotlyFigure energy("Energy of the unforced, frictionless pendulum", "Time (s)", "Total energy (J)");
  const std::vector<double> times = TimeAxis(kNumStages);
  RESULT_ASSERT(
      energy
          .AddLine(times,
                   energies[0],
                   viz::LineStyle{.name = "Discrete model (25 ms zero-order-hold step)", .color_role = kSeries[0]})
          .has_value(),
      "MakeEnergyCheck: the discrete energy could not be plotted.");
  RESULT_ASSERT(
      energy
          .AddLine(times,
                   energies[1],
                   viz::LineStyle{.name = "RK4 reference", .color_role = kSeries[1], .width = 4.0, .opacity = 0.35})
          .has_value(),
      "MakeEnergyCheck: the reference energy could not be plotted.");

  // Error at a fixed time as the step halves: fourth order -- the error falling sixteenfold per
  // halving -- is the signature of a correctly implemented RK4 step, and anything else the
  // signature of a bug. Floored where it reaches rounding, which a log axis cannot draw below.
  constexpr double kCheckTime = 0.5;
  const State truth = IntegrateReferenceStep(*dynamics, released, 0.0, kCheckTime, 20000);
  std::vector<double> steps;
  std::vector<double> errors;
  std::vector<double> fourth_order;
  for (std::size_t count = 5; count <= 320; count *= 2) {
    const double time_step = kCheckTime / static_cast<double>(count);
    State state = released;
    for (std::size_t step = 0; step < count; ++step) {
      state = discrete_step(state, time_step);
    }
    steps.push_back(time_step);
    errors.push_back(std::max((state - truth).norm(), 1e-14));
  }
  fourth_order.reserve(steps.size());
  for (const double time_step : steps) {
    fourth_order.push_back(errors.front() * std::pow(time_step / steps.front(), 4.0));
  }
  viz::PlotlyFigure convergence(
      "Discretization error against the reference, as the step shrinks", "Time step (s)", "State error at t = 0.5 s");
  convergence.UseLogarithmicYAxis();
  RESULT_ASSERT(
      convergence
          .AddLine(
              steps, errors, viz::LineStyle{.name = "Discrete model", .color_role = kSeries[0], .show_markers = true})
          .has_value(),
      "MakeEnergyCheck: the convergence curve could not be plotted.");
  RESULT_ASSERT(convergence
                    .AddLine(steps,
                             fourth_order,
                             viz::LineStyle{.name = "Slope of fourth order",
                                            .color_role = viz::PlotColorRole::kMuted,
                                            .width = 1.5,
                                            .show_on_hover = false})
                    .has_value(),
                "MakeEnergyCheck: the reference slope could not be plotted.");

  return SuccessResult(EnergyCheck{
      .energy = std::move(energy), .convergence = std::move(convergence), .max_relative_energy_error = max_error});
}

// ---------------------------------------------------------------------------------------------
// Was the L1 cost minimized?
// ---------------------------------------------------------------------------------------------

auto ControlPaths(const PlotBatch& batch) noexcept -> std::vector<std::vector<double>> {
  std::vector<std::vector<double>> paths(kNumPlotTrajectories, std::vector<double>(kNumStages - 1));
  for (std::size_t trajectory = 0; trajectory < kNumPlotTrajectories; ++trajectory) {
    for (std::size_t stage = 0; stage + 1 < kNumStages; ++stage) {
      paths[trajectory][stage] = batch.ControlAt(trajectory, stage)[0];
    }
  }
  return paths;
}

auto MakeControlFigure(const PlotBatch& batch, std::size_t representative) noexcept -> Result<viz::PlotlyFigure> {
  viz::PlotlyFigure figure("The control signal", "Time (s)", "Control u (fraction of the 5 N m limit)");
  const std::vector<double> times = TimeAxis(kNumStages - 1);
  const std::vector<std::vector<double>> paths = ControlPaths(batch);
  if (const Result<> added = AddCloud(figure, times, paths, 24, "u", kSeries[0]); !added.has_value()) {
    return ErrorResult(added.error().message, added.error().location);
  }
  RESULT_ASSERT(
      figure
          .AddLine(times,
                   paths[representative],
                   viz::LineStyle{.name = "u of the animated rollout", .color_role = kSeries[1], .width = 1.5})
          .has_value(),
      "MakeControlFigure: the representative control could not be plotted.");
  return SuccessResult(std::move(figure));
}

// Which of the four levels a control sits at: 0 for +1, 1 for -1, 2 for coasting at zero, and 3
// for anything in between.
auto ControlLevel(double control) noexcept -> std::size_t {
  constexpr double kTolerance = 1e-9;
  if (control >= 1.0 - kTolerance) {
    return 0;
  }
  if (control <= -1.0 + kTolerance) {
    return 1;
  }
  if (std::abs(control) <= kTolerance) {
    return 2;
  }
  return 3;
}

auto MakeControlShareFigure(const PlotBatch& batch) noexcept -> Result<viz::PlotlyFigure> {
  viz::PlotlyFigure figure("How often the control is at each level", "Time (s)", "Share of rollouts");
  const std::vector<double> times = TimeAxis(kNumStages - 1);
  std::array<std::vector<double>, 4> shares;
  for (std::vector<double>& share : shares) {
    share.assign(kNumStages - 1, 0.0);
  }
  const double increment = 1.0 / static_cast<double>(kNumPlotTrajectories);
  for (std::size_t stage = 0; stage + 1 < kNumStages; ++stage) {
    for (std::size_t trajectory = 0; trajectory < kNumPlotTrajectories; ++trajectory) {
      const double control = batch.ControlAt(trajectory, stage)[0];
      shares[ControlLevel(control)][stage] += increment;
    }
  }
  const std::array<std::string, 4> names{"Full torque, +1", "Full torque, -1", "Coasting, 0", "In between"};
  for (std::size_t level = 0; level < shares.size(); ++level) {
    RESULT_ASSERT(
        figure.AddLine(times, shares[level], viz::LineStyle{.name = names[level], .color_role = kSeries[level]})
            .has_value(),
        "MakeControlShareFigure: a share curve could not be plotted.");
  }
  return SuccessResult(std::move(figure));
}

auto MakeFuelFigure(const PlotBatch& batch) noexcept -> Result<viz::PlotlyFigure> {
  viz::PlotlyFigure figure("Fuel spent: the L1 running cost, accumulated", "Time (s)", "Accumulated c0 |u| dt");
  const std::vector<double> times = TimeAxis(kNumStages);
  std::vector<double> spent(kNumStages, 0.0);
  std::vector<double> saturated(kNumStages, 0.0);
  for (std::size_t stage = 1; stage < kNumStages; ++stage) {
    double step = 0.0;
    for (std::size_t trajectory = 0; trajectory < kNumPlotTrajectories; ++trajectory) {
      step += kFuelWeight * kTimeStep * std::abs(batch.ControlAt(trajectory, stage - 1)[0]);
    }
    spent[stage] = spent[stage - 1] + (step / static_cast<double>(kNumPlotTrajectories));
    saturated[stage] = kFuelWeight * StageTime(stage);
  }
  RESULT_ASSERT(
      figure.AddLine(times, spent, viz::LineStyle{.name = "The policy (mean)", .color_role = kSeries[0]}).has_value(),
      "MakeFuelFigure: the fuel curve could not be plotted.");
  RESULT_ASSERT(
      figure
          .AddLine(
              times,
              saturated,
              viz::LineStyle{.name = "Full torque throughout", .color_role = viz::PlotColorRole::kMuted, .width = 1.5})
          .has_value(),
      "MakeFuelFigure: the reference curve could not be plotted.");
  return SuccessResult(std::move(figure));
}

// The policy as a map: u over a window of (alpha, omega) around the mean path, with the second
// link's angle and rate held at the mean path's, animated over the horizon.
auto MakePolicyMap(const Policy& policy, const PlotBatch& batch) noexcept -> Result<viz::PlotlyFigure> {
  constexpr std::size_t kGrid = 41;
  constexpr double kAngleReach = 1.0;
  constexpr double kRateReach = 4.0;
  std::vector<double> angle_offsets(kGrid);
  std::vector<double> rate_offsets(kGrid);
  for (std::size_t index = 0; index < kGrid; ++index) {
    const double fraction = (2.0 * static_cast<double>(index) / static_cast<double>(kGrid - 1)) - 1.0;
    angle_offsets[index] = kAngleReach * fraction;
    rate_offsets[index] = kRateReach * fraction;
  }

  std::vector<viz::AnimationFrame> frames;
  for (std::size_t stage = 0; stage + 1 < kNumStages; stage += 4) {
    const State mean = batch.MeanStateAtStage(stage);
    std::vector<double> controls;
    controls.reserve(kGrid * kGrid);
    for (const double rate_offset : rate_offsets) {
      for (const double angle_offset : angle_offsets) {
        State state = mean;
        state[DoubleInvertedPendulumDynamics::kAlpha] += angle_offset;
        state[DoubleInvertedPendulumDynamics::kOmega] += rate_offset;
        controls.push_back(policy(stage, state)[0]);
      }
    }
    frames.push_back(viz::AnimationFrame{.label = std::format("t = {:.2f} s", StageTime(stage)),
                                         .trace_values = {std::move(controls)}});
  }

  viz::PlotlyFigure figure("The policy around the mean path",
                           "First link angle, offset from the mean path (rad)",
                           "First link rate, offset from the mean path (rad/s)");
  RESULT_ASSERT(figure
                    .AddHeatmap(angle_offsets,
                                rate_offsets,
                                frames.front().trace_values[0],
                                viz::HeatmapStyle{.name = "u",
                                                  .colorscale_role = viz::PlotColorscaleRole::kDiverging,
                                                  .value_label = "Control u",
                                                  .centered_on_zero = true})
                    .has_value(),
                "MakePolicyMap: the heatmap could not be built.");
  constexpr std::array<std::size_t, 1> kAnimated{0};
  RESULT_ASSERT(figure.Animate(kAnimated, frames, viz::AnimationStyle{.frame_duration_ms = 200.0, .slider_prefix = ""})
                    .has_value(),
                "MakePolicyMap: the animation could not be built.");
  return SuccessResult(std::move(figure));
}

// ---------------------------------------------------------------------------------------------
// Convergence
// ---------------------------------------------------------------------------------------------

// What one iteration's policy does, recorded right after the iteration while the solver's table
// still holds it: the policy reads the live table, so it cannot be kept and asked later.
struct IterationSnapshot {
  // The nominal rollout: the policy from hanging at rest with the noise switched off.
  std::vector<State> states{};
  std::vector<double> controls{};

  // The policy over time and the first link's angle, around the nominal rollout (see
  // MakePolicySliceAnimation), row-major over kSliceOffsets by kSliceStages.
  std::vector<double> policy_slice{};

  // The value function's own estimate of the expected cost from the start, V~_0(x0).
  double predicted_cost = 0.0;
};

// The policy slice's grid: every other control stage, and the first link's angle within a radian
// of the nominal rollout's.
constexpr std::size_t kSliceStageStride = 2;
constexpr std::size_t kSliceStages = (kNumStages - 1) / kSliceStageStride;
constexpr std::size_t kSliceOffsets = 25;
constexpr double kSliceAngleReach = 1.0;

auto SliceAngleOffset(std::size_t row) noexcept -> double {
  return kSliceAngleReach * ((2.0 * static_cast<double>(row) / static_cast<double>(kSliceOffsets - 1)) - 1.0);
}

auto SnapshotIteration(const Problem& problem,
                       const Policy& policy,
                       const ValueFunction& initial_value_function) noexcept -> IterationSnapshot {
  IterationSnapshot snapshot;
  const State start = DoubleInvertedPendulumHangingState();
  snapshot.predicted_cost = initial_value_function(start);
  snapshot.states.reserve(kNumStages);
  snapshot.controls.reserve(kNumStages - 1);
  snapshot.states.push_back(start);
  for (std::size_t stage = 0; stage + 1 < kNumStages; ++stage) {
    const Control control = policy(stage, snapshot.states.back());
    snapshot.controls.push_back(control[0]);
    snapshot.states.push_back(problem.forward_model(stage, snapshot.states.back(), control, State::Zero()));
  }

  snapshot.policy_slice.assign(kSliceOffsets * kSliceStages, 0.0);
  ParallelFor(snapshot.policy_slice.size(), [&snapshot, &policy](std::size_t cell) noexcept {
    const std::size_t row = cell / kSliceStages;
    const std::size_t stage = (cell % kSliceStages) * kSliceStageStride;
    State state = snapshot.states[stage];
    state[DoubleInvertedPendulumDynamics::kAlpha] += SliceAngleOffset(row);
    snapshot.policy_slice[cell] = policy(stage, state)[0];
  });
  return snapshot;
}

auto IterationLabel(const DtFbsdeIterationReport<double>& report) noexcept -> std::string {
  return std::format("iteration {}, mean cost {:.3f}", report.iteration + 1, report.mean_cost);
}

// The expected cost as the iterations go: the policy's mean cost on the evaluation rollouts, its
// two parts, and the value function's own prediction of it -- the two agreeing is the backward
// pass's fit being consistent with what the policy actually does.
auto MakeConvergenceFigure(const std::vector<DtFbsdeIterationReport<double>>& iterations,
                           const std::vector<IterationSnapshot>& snapshots) noexcept -> Result<viz::PlotlyFigure> {
  viz::PlotlyFigure figure("Expected cost over the iterations", "Iteration", "Expected cost");
  figure.UseLogarithmicYAxis();
  std::vector<double> axis;
  std::array<std::vector<double>, 5> series;
  for (std::size_t index = 0; index < iterations.size(); ++index) {
    const DtFbsdeIterationReport<double>& report = iterations[index];
    axis.push_back(static_cast<double>(report.iteration + 1));
    series[0].push_back(report.mean_cost);
    series[1].push_back(report.mean_terminal_cost);
    series[2].push_back(report.mean_running_cost);
    // Floored for the log axis: early fits can predict nonsense, even below zero.
    series[3].push_back(std::max(snapshots[index].predicted_cost, 1e-3));
    series[4].push_back(report.sampled_mean_cost);
  }
  const std::array<viz::LineStyle, 5> styles{
      viz::LineStyle{.name = "Mean cost of the policy (evaluation rollouts)", .color_role = kSeries[0]},
      viz::LineStyle{.name = "Terminal cost part", .color_role = kSeries[1], .width = 1.5},
      viz::LineStyle{.name = "Fuel part (L1 running cost)", .color_role = kSeries[2], .width = 1.5},
      viz::LineStyle{.name = "Value function's prediction, V~_0(x0)", .color_role = kSeries[3]},
      viz::LineStyle{.name = "Forward-pass samples, exploration included",
                     .color_role = viz::PlotColorRole::kMuted,
                     .width = 1.5,
                     .opacity = 0.6},
  };
  for (std::size_t index = 0; index < series.size(); ++index) {
    RESULT_ASSERT(figure.AddLine(axis, series[index], styles[index]).has_value(),
                  "MakeConvergenceFigure: a curve could not be plotted.");
  }
  return SuccessResult(std::move(figure));
}

// Each iteration's nominal rollout -- both link angles and the control -- animated over the
// iterations, so the swing-up can be watched emerging.
auto MakeNominalRolloutAnimation(const std::vector<DtFbsdeIterationReport<double>>& iterations,
                                 const std::vector<IterationSnapshot>& snapshots) noexcept
    -> Result<viz::PlotlyFigure> {
  viz::PlotlyFigure figure("The policy at each iteration: its noise-free rollout from hanging",
                           "Time (s)",
                           "Angle (rad) and control u (fraction of the limit)");
  const std::vector<double> times = TimeAxis(kNumStages);
  const std::vector<double> control_times = TimeAxis(kNumStages - 1);

  std::vector<viz::AnimationFrame> frames;
  frames.reserve(snapshots.size());
  for (std::size_t index = 0; index < snapshots.size(); ++index) {
    std::vector<double> alpha;
    std::vector<double> beta;
    for (const State& state : snapshots[index].states) {
      alpha.push_back(state[DoubleInvertedPendulumDynamics::kAlpha]);
      beta.push_back(state[DoubleInvertedPendulumDynamics::kBeta]);
    }
    frames.push_back(
        viz::AnimationFrame{.label = IterationLabel(iterations[index]),
                            .trace_values = {std::move(alpha), std::move(beta), snapshots[index].controls}});
  }

  RESULT_ASSERT(
      figure
          .AddLine(
              times,
              std::vector<double>(times.size(), 0.0),
              viz::LineStyle{
                  .name = "Inverted", .color_role = viz::PlotColorRole::kMuted, .width = 1.5, .show_on_hover = false})
          .has_value(),
      "MakeNominalRolloutAnimation: the reference line could not be plotted.");
  const viz::AnimationFrame& last = frames.back();
  RESULT_ASSERT(
      figure.AddLine(times, last.trace_values[0], viz::LineStyle{.name = "First link, alpha", .color_role = kSeries[0]})
              .has_value() &&
          figure
              .AddLine(times,
                       last.trace_values[1],
                       viz::LineStyle{.name = "Second link relative to first, beta", .color_role = kSeries[1]})
              .has_value() &&
          figure
              .AddLine(control_times,
                       last.trace_values[2],
                       viz::LineStyle{.name = "Control u", .color_role = kSeries[2], .width = 1.5})
              .has_value(),
      "MakeNominalRolloutAnimation: a curve could not be plotted.");
  constexpr std::array<std::size_t, 3> kAnimated{1, 2, 3};
  RESULT_ASSERT(figure.Animate(kAnimated, frames, viz::AnimationStyle{.frame_duration_ms = 120.0, .slider_prefix = ""})
                    .has_value(),
                "MakeNominalRolloutAnimation: the animation could not be built.");
  return SuccessResult(std::move(figure));
}

// Each iteration's policy over time and the first link's angle, the other three coordinates held at
// the nominal rollout's: the band the rollout runs through, animated over the iterations.
auto MakePolicySliceAnimation(const std::vector<DtFbsdeIterationReport<double>>& iterations,
                              const std::vector<IterationSnapshot>& snapshots) noexcept -> Result<viz::PlotlyFigure> {
  std::vector<double> stage_times(kSliceStages);
  for (std::size_t column = 0; column < kSliceStages; ++column) {
    stage_times[column] = StageTime(column * kSliceStageStride);
  }
  std::vector<double> offsets(kSliceOffsets);
  for (std::size_t row = 0; row < kSliceOffsets; ++row) {
    offsets[row] = SliceAngleOffset(row);
  }

  std::vector<viz::AnimationFrame> frames;
  frames.reserve(snapshots.size());
  for (std::size_t index = 0; index < snapshots.size(); ++index) {
    frames.push_back(viz::AnimationFrame{.label = IterationLabel(iterations[index]),
                                         .trace_values = {snapshots[index].policy_slice}});
  }

  viz::PlotlyFigure figure("The policy at each iteration, around its noise-free rollout",
                           "Time (s)",
                           "First link angle, offset from the rollout (rad)");
  RESULT_ASSERT(figure
                    .AddHeatmap(stage_times,
                                offsets,
                                frames.back().trace_values[0],
                                viz::HeatmapStyle{.name = "u",
                                                  .colorscale_role = viz::PlotColorscaleRole::kDiverging,
                                                  .value_label = "Control u",
                                                  .centered_on_zero = true})
                    .has_value(),
                "MakePolicySliceAnimation: the heatmap could not be built.");
  constexpr std::array<std::size_t, 1> kAnimated{0};
  RESULT_ASSERT(figure.Animate(kAnimated, frames, viz::AnimationStyle{.frame_duration_ms = 120.0, .slider_prefix = ""})
                    .has_value(),
                "MakePolicySliceAnimation: the animation could not be built.");
  return SuccessResult(std::move(figure));
}

// Every iteration's nominal control at once: one row per iteration, so the bang-off-bang pattern
// can be seen settling.
auto MakeControlHistoryFigure(const std::vector<IterationSnapshot>& snapshots) noexcept -> Result<viz::PlotlyFigure> {
  viz::PlotlyFigure figure("The noise-free rollout's control, iteration by iteration", "Time (s)", "Iteration");
  const std::vector<double> times = TimeAxis(kNumStages - 1);
  std::vector<double> rows;
  std::vector<double> controls;
  controls.reserve(snapshots.size() * times.size());
  for (std::size_t index = 0; index < snapshots.size(); ++index) {
    rows.push_back(static_cast<double>(index + 1));
    controls.insert(controls.end(), snapshots[index].controls.begin(), snapshots[index].controls.end());
  }
  RESULT_ASSERT(figure
                    .AddHeatmap(times,
                                rows,
                                controls,
                                viz::HeatmapStyle{.name = "u",
                                                  .colorscale_role = viz::PlotColorscaleRole::kDiverging,
                                                  .value_label = "Control u",
                                                  .centered_on_zero = true})
                    .has_value(),
                "MakeControlHistoryFigure: the heatmap could not be built.");
  return SuccessResult(std::move(figure));
}

}  // namespace

auto RunDoubleInvertedPendulumExperiment(const DoubleInvertedPendulumExperimentOptions& options,
                                         const DoubleInvertedPendulumProgress& progress) noexcept
    -> Result<DoubleInvertedPendulumReport> {
  const DoubleInvertedPendulumPhysicalParameters physical;
  const Result<Problem> problem = MakeProblem(physical);
  if (!problem.has_value()) {
    return ErrorResult(problem.error().message, problem.error().location);
  }

  const Result<DiagonalCovarianceNormalDistribution<4>> initial_distribution =
      DiagonalCovarianceNormalDistribution<4>::Make(DoubleInvertedPendulumHangingState(), kInitialVariance);
  const Result<ValueFunction> blank =
      ValueFunction::Make(ValueFunction::ParameterVector::Zero(), kNormalizationCenter, kNormalizationScale);
  RESULT_ASSERT(initial_distribution.has_value() && blank.has_value(),
                "RunDoubleInvertedPendulumExperiment: the initial distribution or value function could not be built.");
  ValueFunction terminal_value_function = *blank;
  if (const Result<> set = terminal_value_function.SetParameters(
          blank->QuadraticParameters(Eigen::Matrix4d(kTerminalWeights.asDiagonal()), 0.0));
      !set.has_value()) {
    return ErrorResult(set.error().message, set.error().location);
  }

  DtFbsdeIterativeOptions<double> solver_options;
  solver_options.backward_target = DtFbsdeBackwardTarget::kDoubleGreedy;
  solver_options.exploration_probability = 0.03;
  solver_options.initial_exploration_probability = 1.0;
  solver_options.exploration_decay = 0.8;
  solver_options.probe_fraction = 0.25;
  solver_options.target_floor = 0.0;
  solver_options.entropy_temperature = 2.0;

  // The solver's tables and batch run to tens of megabytes; it lives on the heap.
  auto solver = std::make_unique<Result<Solver>>(Solver::Make(problem->forward_model,
                                                              problem->cost_model.running_cost(),
                                                              problem->cost_model.terminal_cost(),
                                                              *initial_distribution,
                                                              Control(-1.0),
                                                              Control(1.0),
                                                              terminal_value_function,
                                                              terminal_value_function,
                                                              solver_options,
                                                              options.seed));
  if (!solver->has_value()) {
    return ErrorResult(solver->error().message, solver->error().location);
  }

  DoubleInvertedPendulumReport report;
  std::vector<IterationSnapshot> snapshots;
  snapshots.reserve(options.iterations);
  for (std::size_t iteration = 0; iteration < options.iterations; ++iteration) {
    const Result<DtFbsdeIterationReport<double>> step = (*solver)->Iterate();
    if (!step.has_value()) {
      return ErrorResult(step.error().message, step.error().location);
    }
    report.iterations.push_back(*step);
    snapshots.push_back(SnapshotIteration(*problem, (*solver)->CurrentPolicy(), (*solver)->ValueFunctions()[0]));
    report.predicted_costs.push_back(snapshots.back().predicted_cost);
    if (progress) {
      progress(*step);
    }
  }

  // The final policy on fresh noise the method never saw.
  const Policy policy = (*solver)->CurrentPolicy();
  const auto batch = std::make_unique<Result<PlotBatch>>(
      PlotBatch::Make(problem->forward_model, policy, *initial_distribution, options.seed ^ 0xA5A5A5A5ULL));
  if (!batch->has_value()) {
    return ErrorResult(batch->error().message, batch->error().location);
  }
  const PlotBatch& rollouts = **batch;

  report.final_mean_cost = rollouts.ExpectedCostToGo(problem->cost_model)[0];
  double terminal = 0.0;
  std::size_t inverted = 0;
  constexpr double kInvertedTolerance = 20.0 * std::numbers::pi / 180.0;
  for (std::size_t trajectory = 0; trajectory < kNumPlotTrajectories; ++trajectory) {
    const State final_state = rollouts.StateAt(trajectory, kNumStages - 1);
    terminal += problem->cost_model.terminal_cost()(final_state);
    report.terminal_mean += final_state / static_cast<double>(kNumPlotTrajectories);
    if (final_state.head<2>().cwiseAbs().maxCoeff() <= kInvertedTolerance) {
      ++inverted;
    }
  }
  report.final_mean_terminal_cost = terminal / static_cast<double>(kNumPlotTrajectories);
  report.final_mean_fuel = report.final_mean_cost - report.final_mean_terminal_cost;
  report.fraction_inverted = static_cast<double>(inverted) / static_cast<double>(kNumPlotTrajectories);

  const std::size_t representative = MedianTrajectory(rollouts, problem->cost_model);

  Result<viz::PlotlyFigure> linkage = MakeLinkageAnimation(problem->dynamics, rollouts, representative);
  Result<viz::PlotlyFigure> angles = MakeAngleFigure(rollouts);
  Result<viz::PlotlyFigure> control = MakeControlFigure(rollouts, representative);
  Result<viz::PlotlyFigure> shares = MakeControlShareFigure(rollouts);
  Result<viz::PlotlyFigure> fuel = MakeFuelFigure(rollouts);
  Result<viz::PlotlyFigure> policy_map = MakePolicyMap(policy, rollouts);
  Result<viz::PlotlyFigure> convergence = MakeConvergenceFigure(report.iterations, snapshots);
  Result<viz::PlotlyFigure> nominal = MakeNominalRolloutAnimation(report.iterations, snapshots);
  Result<viz::PlotlyFigure> slice = MakePolicySliceAnimation(report.iterations, snapshots);
  Result<viz::PlotlyFigure> history = MakeControlHistoryFigure(snapshots);
  Result<ReferenceComparison> reference = MakeReferenceComparison(*problem, rollouts, representative);
  Result<EnergyCheck> energy = MakeEnergyCheck(physical);
  for (const Result<viz::PlotlyFigure>* figure :
       {&linkage, &angles, &control, &shares, &fuel, &policy_map, &convergence, &nominal, &slice, &history}) {
    if (!figure->has_value()) {
      return ErrorResult(figure->error().message, figure->error().location);
    }
  }
  if (!reference.has_value()) {
    return ErrorResult(reference.error().message, reference.error().location);
  }
  if (!energy.has_value()) {
    return ErrorResult(energy.error().message, energy.error().location);
  }
  report.max_reference_angle_error = reference->max_angle_error;
  report.max_one_step_angle_error = reference->max_one_step_angle_error;
  report.max_one_step_rate_error = reference->max_one_step_rate_error;
  report.max_relative_energy_error = energy->max_relative_energy_error;

  report.name = "L1 double inverted pendulum swing-up by DT-FBSDE";
  report.description = std::format(
      "Two 1 kg, 0.5 m links, a 5 N m motor at the base joint only, starting from hanging at rest, with the thesis's "
      "minimum-fuel objective: c0 |u| per second plus a quadratic terminal cost at inverted, over {:.1f} s in {} "
      "steps. Solved by the DT-FBSDE iterative method -- {} parallel samples a pass, a quartic value function fitted "
      "at every stage, the policy improved through the Taylor Q-function -- over {} iterations. The first figure plays "
      "the median rollout of the final policy; the next group shows how the expected cost and the policy evolved over "
      "the iterations -- scrub the sliders to see each iteration's policy; the next checks the simulation against a "
      "Runge-Kutta reference "
      "and energy conservation; the last group shows the control, which an L1-optimal policy keeps at -1, 0 or +1. "
      "On {} fresh rollouts: mean cost {:.3f} (fuel {:.3f}, terminal {:.3f}); {:.0f}% end with both links within 20 "
      "degrees of inverted. For reference, the best noise-free open-loop control sequence found by direct optimization "
      "costs {:.2f}.",
      kHorizon,
      kNumStages - 1,
      kNumSamples,
      options.iterations,
      kNumPlotTrajectories,
      report.final_mean_cost,
      report.final_mean_fuel,
      report.final_mean_terminal_cost,
      100.0 * report.fraction_inverted,
      kOpenLoopReferenceCost);
  report.summary = std::format(
      "final policy: mean cost {:.3f} = fuel {:.3f} + terminal {:.3f} (open-loop noise-free reference {:.2f}); {:.0f}% "
      "of rollouts end within 20 deg of inverted\n"
      "  mean terminal state [alpha beta omega psi] = [{:.3f} {:.3f} {:.3f} {:.3f}]\n"
      "  simulation: one-step error against RK4 at most {:.1e} rad and {:.1e} rad/s (per-step noise {:.1e} and "
      "{:.1e}); frictionless energy drift {:.1e} of the energy scale; open-loop replay of the whole horizon drifts "
      "{:.2f} rad, the pendulum's chaos amplifying the one-step error",
      report.final_mean_cost,
      report.final_mean_fuel,
      report.final_mean_terminal_cost,
      kOpenLoopReferenceCost,
      100.0 * report.fraction_inverted,
      report.terminal_mean[0],
      report.terminal_mean[1],
      report.terminal_mean[2],
      report.terminal_mean[3],
      report.max_one_step_angle_error,
      report.max_one_step_rate_error,
      kContinuousDiffusion[0] * std::sqrt(kTimeStep),
      kContinuousDiffusion[2] * std::sqrt(kTimeStep),
      report.max_relative_energy_error,
      report.max_reference_angle_error);

  report.figures.push_back(std::move(*linkage));
  report.figures.push_back(std::move(*angles));
  report.figures.push_back(std::move(*convergence));
  report.figures.push_back(std::move(*nominal));
  report.figures.push_back(std::move(*slice));
  report.figures.push_back(std::move(*history));
  report.figures.push_back(std::move(reference->one_step));
  report.figures.push_back(std::move(reference->open_loop));
  report.figures.push_back(std::move(energy->energy));
  report.figures.push_back(std::move(energy->convergence));
  report.figures.push_back(std::move(*control));
  report.figures.push_back(std::move(*shares));
  report.figures.push_back(std::move(*fuel));
  report.figures.push_back(std::move(*policy_map));
  return SuccessResult(std::move(report));
}

}  // namespace fbsde_traj_opt::examples
