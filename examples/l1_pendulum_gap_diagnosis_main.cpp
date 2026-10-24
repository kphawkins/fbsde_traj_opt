// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

// Why DT-FBSDE stops short of the optimum on the L1 inverted pendulum: separates what the value
// function representation can express from what the method learns.
//
//   bazel run -c opt //examples:l1_pendulum_gap_diagnosis
//
// 1. The representation's ceiling. The soft-min model is fitted directly to the grid optimum's
//    value function, stage by stage, on states from the optimal policy's rollouts -- the best fit
//    the method could hope to learn -- and the Taylor Q-function policy over that fit is scored.
//    If it falls short with K components, no sampling strategy closes the gap with K components.
// 2. What the method learns. DT-FBSDE runs in the best configuration found so far, and its policy
//    is compared with the optimum: cost split, the noise-free control sequence, the final state,
//    and the learned value function against the optimal one along the learned trajectory.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <format>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include "fbsde_traj_opt/policies/taylor_q_function_l1_policy.hpp"
#include "fbsde_traj_opt/sampling/counter_based_normal_sampler.hpp"
#include "fbsde_traj_opt/solvers/dt_fbsde_iterative_solver.hpp"
#include "fbsde_traj_opt/solvers/policy_rollout_evaluation.hpp"
#include "fbsde_traj_opt/utils/result.hpp"
#include "fbsde_traj_opt/value_function/polynomial_value_function_approx.hpp"
#include "fbsde_traj_opt/value_function/soft_min_quadratic_value_function_approx.hpp"
#include "fbsde_traj_opt/value_function/value_function_sgd_fitter.hpp"
#include "examples/fbrrt_comparison_experiment.hpp"
#include "examples/fbrrt_comparison_setup.hpp"
#include "examples/l1_benchmark_models.hpp"
#include "examples/l1_grid_dp_reference.hpp"

namespace fbsde_traj_opt::examples {
namespace {

using Clock = std::chrono::steady_clock;
using Control = Eigen::Matrix<double, 1, 1>;
using Grid = L1GridDpReference<L1Pendulum>;

constexpr std::size_t kStages = kL1PendulumNumStages;
constexpr std::uint64_t kSeed = 20260928;
constexpr std::uint64_t kHeldOutStream = 0x8CB92BA72F3D8DD7ULL;  // as the batch sweep's
constexpr std::size_t kHeldOutRollouts = 1024;

// Fit samples per stage: half from the optimal policy's rollouts, half from rollouts with three
// times the noise, so the fit covers where the Q-function minimization looks as well as where the
// optimal trajectories go.
constexpr int kFitSamples = 2048;

template <typename PolicyT>
auto HeldOut(const L1Pendulum& pendulum, const PolicyT& policy) noexcept -> PolicyRolloutCosts<double> {
  const Eigen::Matrix2d factor = Eigen::LLT<Eigen::Matrix2d>(pendulum.initial_distribution.Covariance()).matrixL();
  return EvaluatePolicyByRollouts(pendulum.forward_model,
                                  policy,
                                  pendulum.running_cost,
                                  pendulum.terminal_cost,
                                  Eigen::Vector2d(pendulum.initial_distribution.Mean()),
                                  factor,
                                  kHeldOutRollouts,
                                  kStages - 1,
                                  kSeed ^ kHeldOutStream);
}

// The noise-free rollout from the initial mean: the control sequence as runs of -, 0, + (with
// interior controls marked by their sign in lower case), the fuel, and the final state.
template <typename PolicyT>
auto DescribeRollout(const L1Pendulum& pendulum, const PolicyT& policy) noexcept -> std::string {
  Eigen::Vector2d state = pendulum.initial_distribution.Mean();
  std::string runs;
  char previous = '\0';
  std::size_t length = 0;
  double fuel = 0.0;
  const auto flush = [&] {
    if (length > 0) {
      runs += std::format("{}{} ", previous, length);
    }
  };
  for (std::size_t stage = 0; stage + 1 < kStages; ++stage) {
    const Control control = policy(stage, state);
    const double u = control[0];
    const char symbol = u >= 1.0 - 1e-9 ? '+' : u <= -1.0 + 1e-9 ? '-' : u > 1e-9 ? 'p' : u < -1e-9 ? 'm' : '0';
    if (symbol != previous) {
      flush();
      previous = symbol;
      length = 0;
    }
    ++length;
    fuel += pendulum.running_cost(stage, state, control);
    state = pendulum.forward_model(stage, state, control, Eigen::Vector2d::Zero());
  }
  flush();
  return std::format("fuel {:.3f}, terminal {:.3f}, final ({:+.3f} rad, {:+.3f} rad/s)\n      {}",
                     fuel,
                     pendulum.terminal_cost(state),
                     state[0],
                     state[1],
                     runs);
}

// One stage of the grid's value function as a ValueFunctionApprox, its derivatives by central
// differences over two grid cells, so the Taylor Q-function policy can run on the optimal value
// function itself.
class GridStageValueFunction {
 public:
  GridStageValueFunction(const Grid& grid, std::size_t stage, Eigen::Vector2d step) noexcept
      : grid_(&grid), stage_(stage), step_(std::move(step)) {}

  auto operator()(const Eigen::Vector2d& state) const noexcept -> double { return grid_->Value(stage_, state); }

  [[nodiscard]] auto Gradient(const Eigen::Vector2d& state) const noexcept -> Eigen::Vector2d {
    Eigen::Vector2d gradient;
    for (Eigen::Index axis = 0; axis < 2; ++axis) {
      const Eigen::Vector2d offset = step_[axis] * Eigen::Vector2d::Unit(axis);
      gradient[axis] = ((*this)(state + offset) - (*this)(state - offset)) / (2.0 * step_[axis]);
    }
    return gradient;
  }

  [[nodiscard]] auto Hessian(const Eigen::Vector2d& state) const noexcept -> Eigen::Matrix2d {
    Eigen::Matrix2d hessian;
    const double center = (*this)(state);
    const Eigen::Vector2d along_first = step_[0] * Eigen::Vector2d::UnitX();
    const Eigen::Vector2d along_second = step_[1] * Eigen::Vector2d::UnitY();
    hessian(0, 0) =
        ((*this)(state + along_first) - (2.0 * center) + (*this)(state - along_first)) / (step_[0] * step_[0]);
    hessian(1, 1) =
        ((*this)(state + along_second) - (2.0 * center) + (*this)(state - along_second)) / (step_[1] * step_[1]);
    hessian(0, 1) = ((*this)(state + along_first + along_second) - (*this)(state + along_first - along_second) -
                     (*this)(state - along_first + along_second) + (*this)(state - along_first - along_second)) /
                    (4.0 * step_[0] * step_[1]);
    hessian(1, 0) = hessian(0, 1);
    return hessian;
  }

 private:
  const Grid* grid_;
  std::size_t stage_;
  Eigen::Vector2d step_;
};

// Samples the fit states at every stage from rollouts of the grid policy.
auto SampleFitStates(const L1Pendulum& pendulum, const Grid& grid) noexcept
    -> std::vector<ValueFunctionStateBatch<Eigen::Vector2d, kFitSamples>> {
  std::vector<ValueFunctionStateBatch<Eigen::Vector2d, kFitSamples>> states(kStages);
  const CounterBasedNormalSampler sampler(kSeed ^ 0x2545F4914F6CDD1DULL);
  const Eigen::Matrix2d factor = Eigen::LLT<Eigen::Matrix2d>(pendulum.initial_distribution.Covariance()).matrixL();
  ParallelFor(kFitSamples, [&](std::size_t sample) noexcept {
    const double inflation = sample % 2 == 0 ? 1.0 : 3.0;
    const auto column = static_cast<Eigen::Index>(sample);
    Eigen::Vector2d noise;
    sampler.FillNoise(sample, 0, noise);
    Eigen::Vector2d state = pendulum.initial_distribution.Mean() + (factor * noise);
    states[0].col(column) = state;
    for (std::size_t stage = 0; stage + 1 < kStages; ++stage) {
      sampler.FillNoise(sample, stage + 1, noise);
      state = pendulum.forward_model(stage, state, grid(stage, state), Eigen::Vector2d(inflation * noise));
      states[stage + 1].col(column) = state;
    }
  });
  return states;
}

// Fits `table` to the grid's value function stage by stage, backward, with `fit(stage, table,
// targets)`, and scores the Taylor Q-function policy over the result. The fit error is reported on
// the samples from the optimal policy's own rollouts -- where the policy needs the value function to
// be right -- and on all of them.
template <typename ValueFunction, typename FitT>
auto ScoreFit(const std::string& name,
              const L1Pendulum& pendulum,
              const Grid& grid,
              const std::vector<ValueFunctionStateBatch<Eigen::Vector2d, kFitSamples>>& states,
              std::vector<ValueFunction> table,
              const FitT& fit) noexcept -> Result<> {
  double on_path = 0.0;
  double overall = 0.0;
  double worst_on_path = 0.0;
  double path_absolute = 0.0;
  const Clock::time_point begin = Clock::now();
  for (std::size_t stage = kStages - 1; stage-- > 0;) {
    ValueFunctionValueBatch<Eigen::Vector2d, kFitSamples> targets;
    for (std::size_t sample = 0; sample < static_cast<std::size_t>(kFitSamples); ++sample) {
      targets[static_cast<Eigen::Index>(sample)] =
          grid.Value(stage, states[stage].col(static_cast<Eigen::Index>(sample)));
    }
    RESULT_RETURN_IF_ERROR(fit(stage, table, targets));
    ValueFunctionValueBatch<Eigen::Vector2d, kFitSamples> values;
    table[stage].template Values<kFitSamples>(states[stage], values);
    double path_error = 0.0;
    double path_scale = 0.0;
    for (std::size_t sample = 0; sample < static_cast<std::size_t>(kFitSamples); sample += 2) {
      const auto column = static_cast<Eigen::Index>(sample);
      path_error += (values[column] - targets[column]) * (values[column] - targets[column]);
      path_scale += targets[column] * targets[column];
    }
    const double path_relative = std::sqrt(path_error / path_scale);
    path_absolute += std::sqrt(path_error / (kFitSamples / 2)) / static_cast<double>(kStages - 1);
    on_path += path_relative / static_cast<double>(kStages - 1);
    worst_on_path = std::max(worst_on_path, path_relative);
    overall += ((values - targets).norm() / targets.norm()) / static_cast<double>(kStages - 1);
  }
  const double fit_seconds = std::chrono::duration<double>(Clock::now() - begin).count();
  RESULT_ASSIGN_OR_RETURN(
      const auto policy,
      (TaylorQFunctionL1Policy<2, 1, L1Pendulum::ForwardModel, ValueFunction>::Make(
          pendulum.forward_model, pendulum.running_cost.L1Weight(), Control(-1.0), Control(1.0), table)));
  const PolicyRolloutCosts<double> costs = HeldOut(pendulum, policy);
  std::cout << std::format(
      "  {:<38} ({:5.1f} s): fit error on the optimal paths {:.3f} RMS, {:5.1f}% (worst stage {:5.1f}%), "
      "everywhere {:5.1f}%\n"
      "      held-out cost {:.4f} (fuel {:.4f}, terminal {:.4f})\n      {}\n",
      name,
      fit_seconds,
      path_absolute,
      100.0 * on_path,
      100.0 * worst_on_path,
      100.0 * overall,
      costs.total,
      costs.running,
      costs.terminal,
      DescribeRollout(pendulum, policy));
  return SuccessResult();
}

// The K-component soft-min model at sharpness `sharpness`, fitted by SGD, each stage warm-started
// from the one after it.
template <int K>
auto SoftMinCeiling(const L1Pendulum& pendulum,
                    const Grid& grid,
                    const std::vector<ValueFunctionStateBatch<Eigen::Vector2d, kFitSamples>>& states,
                    double sharpness,
                    double learning_rate,
                    std::size_t epochs) noexcept -> Result<> {
  using ValueFunction = SoftMinQuadraticValueFunctionApprox<2, K>;
  FbrrtComparisonOptions comparison;
  comparison.soft_min_sharpness = sharpness;
  RESULT_ASSIGN_OR_RETURN(const StartingTable<ValueFunction> start,
                          MakeStartingTable(pendulum, comparison, static_cast<const ValueFunction*>(nullptr)));
  std::vector<ValueFunction> table(kStages, start.initial);
  table.back() = start.terminal;
  ValueFunctionSgdOptions<double> sgd;
  sgd.learning_rate = learning_rate;
  const auto fit = [&](std::size_t stage,
                       std::vector<ValueFunction>& fitted,
                       const ValueFunctionValueBatch<Eigen::Vector2d, kFitSamples>& targets) noexcept -> Result<> {
    if (stage + 1 < kStages - 1) {
      fitted[stage] = fitted[stage + 1];
    }
    using Fitter = ValueFunctionSgdFitter<ValueFunction, 64>;
    RESULT_ASSIGN_OR_RETURN(Fitter fitter, Fitter::Make(sgd, fitted[stage].Parameters()));
    std::vector<typename Fitter::StepReport> history;
    return fitter.template FitEpochs<kFitSamples>(fitted[stage], states[stage], targets, epochs, stage, history);
  };
  return ScoreFit(std::format("soft-min K={} beta={:g} lr={:g} x{}", K, sharpness, learning_rate, epochs),
                  pendulum,
                  grid,
                  states,
                  std::move(table),
                  fit);
}

// The polynomial model of degree `Degree`, fitted in closed form: no optimizer stands between the
// representation and its best fit.
template <int Degree>
auto PolynomialCeiling(const L1Pendulum& pendulum,
                       const Grid& grid,
                       const std::vector<ValueFunctionStateBatch<Eigen::Vector2d, kFitSamples>>& states,
                       double off_path_weight) noexcept -> Result<> {
  using ValueFunction = PolynomialValueFunctionApprox<2, Degree>;
  RESULT_ASSIGN_OR_RETURN(
      ValueFunction blank,
      ValueFunction::Make(
          ValueFunction::ParameterVector::Zero(), pendulum.normalization_center, pendulum.normalization_scale));
  ValueFunction terminal = blank;
  RESULT_RETURN_IF_ERROR(terminal.SetParameters(blank.QuadraticParameters(pendulum.terminal_cost_mat, 0.0)));
  std::vector<ValueFunction> table(kStages, blank);
  table.back() = terminal;
  const auto fit = [&](std::size_t stage,
                       std::vector<ValueFunction>& fitted,
                       const ValueFunctionValueBatch<Eigen::Vector2d, kFitSamples>& targets) noexcept -> Result<> {
    ValueFunctionValueBatch<Eigen::Vector2d, kFitSamples> weights;
    for (Eigen::Index sample = 0; sample < kFitSamples; ++sample) {
      weights[sample] = sample % 2 == 0 ? 1.0 : off_path_weight;
    }
    return fitted[stage].template FitWeightedLeastSquares<kFitSamples>(states[stage], targets, weights, 1e-6);
  };
  return ScoreFit(std::format("polynomial degree {}, off-path weight {:g}", Degree, off_path_weight),
                  pendulum,
                  grid,
                  states,
                  std::move(table),
                  fit);
}

// The polynomial model fitted in closed form to the optimal value function's values and to its
// slope along the control column `b`: the weighted least squares of
//
//   sum_j w_j [ (V(x_j) - Y_j)^2 + gamma (b' grad V(x_j) - b' G_j)^2 ],
//
// where `b' grad V` is the value difference across one full control step, in cost units, and `G_j`
// is the optimal value function's gradient by central differences over two grid cells. Both rows
// are linear in the coefficients; a feature's slope along `b` is a central difference of the
// features, exact for a polynomial up to rounding.
template <int Degree>
auto SobolevPolynomialCeiling(const L1Pendulum& pendulum,
                              const Grid& grid,
                              const std::vector<ValueFunctionStateBatch<Eigen::Vector2d, kFitSamples>>& states,
                              const Eigen::Vector2d& difference_step,
                              double off_path_weight,
                              double gamma) noexcept -> Result<> {
  using ValueFunction = PolynomialValueFunctionApprox<2, Degree>;
  using Parameters = typename ValueFunction::ParameterVector;
  using ParameterMat = typename ValueFunction::ParameterMat;
  constexpr double kRidge = 1e-6;
  constexpr double kFeatureStep = 1e-4;
  RESULT_ASSIGN_OR_RETURN(
      ValueFunction blank,
      ValueFunction::Make(Parameters::Zero(), pendulum.normalization_center, pendulum.normalization_scale));
  ValueFunction terminal = blank;
  RESULT_RETURN_IF_ERROR(terminal.SetParameters(blank.QuadraticParameters(pendulum.terminal_cost_mat, 0.0)));
  std::vector<ValueFunction> table(kStages, blank);
  table.back() = terminal;
  const auto fit = [&](std::size_t stage,
                       std::vector<ValueFunction>& fitted,
                       const ValueFunctionValueBatch<Eigen::Vector2d, kFitSamples>& targets) noexcept -> Result<> {
    const GridStageValueFunction exact(grid, stage, difference_step);
    // The ridge as FitWeightedLeastSquares applies it, scaled by the sample count.
    ParameterMat normal = (kRidge * kFitSamples) * ParameterMat::Identity();
    Parameters right = Parameters::Zero();
    for (Eigen::Index sample = 0; sample < kFitSamples; ++sample) {
      const Eigen::Vector2d state = states[stage].col(sample);
      const double weight = sample % 2 == 0 ? 1.0 : off_path_weight;
      const Parameters features = blank.ParameterGradient(state);
      normal += weight * features * features.transpose();
      right += weight * targets[sample] * features;
      if (gamma > 0.0) {
        const Eigen::Vector2d column = pendulum.forward_model.control_drift_mat()(stage, state);
        const Parameters slope_features = (blank.ParameterGradient(state + (kFeatureStep * column)) -
                                           blank.ParameterGradient(state - (kFeatureStep * column))) /
                                          (2.0 * kFeatureStep);
        const double slope_target = column.dot(exact.Gradient(state));
        normal += gamma * weight * slope_features * slope_features.transpose();
        right += gamma * weight * slope_target * slope_features;
      }
    }
    const Eigen::LDLT<ParameterMat> factorization(normal);
    RESULT_ASSERT(factorization.info() == Eigen::Success, "SobolevPolynomialCeiling: the normal equations failed.");
    return fitted[stage].SetParameters(factorization.solve(right));
  };
  return ScoreFit(std::format("poly deg {} w_off {:g} gamma {:g}", Degree, off_path_weight, gamma),
                  pendulum,
                  grid,
                  states,
                  std::move(table),
                  fit);
}

// Runs DT-FBSDE in the best configuration found so far and compares what it learned with the
// optimum.
template <std::size_t Batch>
auto Learned(const L1Pendulum& pendulum, const Grid& grid, double seconds, std::uint64_t seed) noexcept -> Result<> {
  using ValueFunction = SoftMinQuadraticValueFunctionApprox<2, kSoftMinComponents>;
  using Solver = DtFbsdeIterativeSolver<2,
                                        1,
                                        Batch,
                                        kStages,
                                        kComparisonEvaluationSamples,
                                        L1Pendulum::ForwardModel,
                                        QuadraticRegulatorSdeTerminalCostTerm<2>,
                                        ValueFunction>;
  FbrrtComparisonOptions comparison;
  RESULT_ASSIGN_OR_RETURN(const StartingTable<ValueFunction> start,
                          MakeStartingTable(pendulum, comparison, static_cast<const ValueFunction*>(nullptr)));
  DtFbsdeIterativeOptions<double> options = DtFbsdeOptions(comparison);
  options.stop_uphill_line_search = true;
  options.screening_rollouts = 32;
  options.base_diffusion_inflation = 6.0;
  options.max_diffusion_inflation = 6.0;
  auto solver = std::make_unique<Result<Solver>>(Solver::Make(pendulum.forward_model,
                                                              pendulum.running_cost,
                                                              pendulum.terminal_cost,
                                                              pendulum.initial_distribution,
                                                              Control(-1.0),
                                                              Control(1.0),
                                                              start.initial,
                                                              start.terminal,
                                                              options,
                                                              seed));
  if (!solver->has_value()) {
    return ErrorResult(solver->error().message, solver->error().location);
  }
  Solver& dt = **solver;
  const Clock::time_point begin = Clock::now();
  std::size_t iterations = 0;
  while (std::chrono::duration<double>(Clock::now() - begin).count() < seconds) {
    RESULT_RETURN_IF_ERROR(dt.Iterate());
    ++iterations;
  }
  const PolicyRolloutCosts<double> costs = HeldOut(pendulum, dt.CurrentPolicy());
  std::cout << std::format(
      "  batch {}, seed {}, {} iterations in {:.0f} s: held-out cost {:.4f} (fuel {:.4f}, terminal {:.4f}); "
      "its own V_0(x_0) {:.4f}\n      {}\n",
      Batch,
      seed,
      iterations,
      seconds,
      costs.total,
      costs.running,
      costs.terminal,
      dt.ValueFunctions()[0](pendulum.initial_distribution.Mean()),
      DescribeRollout(pendulum, dt.CurrentPolicy()));

  // The learned value function against the optimal one along the learned noise-free trajectory.
  std::cout << "      along it, stage: state -> learned V / optimal V\n";
  Eigen::Vector2d state = pendulum.initial_distribution.Mean();
  const auto policy = dt.CurrentPolicy();
  for (std::size_t stage = 0; stage < kStages; ++stage) {
    if (stage % 8 == 0 || stage == kStages - 1) {
      std::cout << std::format("        {:>2}: ({:+.2f}, {:+.2f}) -> {:7.3f} / {:7.3f}\n",
                               stage,
                               state[0],
                               state[1],
                               dt.ValueFunctions()[stage](state),
                               grid.Value(stage, state));
    }
    if (stage + 1 < kStages) {
      state = pendulum.forward_model(stage, state, policy(stage, state), Eigen::Vector2d::Zero());
    }
  }
  return SuccessResult();
}

auto Run(int argc, char** argv) -> Result<> {
  std::size_t epochs = 300;
  double seconds = 60.0;
  std::size_t learned_seeds = 2;
  bool sobolev_only = false;
  for (int index = 1; index < argc; index += 2) {
    const std::string argument = argv[index];
    if (argument == "--sobolev-only") {
      sobolev_only = true;
      --index;
      continue;
    }
    if (index + 1 >= argc) {
      return ErrorResult("every option but --sobolev-only needs a value");
    }
    if (argument == "--epochs") {
      epochs = std::strtoull(argv[index + 1], nullptr, 10);
    } else if (argument == "--seconds") {
      seconds = std::strtod(argv[index + 1], nullptr);
    } else if (argument == "--learned-seeds") {
      learned_seeds = std::strtoull(argv[index + 1], nullptr, 10);
    } else {
      return ErrorResult("options: --epochs N, --seconds S, --learned-seeds N");
    }
  }

  RESULT_ASSIGN_OR_RETURN(const L1Pendulum pendulum, MakeL1Pendulum());
  Grid::Options grid_options;
  grid_options.lower = Eigen::Vector2d(-2.0, -9.0);
  grid_options.upper = Eigen::Vector2d(5.0, 9.0);
  RESULT_ASSIGN_OR_RETURN(const Grid grid, Grid::Make(pendulum, kStages, grid_options));
  const PolicyRolloutCosts<double> optimum = HeldOut(pendulum, grid);
  std::cout << std::format(
      "Grid optimum: held-out cost {:.4f} (fuel {:.4f}, terminal {:.4f}); V_0(x_0) {:.4f}\n      {}\n",
      optimum.total,
      optimum.running,
      optimum.terminal,
      grid.Value(0, pendulum.initial_distribution.Mean()),
      DescribeRollout(pendulum, grid));

  std::cout << "\n0. The Taylor-Q policy over the optimal value function itself:\n";
  {
    std::vector<GridStageValueFunction> exact;
    const Eigen::Vector2d step =
        2.0 * (grid_options.upper - grid_options.lower) / static_cast<double>(grid_options.cells_per_axis);
    for (std::size_t stage = 0; stage < kStages; ++stage) {
      exact.emplace_back(grid, stage, step);
    }
    RESULT_ASSIGN_OR_RETURN(
        const auto policy,
        (TaylorQFunctionL1Policy<2, 1, L1Pendulum::ForwardModel, GridStageValueFunction>::Make(
            pendulum.forward_model, pendulum.running_cost.L1Weight(), Control(-1.0), Control(1.0), exact)));
    const PolicyRolloutCosts<double> costs = HeldOut(pendulum, policy);
    std::cout << std::format("  held-out cost {:.4f} (fuel {:.4f}, terminal {:.4f})\n      {}\n",
                             costs.total,
                             costs.running,
                             costs.terminal,
                             DescribeRollout(pendulum, policy));
  }

  if (sobolev_only) {
    std::cout << "\n3. The polynomial model fitted to the optimal values and slopes along b, by gamma:\n";
    const auto fit_states = SampleFitStates(pendulum, grid);
    const Eigen::Vector2d step =
        2.0 * (grid_options.upper - grid_options.lower) / static_cast<double>(grid_options.cells_per_axis);
    for (const double off_path_weight : {1.0, 0.01}) {
      for (const double gamma : {0.0, 0.1, 1.0, 10.0, 100.0}) {
        RESULT_RETURN_IF_ERROR(SobolevPolynomialCeiling<6>(pendulum, grid, fit_states, step, off_path_weight, gamma));
        RESULT_RETURN_IF_ERROR(SobolevPolynomialCeiling<8>(pendulum, grid, fit_states, step, off_path_weight, gamma));
      }
    }
    return SuccessResult();
  }

  std::cout << "\n1. The soft-min model fitted to the optimal value function, and its Taylor-Q policy:\n";
  const auto states = SampleFitStates(pendulum, grid);
  for (const double sharpness : {1.0, 4.0}) {
    for (const double learning_rate : {0.05, 0.01}) {
      RESULT_RETURN_IF_ERROR(SoftMinCeiling<4>(pendulum, grid, states, sharpness, learning_rate, epochs));
    }
  }
  RESULT_RETURN_IF_ERROR(SoftMinCeiling<4>(pendulum, grid, states, 4.0, 0.01, 4 * epochs));
  RESULT_RETURN_IF_ERROR(SoftMinCeiling<8>(pendulum, grid, states, 4.0, 0.01, epochs));
  RESULT_RETURN_IF_ERROR(SoftMinCeiling<16>(pendulum, grid, states, 4.0, 0.01, epochs));
  for (const double off_path_weight : {1.0, 0.01}) {
    RESULT_RETURN_IF_ERROR(PolynomialCeiling<4>(pendulum, grid, states, off_path_weight));
    RESULT_RETURN_IF_ERROR(PolynomialCeiling<6>(pendulum, grid, states, off_path_weight));
    RESULT_RETURN_IF_ERROR(PolynomialCeiling<8>(pendulum, grid, states, off_path_weight));
  }

  std::cout << "\n2. What DT-FBSDE learns (constant noise x6, the line-search options):\n";
  for (std::size_t index = 0; index < learned_seeds; ++index) {
    RESULT_RETURN_IF_ERROR(Learned<64>(pendulum, grid, seconds, kSeed + (1000U * index)));
  }
  return SuccessResult();
}

}  // namespace
}  // namespace fbsde_traj_opt::examples

auto main(int argc, char** argv) -> int {
  const fbsde_traj_opt::Result<> result = fbsde_traj_opt::examples::Run(argc, argv);
  if (!result.has_value()) {
    RESULT_REPORT_RESULT(result);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
