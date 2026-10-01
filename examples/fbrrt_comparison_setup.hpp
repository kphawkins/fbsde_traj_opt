// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#ifndef FBSDE_TRAJ_OPT_EXAMPLES_FBRRT_COMPARISON_SETUP_HPP_
#define FBSDE_TRAJ_OPT_EXAMPLES_FBRRT_COMPARISON_SETUP_HPP_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include <Eigen/Core>

#include "fbsde_traj_opt/costs/quadratic_regulator_sde_terminal_cost_term.hpp"
#include "fbsde_traj_opt/solvers/dt_fbsde_iterative_solver.hpp"
#include "fbsde_traj_opt/solvers/fbrrt_solver.hpp"
#include "fbsde_traj_opt/utils/result.hpp"
#include "fbsde_traj_opt/value_function/nonnegative_value_function_approx.hpp"
#include "fbsde_traj_opt/value_function/polynomial_value_function_approx.hpp"
#include "fbsde_traj_opt/value_function/soft_min_quadratic_value_function_approx.hpp"
#include "examples/fbrrt_comparison_experiment.hpp"

namespace fbsde_traj_opt::examples {

// How the FBRRT comparison sets up a problem -- its solver types, starting value functions, and
// solver options -- shared by the experiments that build on it, so that what they show is the
// configuration the comparison measures.

inline constexpr std::size_t kComparisonEvaluationSamples = 256;

// The solver types for one benchmark: `Capacity` samples per stage for both methods, and value
// functions of type `ValueFunctionT`.
template <typename BenchmarkT, std::size_t Capacity, std::size_t NumStages, typename ValueFunctionT>
struct Setup {
  static constexpr int kStateDim = BenchmarkT::State::RowsAtCompileTime;
  using ForwardModel = typename BenchmarkT::ForwardModel;
  using ValueFunction = ValueFunctionT;
  using TerminalCost = QuadraticRegulatorSdeTerminalCostTerm<kStateDim>;
  using Components = DefaultFbrrtComponents<kStateDim, 1, Capacity, ForwardModel, ValueFunction>;
  using Fbrrt = FbrrtSolver<kStateDim,
                            1,
                            Capacity,
                            NumStages,
                            kComparisonEvaluationSamples,
                            ForwardModel,
                            TerminalCost,
                            ValueFunction,
                            Components>;
  using DtFbsde = DtFbsdeIterativeSolver<kStateDim,
                                         1,
                                         Capacity,
                                         NumStages,
                                         kComparisonEvaluationSamples,
                                         ForwardModel,
                                         TerminalCost,
                                         ValueFunction>;
};

// The terminal value function and the starting guess at every other stage.
template <typename ValueFunctionT>
struct StartingTable {
  ValueFunctionT terminal;
  ValueFunctionT initial;
};

// The polynomial model: the terminal cost, exactly, at every stage.
template <int N, int Degree, typename BenchmarkT>
auto MakeStartingTable(const BenchmarkT& benchmark,
                       const FbrrtComparisonOptions& /*options*/,
                       const PolynomialValueFunctionApprox<N, Degree>* /*tag*/) noexcept
    -> Result<StartingTable<PolynomialValueFunctionApprox<N, Degree>>> {
  using ValueFunction = PolynomialValueFunctionApprox<N, Degree>;
  RESULT_ASSIGN_OR_RETURN(
      ValueFunction value_function,
      ValueFunction::Make(
          ValueFunction::ParameterVector::Zero(), benchmark.normalization_center, benchmark.normalization_scale));
  RESULT_RETURN_IF_ERROR(
      value_function.SetParameters(value_function.QuadraticParameters(benchmark.terminal_cost_mat, 0.0)));
  return SuccessResult(StartingTable<ValueFunction>{.terminal = value_function, .initial = value_function});
}

// The soft-min model: the terminal cost exactly at the terminal stage, and its bowl spread along
// the way from the goal to the start everywhere else.
template <int N, int K, typename BenchmarkT>
auto MakeStartingTable(const BenchmarkT& benchmark,
                       const FbrrtComparisonOptions& options,
                       const SoftMinQuadraticValueFunctionApprox<N, K>* /*tag*/) noexcept
    -> Result<StartingTable<SoftMinQuadraticValueFunctionApprox<N, K>>> {
  using ValueFunction = SoftMinQuadraticValueFunctionApprox<N, K>;
  using State = typename ValueFunction::State;
  using Parameters = typename ValueFunction::ParameterVector;
  const double sharpness = options.soft_min_sharpness;
  const double offset = options.soft_min_diagonal_offset;
  RESULT_ASSIGN_OR_RETURN(
      const Parameters terminal_parameters,
      ValueFunction::QuadraticParameters(benchmark.terminal_cost_mat, State::Zero(), 0.0, sharpness, offset));
  std::array<State, static_cast<std::size_t>(K)> centers{};
  const State start = benchmark.initial_distribution.Mean();
  for (std::size_t component = 0; component < centers.size(); ++component) {
    centers[component] =
        (options.soft_min_center_spread * static_cast<double>(component) / static_cast<double>(K - 1)) * start;
  }
  RESULT_ASSIGN_OR_RETURN(
      const Parameters initial_parameters,
      ValueFunction::QuadraticParametersAt(benchmark.terminal_cost_mat, centers, 0.0, sharpness, offset));
  RESULT_ASSIGN_OR_RETURN(ValueFunction terminal, ValueFunction::Make(terminal_parameters, sharpness, offset));
  RESULT_ASSIGN_OR_RETURN(ValueFunction initial, ValueFunction::Make(initial_parameters, sharpness, offset));
  return SuccessResult(StartingTable<ValueFunction>{.terminal = terminal, .initial = initial});
}

// The nonnegative soft-min model: the soft-min model's starting table, each entry wrapped in the
// softplus. The terminal entry is then the terminal cost to within `s log 2` near the goal.
template <int N, int K, typename Scalar>
using NonnegativeSoftMinQuadraticValueFunctionApprox =
    NonnegativeValueFunctionApprox<SoftMinQuadraticValueFunctionApprox<N, K, Scalar>>;

template <int N, int K, typename BenchmarkT>
auto MakeStartingTable(const BenchmarkT& benchmark,
                       const FbrrtComparisonOptions& options,
                       const NonnegativeSoftMinQuadraticValueFunctionApprox<N, K, double>* /*tag*/) noexcept
    -> Result<StartingTable<NonnegativeSoftMinQuadraticValueFunctionApprox<N, K, double>>> {
  using ValueFunction = NonnegativeSoftMinQuadraticValueFunctionApprox<N, K, double>;
  using Inner = SoftMinQuadraticValueFunctionApprox<N, K>;
  RESULT_ASSIGN_OR_RETURN(const StartingTable<Inner> inner,
                          MakeStartingTable(benchmark, options, static_cast<const Inner*>(nullptr)));
  RESULT_ASSIGN_OR_RETURN(ValueFunction terminal, ValueFunction::Make(inner.terminal, options.nonnegative_softness));
  RESULT_ASSIGN_OR_RETURN(ValueFunction initial, ValueFunction::Make(inner.initial, options.nonnegative_softness));
  return SuccessResult(StartingTable<ValueFunction>{.terminal = terminal, .initial = initial});
}

// How connected one particle set's stages are: the fractions MethodRun records.
struct Connectivity {
  double disconnected = 0.0;
  double worst_stage_disconnected = 0.0;
  double orphaned = 0.0;
};

template <typename ParticleSetT, typename State>
auto ParticleConnectivity(const ParticleSetT& particles, const State& region_lower, const State& region_upper) noexcept
    -> Connectivity {
  Connectivity connectivity;
  const State scale = (2.0 / (region_upper - region_lower).array()).matrix();
  std::size_t total = 0;
  std::size_t disconnected = 0;
  std::size_t orphaned = 0;
  for (std::size_t stage = 2; stage < ParticleSetT::kNumStages; ++stage) {
    std::size_t stage_disconnected = 0;
    for (std::size_t slot = 0; slot < particles.count(stage); ++slot) {
      ++total;
      orphaned += particles.parents(stage)[slot] == ParticleSetT::kDetachedParent ? 1 : 0;
      const State source = particles.source_states(stage).col(static_cast<Eigen::Index>(slot));
      bool reached = false;
      for (std::size_t parent = 0; parent < particles.count(stage - 1) && !reached; ++parent) {
        const State difference = source - State(particles.states(stage - 1).col(static_cast<Eigen::Index>(parent)));
        reached = scale.cwiseProduct(difference).norm() <= kDisconnectionRadius;
      }
      stage_disconnected += reached ? 0 : 1;
    }
    disconnected += stage_disconnected;
    if (particles.count(stage) > 0) {
      connectivity.worst_stage_disconnected =
          std::max(connectivity.worst_stage_disconnected,
                   static_cast<double>(stage_disconnected) / static_cast<double>(particles.count(stage)));
    }
  }
  const double count = total == 0 ? 1.0 : static_cast<double>(total);
  connectivity.disconnected = static_cast<double>(disconnected) / count;
  connectivity.orphaned = static_cast<double>(orphaned) / count;
  return connectivity;
}

// The DT-FBSDE options the double inverted pendulum example was tuned with.
inline auto DtFbsdeOptions(const FbrrtComparisonOptions& comparison) noexcept -> DtFbsdeIterativeOptions<double> {
  DtFbsdeIterativeOptions<double> options;
  options.sgd = comparison.dt_fbsde_sgd;
  options.sgd_epochs = comparison.dt_fbsde_sgd_epochs;
  options.nonlinear_fit_method = comparison.dt_fbsde_fit_method;
  options.backward_target = DtFbsdeBackwardTarget::kDoubleGreedy;
  options.exploration_probability = 0.03;
  options.initial_exploration_probability = 1.0;
  options.exploration_decay = 0.8;
  options.probe_fraction = 0.25;
  options.target_floor = 0.0;
  options.entropy_temperature = 2.0;
  return options;
}

// Builds the FBRRT solver the comparison runs, on the heap: its particle set and value function
// tables run to megabytes.
template <typename SetupT, typename BenchmarkT>
auto MakeComparisonFbrrtSolver(const BenchmarkT& benchmark,
                               const StartingTable<typename SetupT::ValueFunction>& table,
                               const FbrrtComparisonOptions& options,
                               const FbrrtOptions<double>& fbrrt_options,
                               double keep_fraction,
                               std::uint64_t seed) noexcept -> Result<std::unique_ptr<typename SetupT::Fbrrt>> {
  using Control = Eigen::Matrix<double, 1, 1>;
  constexpr auto kCapacity = static_cast<std::size_t>(SetupT::Fbrrt::Particles::kCapacity);
  constexpr std::size_t kNumStages = SetupT::Fbrrt::Particles::kNumStages;
  const auto keep_count = static_cast<std::size_t>(keep_fraction * static_cast<double>(kCapacity));
  RESULT_ASSIGN_OR_RETURN(typename SetupT::Components components,
                          (MakeDefaultFbrrtComponents<kCapacity, kNumStages>(benchmark.forward_model,
                                                                             benchmark.running_cost,
                                                                             benchmark.region_lower,
                                                                             benchmark.region_upper,
                                                                             Control(-1.0),
                                                                             Control(1.0),
                                                                             options.backward,
                                                                             keep_count,
                                                                             table.initial,
                                                                             options.erode)));
  auto solver = std::make_unique<Result<typename SetupT::Fbrrt>>(SetupT::Fbrrt::Make(benchmark.forward_model,
                                                                                     benchmark.running_cost,
                                                                                     benchmark.terminal_cost,
                                                                                     benchmark.initial_distribution,
                                                                                     Control(-1.0),
                                                                                     Control(1.0),
                                                                                     table.initial,
                                                                                     table.terminal,
                                                                                     std::move(components),
                                                                                     fbrrt_options,
                                                                                     seed));
  if (!solver->has_value()) {
    return ErrorResult(solver->error().message, solver->error().location);
  }
  return SuccessResult(std::make_unique<typename SetupT::Fbrrt>(std::move(**solver)));
}

}  // namespace fbsde_traj_opt::examples

#endif  // FBSDE_TRAJ_OPT_EXAMPLES_FBRRT_COMPARISON_SETUP_HPP_
