// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#include "examples/fbrrt_comparison_experiment.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "fbsde_traj_opt/costs/quadratic_regulator_sde_terminal_cost_term.hpp"
#include "fbsde_traj_opt/fbrrt/sgd_taylor_backward_pass.hpp"
#include "fbsde_traj_opt/solvers/dt_fbsde_iterative_solver.hpp"
#include "fbsde_traj_opt/solvers/fbrrt_solver.hpp"
#include "fbsde_traj_opt/utils/result.hpp"
#include "fbsde_traj_opt/value_function/polynomial_value_function_approx.hpp"
#include "examples/l1_benchmark_models.hpp"

namespace fbsde_traj_opt::examples {
namespace {

using Clock = std::chrono::steady_clock;
using Control = Eigen::Matrix<double, 1, 1>;

constexpr std::size_t kEvaluationSamples = 256;
constexpr double kInfinity = std::numeric_limits<double>::infinity();

// The solver types for one benchmark: `Capacity` samples per stage for both methods, and value
// functions of polynomial degree `Degree`.
template <typename BenchmarkT, std::size_t Capacity, std::size_t NumStages, int Degree>
struct Setup {
  static constexpr int kStateDim = BenchmarkT::State::RowsAtCompileTime;
  using ForwardModel = typename BenchmarkT::ForwardModel;
  using ValueFunction = PolynomialValueFunctionApprox<kStateDim, Degree>;
  using TerminalCost = QuadraticRegulatorSdeTerminalCostTerm<kStateDim>;
  using Components = DefaultFbrrtComponents<kStateDim, 1, Capacity, ForwardModel, ValueFunction>;
  using Fbrrt = FbrrtSolver<kStateDim,
                            1,
                            Capacity,
                            NumStages,
                            kEvaluationSamples,
                            ForwardModel,
                            TerminalCost,
                            ValueFunction,
                            Components>;
  using DtFbsde = DtFbsdeIterativeSolver<kStateDim,
                                         1,
                                         Capacity,
                                         NumStages,
                                         kEvaluationSamples,
                                         ForwardModel,
                                         TerminalCost,
                                         ValueFunction>;
};

// The terminal cost in the value function's basis: the starting table at every stage for both
// methods.
template <typename ValueFunctionT, typename BenchmarkT>
auto MakeTerminalValueFunction(const BenchmarkT& benchmark) noexcept -> Result<ValueFunctionT> {
  RESULT_ASSIGN_OR_RETURN(
      ValueFunctionT value_function,
      ValueFunctionT::Make(
          ValueFunctionT::ParameterVector::Zero(), benchmark.normalization_center, benchmark.normalization_scale));
  RESULT_RETURN_IF_ERROR(
      value_function.SetParameters(value_function.QuadraticParameters(benchmark.terminal_cost_mat, 0.0)));
  return SuccessResult(std::move(value_function));
}

// The DT-FBSDE options the double inverted pendulum example was tuned with.
auto DtFbsdeOptions() noexcept -> DtFbsdeIterativeOptions<double> {
  DtFbsdeIterativeOptions<double> options;
  options.backward_target = DtFbsdeBackwardTarget::kDoubleGreedy;
  options.exploration_probability = 0.03;
  options.initial_exploration_probability = 1.0;
  options.exploration_decay = 0.8;
  options.probe_fraction = 0.25;
  options.target_floor = 0.0;
  options.entropy_temperature = 2.0;
  return options;
}

template <typename SetupT, typename BenchmarkT>
auto RunFbrrt(const BenchmarkT& benchmark,
              const typename SetupT::ValueFunction& terminal_value_function,
              const FbrrtComparisonOptions& options,
              double keep_fraction,
              std::size_t iterations,
              std::uint64_t seed) noexcept -> Result<MethodRun> {
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
                                                                             terminal_value_function)));
  // The particle set and value function tables run to megabytes; the solver lives on the heap.
  auto solver = std::make_unique<Result<typename SetupT::Fbrrt>>(SetupT::Fbrrt::Make(benchmark.forward_model,
                                                                                     benchmark.running_cost,
                                                                                     benchmark.terminal_cost,
                                                                                     benchmark.initial_distribution,
                                                                                     Control(-1.0),
                                                                                     Control(1.0),
                                                                                     terminal_value_function,
                                                                                     terminal_value_function,
                                                                                     std::move(components),
                                                                                     options.fbrrt,
                                                                                     seed));
  if (!solver->has_value()) {
    return ErrorResult(solver->error().message, solver->error().location);
  }

  MethodRun run;
  run.costs.push_back((*solver)->Evaluate().total);
  run.seconds.push_back(0.0);
  for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
    RESULT_ASSIGN_OR_RETURN(const FbrrtIterationReport<double> report, (*solver)->Iterate());
    run.costs.push_back(report.mean_cost);
    run.seconds.push_back(run.seconds.back() + report.expand_seconds + report.backward_seconds + report.erode_seconds);
  }
  return SuccessResult(std::move(run));
}

template <typename SetupT, typename BenchmarkT>
auto RunDtFbsde(const BenchmarkT& benchmark,
                const typename SetupT::ValueFunction& terminal_value_function,
                std::size_t iterations,
                std::uint64_t seed) noexcept -> Result<MethodRun> {
  auto solver = std::make_unique<Result<typename SetupT::DtFbsde>>(SetupT::DtFbsde::Make(benchmark.forward_model,
                                                                                         benchmark.running_cost,
                                                                                         benchmark.terminal_cost,
                                                                                         benchmark.initial_distribution,
                                                                                         Control(-1.0),
                                                                                         Control(1.0),
                                                                                         terminal_value_function,
                                                                                         terminal_value_function,
                                                                                         DtFbsdeOptions(),
                                                                                         seed));
  if (!solver->has_value()) {
    return ErrorResult(solver->error().message, solver->error().location);
  }

  MethodRun run;
  run.costs.push_back((*solver)->CurrentCost());
  run.seconds.push_back(0.0);
  for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
    const Clock::time_point start = Clock::now();
    RESULT_ASSIGN_OR_RETURN(const DtFbsdeIterationReport<double> report, (*solver)->Iterate());
    run.costs.push_back(report.mean_cost);
    run.seconds.push_back(run.seconds.back() + std::chrono::duration<double>(Clock::now() - start).count());
  }
  return SuccessResult(std::move(run));
}

auto Median(std::vector<double> values) noexcept -> double {
  if (values.empty()) {
    return kInfinity;
  }
  const auto middle = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2);
  std::ranges::nth_element(values, middle);
  if (values.size() % 2 == 1) {
    return *middle;
  }
  const double upper = *middle;
  return 0.5 * (upper + *std::ranges::max_element(values.begin(), middle));
}

// Fills in a method's medians against the problem's convergence threshold.
auto Summarize(MethodResult& result, double threshold) noexcept -> void {
  std::vector<double> final_costs;
  std::vector<double> best_costs;
  std::vector<double> iterations;
  std::vector<double> seconds;
  std::vector<double> per_iteration;
  std::size_t converged = 0;
  for (const MethodRun& run : result.runs) {
    final_costs.push_back(run.costs.back());
    best_costs.push_back(*std::ranges::min_element(run.costs));
    per_iteration.push_back(run.seconds.back() / static_cast<double>(run.costs.size() - 1));
    const auto first = std::ranges::find_if(run.costs, [threshold](double cost) { return cost <= threshold; });
    if (first == run.costs.end()) {
      iterations.push_back(kInfinity);
      seconds.push_back(kInfinity);
    } else {
      const auto index = static_cast<std::size_t>(first - run.costs.begin());
      iterations.push_back(static_cast<double>(index));
      seconds.push_back(run.seconds[index]);
      ++converged;
    }
  }
  result.median_final_cost = Median(final_costs);
  result.median_best_cost = Median(best_costs);
  result.median_iterations_to_converge = Median(iterations);
  result.median_seconds_to_converge = Median(seconds);
  result.median_seconds_per_iteration = Median(per_iteration);
  result.fraction_converged = static_cast<double>(converged) / static_cast<double>(result.runs.size());
}

template <typename SetupT, typename BenchmarkT>
auto Compare(const BenchmarkT& benchmark,
             const FbrrtComparisonOptions& options,
             double keep_fraction,
             std::size_t iterations,
             const FbrrtComparisonProgress& progress) noexcept -> Result<BenchmarkComparison> {
  RESULT_ASSIGN_OR_RETURN(const typename SetupT::ValueFunction terminal_value_function,
                          MakeTerminalValueFunction<typename SetupT::ValueFunction>(benchmark));
  BenchmarkComparison comparison;
  comparison.name = benchmark.name;
  comparison.description = benchmark.description;
  comparison.fbrrt.method = "FBRRT";
  comparison.dt_fbsde.method = "DT-FBSDE";

  for (std::size_t index = 0; index < options.num_seeds; ++index) {
    const std::uint64_t seed = options.seed + (1000U * static_cast<std::uint64_t>(index));
    RESULT_ASSIGN_OR_RETURN(
        MethodRun fbrrt,
        RunFbrrt<SetupT>(benchmark, terminal_value_function, options, keep_fraction, iterations, seed));
    if (progress) {
      progress(benchmark.name, comparison.fbrrt.method, index, fbrrt);
    }
    comparison.fbrrt.runs.push_back(std::move(fbrrt));

    RESULT_ASSIGN_OR_RETURN(MethodRun dt_fbsde,
                            RunDtFbsde<SetupT>(benchmark, terminal_value_function, iterations, seed));
    if (progress) {
      progress(benchmark.name, comparison.dt_fbsde.method, index, dt_fbsde);
    }
    comparison.dt_fbsde.runs.push_back(std::move(dt_fbsde));
  }

  comparison.initial_cost = comparison.fbrrt.runs.front().costs.front();
  comparison.best_cost = kInfinity;
  for (const MethodResult* method : {&comparison.fbrrt, &comparison.dt_fbsde}) {
    for (const MethodRun& run : method->runs) {
      comparison.best_cost = std::min(comparison.best_cost, *std::ranges::min_element(run.costs));
    }
  }
  comparison.convergence_threshold = (1.0 + options.convergence_tolerance) * comparison.best_cost;
  Summarize(comparison.fbrrt, comparison.convergence_threshold);
  Summarize(comparison.dt_fbsde, comparison.convergence_threshold);
  return SuccessResult(std::move(comparison));
}

}  // namespace

auto RunFbrrtComparison(const FbrrtComparisonOptions& options, const FbrrtComparisonProgress& progress) noexcept
    -> Result<std::vector<BenchmarkComparison>> {
  RESULT_ASSERT(options.num_seeds >= 1 && options.iterations >= 1,
                "RunFbrrtComparison: the comparison needs at least one seed and one iteration.");
  std::vector<BenchmarkComparison> comparisons;

  RESULT_ASSIGN_OR_RETURN(const L1DoubleIntegrator integrator, MakeL1DoubleIntegrator());
  RESULT_ASSIGN_OR_RETURN(BenchmarkComparison integrator_comparison,
                          (Compare<Setup<L1DoubleIntegrator, 1024, kL1DoubleIntegratorNumStages, 2>>(
                              integrator, options, options.keep_fraction, options.iterations, progress)));
  comparisons.push_back(std::move(integrator_comparison));

  RESULT_ASSIGN_OR_RETURN(const L1Pendulum pendulum, MakeL1Pendulum());
  RESULT_ASSIGN_OR_RETURN(BenchmarkComparison pendulum_comparison,
                          (Compare<Setup<L1Pendulum, 1024, kL1PendulumNumStages, 4>>(
                              pendulum, options, options.keep_fraction, options.iterations, progress)));
  comparisons.push_back(std::move(pendulum_comparison));

  if (options.include_double_inverted_pendulum) {
    RESULT_ASSIGN_OR_RETURN(const L1DoubleInvertedPendulum double_pendulum, MakeL1DoubleInvertedPendulum());
    RESULT_ASSIGN_OR_RETURN(BenchmarkComparison double_pendulum_comparison,
                            (Compare<Setup<L1DoubleInvertedPendulum, 4096, kL1DoubleInvertedPendulumNumStages, 4>>(
                                double_pendulum,
                                options,
                                options.double_inverted_pendulum_keep_fraction,
                                options.double_inverted_pendulum_iterations,
                                progress)));
    comparisons.push_back(std::move(double_pendulum_comparison));
  }
  return SuccessResult(std::move(comparisons));
}

auto SummarizeFbrrtComparison(const BenchmarkComparison& comparison) noexcept -> std::string {
  std::string summary = std::format("{}: initial cost {:.3f}, best cost {:.3f}, converged = within {:.3f}\n",
                                    comparison.name,
                                    comparison.initial_cost,
                                    comparison.best_cost,
                                    comparison.convergence_threshold);
  for (const MethodResult* method : {&comparison.fbrrt, &comparison.dt_fbsde}) {
    summary += std::format(
        "  {:<8} median final {:.3f}, median best {:.3f}; converged in {:.0f}% of {} runs, median {} iterations / "
        "{:.3f} s ({:.4f} s per iteration)\n",
        method->method,
        method->median_final_cost,
        method->median_best_cost,
        100.0 * method->fraction_converged,
        method->runs.size(),
        std::isfinite(method->median_iterations_to_converge)
            ? std::format("{:.0f}", method->median_iterations_to_converge)
            : std::string("never"),
        method->median_seconds_to_converge,
        method->median_seconds_per_iteration);
  }
  return summary;
}

}  // namespace fbsde_traj_opt::examples
