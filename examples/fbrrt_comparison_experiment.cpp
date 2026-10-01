// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#include "examples/fbrrt_comparison_experiment.hpp"

#include <algorithm>
#include <array>
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
#include "fbsde_traj_opt/fbrrt/taylor_backward_pass.hpp"
#include "fbsde_traj_opt/solvers/dt_fbsde_iterative_solver.hpp"
#include "fbsde_traj_opt/solvers/fbrrt_solver.hpp"
#include "fbsde_traj_opt/utils/result.hpp"
#include "fbsde_traj_opt/value_function/polynomial_value_function_approx.hpp"
#include "fbsde_traj_opt/value_function/soft_min_quadratic_value_function_approx.hpp"
#include "examples/fbrrt_comparison_setup.hpp"
#include "examples/l1_benchmark_models.hpp"

namespace fbsde_traj_opt::examples {
namespace {

using Clock = std::chrono::steady_clock;
using Control = Eigen::Matrix<double, 1, 1>;

constexpr double kInfinity = std::numeric_limits<double>::infinity();

template <typename SetupT, typename BenchmarkT>
auto RunFbrrt(const BenchmarkT& benchmark,
              const StartingTable<typename SetupT::ValueFunction>& table,
              const FbrrtComparisonOptions& options,
              double keep_fraction,
              std::size_t iterations,
              std::uint64_t seed) noexcept -> Result<MethodRun> {
  RESULT_ASSIGN_OR_RETURN(
      const std::unique_ptr<typename SetupT::Fbrrt> solver,
      MakeComparisonFbrrtSolver<SetupT>(benchmark, table, options, options.fbrrt, keep_fraction, seed));

  MethodRun run;
  run.costs.push_back(solver->Evaluate().total);
  run.seconds.push_back(0.0);
  for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
    RESULT_ASSIGN_OR_RETURN(const FbrrtIterationReport<double> report, solver->Iterate());
    run.costs.push_back(report.mean_cost);
    const double evaluation = options.fbrrt.line_search ? report.evaluate_seconds : 0.0;
    const Connectivity connectivity =
        ParticleConnectivity(solver->particles(), benchmark.region_lower, benchmark.region_upper);
    run.disconnected_fractions.push_back(connectivity.disconnected);
    run.worst_stage_disconnected_fractions.push_back(connectivity.worst_stage_disconnected);
    run.orphan_fractions.push_back(connectivity.orphaned);
    run.seconds.push_back(run.seconds.back() + report.expand_seconds + report.backward_seconds + evaluation +
                          report.erode_seconds);
  }
  return SuccessResult(std::move(run));
}

template <typename SetupT, typename BenchmarkT>
auto RunDtFbsde(const BenchmarkT& benchmark,
                const StartingTable<typename SetupT::ValueFunction>& table,
                const FbrrtComparisonOptions& options,
                std::size_t iterations,
                std::uint64_t seed) noexcept -> Result<MethodRun> {
  auto solver = std::make_unique<Result<typename SetupT::DtFbsde>>(SetupT::DtFbsde::Make(benchmark.forward_model,
                                                                                         benchmark.running_cost,
                                                                                         benchmark.terminal_cost,
                                                                                         benchmark.initial_distribution,
                                                                                         Control(-1.0),
                                                                                         Control(1.0),
                                                                                         table.initial,
                                                                                         table.terminal,
                                                                                         DtFbsdeOptions(options),
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
  std::vector<double> disconnected;
  std::vector<double> worst;
  std::vector<double> orphaned;
  for (const MethodRun& run : result.runs) {
    if (!run.disconnected_fractions.empty()) {
      disconnected.push_back(run.disconnected_fractions.back());
      worst.push_back(run.worst_stage_disconnected_fractions.back());
      orphaned.push_back(run.orphan_fractions.back());
    }
  }
  result.median_disconnected_fraction = disconnected.empty() ? 0.0 : Median(disconnected);
  result.median_worst_stage_disconnected_fraction = worst.empty() ? 0.0 : Median(worst);
  result.median_orphan_fraction = orphaned.empty() ? 0.0 : Median(orphaned);
  result.fraction_converged = static_cast<double>(converged) / static_cast<double>(result.runs.size());
}

template <typename SetupT, typename BenchmarkT>
auto Compare(const BenchmarkT& benchmark,
             const FbrrtComparisonOptions& options,
             double keep_fraction,
             std::size_t iterations,
             const FbrrtComparisonProgress& progress) noexcept -> Result<BenchmarkComparison> {
  RESULT_ASSIGN_OR_RETURN(
      const StartingTable<typename SetupT::ValueFunction> table,
      MakeStartingTable(benchmark, options, static_cast<const typename SetupT::ValueFunction*>(nullptr)));
  BenchmarkComparison comparison;
  comparison.name = benchmark.name;
  comparison.description = benchmark.description;
  comparison.fbrrt.method = "FBRRT";
  comparison.dt_fbsde.method = "DT-FBSDE";

  for (std::size_t index = 0; index < options.num_seeds; ++index) {
    const std::uint64_t seed = options.seed + (1000U * static_cast<std::uint64_t>(index));
    RESULT_ASSIGN_OR_RETURN(MethodRun fbrrt,
                            RunFbrrt<SetupT>(benchmark, table, options, keep_fraction, iterations, seed));
    if (progress) {
      progress(benchmark.name, comparison.fbrrt.method, index, fbrrt);
    }
    comparison.fbrrt.runs.push_back(std::move(fbrrt));

    RESULT_ASSIGN_OR_RETURN(MethodRun dt_fbsde, RunDtFbsde<SetupT>(benchmark, table, options, iterations, seed));
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

namespace {

// The comparison on every benchmark, with value functions `Model<N, Small>` on the double
// integrator and `Model<N, Large>` on the pendulums: a polynomial's degree, or a soft minimum's
// component count.
template <template <int, int, typename> class Model, int Small, int Large>
auto CompareAll(const FbrrtComparisonOptions& options, const FbrrtComparisonProgress& progress) noexcept
    -> Result<std::vector<BenchmarkComparison>> {
  std::vector<BenchmarkComparison> comparisons;

  RESULT_ASSIGN_OR_RETURN(const L1DoubleIntegrator integrator, MakeL1DoubleIntegrator());
  RESULT_ASSIGN_OR_RETURN(
      BenchmarkComparison integrator_comparison,
      (Compare<Setup<L1DoubleIntegrator, 1024, kL1DoubleIntegratorNumStages, Model<2, Small, double>>>(
          integrator, options, options.keep_fraction, options.iterations, progress)));
  comparisons.push_back(std::move(integrator_comparison));

  RESULT_ASSIGN_OR_RETURN(const L1Pendulum pendulum, MakeL1Pendulum());
  RESULT_ASSIGN_OR_RETURN(BenchmarkComparison pendulum_comparison,
                          (Compare<Setup<L1Pendulum, 1024, kL1PendulumNumStages, Model<2, Large, double>>>(
                              pendulum, options, options.keep_fraction, options.iterations, progress)));
  comparisons.push_back(std::move(pendulum_comparison));

  if (options.include_double_inverted_pendulum) {
    RESULT_ASSIGN_OR_RETURN(const L1DoubleInvertedPendulum double_pendulum, MakeL1DoubleInvertedPendulum());
    RESULT_ASSIGN_OR_RETURN(
        BenchmarkComparison double_pendulum_comparison,
        (Compare<Setup<L1DoubleInvertedPendulum, 4096, kL1DoubleInvertedPendulumNumStages, Model<4, Large, double>>>(
            double_pendulum,
            options,
            options.double_inverted_pendulum_keep_fraction,
            options.double_inverted_pendulum_iterations,
            progress)));
    comparisons.push_back(std::move(double_pendulum_comparison));
  }
  return SuccessResult(std::move(comparisons));
}

}  // namespace

auto RunFbrrtComparison(const FbrrtComparisonOptions& options, const FbrrtComparisonProgress& progress) noexcept
    -> Result<std::vector<BenchmarkComparison>> {
  RESULT_ASSERT(options.num_seeds >= 1 && options.iterations >= 1,
                "RunFbrrtComparison: the comparison needs at least one seed and one iteration.");
  if (options.model == ComparisonValueFunctionModel::kPolynomial) {
    return CompareAll<PolynomialValueFunctionApprox, 2, 4>(options, progress);
  }
  if (options.model == ComparisonValueFunctionModel::kNonnegativeSoftMin) {
    return CompareAll<NonnegativeSoftMinQuadraticValueFunctionApprox, kSoftMinComponents, kSoftMinComponents>(options,
                                                                                                              progress);
  }
  return CompareAll<SoftMinQuadraticValueFunctionApprox, kSoftMinComponents, kSoftMinComponents>(options, progress);
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
    if (method == &comparison.fbrrt) {
      summary += std::format(
          "  {:<8} last iteration: {:.1f}% of particles disconnected from the stage before ({:.1f}% "
          "in the worst stage), {:.1f}% orphaned\n",
          "",
          100.0 * method->median_disconnected_fraction,
          100.0 * method->median_worst_stage_disconnected_fraction,
          100.0 * method->median_orphan_fraction);
    }
  }
  return summary;
}

}  // namespace fbsde_traj_opt::examples
