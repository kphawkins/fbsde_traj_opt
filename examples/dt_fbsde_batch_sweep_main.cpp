// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

// Sweeps the DT-FBSDE iterative method's batch size on the two 2-D L1 benchmarks, and measures
// each configuration against a grid dynamic-programming optimum: how long it takes to settle, and
// whether what it settles on is the optimum.
//
//   bazel run -c opt //examples:dt_fbsde_batch_sweep
//   bazel run -c opt //examples:dt_fbsde_batch_sweep -- --batch-sizes 16,64 --line-search-samples 64
//
// Every run is given the same budget of forward samples, so a smaller batch runs proportionally
// more iterations: more frequent backward passes and policy updates on less data each. The
// solver is the comparison's DT-FBSDE (fbrrt_comparison_setup.hpp) with only the batch -- and, if
// asked, the line search's rollout count -- changed.
//
// Costs are measured on held-out rollouts: noise the line search never sees, on more rollouts than
// it uses. The line search accepts a step only if it lowers the cost on its own fixed rollouts, and
// a run of thousands of iterations could otherwise improve on those rollouts by fitting their noise.
// The grid policy is scored on the same held-out rollouts, seed by seed, so a gap between them is a
// gap between policies.
//
// The time charged to a run is the time inside Iterate(); held-out evaluation is not charged.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <format>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include "fbsde_traj_opt/sampling/counter_based_normal_sampler.hpp"
#include "fbsde_traj_opt/solvers/dt_fbsde_iterative_solver.hpp"
#include "fbsde_traj_opt/solvers/policy_rollout_evaluation.hpp"
#include "fbsde_traj_opt/utils/result.hpp"
#include "fbsde_traj_opt/value_function/polynomial_value_function_approx.hpp"
#include "fbsde_traj_opt/value_function/soft_min_quadratic_value_function_approx.hpp"
#include "examples/fbrrt_comparison_experiment.hpp"
#include "examples/fbrrt_comparison_setup.hpp"
#include "examples/l1_benchmark_models.hpp"
#include "examples/l1_grid_dp_reference.hpp"

namespace fbsde_traj_opt::examples {
namespace {

using Clock = std::chrono::steady_clock;
using Control = Eigen::Matrix<double, 1, 1>;

constexpr double kInfinity = std::numeric_limits<double>::infinity();

// Held-out rollouts draw from their own stream, apart from the line search's.
constexpr std::uint64_t kHeldOutStream = 0x8CB92BA72F3D8DD7ULL;

// The batch sizes and line-search rollout counts compiled in.
constexpr std::array<std::size_t, 6> kBatchSizes = {16, 32, 64, 128, 256, 1024};
constexpr std::array<std::size_t, 2> kLineSearchSamples = {64, 256};

// The polynomial model's degree, when it is the one swept: the degree that matched the optimum best
// in the pendulum diagnosis (l1_pendulum_gap_diagnosis_main.cpp).
constexpr int kPolynomialDegree = 6;

struct SweepOptions {
  std::uint64_t seed = 20260928;
  std::size_t num_seeds = 5;
  std::vector<std::size_t> batch_sizes = {16, 32, 64, 128, 1024};
  std::size_t line_search_samples = 256;
  // Forward samples per run: 30 iterations at the comparison's 1024.
  std::size_t sample_budget = 30 * 1024;
  std::size_t max_iterations = 4000;
  // If positive, run each configuration for this many charged seconds instead of a sample budget.
  double time_budget = 0.0;
  // Hold the exploration schedule fixed in samples rather than in iterations: decay^(B / 1024) per
  // iteration, so a batch of B explores for as many samples as a batch of 1024 does.
  bool scale_exploration = false;
  // The solver's line-search and diffusion-inflation options (DtFbsdeIterativeOptions).
  bool stop_uphill_line_search = false;
  std::size_t screening_rollouts = 0;
  double inflation_growth = 1.0;
  double base_inflation = 1.0;
  double max_inflation = 8.0;
  double stall_threshold = 0.0;
  std::size_t stall_patience = 1;
  double uninflated_fraction = 0.0;
  // The slope-matching weight (DtFbsdeIterativeOptions::slope_weight); zero fits values alone.
  double slope_weight = 0.0;
  // Fit the polynomial model of degree kPolynomialDegree rather than the soft-min one.
  bool polynomial = false;
  // Annealed exploration: the inflation's base falls geometrically from `anneal_start` to one over
  // the first `anneal_fraction` of the time budget. One disables it.
  double anneal_start = 1.0;
  double anneal_fraction = 0.5;
  // Diffusion continuation: the problem's diffusion scale falls the same way from `homotopy_start`
  // to one over the first `homotopy_fraction` of the time budget, in steps of at least 5%.
  double homotopy_start = 1.0;
  double homotopy_fraction = 0.5;
  std::size_t held_out_rollouts = 1024;
  std::size_t checkpoints = 100;
  double tolerance = 0.05;
  bool include_integrator = true;
  bool include_pendulum = true;
  std::size_t dp_cells = 400;
  std::size_t dp_controls = 41;
  std::size_t dp_quadrature = 3;
  std::string csv_path = "dt_fbsde_batch_sweep.csv";
  FbrrtComparisonOptions comparison{};
};

// One run's trace: held-out cost at each checkpoint, and where the time went.
struct Run {
  std::size_t batch = 0;
  std::size_t seed_index = 0;
  std::vector<std::size_t> iterations{};
  std::vector<double> seconds{};
  std::vector<double> held_out{};
  std::vector<double> line_search_cost{};
  double forward_seconds = 0.0;
  double backward_seconds = 0.0;
  double line_search_seconds = 0.0;
  std::size_t total_iterations = 0;
  std::size_t accepted = 0;
  std::size_t candidates = 0;
  std::size_t inflated = 0;
  double inflation_sum = 0.0;
};

auto Median(std::vector<double> values) noexcept -> double {
  if (values.empty()) {
    return kInfinity;
  }
  std::ranges::sort(values);
  const std::size_t middle = values.size() / 2;
  return values.size() % 2 == 1 ? values[middle] : 0.5 * (values[middle - 1] + values[middle]);
}

template <typename BenchmarkT, typename PolicyT>
auto HeldOutCost(const BenchmarkT& benchmark,
                 const PolicyT& policy,
                 std::size_t num_stages,
                 std::size_t rollouts,
                 std::uint64_t seed) noexcept -> double {
  using StateMat = Eigen::Matrix<double, BenchmarkT::State::RowsAtCompileTime, BenchmarkT::State::RowsAtCompileTime>;
  const StateMat factor = Eigen::LLT<StateMat>(benchmark.initial_distribution.Covariance()).matrixL();
  return EvaluatePolicyByRollouts(benchmark.forward_model,
                                  policy,
                                  benchmark.running_cost,
                                  benchmark.terminal_cost,
                                  typename BenchmarkT::State(benchmark.initial_distribution.Mean()),
                                  factor,
                                  rollouts,
                                  num_stages - 1,
                                  seed ^ kHeldOutStream)
      .total;
}

template <typename BenchmarkT,
          std::size_t NumStages,
          typename ValueFunction,
          std::size_t Batch,
          std::size_t LineSearchSamples>
auto RunOne(const BenchmarkT& benchmark,
            const StartingTable<ValueFunction>& table,
            const SweepOptions& options,
            std::size_t seed_index) noexcept -> Result<Run> {
  using Solver = DtFbsdeIterativeSolver<2,
                                        1,
                                        Batch,
                                        NumStages,
                                        LineSearchSamples,
                                        typename BenchmarkT::ForwardModel,
                                        QuadraticRegulatorSdeTerminalCostTerm<2>,
                                        ValueFunction>;
  const std::uint64_t seed = options.seed + (1000U * static_cast<std::uint64_t>(seed_index));
  DtFbsdeIterativeOptions<double> solver_options = DtFbsdeOptions(options.comparison);
  if (options.scale_exploration) {
    solver_options.exploration_decay = std::pow(solver_options.exploration_decay, static_cast<double>(Batch) / 1024.0);
  }
  solver_options.stop_uphill_line_search = options.stop_uphill_line_search;
  solver_options.screening_rollouts = std::min(options.screening_rollouts, LineSearchSamples);
  solver_options.diffusion_inflation_growth = options.inflation_growth;
  solver_options.base_diffusion_inflation = options.base_inflation;
  solver_options.max_diffusion_inflation = options.max_inflation;
  solver_options.stall_relative_improvement = options.stall_threshold;
  solver_options.stall_patience = options.stall_patience;
  solver_options.uninflated_path_fraction = options.uninflated_fraction;
  solver_options.slope_weight = options.slope_weight;
  RESULT_ASSERT((options.anneal_start <= 1.0 && options.homotopy_start <= 1.0) || options.time_budget > 0.0,
                "the annealing and continuation schedules run on the time budget; pass --time-budget.");
  auto solver = std::make_unique<Result<Solver>>(Solver::Make(benchmark.forward_model,
                                                              benchmark.running_cost,
                                                              benchmark.terminal_cost,
                                                              benchmark.initial_distribution,
                                                              Control(-1.0),
                                                              Control(1.0),
                                                              table.initial,
                                                              table.terminal,
                                                              solver_options,
                                                              seed));
  if (!solver->has_value()) {
    return ErrorResult(solver->error().message, solver->error().location);
  }
  Solver& dt = **solver;
  if (options.homotopy_start > 1.0) {
    RESULT_RETURN_IF_ERROR(dt.SetDiffusionScale(options.homotopy_start));
  }

  const bool timed = options.time_budget > 0.0;
  const std::size_t iterations =
      timed ? options.max_iterations
            : std::clamp<std::size_t>(options.sample_budget / Batch, 1, options.max_iterations);
  const std::size_t interval = std::max<std::size_t>(1, iterations / options.checkpoints);
  const double checkpoint_seconds = options.time_budget / static_cast<double>(options.checkpoints);

  Run run;
  run.batch = Batch;
  run.seed_index = seed_index;
  double charged = 0.0;
  double next_checkpoint = checkpoint_seconds;
  const auto checkpoint = [&](std::size_t iteration) noexcept {
    run.iterations.push_back(iteration);
    run.seconds.push_back(charged);
    run.held_out.push_back(HeldOutCost(benchmark, dt.CurrentPolicy(), NumStages, options.held_out_rollouts, seed));
    run.line_search_cost.push_back(dt.CurrentCost());
  };
  checkpoint(0);
  // A schedule's multiplier at charged time `t`: `start` falling geometrically to one at `fraction`
  // of the budget.
  const auto scheduled = [&options](double start, double fraction, double t) noexcept {
    return std::pow(start, std::max(0.0, 1.0 - (t / (fraction * options.time_budget))));
  };
  for (std::size_t iteration = 1; iteration <= iterations; ++iteration) {
    const Clock::time_point start = Clock::now();
    if (options.anneal_start > 1.0) {
      RESULT_RETURN_IF_ERROR(
          dt.SetExplorationInflation(scheduled(options.anneal_start, options.anneal_fraction, charged)));
    }
    if (options.homotopy_start > 1.0) {
      const double scale = scheduled(options.homotopy_start, options.homotopy_fraction, charged);
      if (scale == 1.0 || scale <= dt.DiffusionScale() / 1.05) {
        RESULT_RETURN_IF_ERROR(dt.SetDiffusionScale(scale));
      }
    }
    RESULT_ASSIGN_OR_RETURN(const DtFbsdeIterationReport<double> report, dt.Iterate());
    charged += std::chrono::duration<double>(Clock::now() - start).count();
    run.forward_seconds += report.forward_seconds;
    run.backward_seconds += report.backward_seconds;
    run.line_search_seconds += report.line_search_seconds;
    run.accepted += report.accepted_step > 0.0 ? 1 : 0;
    run.candidates += report.candidates_tried;
    run.inflated += report.diffusion_inflation > 1.0 ? 1 : 0;
    run.inflation_sum += report.diffusion_inflation;
    run.total_iterations = iteration;
    const bool done = iteration == iterations || (timed && charged >= options.time_budget);
    if (done || (timed ? charged >= next_checkpoint : iteration % interval == 0)) {
      checkpoint(iteration);
      while (next_checkpoint <= charged) {
        next_checkpoint += checkpoint_seconds;
      }
    }
    if (done) {
      break;
    }
  }
  return SuccessResult(std::move(run));
}

// Dispatches a runtime batch size and line-search count to their compiled instantiations.
template <typename BenchmarkT,
          std::size_t NumStages,
          typename ValueFunction,
          std::size_t LineSearchSamples,
          std::size_t... Indices>
auto DispatchBatch(const BenchmarkT& benchmark,
                   const StartingTable<ValueFunction>& table,
                   const SweepOptions& options,
                   std::size_t batch,
                   std::size_t seed_index,
                   std::index_sequence<Indices...> /*indices*/) noexcept -> Result<Run> {
  Result<Run> result = ErrorResult("unsupported batch size; the compiled ones are 16, 32, 64, 128, 256, and 1024.");
  (void)((batch == kBatchSizes[Indices] &&
          (result = RunOne<BenchmarkT, NumStages, ValueFunction, kBatchSizes[Indices], LineSearchSamples>(
               benchmark, table, options, seed_index),
           true)) ||
         ...);
  return result;
}

template <typename BenchmarkT, std::size_t NumStages, typename ValueFunction>
auto Dispatch(const BenchmarkT& benchmark,
              const StartingTable<ValueFunction>& table,
              const SweepOptions& options,
              std::size_t batch,
              std::size_t seed_index) noexcept -> Result<Run> {
  constexpr auto kIndices = std::make_index_sequence<kBatchSizes.size()>();
  if (options.line_search_samples == kLineSearchSamples[0]) {
    return DispatchBatch<BenchmarkT, NumStages, ValueFunction, kLineSearchSamples[0]>(
        benchmark, table, options, batch, seed_index, kIndices);
  }
  return DispatchBatch<BenchmarkT, NumStages, ValueFunction, kLineSearchSamples[1]>(
      benchmark, table, options, batch, seed_index, kIndices);
}

// The box the grid reference covers on each benchmark: the FBRRT region, widened, which the optimal
// trajectories stay well inside (the sweep prints the box they actually span).
struct GridBox {
  Eigen::Vector2d lower;
  Eigen::Vector2d upper;
};

// The box that 256 of the grid policy's held-out rollouts span, to check that the grid contains them.
template <typename BenchmarkT, typename PolicyT>
auto TrajectoryBox(const BenchmarkT& benchmark, const PolicyT& policy, std::size_t num_stages, std::uint64_t seed)
    -> GridBox {
  const CounterBasedNormalSampler sampler(seed ^ kHeldOutStream);
  GridBox box{.lower = Eigen::Vector2d::Constant(kInfinity), .upper = Eigen::Vector2d::Constant(-kInfinity)};
  const Eigen::Matrix2d factor = Eigen::LLT<Eigen::Matrix2d>(benchmark.initial_distribution.Covariance()).matrixL();
  for (std::size_t rollout = 0; rollout < 256; ++rollout) {
    Eigen::Vector2d noise;
    sampler.FillNoise(rollout, 0, noise);
    Eigen::Vector2d state = benchmark.initial_distribution.Mean() + (factor * noise);
    for (std::size_t stage = 0; stage + 1 < num_stages; ++stage) {
      box.lower = box.lower.cwiseMin(state);
      box.upper = box.upper.cwiseMax(state);
      sampler.FillNoise(rollout, stage + 1, noise);
      state = benchmark.forward_model(stage, state, policy(stage, state), noise);
    }
    box.lower = box.lower.cwiseMin(state);
    box.upper = box.upper.cwiseMax(state);
  }
  return box;
}

template <std::size_t NumStages, typename BenchmarkT>
auto Sweep(const BenchmarkT& benchmark, const GridBox& grid_box, const SweepOptions& options, std::ofstream& csv)
    -> Result<> {
  std::cout << std::format("\n=== {} ===\n", benchmark.name);

  // The optimum, by dynamic programming on the grid.
  typename L1GridDpReference<BenchmarkT>::Options grid_options;
  grid_options.lower = grid_box.lower;
  grid_options.upper = grid_box.upper;
  grid_options.cells_per_axis = options.dp_cells;
  grid_options.num_controls = options.dp_controls;
  grid_options.quadrature_order = options.dp_quadrature;
  const Clock::time_point grid_start = Clock::now();
  RESULT_ASSIGN_OR_RETURN(const L1GridDpReference<BenchmarkT> grid,
                          L1GridDpReference<BenchmarkT>::Make(benchmark, NumStages, grid_options));
  const double grid_seconds = std::chrono::duration<double>(Clock::now() - grid_start).count();
  const GridBox span = TrajectoryBox(benchmark, grid, NumStages, options.seed);
  std::cout << std::format(
      "grid DP ({}^2 cells, {} controls, {}-point quadrature) in {:.1f} s: V_0(x_0) = {:.4f}\n"
      "  its trajectories span [{:.2f}, {:.2f}] x [{:.2f}, {:.2f}] of the grid's [{:.2f}, {:.2f}] x [{:.2f}, "
      "{:.2f}]\n",
      options.dp_cells,
      options.dp_controls,
      options.dp_quadrature,
      grid_seconds,
      grid.Value(0, benchmark.initial_distribution.Mean()),
      span.lower[0],
      span.upper[0],
      span.lower[1],
      span.upper[1],
      grid_box.lower[0],
      grid_box.upper[0],
      grid_box.lower[1],
      grid_box.upper[1]);

  std::vector<double> optimum(options.num_seeds);
  for (std::size_t index = 0; index < options.num_seeds; ++index) {
    const std::uint64_t seed = options.seed + (1000U * static_cast<std::uint64_t>(index));
    optimum[index] = HeldOutCost(benchmark, grid, NumStages, options.held_out_rollouts, seed);
  }
  std::cout << std::format(
      "grid policy held-out cost: median {:.4f} over {} seeds\n", Median(optimum), options.num_seeds);

  using SoftMin = SoftMinQuadraticValueFunctionApprox<2, kSoftMinComponents>;
  using Polynomial = PolynomialValueFunctionApprox<2, kPolynomialDegree>;
  RESULT_ASSIGN_OR_RETURN(const StartingTable<SoftMin> soft_min_table,
                          MakeStartingTable(benchmark, options.comparison, static_cast<const SoftMin*>(nullptr)));
  RESULT_ASSIGN_OR_RETURN(const StartingTable<Polynomial> polynomial_table,
                          MakeStartingTable(benchmark, options.comparison, static_cast<const Polynomial*>(nullptr)));

  std::cout << std::format(
      "\n{:>5} {:>5} {:>8} {:>7} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>8} {:>8} {:>8} {:>9} {:>9} {:>6}\n",
      "batch",
      "iters",
      "total s",
      "ms/it",
      "fwd%",
      "bwd%",
      "ls%",
      "acc%",
      "cand",
      "infl%",
      "infl",
      "final",
      "best",
      "gap%",
      "settle s",
      "optimal s",
      "reach");
  for (const std::size_t batch : options.batch_sizes) {
    std::vector<Run> runs;
    for (std::size_t index = 0; index < options.num_seeds; ++index) {
      RESULT_ASSIGN_OR_RETURN(
          Run run,
          options.polynomial
              ? (Dispatch<BenchmarkT, NumStages, Polynomial>(benchmark, polynomial_table, options, batch, index))
              : (Dispatch<BenchmarkT, NumStages, SoftMin>(benchmark, soft_min_table, options, batch, index)));
      for (std::size_t point = 0; point < run.iterations.size(); ++point) {
        csv << std::format("{},{},{},{},{:.6f},{:.6f},{:.6f},{:.6f}\n",
                           benchmark.name,
                           batch,
                           index,
                           run.iterations[point],
                           run.seconds[point],
                           run.held_out[point],
                           run.line_search_cost[point],
                           optimum[index]);
      }
      runs.push_back(std::move(run));
    }

    std::vector<double> total;
    std::vector<double> iteration_counts;
    std::vector<double> per_iteration;
    std::vector<double> forward;
    std::vector<double> backward;
    std::vector<double> line_search;
    std::vector<double> accepted;
    std::vector<double> candidates;
    std::vector<double> inflated;
    std::vector<double> inflation;
    std::vector<double> final_cost;
    std::vector<double> best_cost;
    std::vector<double> gap;
    std::vector<double> settle;
    std::vector<double> reach_optimal;
    std::size_t reached = 0;
    for (const Run& run : runs) {
      const double seconds = run.seconds.back();
      const double phases = run.forward_seconds + run.backward_seconds + run.line_search_seconds;
      total.push_back(seconds);
      iteration_counts.push_back(static_cast<double>(run.total_iterations));
      per_iteration.push_back(1e3 * seconds / static_cast<double>(run.total_iterations));
      forward.push_back(100.0 * run.forward_seconds / phases);
      backward.push_back(100.0 * run.backward_seconds / phases);
      line_search.push_back(100.0 * run.line_search_seconds / phases);
      accepted.push_back(100.0 * static_cast<double>(run.accepted) / static_cast<double>(run.total_iterations));
      candidates.push_back(static_cast<double>(run.candidates) / static_cast<double>(run.total_iterations));
      inflated.push_back(100.0 * static_cast<double>(run.inflated) / static_cast<double>(run.total_iterations));
      inflation.push_back(run.inflation_sum / static_cast<double>(run.total_iterations));
      final_cost.push_back(run.held_out.back());
      const double best = *std::ranges::min_element(run.held_out);
      best_cost.push_back(best);
      const double opt = optimum[run.seed_index];
      gap.push_back(100.0 * (run.held_out.back() - opt) / opt);
      const auto settled =
          std::ranges::find_if(run.held_out, [&](double cost) { return cost <= (1.0 + options.tolerance) * best; });
      settle.push_back(run.seconds[static_cast<std::size_t>(settled - run.held_out.begin())]);
      const auto near =
          std::ranges::find_if(run.held_out, [&](double cost) { return cost <= (1.0 + options.tolerance) * opt; });
      if (near == run.held_out.end()) {
        reach_optimal.push_back(kInfinity);
      } else {
        reach_optimal.push_back(run.seconds[static_cast<std::size_t>(near - run.held_out.begin())]);
        ++reached;
      }
    }
    const double reach_median = Median(reach_optimal);
    std::cout << std::format(
        "{:>5} {:>5.0f} {:>8.2f} {:>7.2f} {:>6.1f} {:>6.1f} {:>6.1f} {:>6.1f} {:>6.2f} {:>6.1f} {:>6.2f} {:>8.4f} "
        "{:>8.4f} {:>8.1f} {:>9.2f} {:>9} "
        "{:>3}/{:<2}\n",
        batch,
        Median(iteration_counts),
        Median(total),
        Median(per_iteration),
        Median(forward),
        Median(backward),
        Median(line_search),
        Median(accepted),
        Median(candidates),
        Median(inflated),
        Median(inflation),
        Median(final_cost),
        Median(best_cost),
        Median(gap),
        Median(settle),
        std::isfinite(reach_median) ? std::format("{:.2f}", reach_median) : std::string("never"),
        reached,
        runs.size());
    std::cout.flush();
  }
  std::cout << std::format(
      "\nColumns are medians over seeds. cand: line-search candidates per iteration. infl%: iterations whose\n"
      "forward pass was inflated; infl: their mean inflation factor over all iterations. final/best: held-out cost at "
      "the end and at the best checkpoint.\n"
      "gap%: final cost above the grid optimum on the same rollouts. settle s: charged time until the held-out\n"
      "cost first comes within {0:.0f}% of the run's own best. optimal s: until within {0:.0f}% of the grid\n"
      "optimum (\"never\" unless at least half the runs got there); reach: how many runs did.\n",
      100.0 * options.tolerance);
  return SuccessResult();
}

auto ParseList(std::string_view value) noexcept -> std::vector<std::size_t> {
  std::vector<std::size_t> list;
  std::size_t begin = 0;
  while (begin <= value.size()) {
    const std::size_t end = std::min(value.find(',', begin), value.size());
    list.push_back(std::strtoull(std::string(value.substr(begin, end - begin)).c_str(), nullptr, 10));
    begin = end + 1;
  }
  return list;
}

auto PrintUsage() noexcept -> void {
  std::cout
      << "usage: dt_fbsde_batch_sweep [options]\n"
      << "  --seed N                  the seed all sampling derives from\n"
      << "  --seeds N                 runs per configuration (default 5)\n"
      << "  --batch-sizes L           comma-separated, from 16,32,64,128,256,1024 (default 16,32,64,128,1024)\n"
      << "  --line-search-samples N   rollouts per line-search candidate: 64 or 256 (default 256)\n"
      << "  --sample-budget N         forward samples per run (default 30720 = 30 x 1024)\n"
      << "  --max-iterations N        cap on iterations per run (default 4000)\n"
      << "  --time-budget S           run each configuration for S charged seconds instead\n"
      << "  --scale-exploration       decay exploration per sample rather than per iteration\n"
      << "  --ls-stop-uphill          stop the line search once the update looks uphill\n"
      << "  --ls-screen N             screen line-search candidates on N rollouts first (default 0: off)\n"
      << "  --inflate G               multiply the forward noise by G per stalled iteration (default 1: off)\n"
      << "  --inflate-base B          the inflation it starts at and resets to (default 1)\n"
      << "  --inflate-max M           cap on the noise inflation (default 8)\n"
      << "  --uninflated-fraction F   share of paths always sampled at the problem's own noise (default 0)\n"
      << "  --slope-weight G          also fit value slopes along the control columns, weighted G (default 0: off)\n"
      << "  --model M                 soft-min (default) or polynomial (degree 6)\n"
      << "  --anneal S                anneal the exploration inflation from S down to 1 (default 1: off)\n"
      << "  --anneal-fraction F       ...over this fraction of the time budget (default 0.5)\n"
      << "  --homotopy S              solve with S times the diffusion, walked down to 1 (default 1: off)\n"
      << "  --homotopy-fraction F     ...over this fraction of the time budget (default 0.5)\n"
      << "  --stall-threshold T       an iteration stalls unless the cost falls by more than T of itself (default 0)\n"
      << "  --stall-patience N        stalled iterations in a row before inflating (default 1)\n"
      << "  --dt-epochs N             SGD epochs per stage per fit (default 50)\n"
      << "  --dt-learning-rate R      SGD learning rate (default 0.05)\n"
      << "  --held-out N              held-out rollouts per checkpoint (default 1024)\n"
      << "  --checkpoints N           held-out checkpoints per run (default 100)\n"
      << "  --only integrator|pendulum\n"
      << "  --dp-cells N              grid cells per axis (default 400)\n"
      << "  --dp-controls N           grid controls, odd (default 41)\n"
      << "  --dp-quadrature N         Gauss-Hermite points per noise axis: 3 or 5 (default 3)\n"
      << "  --csv PATH                where to write every checkpoint (default dt_fbsde_batch_sweep.csv)\n";
}

auto ParseOptions(int argc, char** argv, SweepOptions& options) noexcept -> Result<> {
  const std::vector<std::string_view> arguments(argv + 1, argv + argc);
  for (std::size_t index = 0; index < arguments.size(); ++index) {
    const std::string_view argument = arguments[index];
    if (argument == "--scale-exploration") {
      options.scale_exploration = true;
      continue;
    }
    if (argument == "--ls-stop-uphill") {
      options.stop_uphill_line_search = true;
      continue;
    }
    RESULT_ASSERT(index + 1 < arguments.size(), "every option but the flags needs a value.");
    const std::string value(arguments[++index]);
    const auto count = static_cast<std::size_t>(std::strtoull(value.c_str(), nullptr, 10));
    if (argument == "--seed") {
      options.seed = std::strtoull(value.c_str(), nullptr, 10);
    } else if (argument == "--seeds") {
      options.num_seeds = count;
    } else if (argument == "--batch-sizes") {
      options.batch_sizes = ParseList(value);
      for (const std::size_t batch : options.batch_sizes) {
        RESULT_ASSERT(std::ranges::find(kBatchSizes, batch) != kBatchSizes.end(),
                      "--batch-sizes takes values from 16, 32, 64, 128, 256, 1024.");
      }
    } else if (argument == "--line-search-samples") {
      RESULT_ASSERT(std::ranges::find(kLineSearchSamples, count) != kLineSearchSamples.end(),
                    "--line-search-samples is 64 or 256.");
      options.line_search_samples = count;
    } else if (argument == "--sample-budget") {
      options.sample_budget = count;
    } else if (argument == "--ls-screen") {
      options.screening_rollouts = count;
    } else if (argument == "--inflate") {
      options.inflation_growth = std::strtod(value.c_str(), nullptr);
    } else if (argument == "--inflate-base") {
      options.base_inflation = std::strtod(value.c_str(), nullptr);
    } else if (argument == "--slope-weight") {
      options.slope_weight = std::strtod(value.c_str(), nullptr);
    } else if (argument == "--model") {
      RESULT_ASSERT(value == "soft-min" || value == "polynomial", "--model is soft-min or polynomial.");
      options.polynomial = value == "polynomial";
    } else if (argument == "--uninflated-fraction") {
      options.uninflated_fraction = std::strtod(value.c_str(), nullptr);
    } else if (argument == "--anneal") {
      options.anneal_start = std::strtod(value.c_str(), nullptr);
    } else if (argument == "--anneal-fraction") {
      options.anneal_fraction = std::strtod(value.c_str(), nullptr);
    } else if (argument == "--homotopy") {
      options.homotopy_start = std::strtod(value.c_str(), nullptr);
    } else if (argument == "--homotopy-fraction") {
      options.homotopy_fraction = std::strtod(value.c_str(), nullptr);
    } else if (argument == "--inflate-max") {
      options.max_inflation = std::strtod(value.c_str(), nullptr);
    } else if (argument == "--stall-threshold") {
      options.stall_threshold = std::strtod(value.c_str(), nullptr);
    } else if (argument == "--stall-patience") {
      options.stall_patience = count;
    } else if (argument == "--time-budget") {
      options.time_budget = std::strtod(value.c_str(), nullptr);
    } else if (argument == "--max-iterations") {
      options.max_iterations = count;
    } else if (argument == "--dt-epochs") {
      options.comparison.dt_fbsde_sgd_epochs = count;
    } else if (argument == "--dt-learning-rate") {
      options.comparison.dt_fbsde_sgd.learning_rate = std::strtod(value.c_str(), nullptr);
    } else if (argument == "--held-out") {
      options.held_out_rollouts = count;
    } else if (argument == "--checkpoints") {
      options.checkpoints = std::max<std::size_t>(count, 1);
    } else if (argument == "--only") {
      RESULT_ASSERT(value == "integrator" || value == "pendulum", "--only is integrator or pendulum.");
      options.include_integrator = value == "integrator";
      options.include_pendulum = value == "pendulum";
    } else if (argument == "--dp-cells") {
      options.dp_cells = count;
    } else if (argument == "--dp-controls") {
      options.dp_controls = count;
    } else if (argument == "--dp-quadrature") {
      options.dp_quadrature = count;
    } else if (argument == "--csv") {
      options.csv_path = value;
    } else {
      return ErrorResult("unrecognized argument; run with --help for the options.");
    }
  }
  return SuccessResult();
}

auto Run(int argc, char** argv) -> Result<> {
  SweepOptions options;
  RESULT_RETURN_IF_ERROR(ParseOptions(argc, argv, options));
  std::ofstream csv(options.csv_path);
  RESULT_ASSERT(csv.good(), "could not open the CSV file for writing.");
  csv << "benchmark,batch,seed,iteration,seconds,held_out_cost,line_search_cost,grid_optimum\n";
  std::cout << std::format(
      "DT-FBSDE batch sweep: {} seeds, {} per run, {} line-search rollouts, {} held-out rollouts, "
      "exploration decay per {}; line search{}{}; inflation from {:g}, x{:g} up to {:g}, stall below {:g} for {}, "
      "{:g} of paths uninflated; anneal {:g} over {:g}; homotopy {:g} over {:g}; {} model, slope weight {:g}\n",
      options.num_seeds,
      options.time_budget > 0.0 ? std::format("{:.0f} charged seconds", options.time_budget)
                                : std::format("{} forward samples", options.sample_budget),
      options.line_search_samples,
      options.held_out_rollouts,
      options.scale_exploration ? "sample" : "iteration",
      options.stop_uphill_line_search ? " stops uphill" : "",
      options.screening_rollouts > 0 ? std::format(" screens on {}", options.screening_rollouts) : std::string(),
      options.base_inflation,
      options.inflation_growth,
      options.max_inflation,
      options.stall_threshold,
      options.stall_patience,
      options.uninflated_fraction,
      options.anneal_start,
      options.anneal_fraction,
      options.homotopy_start,
      options.homotopy_fraction,
      options.polynomial ? std::format("polynomial degree {}", kPolynomialDegree) : std::string("soft-min"),
      options.slope_weight);

  if (options.include_integrator) {
    RESULT_ASSIGN_OR_RETURN(const L1DoubleIntegrator integrator, MakeL1DoubleIntegrator());
    RESULT_RETURN_IF_ERROR((Sweep<kL1DoubleIntegratorNumStages>(
        integrator, GridBox{.lower = {-2.5, -3.5}, .upper = {4.5, 2.5}}, options, csv)));
  }
  if (options.include_pendulum) {
    RESULT_ASSIGN_OR_RETURN(const L1Pendulum pendulum, MakeL1Pendulum());
    RESULT_RETURN_IF_ERROR(
        (Sweep<kL1PendulumNumStages>(pendulum, GridBox{.lower = {-2.0, -9.0}, .upper = {5.0, 9.0}}, options, csv)));
  }
  std::cout << std::format("\nEvery checkpoint is in {}\n", options.csv_path);
  return SuccessResult();
}

}  // namespace
}  // namespace fbsde_traj_opt::examples

auto main(int argc, char** argv) -> int {
  for (int index = 1; index < argc; ++index) {
    if (std::string_view(argv[index]) == "--help") {
      fbsde_traj_opt::examples::PrintUsage();
      return EXIT_SUCCESS;
    }
  }
  const fbsde_traj_opt::Result<> result = fbsde_traj_opt::examples::Run(argc, argv);
  if (!result.has_value()) {
    RESULT_REPORT_RESULT(result);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
