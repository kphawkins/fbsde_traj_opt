// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

// Compares the two FBSDE methods -- DT-FBSDE and FBRRT -- with two standard trajectory optimizers
// from outside that family -- box-constrained iLQR and MPPI -- on the three L1 benchmarks
// (l1_benchmark_models.hpp): how fast each finds a near-optimal policy, how close it gets, and how
// reliably, over several seeds.
//
//   bazel run -c opt //examples:method_comparison
//   bazel run -c opt //examples:method_comparison -- --seeds 3 --only pendulum --methods ilqr,mppi
//
// Every method is given the same budget of charged solver time per run, and every policy is scored
// on the same held-out rollouts: one fixed set of noise, shared by all methods, seeds, and
// checkpoints, which no solver ever samples. On the two 2-D benchmarks the grid dynamic-programming
// policy (l1_grid_dp_reference.hpp) is scored on the same rollouts as the optimum; the double
// inverted pendulum has no such reference, and the analysis takes the best any method reaches.
//
// The methods, and what is charged:
//
//   * DT-FBSDE: DtFbsdeIterativeSolver with the batch sweep's recipe (dt_fbsde_batch_sweep_main.cpp)
//     on every benchmark: a degree-4 polynomial value function, the off-policy Taylor noiseless
//     backward target, half the paths at six times the noise, and a screened, stop-uphill line
//     search; batches of 16 on the 2-D benchmarks and 256 on the double inverted pendulum, where the
//     line search starts at a half step.
//     Charged: Iterate().
//   * DT-FBSDE+iLQR: DT-FBSDE warm-started from iLQR. The run begins as that seed's iLQR run does,
//     from the same random controls. At iLQR iterations spaced geometrically by --seed-growth, and at
//     the last, iLQR's quadratic value expansions are turned into a candidate value function table
//     (ilqr_seeded_table.hpp), scored by the DT-FBSDE solver on its own evaluation rollouts -- never
//     the held-out ones -- and installed if it is the best candidate yet. iLQR's best iterates are not
//     its last on the pendulums, which is why there is more than one candidate. iLQR stops once it
//     converges or has used --seed-ilqr-budget; DT-FBSDE then iterates from the installed table for
//     the rest of the budget, sampling around the seeded policy from the first forward pass (its
//     initial exploration probability is the regular one, not one). Charged: the iLQR iterations,
//     building and scoring every candidate, and Iterate(). Its iteration count is iLQR's iterations
//     followed by DT-FBSDE's.
//   * FBRRT: FbrrtSolver as the FBRRT comparison configures it. Charged: Iterate().
//   * iLQR: BoxIlqrSolver from random initial controls, one draw per seed, plus one run from zero
//     controls; scored as the closed-loop policy that tracks its nominal trajectory with its gains.
//     Charged: Iterate(). A run stops when the solver reports convergence.
//   * MPPI: MppiSolver from zero controls, scored twice at every checkpoint -- as the open-loop
//     sequence it optimizes ("MPPI"), and with tracking gains from one iLQR backward pass about its
//     noiseless trajectory ("MPPI+LQR"). Charged: Iterate(); the gains take milliseconds and are not
//     charged.
//   * iLQR-MPC and MPPI-MPC: receding-horizon iLQR and MPPI, warm-started from each seed's offline
//     solution and re-planned at every stage of every held-out rollout. They are not offline solvers,
//     so they have no convergence trace; each is reported as a cost on the first --mpc-rollouts
//     held-out rollouts and an online compute time per control step.
//
// Output: every checkpoint to --csv, the MPC results to --mpc-csv, and the grid optima to
// --reference-csv, for analysis elsewhere.

#include <algorithm>
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
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include "fbsde_traj_opt/costs/quadratic_regulator_sde_terminal_cost_term.hpp"
#include "fbsde_traj_opt/sampling/counter_based_normal_sampler.hpp"
#include "fbsde_traj_opt/solvers/box_ilqr_solver.hpp"
#include "fbsde_traj_opt/solvers/dt_fbsde_iterative_solver.hpp"
#include "fbsde_traj_opt/solvers/mppi_solver.hpp"
#include "fbsde_traj_opt/solvers/policy_rollout_evaluation.hpp"
#include "fbsde_traj_opt/utils/parallel_for.hpp"
#include "fbsde_traj_opt/utils/phase_profiler.hpp"
#include "fbsde_traj_opt/utils/result.hpp"
#include "fbsde_traj_opt/value_function/polynomial_value_function_approx.hpp"
#include "fbsde_traj_opt/value_function/soft_min_quadratic_value_function_approx.hpp"
#include "examples/fbrrt_comparison_experiment.hpp"
#include "examples/fbrrt_comparison_setup.hpp"
#include "examples/ilqr_seeded_table.hpp"
#include "examples/l1_benchmark_models.hpp"
#include "examples/l1_grid_dp_reference.hpp"

namespace fbsde_traj_opt::examples {
namespace {

using Clock = std::chrono::steady_clock;
using Control = Eigen::Matrix<double, 1, 1>;

// The held-out rollouts' noise: one fixed seed, apart from every solver's.
constexpr std::uint64_t kHeldOutSeed = 0x5851F42D4C957F2DULL;
constexpr std::uint64_t kInitialControlStream = 0x14057B7EF767814FULL;

struct DriverOptions {
  std::uint64_t seed = 20260928;
  std::size_t num_seeds = 8;
  double time_budget = 60.0;
  double double_pendulum_time_budget = 120.0;
  std::size_t max_iterations = 1000000;
  std::size_t held_out_rollouts = 1024;
  // Checkpoints fall at geometrically spaced charged times, from `first_checkpoint` seconds on.
  double first_checkpoint = 1e-3;
  double checkpoint_growth = 1.15;

  bool include_integrator = true;
  bool include_pendulum = true;
  bool include_double_pendulum = true;
  bool run_dt_fbsde = true;
  bool run_seeded_dt_fbsde = true;
  bool run_fbrrt = true;
  bool run_ilqr = true;
  bool run_mppi = true;

  // iLQR: the spread of the random initial controls, the smoothing floor, and the smoothing the
  // tracking gains are computed at (0: the solver's own).
  double ilqr_initial_spread = 0.5;
  BoxIlqrOptions<double> ilqr{};
  // A pilot sweep of 0 (the solver's own, 1e-3), 0.03, 0.1, 0.3, 1, and 3 found 0.3 to 1 best on both
  // 2-D benchmarks: at the solve's own smoothing the gains vanish wherever the plan coasts.
  double ilqr_gain_smoothing = 1.0;

  // The pilot's choice: noiseless samples (sampling the problem's noise made the cost ranking noisier
  // and the result worse on both 2-D benchmarks), 256 samples, and the control noise annealed.
  MppiOptions<double> mppi = [] {
    MppiOptions<double> defaults;
    defaults.num_samples = 256;
    defaults.control_noise = 0.5;
    defaults.noise_decay = 0.99;
    defaults.min_control_noise = 0.02;
    defaults.effective_sample_fraction = 0.05;
    defaults.sample_process_noise = false;
    return defaults;
  }();
  // The smoothing the MPPI+LQR tracking gains are computed at.
  double mppi_gain_smoothing = 1.0;

  // Receding-horizon MPPI: on how many seeds and held-out rollouts, and its per-step effort.
  std::size_t mpc_seeds = 3;
  std::size_t mpc_rollouts = 128;
  std::size_t mpc_samples = 256;
  std::size_t mpc_iterations = 2;
  double mpc_control_noise = 0.1;
  std::size_t mpc_ilqr_iterations = 2;

  // DT-FBSDE+iLQR: the smoothing iLQR's value expansions are taken at (the tracking gains' by
  // default), the cap on iLQR's charged seconds and iterations before DT-FBSDE takes over, the
  // geometric spacing of the iLQR iterations a candidate table is built at, the soft-min model's
  // conversion, and the first forward pass's exploration probability (negative: the regular one).
  double seed_smoothing = 1.0;
  double seed_ilqr_budget = 10.0;
  std::size_t seed_max_ilqr_iterations = 2000;
  double seed_growth = 1.25;
  IlqrSeedOptions ilqr_seed{};
  double seed_exploration = -1.0;

  // DT-FBSDE slope weights on the 2-D benchmarks. Off by default: slope matching gains a few points in 2-D
  // (profile_study/slope_ablation_2d) but costs a large share of each iteration, and none on the DIP.
  double integrator_slope_weight = 0.0;
  double pendulum_slope_weight = 0.0;

  // DT-FBSDE on the double inverted pendulum: "comparison" is the FBRRT comparison's configuration
  // (4096 paths, soft-min); "sweep" carries the 2-D recipe over -- slope weight
  // `double_pendulum_slope_weight` (0: off), mixed exploration, the screened stop-uphill line search -- on
  // the degree-4 polynomial with batches of kDoublePendulumSweepBatch. "sweep" is the default: a profile
  // put it at about a seventh of the comparison configuration's time per iteration.
  bool double_pendulum_sweep_config = true;
  double double_pendulum_slope_weight = 0.0;

  // DT-FBSDE's backward target under the sweep recipe, on every benchmark. The Taylor noiseless target
  // is the default; the double-greedy target was the default before it.
  DtFbsdeBackwardTarget dt_backward_target = DtFbsdeBackwardTarget::kTaylorNoiseless;

  // DT-FBSDE's line search start, batch pooling, and function-space trust region under the sweep
  // recipe; see DtFbsdeIterativeOptions. The first step is 1 in 2-D and 1/2 on the double inverted
  // pendulum, whose full steps overshoot (line_search_study/README.md); pooling and anchoring are off.
  double dt_initial_step = 1.0;
  double double_pendulum_initial_step = 0.5;
  bool dt_adaptive_initial_step = false;
  std::size_t dt_pool_batches = 0;
  double dt_anchor_weight = 0.0;

  // A cap on MPPI's budget below the benchmark's, for runs where its curve has long since flattened.
  double mppi_time_budget = 0.0;

  std::size_t dp_cells = 400;
  std::size_t dp_controls = 41;
  std::size_t dp_quadrature = 3;

  std::string csv_path = "method_comparison.csv";
  std::string mpc_csv_path = "method_comparison_mpc.csv";
  std::string reference_csv_path = "method_comparison_reference.csv";
  // Where DT-FBSDE's per-phase time goes, one row per run and phase; empty disables the phase timers.
  std::string profile_csv_path{};
  FbrrtComparisonOptions comparison{};
};

// One run's held-out cost at each checkpoint.
struct Trace {
  std::string method{};
  std::size_t seed_index = 0;
  std::vector<std::size_t> iterations{};
  std::vector<double> seconds{};
  std::vector<double> held_out{};
  // DT-FBSDE's phase timers over the run, held-out evaluation excluded, when profiling.
  bool profiled = false;
  profiling::PhaseTotals phases{};
};

struct Outputs {
  std::ofstream csv;
  std::ofstream mpc_csv;
  std::ofstream reference_csv;
  std::ofstream profile_csv;
};

template <typename BenchmarkT>
auto InitialFactor(const BenchmarkT& benchmark) noexcept {
  using StateMat = Eigen::Matrix<double, BenchmarkT::State::RowsAtCompileTime, BenchmarkT::State::RowsAtCompileTime>;
  return StateMat(Eigen::LLT<StateMat>(benchmark.initial_distribution.Covariance()).matrixL());
}

template <typename BenchmarkT, typename PolicyT>
auto HeldOutCost(const BenchmarkT& benchmark, const PolicyT& policy, std::size_t num_stages, std::size_t rollouts)
    -> double {
  return EvaluatePolicyByRollouts(benchmark.forward_model,
                                  policy,
                                  benchmark.running_cost,
                                  benchmark.terminal_cost,
                                  typename BenchmarkT::State(benchmark.initial_distribution.Mean()),
                                  InitialFactor(benchmark),
                                  rollouts,
                                  num_stages - 1,
                                  kHeldOutSeed)
      .total;
}

// Runs `step` until the time budget is spent, `step` reports convergence, or the iteration cap, and
// calls `record` at iteration zero and at each checkpoint. `step` returns whether the method is done.
template <typename StepT, typename RecordT>
auto RunTimed(const DriverOptions& options, double budget, StepT&& step, RecordT&& record) -> Result<> {
  double charged = 0.0;
  double next = options.first_checkpoint;
  record(0, 0.0);
  for (std::size_t iteration = 1; iteration <= options.max_iterations; ++iteration) {
    const Clock::time_point start = Clock::now();
    RESULT_ASSIGN_OR_RETURN(const bool converged, step());
    charged += std::chrono::duration<double>(Clock::now() - start).count();
    const bool done = converged || charged >= budget || iteration == options.max_iterations;
    if (done || charged >= next) {
      record(iteration, charged);
      while (next <= charged) {
        next *= options.checkpoint_growth;
      }
    }
    if (done) {
      break;
    }
  }
  return SuccessResult();
}

auto WriteTrace(std::ofstream& csv, const std::string& problem, const Trace& trace) -> void {
  for (std::size_t point = 0; point < trace.iterations.size(); ++point) {
    csv << std::format("{},{},{},{},{:.6f},{:.6f}\n",
                       problem,
                       trace.method,
                       trace.seed_index,
                       trace.iterations[point],
                       trace.seconds[point],
                       trace.held_out[point]);
  }
  csv.flush();
}

// The run's phase totals, and its charged seconds and iterations as the row `charged_total`.
auto WriteProfile(std::ofstream& csv, const std::string& problem, const Trace& trace) -> void {
  if (!trace.profiled || !csv.is_open()) {
    return;
  }
  for (std::size_t index = 0; index < profiling::kPhaseCount; ++index) {
    csv << std::format("{},{},{},{},{:.6e},{}\n",
                       problem,
                       trace.method,
                       trace.seed_index,
                       profiling::kPhaseNames[index],
                       trace.phases.seconds[index],
                       trace.phases.calls[index]);
  }
  csv << std::format("{},{},{},charged_total,{:.6e},{}\n",
                     problem,
                     trace.method,
                     trace.seed_index,
                     trace.seconds.back(),
                     trace.iterations.back());
  csv.flush();
}

auto Report(const std::string& problem, const Trace& trace) -> void {
  std::cout << std::format("{:<28} {:<18} seed {:>2}: {:9.4f} -> {:9.4f} (best {:9.4f}) in {:7.2f} s, {} iterations\n",
                           problem,
                           trace.method,
                           trace.seed_index,
                           trace.held_out.front(),
                           trace.held_out.back(),
                           *std::ranges::min_element(trace.held_out),
                           trace.seconds.back(),
                           trace.iterations.back());
  std::cout.flush();
}

// ---------------------------------------------------------------------------------------------
// The methods
// ---------------------------------------------------------------------------------------------

template <typename SolverT, typename BenchmarkT, std::size_t NumStages>
auto RunDtFbsde(const BenchmarkT& benchmark,
                const auto& table,
                const DtFbsdeIterativeOptions<double>& solver_options,
                const DriverOptions& options,
                double budget,
                std::size_t seed_index) -> Result<Trace> {
  const std::uint64_t seed = options.seed + (1000U * static_cast<std::uint64_t>(seed_index));
  auto solver = std::make_unique<Result<SolverT>>(SolverT::Make(benchmark.forward_model,
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
  SolverT& dt = **solver;
  Trace trace{.method = "DT-FBSDE", .seed_index = seed_index, .profiled = !options.profile_csv_path.empty()};
  if (trace.profiled) {
    profiling::Reset();
    profiling::SetEnabled(true);
  }
  RESULT_RETURN_IF_ERROR(RunTimed(
      options,
      budget,
      [&]() -> Result<bool> {
        RESULT_RETURN_IF_ERROR(dt.Iterate());
        return SuccessResult(false);
      },
      [&](std::size_t iteration, double seconds) {
        trace.iterations.push_back(iteration);
        trace.seconds.push_back(seconds);
        // The held-out evaluation is the driver's, not the solver's: keep it out of the phase totals.
        profiling::SetEnabled(false);
        trace.held_out.push_back(HeldOutCost(benchmark, dt.CurrentPolicy(), NumStages, options.held_out_rollouts));
        profiling::SetEnabled(trace.profiled);
      }));
  if (trace.profiled) {
    profiling::SetEnabled(false);
    trace.phases = profiling::Snapshot();
  }
  return SuccessResult(std::move(trace));
}

template <typename SetupT, typename BenchmarkT, std::size_t NumStages>
auto RunFbrrt(const BenchmarkT& benchmark,
              const StartingTable<typename SetupT::ValueFunction>& table,
              const DriverOptions& options,
              double keep_fraction,
              double budget,
              std::size_t seed_index) -> Result<Trace> {
  const std::uint64_t seed = options.seed + (1000U * static_cast<std::uint64_t>(seed_index));
  RESULT_ASSIGN_OR_RETURN(const std::unique_ptr<typename SetupT::Fbrrt> solver,
                          MakeComparisonFbrrtSolver<SetupT>(
                              benchmark, table, options.comparison, options.comparison.fbrrt, keep_fraction, seed));
  Trace trace{.method = "FBRRT", .seed_index = seed_index};
  RESULT_RETURN_IF_ERROR(RunTimed(
      options,
      budget,
      [&]() -> Result<bool> {
        RESULT_RETURN_IF_ERROR(solver->Iterate());
        return SuccessResult(false);
      },
      [&](std::size_t iteration, double seconds) {
        trace.iterations.push_back(iteration);
        trace.seconds.push_back(seconds);
        trace.held_out.push_back(
            HeldOutCost(benchmark, solver->CurrentPolicy(), NumStages, options.held_out_rollouts));
      }));
  return SuccessResult(std::move(trace));
}

template <typename BenchmarkT, std::size_t NumStages>
using IlqrSolver =
    BoxIlqrSolver<BenchmarkT::State::RowsAtCompileTime, NumStages, typename BenchmarkT::ForwardModel>;

// iLQR from zero controls if `zero_start`, else from a random draw for `seed_index`.
template <typename BenchmarkT, std::size_t NumStages>
auto MakeIlqr(const BenchmarkT& benchmark, const DriverOptions& options, std::size_t seed_index, bool zero_start)
    -> Result<IlqrSolver<BenchmarkT, NumStages>> {
  const std::uint64_t seed = options.seed + (1000U * static_cast<std::uint64_t>(seed_index));
  const CounterBasedNormalSampler sampler(seed ^ kInitialControlStream);
  std::vector<double> controls(NumStages - 1, 0.0);
  if (!zero_start) {
    for (std::size_t stage = 0; stage < controls.size(); ++stage) {
      controls[stage] = options.ilqr_initial_spread * sampler.Sample(0, stage, 0);
    }
  }
  using Solver = IlqrSolver<BenchmarkT, NumStages>;
  return Solver::Make(benchmark.forward_model,
                      benchmark.running_cost.L1Weight(),
                      benchmark.terminal_cost_mat,
                      benchmark.initial_distribution.Mean(),
                      -1.0,
                      1.0,
                      controls,
                      options.ilqr);
}

template <typename BenchmarkT, std::size_t NumStages>
auto RunIlqr(const BenchmarkT& benchmark,
             const DriverOptions& options,
             double budget,
             std::size_t seed_index,
             bool zero_start,
             std::unique_ptr<IlqrSolver<BenchmarkT, NumStages>>* solver_out = nullptr) -> Result<Trace> {
  using Solver = IlqrSolver<BenchmarkT, NumStages>;
  RESULT_ASSIGN_OR_RETURN(Solver solver, (MakeIlqr<BenchmarkT, NumStages>(benchmark, options, seed_index, zero_start)));
  Trace trace{.method = zero_start ? "iLQR (zero init)" : "iLQR", .seed_index = seed_index};
  RESULT_RETURN_IF_ERROR(RunTimed(
      options,
      budget,
      [&]() -> Result<bool> { return SuccessResult(solver.Iterate().converged); },
      [&](std::size_t iteration, double seconds) {
        trace.iterations.push_back(iteration);
        trace.seconds.push_back(seconds);
        trace.held_out.push_back(HeldOutCost(
            benchmark, solver.CurrentPolicy(options.ilqr_gain_smoothing), NumStages, options.held_out_rollouts));
      }));
  if (solver_out != nullptr) {
    *solver_out = std::make_unique<IlqrSolver<BenchmarkT, NumStages>>(std::move(solver));
  }
  return SuccessResult(std::move(trace));
}

// DT-FBSDE+iLQR; see the file comment.
template <typename SolverT, typename BenchmarkT, std::size_t NumStages>
auto RunSeededDtFbsde(const BenchmarkT& benchmark,
                      const auto& table,
                      DtFbsdeIterativeOptions<double> solver_options,
                      const DriverOptions& options,
                      double budget,
                      std::size_t seed_index) -> Result<Trace> {
  using ValueFunction = std::decay_t<decltype(table.initial)>;
  const std::uint64_t seed = options.seed + (1000U * static_cast<std::uint64_t>(seed_index));
  solver_options.initial_exploration_probability =
      options.seed_exploration >= 0.0 ? options.seed_exploration : solver_options.exploration_probability;
  auto solver = std::make_unique<Result<SolverT>>(SolverT::Make(benchmark.forward_model,
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
  SolverT& dt = **solver;
  const double unseeded_cost = dt.CurrentCost();
  using Ilqr = IlqrSolver<BenchmarkT, NumStages>;
  RESULT_ASSIGN_OR_RETURN(Ilqr ilqr, (MakeIlqr<BenchmarkT, NumStages>(benchmark, options, seed_index, false)));

  bool seeding = true;
  std::size_t ilqr_iterations = 0;
  double next_candidate = 1.0;
  double ilqr_seconds = 0.0;
  double best_cost = std::numeric_limits<double>::infinity();
  std::size_t best_iteration = 0;
  std::size_t candidates = 0;
  // The iLQR phase, one iteration per step; returns whether it is over.
  const auto seed_step = [&]() -> Result<bool> {
    const Clock::time_point start = Clock::now();
    const bool converged = ilqr.Iterate().converged;
    ++ilqr_iterations;
    const bool last = converged || ilqr_iterations >= options.seed_max_ilqr_iterations ||
                      ilqr_seconds + std::chrono::duration<double>(Clock::now() - start).count() >=
                          options.seed_ilqr_budget;
    if (last || static_cast<double>(ilqr_iterations) >= next_candidate) {
      while (next_candidate <= static_cast<double>(ilqr_iterations)) {
        next_candidate = std::max(next_candidate * options.seed_growth, next_candidate + 1.0);
      }
      using Expansion = typename Ilqr::ValueExpansion;
      const std::vector<Expansion> expansions = ilqr.ValueExpansions(options.seed_smoothing);
      RESULT_ASSIGN_OR_RETURN(
          const std::vector<ValueFunction> candidate,
          IlqrSeededTable(table.initial, std::span<const Expansion>(expansions), options.ilqr_seed));
      RESULT_ASSIGN_OR_RETURN(const double cost, dt.SeededCost(candidate));
      ++candidates;
      if (cost < best_cost) {
        best_cost = cost;
        best_iteration = ilqr_iterations;
        RESULT_RETURN_IF_ERROR(dt.SeedValueFunctions(candidate));
      }
    }
    ilqr_seconds += std::chrono::duration<double>(Clock::now() - start).count();
    return SuccessResult(last);
  };

  Trace trace{.method = "DT-FBSDE+iLQR", .seed_index = seed_index};
  RESULT_RETURN_IF_ERROR(RunTimed(
      options,
      budget,
      [&]() -> Result<bool> {
        if (seeding) {
          RESULT_ASSIGN_OR_RETURN(const bool over, seed_step());
          seeding = !over;
          return SuccessResult(false);
        }
        RESULT_RETURN_IF_ERROR(dt.Iterate());
        return SuccessResult(false);
      },
      [&](std::size_t iteration, double seconds) {
        trace.iterations.push_back(iteration);
        trace.seconds.push_back(seconds);
        trace.held_out.push_back(HeldOutCost(benchmark, dt.CurrentPolicy(), NumStages, options.held_out_rollouts));
      }));
  std::cout << std::format(
      "{:<28} {:<18} seed {:>2}: seeded from iLQR iteration {} of {} ({} candidates, {:.3f} s): solver cost {:.4f}, "
      "unseeded {:.4f}\n",
      benchmark.name,
      trace.method,
      seed_index,
      best_iteration,
      ilqr_iterations,
      candidates,
      ilqr_seconds,
      best_cost,
      unseeded_cost);
  return SuccessResult(std::move(trace));
}

template <typename BenchmarkT, std::size_t NumStages>
using MppiSolverFor = MppiSolver<BenchmarkT::State::RowsAtCompileTime,
                                 NumStages,
                                 typename BenchmarkT::ForwardModel,
                                 L1ControlSdeRunningCostTerm<BenchmarkT::State::RowsAtCompileTime, 1>,
                                 QuadraticRegulatorSdeTerminalCostTerm<BenchmarkT::State::RowsAtCompileTime>>;

// The tracking policy about a control sequence's noiseless trajectory.
template <typename BenchmarkT, std::size_t NumStages>
auto TrackingPolicy(const BenchmarkT& benchmark, std::span<const double> controls, double smoothing)
    -> Result<typename IlqrSolver<BenchmarkT, NumStages>::Policy> {
  BoxIlqrOptions<double> ilqr;
  ilqr.initial_smoothing = smoothing;
  ilqr.min_smoothing = smoothing;
  using Solver = IlqrSolver<BenchmarkT, NumStages>;
  RESULT_ASSIGN_OR_RETURN(Solver solver,
                          (Solver::Make(benchmark.forward_model,
                                                                   benchmark.running_cost.L1Weight(),
                                                                   benchmark.terminal_cost_mat,
                                                                   benchmark.initial_distribution.Mean(),
                                                                   -1.0,
                                                                   1.0,
                                                                   controls,
                                                                   ilqr)));
  return SuccessResult(solver.CurrentPolicy());
}

// Offline MPPI: the open-loop trace and the tracked trace, and the final plan for MPC.
template <typename BenchmarkT, std::size_t NumStages>
auto RunMppi(const BenchmarkT& benchmark,
             const DriverOptions& options,
             double budget,
             std::size_t seed_index,
             std::vector<double>& plan_out) -> Result<std::pair<Trace, Trace>> {
  using Solver = MppiSolverFor<BenchmarkT, NumStages>;
  const std::uint64_t seed = options.seed + (1000U * static_cast<std::uint64_t>(seed_index));
  const std::vector<double> zeros(NumStages - 1, 0.0);
  RESULT_ASSIGN_OR_RETURN(Solver solver,
                          Solver::Make(benchmark.forward_model,
                                       benchmark.running_cost,
                                       benchmark.terminal_cost,
                                       benchmark.initial_distribution.Mean(),
                                       InitialFactor(benchmark),
                                       -1.0,
                                       1.0,
                                       zeros,
                                       options.mppi,
                                       seed));
  Trace open_loop{.method = "MPPI", .seed_index = seed_index};
  Trace tracked{.method = "MPPI+LQR", .seed_index = seed_index};
  Result<> tracking_status = SuccessResult();
  RESULT_RETURN_IF_ERROR(RunTimed(
      options,
      budget,
      [&]() -> Result<bool> {
        (void)solver.Iterate();
        return SuccessResult(false);
      },
      [&](std::size_t iteration, double seconds) {
        for (Trace* trace : {&open_loop, &tracked}) {
          trace->iterations.push_back(iteration);
          trace->seconds.push_back(seconds);
        }
        open_loop.held_out.push_back(
            HeldOutCost(benchmark, solver.CurrentPolicy(), NumStages, options.held_out_rollouts));
        Result<typename IlqrSolver<BenchmarkT, NumStages>::Policy> policy =
            TrackingPolicy<BenchmarkT, NumStages>(benchmark, solver.NominalControls(), options.mppi_gain_smoothing);
        if (!policy.has_value()) {
          tracking_status = ErrorResult(policy.error().message, policy.error().location);
          tracked.held_out.push_back(open_loop.held_out.back());
          return;
        }
        tracked.held_out.push_back(HeldOutCost(benchmark, *policy, NumStages, options.held_out_rollouts));
      }));
  RESULT_RETURN_IF_ERROR(tracking_status);
  plan_out.assign(solver.NominalControls().begin(), solver.NominalControls().end());
  return SuccessResult(std::make_pair(std::move(open_loop), std::move(tracked)));
}

struct MpcResult {
  double cost = 0.0;
  double seconds_per_step = 0.0;
};

// A receding-horizon controller on the first `mpc_rollouts` held-out rollouts. Each rollout builds
// its own planner with `make(rollout)`, and at every stage `control(planner, stage, state)`
// re-plans from the state reached and returns the control to apply; only that call is timed.
// Rollouts run in parallel, each planner on one thread.
template <std::size_t NumStages, typename BenchmarkT, typename MakeT, typename ControlT>
auto EvaluateMpc(const BenchmarkT& benchmark, const DriverOptions& options, const MakeT& make, const ControlT& control)
    -> Result<MpcResult> {
  using State = typename BenchmarkT::State;
  const std::size_t rollouts = options.mpc_rollouts;
  const CounterBasedNormalSampler sampler(kHeldOutSeed);
  const auto factor = InitialFactor(benchmark);
  std::vector<double> costs(rollouts, 0.0);
  std::vector<double> seconds(rollouts, 0.0);
  std::vector<char> failed(rollouts, 0);
  ParallelFor(rollouts, [&](std::size_t rollout) noexcept {
    auto planner = make(rollout);
    if (!planner.has_value()) {
      failed[rollout] = 1;
      return;
    }
    State noise;
    sampler.FillNoise(rollout, 0, noise);
    State state = benchmark.initial_distribution.Mean() + (factor * noise);
    double cost = 0.0;
    for (std::size_t stage = 0; stage + 1 < NumStages; ++stage) {
      const Clock::time_point start = Clock::now();
      const Control applied(control(*planner, stage, state));
      seconds[rollout] += std::chrono::duration<double>(Clock::now() - start).count();
      cost += benchmark.running_cost(stage, state, applied);
      sampler.FillNoise(rollout, stage + 1, noise);
      state = benchmark.forward_model(stage, state, applied, noise);
    }
    costs[rollout] = cost + benchmark.terminal_cost(state);
  });
  RESULT_ASSERT(std::ranges::find(failed, 1) == failed.end(), "EvaluateMpc: a planner was rejected.");
  MpcResult result;
  for (std::size_t rollout = 0; rollout < rollouts; ++rollout) {
    result.cost += costs[rollout] / static_cast<double>(rollouts);
    result.seconds_per_step += seconds[rollout] / static_cast<double>(rollouts * (NumStages - 1));
  }
  return SuccessResult(result);
}

// Receding-horizon MPPI warm-started from `plan`: `mpc_iterations` iterations of `mpc_samples`
// samples over the rest of the horizon at every stage.
template <typename BenchmarkT, std::size_t NumStages>
auto RunMppiMpc(const BenchmarkT& benchmark,
                const DriverOptions& options,
                const std::vector<double>& plan,
                std::size_t seed_index) -> Result<MpcResult> {
  using Solver = MppiSolverFor<BenchmarkT, NumStages>;
  MppiOptions<double> mpc = options.mppi;
  mpc.num_samples = options.mpc_samples;
  mpc.parallel = false;
  // Re-planning searches near the warm start: begin at the noise floor the offline run ended at.
  mpc.control_noise = std::max(options.mppi.min_control_noise, options.mpc_control_noise);
  const std::uint64_t seed = options.seed + (1000U * static_cast<std::uint64_t>(seed_index));
  const auto factor = InitialFactor(benchmark);
  return EvaluateMpc<NumStages>(
      benchmark,
      options,
      [&](std::size_t rollout) {
        return Solver::Make(benchmark.forward_model,
                            benchmark.running_cost,
                            benchmark.terminal_cost,
                            benchmark.initial_distribution.Mean(),
                            factor,
                            -1.0,
                            1.0,
                            plan,
                            mpc,
                            seed + 7919U + rollout);
      },
      [&](Solver& solver, std::size_t stage, const typename BenchmarkT::State& state) {
        for (std::size_t iteration = 0; iteration < options.mpc_iterations; ++iteration) {
          (void)solver.IterateFrom(stage, state);
        }
        return solver.NominalControls()[stage];
      });
}

// Receding-horizon iLQR warm-started from a converged solver: at every stage, re-plan from the state
// reached and take `mpc_ilqr_iterations` iterations at the final smoothing.
template <typename BenchmarkT, std::size_t NumStages>
auto RunIlqrMpc(const BenchmarkT& benchmark,
                const DriverOptions& options,
                const IlqrSolver<BenchmarkT, NumStages>& converged) -> Result<MpcResult> {
  using Solver = IlqrSolver<BenchmarkT, NumStages>;
  return EvaluateMpc<NumStages>(
      benchmark,
      options,
      [&](std::size_t /*rollout*/) -> Result<Solver> { return SuccessResult(converged); },
      [&](Solver& solver, std::size_t stage, const typename BenchmarkT::State& state) {
        solver.Replan(stage, state);
        for (std::size_t iteration = 0; iteration < options.mpc_ilqr_iterations; ++iteration) {
          (void)solver.Iterate();
        }
        return solver.NominalControls()[stage];
      });
}

auto WriteMpc(Outputs& outputs,
              const std::string& problem,
              const std::string& method,
              std::size_t seed_index,
              const DriverOptions& options,
              const MpcResult& mpc) -> void {
  std::cout << std::format("{:<28} {:<18} seed {:>2}: {:9.4f} on {} rollouts, {:.3f} ms per step\n",
                           problem,
                           method,
                           seed_index,
                           mpc.cost,
                           options.mpc_rollouts,
                           1e3 * mpc.seconds_per_step);
  outputs.mpc_csv << std::format(
      "{},{},{},{},{:.6f},{:.9f}\n", problem, method, seed_index, options.mpc_rollouts, mpc.cost, mpc.seconds_per_step);
  outputs.mpc_csv.flush();
}

// ---------------------------------------------------------------------------------------------
// One benchmark
// ---------------------------------------------------------------------------------------------

// Everything but DT-FBSDE, which differs by benchmark, on one benchmark; `run_dt_fbsde` and
// `run_seeded_dt_fbsde` run it, unseeded and seeded from iLQR.
template <std::size_t NumStages,
          typename FbrrtSetupT,
          typename BenchmarkT,
          typename DtFbsdeT,
          typename SeededDtFbsdeT,
          typename ReferenceT>
auto CompareOn(const BenchmarkT& benchmark,
               const DriverOptions& options,
               double budget,
               double keep_fraction,
               const DtFbsdeT& run_dt_fbsde,
               const SeededDtFbsdeT& run_seeded_dt_fbsde,
               const ReferenceT& reference,
               Outputs& outputs) -> Result<> {
  std::cout << std::format("\n=== {} ({:.0f} s per run) ===\n", benchmark.name, budget);
  const std::string& problem = benchmark.name;
  double reference_cost = 0.0;
  if constexpr (!std::is_same_v<ReferenceT, std::nullptr_t>) {
    reference_cost = HeldOutCost(benchmark, *reference, NumStages, options.held_out_rollouts);
    const double subset = HeldOutCost(benchmark, *reference, NumStages, options.mpc_rollouts);
    std::cout << std::format("grid DP optimum, held out: {:.4f} ({} rollouts), {:.4f} (first {})\n",
                             reference_cost,
                             options.held_out_rollouts,
                             subset,
                             options.mpc_rollouts);
    outputs.reference_csv << std::format("{},{:.6f},{:.6f}\n", problem, reference_cost, subset);
    outputs.reference_csv.flush();
  }

  RESULT_ASSIGN_OR_RETURN(
      const StartingTable<typename FbrrtSetupT::ValueFunction> fbrrt_table,
      MakeStartingTable(
          benchmark, options.comparison, static_cast<const typename FbrrtSetupT::ValueFunction*>(nullptr)));

  if (options.run_ilqr) {
    RESULT_ASSIGN_OR_RETURN(const Trace zero, (RunIlqr<BenchmarkT, NumStages>(benchmark, options, budget, 0, true)));
    Report(problem, zero);
    WriteTrace(outputs.csv, problem, zero);
  }
  for (std::size_t index = 0; index < options.num_seeds; ++index) {
    if (options.run_ilqr) {
      std::unique_ptr<IlqrSolver<BenchmarkT, NumStages>> solver;
      RESULT_ASSIGN_OR_RETURN(const Trace trace,
                              (RunIlqr<BenchmarkT, NumStages>(benchmark, options, budget, index, false, &solver)));
      Report(problem, trace);
      WriteTrace(outputs.csv, problem, trace);
      if (index < options.mpc_seeds) {
        RESULT_ASSIGN_OR_RETURN(const MpcResult mpc, (RunIlqrMpc<BenchmarkT, NumStages>(benchmark, options, *solver)));
        WriteMpc(outputs, problem, "iLQR-MPC", index, options, mpc);
      }
    }
    if (options.run_mppi) {
      std::vector<double> plan;
      const double mppi_budget = options.mppi_time_budget > 0.0 ? std::min(budget, options.mppi_time_budget) : budget;
      RESULT_ASSIGN_OR_RETURN(const auto traces,
                              (RunMppi<BenchmarkT, NumStages>(benchmark, options, mppi_budget, index, plan)));
      for (const Trace* trace : {&traces.first, &traces.second}) {
        Report(problem, *trace);
        WriteTrace(outputs.csv, problem, *trace);
      }
      if (index < options.mpc_seeds) {
        RESULT_ASSIGN_OR_RETURN(const MpcResult mpc,
                                (RunMppiMpc<BenchmarkT, NumStages>(benchmark, options, plan, index)));
        WriteMpc(outputs, problem, "MPPI-MPC", index, options, mpc);
      }
    }
    if (options.run_fbrrt) {
      RESULT_ASSIGN_OR_RETURN(
          const Trace trace,
          (RunFbrrt<FbrrtSetupT, BenchmarkT, NumStages>(benchmark, fbrrt_table, options, keep_fraction, budget, index)));
      Report(problem, trace);
      WriteTrace(outputs.csv, problem, trace);
    }
    if (options.run_dt_fbsde) {
      RESULT_ASSIGN_OR_RETURN(const Trace trace, run_dt_fbsde(index));
      Report(problem, trace);
      WriteTrace(outputs.csv, problem, trace);
      WriteProfile(outputs.profile_csv, problem, trace);
    }
    if (options.run_seeded_dt_fbsde) {
      RESULT_ASSIGN_OR_RETURN(const Trace trace, run_seeded_dt_fbsde(index));
      Report(problem, trace);
      WriteTrace(outputs.csv, problem, trace);
    }
  }
  return SuccessResult();
}

// The DT-FBSDE recipe every benchmark runs by default: the batch sweep's best 2-D configuration, with the
// backward target `options.dt_backward_target`.
auto SweepDtFbsdeOptions(const DriverOptions& options, double slope_weight) -> DtFbsdeIterativeOptions<double> {
  DtFbsdeIterativeOptions<double> solver = DtFbsdeOptions(options.comparison);
  solver.backward_target = options.dt_backward_target;
  solver.stop_uphill_line_search = true;
  solver.screening_rollouts = 32;
  solver.base_diffusion_inflation = 6.0;
  solver.max_diffusion_inflation = 6.0;
  solver.uninflated_path_fraction = 0.5;
  solver.slope_weight = slope_weight;
  solver.initial_step = options.dt_initial_step;
  solver.adaptive_initial_step = options.dt_adaptive_initial_step;
  solver.pool_batches_since_acceptance = options.dt_pool_batches;
  solver.anchor_weight = options.dt_anchor_weight;
  return solver;
}

constexpr int kSweepPolynomialDegree = 4;
constexpr std::size_t kSweepBatch = 16;
constexpr std::size_t kSweepLineSearchSamples = 256;
constexpr std::size_t kDoublePendulumSweepBatch = 256;

template <std::size_t NumStages, typename BenchmarkT>
auto CompareTwoDimensional(const BenchmarkT& benchmark,
                           const L1GridDpReference<BenchmarkT>* reference,
                           const DriverOptions& options,
                           double slope_weight,
                           Outputs& outputs) -> Result<> {
  using Polynomial = PolynomialValueFunctionApprox<2, kSweepPolynomialDegree>;
  using DtSolver = DtFbsdeIterativeSolver<2,
                                          1,
                                          kSweepBatch,
                                          NumStages,
                                          kSweepLineSearchSamples,
                                          typename BenchmarkT::ForwardModel,
                                          QuadraticRegulatorSdeTerminalCostTerm<2>,
                                          Polynomial>;
  using FbrrtSetup = Setup<BenchmarkT, 1024, NumStages, SoftMinQuadraticValueFunctionApprox<2, kSoftMinComponents>>;
  RESULT_ASSIGN_OR_RETURN(const StartingTable<Polynomial> table,
                          MakeStartingTable(benchmark, options.comparison, static_cast<const Polynomial*>(nullptr)));
  const DtFbsdeIterativeOptions<double> solver_options = SweepDtFbsdeOptions(options, slope_weight);
  const auto run_dt_fbsde = [&](std::size_t index) {
    return RunDtFbsde<DtSolver, BenchmarkT, NumStages>(
        benchmark, table, solver_options, options, options.time_budget, index);
  };
  const auto run_seeded_dt_fbsde = [&](std::size_t index) {
    return RunSeededDtFbsde<DtSolver, BenchmarkT, NumStages>(
        benchmark, table, solver_options, options, options.time_budget, index);
  };
  return CompareOn<NumStages, FbrrtSetup>(benchmark,
                                          options,
                                          options.time_budget,
                                          options.comparison.keep_fraction,
                                          run_dt_fbsde,
                                          run_seeded_dt_fbsde,
                                          reference,
                                          outputs);
}

template <std::size_t NumStages, typename BenchmarkT>
auto MakeGrid(const BenchmarkT& benchmark,
              const DriverOptions& options,
              const Eigen::Vector2d& lower,
              const Eigen::Vector2d& upper) -> Result<L1GridDpReference<BenchmarkT>> {
  typename L1GridDpReference<BenchmarkT>::Options grid;
  grid.lower = lower;
  grid.upper = upper;
  grid.cells_per_axis = options.dp_cells;
  grid.num_controls = options.dp_controls;
  grid.quadrature_order = options.dp_quadrature;
  return L1GridDpReference<BenchmarkT>::Make(benchmark, NumStages, grid);
}

auto Run(const DriverOptions& options) -> Result<> {
  Outputs outputs{.csv = std::ofstream(options.csv_path),
                  .mpc_csv = std::ofstream(options.mpc_csv_path),
                  .reference_csv = std::ofstream(options.reference_csv_path),
                  .profile_csv = std::ofstream()};
  RESULT_ASSERT(outputs.csv.good() && outputs.mpc_csv.good() && outputs.reference_csv.good(),
                "could not open the output files for writing.");
  outputs.csv << "problem,method,seed,iteration,seconds,held_out_cost\n";
  outputs.mpc_csv << "problem,method,seed,rollouts,cost,seconds_per_step\n";
  outputs.reference_csv << "problem,grid_optimum,grid_optimum_mpc_subset\n";
  if (!options.profile_csv_path.empty()) {
    outputs.profile_csv.open(options.profile_csv_path);
    RESULT_ASSERT(outputs.profile_csv.good(), "could not open the profile file for writing.");
    outputs.profile_csv << "problem,method,seed,phase,seconds,calls\n";
  }

  // The grid boxes are the batch sweep's.
  if (options.include_integrator) {
    RESULT_ASSIGN_OR_RETURN(const L1DoubleIntegrator integrator, MakeL1DoubleIntegrator());
    RESULT_ASSIGN_OR_RETURN(
        const auto grid,
        MakeGrid<kL1DoubleIntegratorNumStages>(integrator, options, {-2.5, -3.5}, {4.5, 2.5}));
    RESULT_RETURN_IF_ERROR(CompareTwoDimensional<kL1DoubleIntegratorNumStages>(
        integrator, &grid, options, options.integrator_slope_weight, outputs));
  }
  if (options.include_pendulum) {
    RESULT_ASSIGN_OR_RETURN(const L1Pendulum pendulum, MakeL1Pendulum());
    RESULT_ASSIGN_OR_RETURN(const auto grid,
                            MakeGrid<kL1PendulumNumStages>(pendulum, options, {-2.0, -9.0}, {5.0, 9.0}));
    RESULT_RETURN_IF_ERROR(CompareTwoDimensional<kL1PendulumNumStages>(
        pendulum, &grid, options, options.pendulum_slope_weight, outputs));
  }
  if (options.include_double_pendulum) {
    constexpr std::size_t kStages = kL1DoubleInvertedPendulumNumStages;
    using SoftMin = SoftMinQuadraticValueFunctionApprox<4, kSoftMinComponents>;
    using Polynomial = PolynomialValueFunctionApprox<4, kSweepPolynomialDegree>;
    using DipSetup = Setup<L1DoubleInvertedPendulum, 4096, kStages, SoftMin>;
    RESULT_ASSIGN_OR_RETURN(const L1DoubleInvertedPendulum double_pendulum, MakeL1DoubleInvertedPendulum());
    // FBRRT and the comparison configuration keep the soft-min model; the sweep recipe uses the polynomial.
    RESULT_ASSIGN_OR_RETURN(
        const StartingTable<SoftMin> table,
        MakeStartingTable(double_pendulum, options.comparison, static_cast<const SoftMin*>(nullptr)));
    RESULT_ASSIGN_OR_RETURN(
        const StartingTable<Polynomial> sweep_table,
        MakeStartingTable(double_pendulum, options.comparison, static_cast<const Polynomial*>(nullptr)));
    using SweepDtFbsde = DtFbsdeIterativeSolver<4,
                                                1,
                                                kDoublePendulumSweepBatch,
                                                kStages,
                                                kSweepLineSearchSamples,
                                                L1DoubleInvertedPendulum::ForwardModel,
                                                QuadraticRegulatorSdeTerminalCostTerm<4>,
                                                Polynomial>;
    DtFbsdeIterativeOptions<double> solver_options =
        options.double_pendulum_sweep_config ? SweepDtFbsdeOptions(options, options.double_pendulum_slope_weight)
                                             : DtFbsdeOptions(options.comparison);
    if (options.double_pendulum_sweep_config) {
      solver_options.initial_step = options.double_pendulum_initial_step;
    }
    const auto run_dt_fbsde = [&](std::size_t index) {
      return options.double_pendulum_sweep_config
                 ? RunDtFbsde<SweepDtFbsde, L1DoubleInvertedPendulum, kStages>(
                       double_pendulum, sweep_table, solver_options, options, options.double_pendulum_time_budget, index)
                 : RunDtFbsde<typename DipSetup::DtFbsde, L1DoubleInvertedPendulum, kStages>(
                       double_pendulum, table, solver_options, options, options.double_pendulum_time_budget, index);
    };
    const auto run_seeded_dt_fbsde = [&](std::size_t index) {
      return options.double_pendulum_sweep_config
                 ? RunSeededDtFbsde<SweepDtFbsde, L1DoubleInvertedPendulum, kStages>(
                       double_pendulum, sweep_table, solver_options, options, options.double_pendulum_time_budget, index)
                 : RunSeededDtFbsde<typename DipSetup::DtFbsde, L1DoubleInvertedPendulum, kStages>(
                       double_pendulum, table, solver_options, options, options.double_pendulum_time_budget, index);
    };
    RESULT_RETURN_IF_ERROR((CompareOn<kStages, DipSetup>(double_pendulum,
                                                         options,
                                                         options.double_pendulum_time_budget,
                                                         options.comparison.double_inverted_pendulum_keep_fraction,
                                                         run_dt_fbsde,
                                                         run_seeded_dt_fbsde,
                                                         nullptr,
                                                         outputs)));
  }
  std::cout << std::format("\nCheckpoints in {}, MPC results in {}, grid optima in {}\n",
                           options.csv_path,
                           options.mpc_csv_path,
                           options.reference_csv_path);
  return SuccessResult();
}

// ---------------------------------------------------------------------------------------------
// Command line
// ---------------------------------------------------------------------------------------------

auto PrintUsage() -> void {
  std::cout
      << "usage: method_comparison [options]\n"
      << "  --seed N                 the seed all sampling derives from\n"
      << "  --seeds N                runs per method per benchmark (default 8)\n"
      << "  --time-budget S          charged seconds per run on the 2-D benchmarks (default 60)\n"
      << "  --dip-time-budget S      charged seconds per run on the double inverted pendulum (default 120)\n"
      << "  --held-out N             held-out rollouts per checkpoint (default 1024)\n"
      << "  --only P[,P...]          integrator, pendulum, double-pendulum\n"
      << "  --methods M[,M...]       dt-fbsde, dt-fbsde-ilqr, fbrrt, ilqr, mppi (default all)\n"
      << "  --ilqr-spread S          std. dev. of iLQR's random initial controls (default 0.5)\n"
      << "  --ilqr-min-smoothing E   iLQR's final pseudo-Huber smoothing (default 1e-3)\n"
      << "  --ilqr-gain-smoothing E  smoothing iLQR's tracking gains are computed at (default 1; 0: its own)\n"
      << "  --mppi-samples N         MPPI samples per iteration (default 256)\n"
      << "  --mppi-noise S           MPPI control noise std. dev. (default 0.5)\n"
      << "  --mppi-ess F             MPPI's target effective sample fraction (default 0.05)\n"
      << "  --mppi-noise-decay D     MPPI control noise decay per iteration (default 0.99)\n"
      << "  --mppi-min-noise S       MPPI control noise floor (default 0.02)\n"
      << "  --mppi-process-noise on|off  roll MPPI's samples out with the problem's noise (default off)\n"
      << "  --mppi-gain-smoothing E  smoothing MPPI+LQR's gains are computed at (default 1)\n"
      << "  --mpc-seeds N            seeds MPC is evaluated on (default 3; 0 skips it)\n"
      << "  --mpc-rollouts N         held-out rollouts MPC is evaluated on (default 128)\n"
      << "  --mpc-samples N          MPPI-MPC samples per iteration (default 256)\n"
      << "  --mpc-iterations N       MPPI-MPC iterations per control step (default 2)\n"
      << "  --mpc-noise S            MPPI-MPC control noise (default 0.1)\n"
      << "  --mpc-ilqr-iterations N  iLQR-MPC iterations per control step (default 2)\n"
      << "  --integrator-slope-weight G  DT-FBSDE's slope weight on the double integrator (default 0: values alone)\n"
      << "  --pendulum-slope-weight G    DT-FBSDE's slope weight on the pendulum (default 0: values alone)\n"
      << "  --dip-dt-config C        DT-FBSDE on the double inverted pendulum: sweep (default) or comparison\n"
      << "  --dip-slope-weight G     its slope weight under the sweep configuration (default 0)\n"
      << "  --dt-target T            DT-FBSDE's backward target: taylor (default) or double-greedy\n"
      << "  --dt-initial-step S      DT-FBSDE's first line search step in 2-D, in (0, 1] (default 1)\n"
      << "  --dip-initial-step S     its first step on the double inverted pendulum (default 0.5)\n"
      << "  --dt-adaptive-step on|off  start each line search at twice the last accepted step (default off)\n"
      << "  --dt-pool N              pool up to N forward batches since the last acceptance (default 0: off)\n"
      << "  --dt-anchor-weight W     function-space trust region weight (default 0: off)\n"
      << "  --mppi-time-budget S     cap on MPPI's charged seconds per run (default: the benchmark's)\n"
      << "  --seed-smoothing E       smoothing DT-FBSDE+iLQR's value expansions are taken at (default 1)\n"
      << "  --seed-ilqr-budget S     cap on DT-FBSDE+iLQR's charged seconds of iLQR (default 10)\n"
      << "  --seed-ilqr-iterations N cap on DT-FBSDE+iLQR's iLQR iterations (default 2000)\n"
      << "  --seed-growth G          spacing of the iLQR iterations it builds a candidate at (default 1.25)\n"
      << "  --seed-metric-floor E    floor on a soft-min seed's curvature (default 1e-3)\n"
      << "  --seed-center-spread C   value spread of a soft-min seed's components (default 0.5)\n"
      << "  --seed-exploration P     its first forward pass's exploration probability (default: the regular one)\n"
      << "  --csv PATH               checkpoints (default method_comparison.csv)\n"
      << "  --mpc-csv PATH           MPC results (default method_comparison_mpc.csv)\n"
      << "  --reference-csv PATH     grid optima (default method_comparison_reference.csv)\n"
      << "  --profile-csv PATH       time DT-FBSDE's phases and write the totals per run (default: off)\n";
}

auto Contains(std::string_view list, std::string_view item) -> bool {
  std::size_t begin = 0;
  while (begin <= list.size()) {
    const std::size_t end = std::min(list.find(',', begin), list.size());
    if (list.substr(begin, end - begin) == item) {
      return true;
    }
    begin = end + 1;
  }
  return false;
}

auto ParseOptions(int argc, char** argv, DriverOptions& options) -> Result<> {
  const std::vector<std::string_view> arguments(argv + 1, argv + argc);
  for (std::size_t index = 0; index < arguments.size(); ++index) {
    const std::string_view argument = arguments[index];
    RESULT_ASSERT(index + 1 < arguments.size(), "every option needs a value.");
    const std::string value(arguments[++index]);
    const double number = std::strtod(value.c_str(), nullptr);
    const auto count = static_cast<std::size_t>(std::strtoull(value.c_str(), nullptr, 10));
    if (argument == "--seed") {
      options.seed = std::strtoull(value.c_str(), nullptr, 10);
    } else if (argument == "--seeds") {
      options.num_seeds = count;
    } else if (argument == "--time-budget") {
      options.time_budget = number;
    } else if (argument == "--dip-time-budget") {
      options.double_pendulum_time_budget = number;
    } else if (argument == "--held-out") {
      options.held_out_rollouts = count;
    } else if (argument == "--only") {
      options.include_integrator = Contains(value, "integrator");
      options.include_pendulum = Contains(value, "pendulum");
      options.include_double_pendulum = Contains(value, "double-pendulum");
    } else if (argument == "--methods") {
      options.run_dt_fbsde = Contains(value, "dt-fbsde");
      options.run_seeded_dt_fbsde = Contains(value, "dt-fbsde-ilqr");
      options.run_fbrrt = Contains(value, "fbrrt");
      options.run_ilqr = Contains(value, "ilqr");
      options.run_mppi = Contains(value, "mppi");
    } else if (argument == "--ilqr-spread") {
      options.ilqr_initial_spread = number;
    } else if (argument == "--ilqr-min-smoothing") {
      options.ilqr.min_smoothing = number;
    } else if (argument == "--ilqr-gain-smoothing") {
      options.ilqr_gain_smoothing = number;
    } else if (argument == "--mppi-samples") {
      options.mppi.num_samples = count;
    } else if (argument == "--mppi-noise") {
      options.mppi.control_noise = number;
    } else if (argument == "--mppi-ess") {
      options.mppi.effective_sample_fraction = number;
    } else if (argument == "--mppi-noise-decay") {
      options.mppi.noise_decay = number;
    } else if (argument == "--mppi-min-noise") {
      options.mppi.min_control_noise = number;
    } else if (argument == "--mppi-process-noise") {
      RESULT_ASSERT(value == "on" || value == "off", "--mppi-process-noise is on or off.");
      options.mppi.sample_process_noise = value == "on";
    } else if (argument == "--mppi-gain-smoothing") {
      options.mppi_gain_smoothing = number;
    } else if (argument == "--mpc-seeds") {
      options.mpc_seeds = count;
    } else if (argument == "--mpc-rollouts") {
      options.mpc_rollouts = count;
    } else if (argument == "--mpc-samples") {
      options.mpc_samples = count;
    } else if (argument == "--mpc-iterations") {
      options.mpc_iterations = count;
    } else if (argument == "--mpc-noise") {
      options.mpc_control_noise = number;
    } else if (argument == "--mpc-ilqr-iterations") {
      options.mpc_ilqr_iterations = count;
    } else if (argument == "--dip-dt-config") {
      RESULT_ASSERT(value == "comparison" || value == "sweep", "--dip-dt-config is comparison or sweep.");
      options.double_pendulum_sweep_config = value == "sweep";
    } else if (argument == "--dt-target") {
      RESULT_ASSERT(value == "taylor" || value == "double-greedy", "--dt-target is taylor or double-greedy.");
      options.dt_backward_target =
          value == "taylor" ? DtFbsdeBackwardTarget::kTaylorNoiseless : DtFbsdeBackwardTarget::kDoubleGreedy;
    } else if (argument == "--dt-initial-step") {
      options.dt_initial_step = number;
    } else if (argument == "--dip-initial-step") {
      options.double_pendulum_initial_step = number;
    } else if (argument == "--dt-adaptive-step") {
      RESULT_ASSERT(value == "on" || value == "off", "--dt-adaptive-step is on or off.");
      options.dt_adaptive_initial_step = value == "on";
    } else if (argument == "--dt-pool") {
      options.dt_pool_batches = count;
    } else if (argument == "--dt-anchor-weight") {
      options.dt_anchor_weight = number;
    } else if (argument == "--integrator-slope-weight") {
      options.integrator_slope_weight = number;
    } else if (argument == "--pendulum-slope-weight") {
      options.pendulum_slope_weight = number;
    } else if (argument == "--dip-slope-weight") {
      options.double_pendulum_slope_weight = number;
    } else if (argument == "--mppi-time-budget") {
      options.mppi_time_budget = number;
    } else if (argument == "--seed-smoothing") {
      options.seed_smoothing = number;
    } else if (argument == "--seed-ilqr-budget") {
      options.seed_ilqr_budget = number;
    } else if (argument == "--seed-ilqr-iterations") {
      options.seed_max_ilqr_iterations = count;
    } else if (argument == "--seed-growth") {
      options.seed_growth = number;
    } else if (argument == "--seed-metric-floor") {
      options.ilqr_seed.metric_floor = number;
    } else if (argument == "--seed-center-spread") {
      options.ilqr_seed.center_spread = number;
    } else if (argument == "--seed-exploration") {
      options.seed_exploration = number;
    } else if (argument == "--csv") {
      options.csv_path = value;
    } else if (argument == "--mpc-csv") {
      options.mpc_csv_path = value;
    } else if (argument == "--reference-csv") {
      options.reference_csv_path = value;
    } else if (argument == "--profile-csv") {
      options.profile_csv_path = value;
    } else {
      return ErrorResult("unrecognized argument; run with --help for the options.");
    }
  }
  RESULT_ASSERT(options.num_seeds >= 1, "the comparison needs at least one seed.");
  RESULT_ASSERT(options.seed_growth > 1.0 && options.seed_max_ilqr_iterations >= 1,
                "DT-FBSDE+iLQR needs a candidate spacing above one and at least one iLQR iteration.");
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
  fbsde_traj_opt::examples::DriverOptions options;
  fbsde_traj_opt::Result<> result = fbsde_traj_opt::examples::ParseOptions(argc, argv, options);
  if (result.has_value()) {
    result = fbsde_traj_opt::examples::Run(options);
  }
  if (!result.has_value()) {
    RESULT_REPORT_RESULT(result);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
