// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#ifndef FBSDE_TRAJ_OPT_EXAMPLES_FBRRT_COMPARISON_EXPERIMENT_HPP_
#define FBSDE_TRAJ_OPT_EXAMPLES_FBRRT_COMPARISON_EXPERIMENT_HPP_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include "fbsde_traj_opt/fbrrt/path_integral_erode.hpp"
#include "fbsde_traj_opt/fbrrt/taylor_backward_pass.hpp"
#include "fbsde_traj_opt/solvers/fbrrt_solver.hpp"
#include "fbsde_traj_opt/utils/result.hpp"
#include "fbsde_traj_opt/value_function/value_function_gauss_newton_fitter.hpp"
#include "fbsde_traj_opt/value_function/value_function_sgd_fitter.hpp"
#include "fbsde_traj_opt/viz/plotly_figure.hpp"

namespace fbsde_traj_opt::examples {

// FBRRT against the DT-FBSDE iterative method on the three L1 benchmarks of Hawkins (2021),
// Section 4.5 (see l1_benchmark_models.hpp), measured the way the thesis measured it: how fast
// each method's policy cost comes down, in iterations and in wall-clock time, over many seeds.
//
// Both methods use the Taylor noiseless estimator and the Taylor Q-function policy, start from the
// same value functions (the terminal cost at every stage), use the same number of samples per
// stage, and score their policies on the same fixed rollout noise. What differs is the method:
//
//   * FBRRT (FbrrtSolver): a particle set expanded by Voronoi-biased RRT draws with no forward
//     noise, value functions moved a few SGD steps per iteration, and path-integral erosion.
//   * DT-FBSDE (DtFbsdeIterativeSolver): parallel sampled paths and one-step probes, a weighted
//     least-squares fit per stage, and a damped blend accepted by a rollout line search. Its
//     options are the ones the double inverted pendulum example was tuned with.
//
// The time charged to a method is the time its algorithm needs. For DT-FBSDE that is the whole of
// Iterate(), whose line search is itself made of rollouts. For FBRRT it is expansion, the backward
// pass and erosion, plus the evaluation when its line search is on (the default) -- without the
// line search FBRRT evaluates only to report, and that time is not charged.

// The value function representation both methods use.
enum class ComparisonValueFunctionModel : std::uint8_t {
  // SoftMinQuadraticValueFunctionApprox with kSoftMinComponents components: the intended model.
  kSoftMinQuadratic,
  // PolynomialValueFunctionApprox: quadratic on the double integrator, quartic otherwise.
  kPolynomial,
  // The soft-min model passed through NonnegativeValueFunctionApprox, so that it cannot go negative.
  kNonnegativeSoftMin,
};

inline constexpr int kSoftMinComponents = 4;

// The knobs of the comparison. The FBRRT ones are exposed so the design choices can be varied from
// the command line.
struct FbrrtComparisonOptions {
  std::uint64_t seed = 20260928;
  std::size_t num_seeds = 10;
  std::size_t iterations = 30;
  std::size_t double_inverted_pendulum_iterations = 60;
  bool include_double_inverted_pendulum = true;

  ComparisonValueFunctionModel model = ComparisonValueFunctionModel::kSoftMinQuadratic;

  // The soft-min model's sharpness `beta` and diagonal floor `eps`. Its non-terminal stages start
  // as kSoftMinComponents copies of the terminal cost's bowl, centered evenly along the segment
  // from the goal to the initial mean, where the trajectories will be.
  double soft_min_sharpness = 1.0;
  double soft_min_diagonal_offset = 1e-3;
  // How far along that segment the components reach: 1 spreads them from the goal to the start, and
  // a small value keeps the starting table close to the terminal cost while still breaking the
  // symmetry between components.
  double soft_min_center_spread = 1.0;

  // The softness `s` of the nonnegative soft-min model's softplus.
  double nonnegative_softness = 0.25;

  // How DT-FBSDE fits a representation with no closed-form least squares (the soft-min model).
  ValueFunctionSgdOptions<double> dt_fbsde_sgd = [] {
    ValueFunctionSgdOptions<double> defaults;
    defaults.learning_rate = 0.05;
    return defaults;
  }();
  std::size_t dt_fbsde_sgd_epochs = 50;
  ValueFunctionFitMethod dt_fbsde_fit_method = ValueFunctionFitMethod::kSgd;

  // The FBRRT defaults are the best of a small sweep on the two 2-D benchmarks with the polynomial
  // model (three seeds each); a learning rate of 0.05 was steadier on the soft-min pendulum but slower
  // on the double integrator:
  // mostly-uniform parent draws after the first iteration, mostly-policy controls, eight epochs of
  // SGD per stage at a learning rate of 0.2, targets floored at zero, and erosion keeping 3/4.
  FbrrtOptions<double> fbrrt = [] {
    FbrrtOptions<double> defaults;
    defaults.rrt_probability = 0.2;
    defaults.exploit_probability = 0.7;
    defaults.line_search = true;
    return defaults;
  }();
  TaylorBackwardPassOptions<double> backward = [] {
    TaylorBackwardPassOptions<double> defaults;
    defaults.sgd.learning_rate = 0.2;
    defaults.epochs_per_stage = 8;
    defaults.target_floor = 0.0;
    return defaults;
  }();

  // The fraction of each stage erosion keeps, M~ / M. The thesis used 1/2 on the 2-D problems and
  // 3/4 on the double inverted pendulum; 3/4 was steadier on both 2-D problems here.
  double keep_fraction = 0.75;

  // FBRRT's erosion options beyond the keep fraction -- resampling, orphan pruning, density, age --
  // all off by default; see PathIntegralErodeOptions.
  PathIntegralErodeOptions<double> erode{};
  double double_inverted_pendulum_keep_fraction = 0.75;

  // "Converged" means within this fraction of the best cost either method reached on the problem.
  double convergence_tolerance = 0.05;
};

// One run of one method: the policy's cost before the first iteration and after each, and the
// algorithm time spent by then.
struct MethodRun {
  std::vector<double> costs{};
  std::vector<double> seconds{};

  // FBRRT only, after each iteration's erosion: the fraction of particles past the second stage
  // whose edge leaves from farther than kDisconnectionRadius from every particle at the stage before
  // -- populations that nothing at the previous stage can reach -- and the fraction whose parent is
  // gone.
  std::vector<double> disconnected_fractions{};
  std::vector<double> orphan_fractions{};
  // The largest disconnected fraction of any one stage: a population cut off from the stage before
  // is concentrated in a few stages, which the overall fraction averages away.
  std::vector<double> worst_stage_disconnected_fractions{};
};

// The radius, in the RRT's region metric -- fractions of the region's half-width -- beyond which an
// edge counts as disconnected from the stage before: 5% of the region's width.
inline constexpr double kDisconnectionRadius = 0.1;

// One method's runs over all seeds, and what they add up to.
struct MethodResult {
  std::string method{};
  std::vector<MethodRun> runs{};

  // Medians over seeds. A run that never converges counts as infinitely slow, so these are
  // infinite unless at least half the runs converged.
  double median_final_cost = 0.0;
  double median_best_cost = 0.0;
  double median_iterations_to_converge = std::numeric_limits<double>::infinity();
  double median_seconds_to_converge = std::numeric_limits<double>::infinity();
  double fraction_converged = 0.0;
  double median_seconds_per_iteration = 0.0;

  // FBRRT only: medians over seeds of the last iteration's disconnected and orphaned fractions.
  // Medians over seeds of when each run came within the convergence tolerance of its own best cost:
  // how long a method takes to settle, wherever it settles.
  double median_iterations_to_settle = 0.0;
  double median_seconds_to_settle = 0.0;

  double median_disconnected_fraction = 0.0;
  double median_worst_stage_disconnected_fraction = 0.0;
  double median_orphan_fraction = 0.0;
};

// The comparison on one benchmark.
struct BenchmarkComparison {
  std::string name{};
  std::string description{};
  double initial_cost = 0.0;
  double best_cost = 0.0;
  double convergence_threshold = 0.0;
  MethodResult fbrrt{};
  MethodResult dt_fbsde{};
  std::vector<viz::PlotlyFigure> figures{};
};

// Called after every run, so a caller can report progress.
using FbrrtComparisonProgress =
    std::function<void(const std::string& benchmark, const std::string& method, std::size_t seed, const MethodRun&)>;

// Runs the comparison on every benchmark.
//
// Fails if a model, a solver, or a figure is rejected, or if a solver fails mid-run.
[[nodiscard]] auto RunFbrrtComparison(const FbrrtComparisonOptions& options,
                                      const FbrrtComparisonProgress& progress) noexcept
    -> Result<std::vector<BenchmarkComparison>>;

// A few lines per benchmark naming what the runs showed, for the terminal and the report.
[[nodiscard]] auto SummarizeFbrrtComparison(const BenchmarkComparison& comparison) noexcept -> std::string;

}  // namespace fbsde_traj_opt::examples

#endif  // FBSDE_TRAJ_OPT_EXAMPLES_FBRRT_COMPARISON_EXPERIMENT_HPP_
