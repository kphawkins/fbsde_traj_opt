// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#ifndef FBSDE_TRAJ_OPT_SOLVERS_FBRRT_SOLVER_HPP_
#define FBSDE_TRAJ_OPT_SOLVERS_FBRRT_SOLVER_HPP_

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include "fbsde_traj_opt/costs/l1_control_sde_running_cost_term.hpp"
#include "fbsde_traj_opt/fbrrt/epsilon_greedy_l1_control_sampler.hpp"
#include "fbsde_traj_opt/fbrrt/fbrrt_particle_set.hpp"
#include "fbsde_traj_opt/fbrrt/forward_expansion.hpp"
#include "fbsde_traj_opt/fbrrt/path_integral_erode.hpp"
#include "fbsde_traj_opt/fbrrt/rrt_expansion_selector.hpp"
#include "fbsde_traj_opt/fbrrt/taylor_backward_pass.hpp"
#include "fbsde_traj_opt/policies/taylor_q_function_l1_policy.hpp"
#include "fbsde_traj_opt/sampling/counter_based_normal_sampler.hpp"
#include "fbsde_traj_opt/sampling/normal_distribution_concepts.hpp"
#include "fbsde_traj_opt/solvers/policy_rollout_evaluation.hpp"
#include "fbsde_traj_opt/utils/result.hpp"

namespace fbsde_traj_opt {

// The FBRRT method of Hawkins (2021), Section 4.3 (Algorithm 4.1), reworked, for a minimum-fuel
// (L1) problem. Each iteration:
//
//  1. Expand. Refill every stage of the particle set to `Capacity` with new edges: parents chosen
//     by Voronoi-biased RRT draws, controls by the policy or at random, and every child placed at
//     `x + K` with no noise (ExpandParticles).
//  2. Backward. Move each stage's value function a few SGD steps toward its Taylor noiseless
//     targets over the edges into the next stage (TaylorBackwardPass) -- no LSMC solve.
//  3. Evaluate. Score the policy, the Taylor Q-function minimizer over the value functions, by
//     rollouts on fixed noise (EvaluatePolicyByRollouts, with the same stream DtFbsdeIterative-
//     Solver uses, so that the two methods are scored on identical noise).
//  4. Erode. Keep the lowest path-integral-heuristic particles at every stage, each stage on its
//     own (PathIntegralErode), leaving room for the next expansion.
//
// How this departs from the thesis, and why, is documented on each component. The solver itself is
// a thin loop over them: each design choice is a type in `ComponentsT`, so an experiment with a
// different choice is a different type, not a flag threaded through the solver.
// DefaultFbrrtComponents names the nominal choice. `ComponentsT` provides the four member types
// `Selector`, `ControlSampler`, `BackwardPass` and `Eroder`, and the members `selector`,
// `control_sampler`, `backward_pass` and `eroder`.
//
// The first iteration has no policy worth exploiting, so, as in the thesis, it draws every parent
// by RRT and every control at random; later iterations use the options' probabilities.

// The knobs of the loop itself; each component has its own.
template <typename Scalar = double>
struct FbrrtOptions {
  // `eps_rrt`: the probability that a new particle's parent is drawn by RRT rather than uniformly.
  Scalar rrt_probability = Scalar{0.5};

  // `eps_opt`: the probability that a new particle's control is the policy's rather than random.
  Scalar exploit_probability = Scalar{0.5};

  // Whether to copy the particle set just before erosion, for plotting what the heuristic saw.
  bool keep_pre_erosion_snapshot = false;

  // Whether the backward pass's update must pay for itself. With the line search on, the update
  // the SGD steps proposed is scored by rolling out its policy, and adopted only if that costs less
  // than the current policy; otherwise it is shortened -- the parameters blended as
  // `theta_old + step (theta_new - theta_old)` -- halving `step` from one up to
  // `max_step_halvings` times, and if no step improves, the value functions are left as they were.
  // This is DtFbsdeIterativeSolver's safeguard against a jump that carries the policy into states
  // the value functions were never fitted in. The rollouts reuse the evaluation's fixed noise, so
  // a difference in cost is a difference between policies and not between noise draws -- and so,
  // equally, the accepted costs are optimistic about that one noise draw.
  bool line_search = false;
  std::size_t max_step_halvings = 7;

  // With the line search, whether the value functions the backward pass trains are kept apart from
  // the ones the reported policy uses. The line search as above discards a rejected update, and the
  // next backward pass starts over from the same value functions -- so if a few SGD steps are never
  // enough to lower the cost, nothing accumulates and the method stalls. With a working copy, the
  // backward pass trains its own table every iteration and never has an update taken back; that
  // table's policy drives the expansion and the targets, and its value functions the erosion. The
  // *deployed* table, whose policy is reported, returned, and evaluated, moves toward the working
  // one -- `deployed + step (working - deployed)`, `step` halving from one -- only when that rolls
  // out cheaper. Learning never stalls, and the reported policy changes only when it improves.
  bool keep_working_copy = false;

  // Whether the forward expansion adds the SDE's noise to every new particle, as the thesis did,
  // rather than placing it at `x + K`; see ExpandParticles.
  bool sample_forward_noise = false;
};

// What one iteration did.
template <typename Scalar = double>
struct FbrrtIterationReport {
  std::size_t iteration = 0;

  // The mean cost of the policy on the evaluation rollouts, after this iteration's backward pass:
  // the whole objective, and its two parts.
  Scalar mean_cost = Scalar{0};
  Scalar mean_running_cost = Scalar{0};
  Scalar mean_terminal_cost = Scalar{0};

  // The step of the backward pass's update that was adopted: one without the line search, and with
  // it the accepted fraction, or zero if no candidate improved. And how many candidates were scored.
  Scalar accepted_step = Scalar{1};
  std::size_t candidates_tried = 0;

  // The value function's own estimate of that cost, V~_0 at the initial mean.
  Scalar predicted_cost = Scalar{0};

  // The backward pass's root-mean-square estimator error, over its stages.
  Scalar rms_residual = Scalar{0};

  // The probabilities the expansion used.
  Scalar rrt_probability = Scalar{0};
  Scalar exploit_probability = Scalar{0};

  // Wall-clock time spent in each phase, in seconds. With the line search on, the evaluation is part
  // of the algorithm and includes every candidate scored.
  double expand_seconds = 0.0;
  double backward_seconds = 0.0;
  double evaluate_seconds = 0.0;
  double erode_seconds = 0.0;
};

// The nominal components: RRT expansion with brute-force nearest neighbors, epsilon-greedy L1
// controls, the SGD Taylor-noiseless backward pass, and path-integral erosion without the tree
// constraint.
template <int N, int M, std::size_t Capacity, typename ForwardModelT, typename ValueFunctionT, typename Scalar = double>
struct DefaultFbrrtComponents {
  using Selector = RrtExpansionSelector<N, BruteForceNearestNeighbor<N, Scalar>, Scalar>;
  using ControlSampler = EpsilonGreedyL1ControlSampler<N, M, Scalar>;
  using BackwardPass = TaylorBackwardPass<ValueFunctionT,
                                          ForwardModelT,
                                          L1ControlSdeRunningCostTerm<N, M, Scalar>,
                                          static_cast<int>(Capacity)>;
  using Eroder = PathIntegralErode;

  Selector selector;
  ControlSampler control_sampler;
  BackwardPass backward_pass;
  Eroder eroder;
};

// Builds the nominal components for a problem of `NumStages` stages: RRT targets drawn from the box
// [region_lower, region_upper], exploratory controls from the control bounds' vertices and zero,
// and erosion down to `erode_keep_count` particles per stage.
//
// Fails if any component rejects its arguments.
template <std::size_t Capacity,
          std::size_t NumStages,
          int N,
          int M,
          typename ForwardModelT,
          typename ValueFunctionT,
          typename Scalar = double>
[[nodiscard]] auto MakeDefaultFbrrtComponents(const ForwardModelT& forward_model,
                                              const L1ControlSdeRunningCostTerm<N, M, Scalar>& running_cost,
                                              const Eigen::Matrix<Scalar, N, 1>& region_lower,
                                              const Eigen::Matrix<Scalar, N, 1>& region_upper,
                                              const Eigen::Matrix<Scalar, M, 1>& lower_bounds,
                                              const Eigen::Matrix<Scalar, M, 1>& upper_bounds,
                                              const TaylorBackwardPassOptions<Scalar>& backward_options,
                                              std::size_t erode_keep_count,
                                              const ValueFunctionT& initial_value_function) noexcept
    -> Result<DefaultFbrrtComponents<N, M, Capacity, ForwardModelT, ValueFunctionT, Scalar>> {
  using Components = DefaultFbrrtComponents<N, M, Capacity, ForwardModelT, ValueFunctionT, Scalar>;
  RESULT_ASSERT(erode_keep_count < Capacity, "MakeDefaultFbrrtComponents: erosion must leave room to expand.");
  RESULT_ASSIGN_OR_RETURN(typename Components::Selector selector,
                          MakeBruteForceRrtExpansionSelector<N, Scalar>(region_lower, region_upper));
  RESULT_ASSIGN_OR_RETURN(typename Components::ControlSampler control_sampler,
                          Components::ControlSampler::Make(lower_bounds, upper_bounds));
  const std::vector<ValueFunctionT> table(NumStages, initial_value_function);
  RESULT_ASSIGN_OR_RETURN(typename Components::BackwardPass backward_pass,
                          Components::BackwardPass::Make(forward_model, running_cost, backward_options, table));
  RESULT_ASSIGN_OR_RETURN(PathIntegralErode eroder, PathIntegralErode::Make(erode_keep_count));
  return SuccessResult(Components{.selector = std::move(selector),
                                  .control_sampler = std::move(control_sampler),
                                  .backward_pass = std::move(backward_pass),
                                  .eroder = std::move(eroder)});
}

template <int N,
          int M,
          std::size_t Capacity,
          std::size_t NumStages,
          std::size_t NumEvaluationSamples,
          typename ForwardModelT,
          typename TerminalCostT,
          typename ValueFunctionT,
          typename ComponentsT,
          typename Scalar = double>
class FbrrtSolver {
  static_assert(NumEvaluationSamples >= 1, "Scoring a policy needs at least one rollout.");

 public:
  using State = Eigen::Matrix<Scalar, N, 1>;
  using Control = Eigen::Matrix<Scalar, M, 1>;
  using StateMat = Eigen::Matrix<Scalar, N, N>;
  using RunningCost = L1ControlSdeRunningCostTerm<N, M, Scalar>;
  using Policy = TaylorQFunctionL1Policy<N, M, ForwardModelT, ValueFunctionT, Scalar>;
  using Particles = FbrrtParticleSet<N, M, Capacity, NumStages, Scalar>;
  using Options = FbrrtOptions<Scalar>;
  using IterationReport = FbrrtIterationReport<Scalar>;

  static constexpr std::size_t kNumControlStages = NumStages - 1;

  static_assert(FbrrtExpansionSelector<typename ComponentsT::Selector, typename Particles::StateBlock, Scalar>);
  static_assert(FbrrtControlSampler<typename ComponentsT::ControlSampler, Policy, State, Control, Scalar>);
  static_assert(FbrrtEroder<typename ComponentsT::Eroder, Particles, ValueFunctionT>);

  // Builds a solver.
  //
  // `initial_value_function` is the starting guess at every stage before the terminal one, and
  // `terminal_value_function` the terminal cost in the same representation, which is never
  // updated. `seed` fixes every random number the solver draws: the initial particles, each
  // iteration's expansion and minibatches, and -- mixed with kPolicyEvaluationStream -- the
  // evaluation rollouts.
  //
  // Fails if a probability is outside [0, 1], if the initial distribution's covariance is not
  // positive definite, or if the policy cannot be built over the control bounds.
  template <typename InitialDistributionT>
    requires NormalDistribution<InitialDistributionT, State>
  static auto Make(ForwardModelT forward_model,
                   RunningCost running_cost,
                   TerminalCostT terminal_cost,
                   const InitialDistributionT& initial_distribution,
                   const Control& lower_bounds,
                   const Control& upper_bounds,
                   const ValueFunctionT& initial_value_function,
                   const ValueFunctionT& terminal_value_function,
                   ComponentsT components,
                   const Options& options,
                   std::uint64_t seed) noexcept -> Result<FbrrtSolver> {
    RESULT_ASSERT(options.rrt_probability >= Scalar{0} && options.rrt_probability <= Scalar{1},
                  "FbrrtSolver::Make: the RRT probability must lie in [0, 1].");
    RESULT_ASSERT(options.exploit_probability >= Scalar{0} && options.exploit_probability <= Scalar{1},
                  "FbrrtSolver::Make: the exploit probability must lie in [0, 1].");
    const StateMat covariance = initial_distribution.Covariance();
    const Eigen::LLT<StateMat> factorization(covariance);
    RESULT_ASSERT(factorization.info() == Eigen::Success,
                  "FbrrtSolver::Make: the initial distribution's covariance is not positive definite.");

    std::vector<ValueFunctionT> table(NumStages, initial_value_function);
    table.back() = terminal_value_function;
    RESULT_RETURN_IF_ERROR(
        Policy::Make(forward_model, running_cost.L1Weight(), lower_bounds, upper_bounds, std::span(table)));
    RESULT_ASSERT(!options.keep_working_copy || options.line_search,
                  "FbrrtSolver::Make: a working copy only has a meaning with the line search on.");

    FbrrtSolver solver(std::move(forward_model),
                       std::move(running_cost),
                       std::move(terminal_cost),
                       State(initial_distribution.Mean()),
                       StateMat(factorization.matrixL()),
                       lower_bounds,
                       upper_bounds,
                       std::move(table),
                       std::move(components),
                       options,
                       seed);
    solver.SampleInitialParticles();
    if (options.line_search) {
      solver.current_costs_ = solver.Evaluate();
    }
    return SuccessResult(std::move(solver));
  }

  // Runs one iteration: expand, backward, evaluate, erode.
  //
  // Fails if a component fails, leaving the value functions as the failing phase left them.
  auto Iterate() noexcept -> Result<IterationReport> {
    using Clock = std::chrono::steady_clock;
    IterationReport report;
    report.iteration = iteration_;
    report.rrt_probability = iteration_ == 0 ? Scalar{1} : options_.rrt_probability;
    report.exploit_probability = iteration_ == 0 ? Scalar{0} : options_.exploit_probability;
    const Policy policy = MakePolicy(table_);
    const std::uint64_t iteration_seed =
        seed_ + (0x9E3779B97F4A7C15ULL * (static_cast<std::uint64_t>(iteration_) + 1U));

    auto start = Clock::now();
    const auto lap = [&start]() noexcept {
      const auto now = Clock::now();
      const double seconds = std::chrono::duration<double>(now - start).count();
      start = now;
      return seconds;
    };

    RESULT_RETURN_IF_ERROR(ExpandParticles(particles_,
                                           forward_model_,
                                           running_cost_,
                                           components_.selector,
                                           components_.control_sampler,
                                           policy,
                                           report.rrt_probability,
                                           report.exploit_probability,
                                           iteration_seed,
                                           options_.sample_forward_noise));
    report.expand_seconds = lap();

    if (options_.line_search && !options_.keep_working_copy) {
      SaveParameters(previous_parameters_);
    }
    RESULT_ASSIGN_OR_RETURN(const auto backward, components_.backward_pass(table_, particles_, policy, iteration_seed));
    report.rms_residual = backward.rms_residual;
    report.backward_seconds = lap();

    PolicyRolloutCosts<Scalar> costs;
    if (options_.keep_working_copy) {
      RESULT_ASSIGN_OR_RETURN(costs, WorkingCopyLineSearch(report));
    } else if (options_.line_search) {
      RESULT_ASSIGN_OR_RETURN(costs, LineSearch(report));
    } else {
      costs = Evaluate();
      report.candidates_tried = 1;
    }
    report.mean_cost = costs.total;
    report.mean_running_cost = costs.running;
    report.mean_terminal_cost = costs.terminal;
    report.predicted_cost = DeployedTable().front()(initial_mean_);
    report.evaluate_seconds = lap();

    if (options_.keep_pre_erosion_snapshot) {
      pre_erosion_particles_ = particles_;
    }
    RESULT_RETURN_IF_ERROR(components_.eroder(particles_, table_));
    report.erode_seconds = lap();

    ++iteration_;
    return SuccessResult(report);
  }

  // The current policy: the Taylor Q-function minimizer over the current value functions -- the
  // deployed ones, with a working copy. It reads them by reference, so it tracks every later
  // iteration and must not outlive the solver.
  [[nodiscard]] auto CurrentPolicy() const noexcept -> Policy { return MakePolicy(DeployedTable()); }

  // The mean cost of the current policy on the evaluation rollouts.
  [[nodiscard]] auto Evaluate() const noexcept -> PolicyRolloutCosts<Scalar> { return Evaluate(DeployedTable()); }

  // The value functions of the current policy, one per stage 0..K.
  [[nodiscard]] auto value_functions() const noexcept -> std::span<const ValueFunctionT> { return DeployedTable(); }

  // The value functions the backward pass trains: the same as value_functions() unless the solver
  // keeps a working copy.
  [[nodiscard]] auto working_value_functions() const noexcept -> std::span<const ValueFunctionT> { return table_; }

  // The particle set as the last iteration left it, after erosion.
  [[nodiscard]] auto particles() const noexcept -> const Particles& { return particles_; }

  // The particle set just before the last erosion, with its heuristics; empty unless
  // `keep_pre_erosion_snapshot` is set.
  [[nodiscard]] auto pre_erosion_particles() const noexcept -> const Particles& { return pre_erosion_particles_; }

  [[nodiscard]] auto iteration_count() const noexcept -> std::size_t { return iteration_; }

 private:
  // The stream the initial particles are drawn from.
  static constexpr std::uint64_t kInitialParticleStream = 0xBB67AE8584CAA73BULL;

  FbrrtSolver(ForwardModelT forward_model,
              RunningCost running_cost,
              TerminalCostT terminal_cost,
              State initial_mean,
              StateMat initial_covariance_factor,
              Control lower_bounds,
              Control upper_bounds,
              std::vector<ValueFunctionT> table,
              ComponentsT components,
              const Options& options,
              std::uint64_t seed) noexcept
      : forward_model_(std::move(forward_model)),
        running_cost_(std::move(running_cost)),
        terminal_cost_(std::move(terminal_cost)),
        initial_mean_(std::move(initial_mean)),
        initial_covariance_factor_(std::move(initial_covariance_factor)),
        lower_bounds_(std::move(lower_bounds)),
        upper_bounds_(std::move(upper_bounds)),
        table_(std::move(table)),
        components_(std::move(components)),
        options_(options),
        seed_(seed),
        deployed_(table_),
        trial_(table_) {}

  using ParameterVector = typename ValueFunctionT::ParameterVector;

  [[nodiscard]] auto DeployedTable() const noexcept -> const std::vector<ValueFunctionT>& {
    return options_.keep_working_copy ? deployed_ : table_;
  }

  [[nodiscard]] auto MakePolicy(const std::vector<ValueFunctionT>& table) const noexcept -> Policy {
    // Cannot fail: Make() built one over the same bounds and a table of the same size.
    return *Policy::Make(forward_model_, running_cost_.L1Weight(), lower_bounds_, upper_bounds_, std::span(table));
  }

  [[nodiscard]] auto Evaluate(const std::vector<ValueFunctionT>& table) const noexcept -> PolicyRolloutCosts<Scalar> {
    return EvaluatePolicyByRollouts(forward_model_,
                                    MakePolicy(table),
                                    running_cost_,
                                    terminal_cost_,
                                    initial_mean_,
                                    initial_covariance_factor_,
                                    NumEvaluationSamples,
                                    kNumControlStages,
                                    seed_ ^ kPolicyEvaluationStream);
  }

  // Scores `deployed + step (working - deployed)`, halving `step` from one, and deploys the first
  // that improves on the deployed cost. The working table is left as the backward pass left it.
  auto WorkingCopyLineSearch(IterationReport& report) noexcept -> Result<PolicyRolloutCosts<Scalar>> {
    auto step = Scalar{1};
    for (std::size_t halving = 0; halving <= options_.max_step_halvings; ++halving, step *= Scalar{0.5}) {
      for (std::size_t stage = 0; stage < kNumControlStages; ++stage) {
        const ParameterVector& deployed = deployed_[stage].Parameters();
        RESULT_RETURN_IF_ERROR(
            trial_[stage].SetParameters(deployed + (step * (table_[stage].Parameters() - deployed))));
      }
      ++report.candidates_tried;
      const PolicyRolloutCosts<Scalar> costs = Evaluate(trial_);
      if (costs.total < current_costs_.total) {
        for (std::size_t stage = 0; stage < kNumControlStages; ++stage) {
          RESULT_RETURN_IF_ERROR(deployed_[stage].SetParameters(trial_[stage].Parameters()));
        }
        current_costs_ = costs;
        report.accepted_step = step;
        return SuccessResult(costs);
      }
    }
    report.accepted_step = Scalar{0};
    return SuccessResult(current_costs_);
  }

  // Copies every stage's parameters but the terminal one's, which never changes.
  auto SaveParameters(std::vector<ParameterVector>& parameters_out) const noexcept -> void {
    parameters_out.resize(kNumControlStages);
    for (std::size_t stage = 0; stage < kNumControlStages; ++stage) {
      parameters_out[stage] = table_[stage].Parameters();
    }
  }

  // Sets every non-terminal stage to `previous + step (proposed - previous)`, in place, so that a
  // policy's span over the table stays valid.
  auto Blend(const std::vector<ParameterVector>& proposed, Scalar step) noexcept -> Result<> {
    for (std::size_t stage = 0; stage < kNumControlStages; ++stage) {
      const ParameterVector blended =
          previous_parameters_[stage] + (step * (proposed[stage] - previous_parameters_[stage]));
      RESULT_RETURN_IF_ERROR(table_[stage].SetParameters(blended));
    }
    return SuccessResult();
  }

  // Scores the backward pass's update, shortening it until it improves on the current cost or the
  // halvings run out, and returns the costs of the policy the table is left with.
  auto LineSearch(IterationReport& report) noexcept -> Result<PolicyRolloutCosts<Scalar>> {
    std::vector<ParameterVector> proposed;
    SaveParameters(proposed);
    auto step = Scalar{1};
    for (std::size_t halving = 0; halving <= options_.max_step_halvings; ++halving, step *= Scalar{0.5}) {
      RESULT_RETURN_IF_ERROR(Blend(proposed, step));
      ++report.candidates_tried;
      const PolicyRolloutCosts<Scalar> costs = Evaluate();
      if (costs.total < current_costs_.total) {
        current_costs_ = costs;
        report.accepted_step = step;
        return SuccessResult(costs);
      }
    }
    report.accepted_step = Scalar{0};
    RESULT_RETURN_IF_ERROR(Blend(proposed, Scalar{0}));
    return SuccessResult(current_costs_);
  }

  auto SampleInitialParticles() noexcept -> void {
    const CounterBasedNormalSampler sampler(seed_ ^ kInitialParticleStream);
    typename Particles::StateBlock states;
    State noise;
    for (std::size_t slot = 0; slot < Capacity; ++slot) {
      sampler.FillNoise(slot, 0, noise);
      states.col(static_cast<Eigen::Index>(slot)) = initial_mean_ + (initial_covariance_factor_ * noise);
    }
    particles_.SetInitialStates(states);
  }

  ForwardModelT forward_model_;
  RunningCost running_cost_;
  TerminalCostT terminal_cost_;
  State initial_mean_;
  StateMat initial_covariance_factor_;
  Control lower_bounds_;
  Control upper_bounds_;
  // One value function per stage 0..K. Never resized, so a policy's span over it stays valid.
  std::vector<ValueFunctionT> table_;
  ComponentsT components_;
  Options options_;
  std::uint64_t seed_;

  Particles particles_;
  Particles pre_erosion_particles_;
  std::size_t iteration_ = 0;

  // The line search's state: the cost of the current policy, and the parameters the backward pass
  // started from.
  PolicyRolloutCosts<Scalar> current_costs_{.total = std::numeric_limits<Scalar>::infinity()};

  // With a working copy: the deployed table, and the candidate the line search is scoring. Never
  // resized, like `table_`.
  std::vector<ValueFunctionT> deployed_;
  std::vector<ValueFunctionT> trial_;
  std::vector<ParameterVector> previous_parameters_;
};

}  // namespace fbsde_traj_opt

#endif  // FBSDE_TRAJ_OPT_SOLVERS_FBRRT_SOLVER_HPP_
