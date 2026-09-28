// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#ifndef FBSDE_TRAJ_OPT_DT_FBSDE_ITERATIVE_SOLVER_HPP_
#define FBSDE_TRAJ_OPT_DT_FBSDE_ITERATIVE_SOLVER_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <span>
#include <utility>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include "fbsde_traj_opt/counter_based_normal_sampler.hpp"
#include "fbsde_traj_opt/l1_control_sde_running_cost_term.hpp"
#include "fbsde_traj_opt/normal_distribution_concepts.hpp"
#include "fbsde_traj_opt/sde_term_concepts.hpp"
#include "fbsde_traj_opt/taylor_q_function_l1_policy.hpp"
#include "fbsde_traj_opt/utils/parallel_for.hpp"
#include "fbsde_traj_opt/utils/result.hpp"
#include "fbsde_traj_opt/value_function_approx_concepts.hpp"

namespace fbsde_traj_opt {

// The DT-FBSDE iterative method of Hawkins (2021), Section 4.6, for a minimum-fuel (L1) problem:
// sample a batch of parallel trajectories, fit a value function at every stage by a backward
// pass, improve the policy through the Taylor Q-function, and resample with the improved policy.
//
// One iteration, in the thesis's terms:
//
//  1. Forward pass. The batch's paths are sampled in parallel under the drift
//
//       K_i = F_i(X_i, u_i),   u_i = mu_i(X_i)          with probability 1 - epsilon,   (5.13)
//                              u_i ~ U_rand             with probability epsilon,       (5.14)
//
//     where `mu` is the current policy and `U_rand` is the vertex set of the control box together
//     with zero -- `{-1, 0, 1}` for a unit bound. That set is the thesis's own choice for L1
//     problems: an L1-optimal policy only ever returns values in it. The rest of the batch is
//     one-step *probes* -- a path's state stepped once under a control from `U_rand` -- which put
//     samples wherever the Q-function minimization of step 3 will query the next stage's value
//     function without letting exploration compound along a path; see SampleForwardPass(). The
//     first iteration, which has no policy to exploit, explores with probability one, as the
//     thesis's first FBRRT iteration does, and the probability then decays on a schedule.
//
//  2. Backward pass. Starting from the terminal cost, each stage's value function is fitted by
//     the least squares of (5.12) -- optionally with its local-entropy weights -- to noiseless
//     Taylor targets (4.37),
//
//       Y^_i = Q~_i(X_i, u) = l(X_i, u) + V~_{i+1}(X_i + F_i(X_i, u)) + 1/2 tr(M_{i+1}),
//
//     taken at every sampled state regardless of how the sample got there, which is what makes
//     the pass off-policy. Which control `u` is a choice (DtFbsdeBackwardTarget): the current
//     policy's, which is Section 4.6's policy evaluation, or the greedy one chosen and scored by
//     two independently fitted halves of the batch. Either way the choice of control is kept
//     independent of the errors in the table being fitted; BackwardPass() explains why that
//     matters over a long horizon.
//
//  3. Policy improvement, damped. The thesis names the chief failure of the iterative method
//     (Appendix B.7): a large jump in the policy carries the trajectories into states where the
//     value function was never fitted, and the method diverges. So the new value functions are
//     not adopted outright. They are blended into the current ones coefficient by coefficient,
//
//       V~_i <- (1 - eta) V~_i + eta V~_i^new,
//
//     and `eta` is halved from one until the blended policy's cost, measured by rolling it out,
//     is lower than the current policy's. Section 4.4's observation that `V^pi <= V^mu` makes pi
//     an improvement over mu is thereby checked rather than assumed. The rollouts that make the
//     comparison use the same noise for every candidate -- the evaluation seed is fixed -- so a
//     difference in cost is a difference between policies and not between noise draws.
//
// The value function representation must be a linear-in-parameters model whose coefficient
// vectors can be blended (PolynomialValueFunctionApprox with its fixed normalization is one) and
// that can fit itself by weighted least squares.
//
// `N` and `M` are the state and control dimensions, `NumSamples` the width of the forward batch,
// `NumStages` the number of stages including the terminal one, `NumEvaluationSamples` the
// number of rollouts used to score a candidate policy, and `ForwardModelT` a
// ComposedForwardSdeModel of dimensions `N` and `M`.

// Which control the backward pass's targets are taken at; see BackwardPass().
enum class DtFbsdeBackwardTarget : std::uint8_t {
  // The current policy's control: the targets estimate V^mu, and improvement happens between
  // iterations. Section 4.6's scheme.
  kCurrentPolicy,
  // The greedy control, chosen and evaluated by two independently fitted halves of the batch.
  kDoubleGreedy,
};

// The knobs of the method.
template <typename Scalar = double>
struct DtFbsdeIterativeOptions {
  DtFbsdeBackwardTarget backward_target = DtFbsdeBackwardTarget::kCurrentPolicy;

  // `epsilon` above: the probability that a (sample, stage) pair takes an exploratory control
  // rather than the policy's. It follows a geometric schedule from the initial value down to this
  // one,
  //
  //   epsilon_j = epsilon + (epsilon_0 - epsilon) * decay^j,
  //
  // so that the early iterations sample broadly -- they have no trustworthy policy to exploit,
  // and a broad batch is what lets the fit find a better basin than the one the first policy
  // happened to fall into -- and the later ones sample tightly around the policy they are
  // refining. A decay of zero explores fully on the first iteration only.
  Scalar exploration_probability = Scalar{0.03};

  // `epsilon_0`: the exploration probability of the first iteration, before any policy exists.
  Scalar initial_exploration_probability = Scalar{1};

  // The per-iteration decay of the schedule, in [0, 1).
  Scalar exploration_decay = Scalar{0};

  // The ridge weight of the least-squares fit; see PolynomialValueFunctionApprox.
  Scalar ridge = Scalar{1e-6};

  // The fraction of the batch given over to one-step probes rather than full paths; see
  // SampleForwardPass() for what a probe is and why. In [0, 1).
  Scalar probe_fraction = Scalar{0.25};

  // A floor under every target. The true value function of a problem whose costs are all
  // non-negative is itself non-negative, so a target below zero is extrapolation error by
  // construction; flooring it stops a fitted function from learning a basin that does not exist
  // and the policy from steering into it. Negative infinity disables the floor.
  Scalar target_floor = -std::numeric_limits<Scalar>::infinity();

  // `lambda` of the local-entropy weights of (5.12), in units of the heuristic's standard
  // deviation over the batch; see FitStage(). Zero weights every sample equally.
  Scalar entropy_temperature = Scalar{0};

  // How many times the blending step may be halved before the iteration gives up and keeps the
  // current value functions: 7 tries steps down to 1/128.
  std::size_t max_step_halvings = 7;
};

// What one iteration did.
template <typename Scalar = double>
struct DtFbsdeIterationReport {
  std::size_t iteration = 0;

  // The blending step that was accepted, or zero if none improved on the current policy.
  Scalar accepted_step = Scalar{0};

  // The mean cost of the current policy on the evaluation rollouts, after this iteration's update:
  // the whole objective, and its two parts.
  Scalar mean_cost = Scalar{0};
  Scalar mean_running_cost = Scalar{0};
  Scalar mean_terminal_cost = Scalar{0};

  // The mean cost the forward batch itself incurred, exploration included -- the cost of the
  // sampling distribution rather than of the policy.
  Scalar sampled_mean_cost = Scalar{0};

  // The exploration probability the forward pass used.
  Scalar exploration_probability = Scalar{0};

  // How many candidate steps were rolled out before one was accepted or the halvings ran out.
  std::size_t candidates_tried = 0;
};

template <int N,
          int M,
          std::size_t NumSamples,
          std::size_t NumStages,
          std::size_t NumEvaluationSamples,
          typename ForwardModelT,
          typename TerminalCostT,
          typename ValueFunctionT,
          typename Scalar = double>
  requires SdeTerminalCostTerm<TerminalCostT, Eigen::Matrix<Scalar, N, 1>> &&
           ParameterizedValueFunctionApprox<ValueFunctionT, Eigen::Matrix<Scalar, N, 1>>
class DtFbsdeIterativeSolver {
  static_assert(NumSamples >= 1, "The forward pass needs at least one sample.");
  static_assert(NumStages >= 2, "A problem needs at least an initial and a terminal stage.");
  static_assert(NumEvaluationSamples >= 1, "Scoring a policy needs at least one rollout.");
  static_assert(NumSamples % 2 == 0, "Double estimation splits the batch into two equal halves.");

 public:
  using State = Eigen::Matrix<Scalar, N, 1>;
  using Control = Eigen::Matrix<Scalar, M, 1>;
  using StateMat = Eigen::Matrix<Scalar, N, N>;
  using RunningCost = L1ControlSdeRunningCostTerm<N, M, Scalar>;
  using Policy = TaylorQFunctionL1Policy<N, M, ForwardModelT, ValueFunctionT, Scalar>;
  using Options = DtFbsdeIterativeOptions<Scalar>;
  using IterationReport = DtFbsdeIterationReport<Scalar>;

  static constexpr int kSampleCount = static_cast<int>(NumSamples);
  static constexpr auto kControlDim = static_cast<std::size_t>(M);
  static constexpr std::size_t kNumControlStages = NumStages - 1;

  using StateBlock = ValueFunctionStateBatch<State, kSampleCount>;
  using ValueBlock = ValueFunctionValueBatch<State, kSampleCount>;
  using ControlBlock = Eigen::Matrix<Scalar, M, kSampleCount>;

  // Builds a solver.
  //
  // `initial_value_function` is both the starting guess at every stage before the terminal one
  // and, through its normalization, the basis every later fit shares; `terminal_value_function`
  // is the terminal cost expressed in that same representation, which the backward pass starts
  // from and never refits. `seed` fixes every random number the solver draws.
  //
  // Fails if an option is out of range, if the initial distribution's covariance is not positive
  // definite, or if the policy cannot be built over the control bounds.
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
                   const Options& options,
                   std::uint64_t seed) noexcept -> Result<DtFbsdeIterativeSolver> {
    RESULT_ASSERT(options.exploration_probability >= Scalar{0} && options.exploration_probability <= Scalar{1} &&
                      options.initial_exploration_probability >= Scalar{0} &&
                      options.initial_exploration_probability <= Scalar{1},
                  "DtFbsdeIterativeSolver::Make: exploration probabilities must lie in [0, 1].");
    RESULT_ASSERT(options.exploration_decay >= Scalar{0} && options.exploration_decay < Scalar{1},
                  "DtFbsdeIterativeSolver::Make: the exploration decay must lie in [0, 1).");
    RESULT_ASSERT(std::isfinite(options.ridge) && options.ridge >= Scalar{0},
                  "DtFbsdeIterativeSolver::Make: the ridge weight must be finite and non-negative.");
    RESULT_ASSERT(options.probe_fraction >= Scalar{0} && options.probe_fraction < Scalar{1},
                  "DtFbsdeIterativeSolver::Make: the probe fraction must lie in [0, 1).");
    RESULT_ASSERT(std::isfinite(options.entropy_temperature) && options.entropy_temperature >= Scalar{0},
                  "DtFbsdeIterativeSolver::Make: the entropy temperature must be finite and non-negative.");
    RESULT_ASSERT(!std::isnan(options.target_floor), "DtFbsdeIterativeSolver::Make: the target floor may not be NaN.");
    RESULT_ASSERT((lower_bounds.array() <= Scalar{0}).all() && (upper_bounds.array() >= Scalar{0}).all() &&
                      lower_bounds.allFinite() && upper_bounds.allFinite(),
                  "DtFbsdeIterativeSolver::Make: every control's bounds must be finite and contain zero.");

    const StateMat covariance = initial_distribution.Covariance();
    const Eigen::LLT<StateMat> factorization(covariance);
    RESULT_ASSERT(factorization.info() == Eigen::Success,
                  "DtFbsdeIterativeSolver::Make: the initial distribution's covariance is not positive definite.");

    DtFbsdeIterativeSolver solver(std::move(forward_model),
                                  std::move(running_cost),
                                  std::move(terminal_cost),
                                  State(initial_distribution.Mean()),
                                  StateMat(factorization.matrixL()),
                                  lower_bounds,
                                  upper_bounds,
                                  initial_value_function,
                                  terminal_value_function,
                                  options,
                                  seed);
    solver.current_cost_ = solver.Evaluate(solver.current_).total;
    return SuccessResult(std::move(solver));
  }

  // Runs one iteration: forward pass, backward pass, damped policy improvement.
  //
  // Fails if a stage's regression fails, which leaves the current value functions as they were.
  auto Iterate() noexcept -> Result<IterationReport> {
    IterationReport report;
    report.iteration = iteration_;

    report.exploration_probability = ExplorationProbability(iteration_);
    report.sampled_mean_cost = SampleForwardPass(report.exploration_probability, SamplingSeed(iteration_));

    if (const Result<> fitted = BackwardPass(); !fitted.has_value()) {
      return ErrorResult(fitted.error().message, fitted.error().location);
    }

    auto step = Scalar{1};
    for (std::size_t halving = 0; halving <= options_.max_step_halvings; ++halving, step *= Scalar{0.5}) {
      if (const Result<> blended = BlendIntoTrial(step); !blended.has_value()) {
        return ErrorResult(blended.error().message, blended.error().location);
      }
      ++report.candidates_tried;
      const Costs trial_cost = Evaluate(trial_);
      if (trial_cost.total < current_cost_) {
        // Copied element by element rather than swapped, so that `current_` keeps its storage and
        // a policy handed out by CurrentPolicy() keeps reading the live table.
        std::ranges::copy(trial_, current_.begin());
        current_cost_ = trial_cost.total;
        report.accepted_step = step;
        break;
      }
    }

    const Costs costs = Evaluate(current_);
    report.mean_cost = costs.total;
    report.mean_running_cost = costs.running;
    report.mean_terminal_cost = costs.terminal;
    ++iteration_;
    return SuccessResult(report);
  }

  // The current policy: the Taylor Q-function minimizer over the current value functions. It
  // reads them by reference, so it tracks every later iteration and must not outlive the solver.
  [[nodiscard]] auto CurrentPolicy() const noexcept -> Policy { return MakePolicy(current_); }

  // The current value functions, one per stage 0..K.
  [[nodiscard]] auto ValueFunctions() const noexcept -> std::span<const ValueFunctionT> { return current_; }

  // The value functions the most recent backward pass fitted, before any blending: the method's
  // estimate of the value of the policy that is greedy with respect to them.
  [[nodiscard]] auto FittedValueFunctions() const noexcept -> std::span<const ValueFunctionT> { return fitted_; }

  // The mean cost of the current policy on the evaluation rollouts.
  [[nodiscard]] auto CurrentCost() const noexcept -> Scalar { return current_cost_; }

  [[nodiscard]] auto IterationCount() const noexcept -> std::size_t { return iteration_; }

  // The states and controls of the most recent forward pass, one block per stage, for plotting
  // what the sampling distribution looked like.
  [[nodiscard]] auto SampledStates() const noexcept -> std::span<const StateBlock> { return states_; }

  [[nodiscard]] auto SampledControls() const noexcept -> std::span<const ControlBlock> { return controls_; }

 private:
  struct Costs {
    Scalar total = Scalar{0};
    Scalar running = Scalar{0};
    Scalar terminal = Scalar{0};
  };

  // Seeds for the forward passes and the evaluation rollouts, kept apart so that no forward pass
  // ever reuses the evaluation's noise and the evaluation's noise never changes.
  static constexpr std::uint64_t kEvaluationStream = 0xD1B54A32D192ED03ULL;

  // The sampler components the forward pass draws its non-noise randomness from, past the N used
  // for the state noise.
  static constexpr std::size_t kExploreComponent = static_cast<std::size_t>(N);
  static constexpr std::size_t kParentPickComponent = static_cast<std::size_t>(N) + 1;
  static constexpr std::size_t kControlPickComponent = static_cast<std::size_t>(N) + 2;

  DtFbsdeIterativeSolver(ForwardModelT forward_model,
                         RunningCost running_cost,
                         TerminalCostT terminal_cost,
                         State initial_mean,
                         StateMat initial_covariance_factor,
                         Control lower_bounds,
                         Control upper_bounds,
                         const ValueFunctionT& initial_value_function,
                         const ValueFunctionT& terminal_value_function,
                         const Options& options,
                         std::uint64_t seed) noexcept
      : forward_model_(std::move(forward_model)),
        running_cost_(std::move(running_cost)),
        terminal_cost_(std::move(terminal_cost)),
        initial_mean_(std::move(initial_mean)),
        initial_covariance_factor_(std::move(initial_covariance_factor)),
        lower_bounds_(std::move(lower_bounds)),
        upper_bounds_(std::move(upper_bounds)),
        options_(options),
        seed_(seed),
        current_(NumStages, initial_value_function),
        fitted_(NumStages, initial_value_function),
        trial_(NumStages, initial_value_function),
        halves_{std::vector<ValueFunctionT>(NumStages, initial_value_function),
                std::vector<ValueFunctionT>(NumStages, initial_value_function)},
        states_(NumStages),
        controls_(kNumControlStages),
        accrued_before_(NumStages, ValueBlock::Zero()) {
    current_.back() = terminal_value_function;
    fitted_.back() = terminal_value_function;
    trial_.back() = terminal_value_function;
    halves_[0].back() = terminal_value_function;
    halves_[1].back() = terminal_value_function;
  }

  [[nodiscard]] auto ExplorationProbability(std::size_t iteration) const noexcept -> Scalar {
    if (iteration == 0) {
      return options_.initial_exploration_probability;
    }
    return options_.exploration_probability +
           ((options_.initial_exploration_probability - options_.exploration_probability) *
            std::pow(options_.exploration_decay, static_cast<Scalar>(iteration)));
  }

  [[nodiscard]] auto SamplingSeed(std::size_t iteration) const noexcept -> std::uint64_t {
    return seed_ + (0x9E3779B97F4A7C15ULL * (static_cast<std::uint64_t>(iteration) + 1U));
  }

  [[nodiscard]] auto MakePolicy(const std::vector<ValueFunctionT>& table) const noexcept -> Policy {
    // Cannot fail: Make() checked the bounds and the table has NumStages >= 2 entries.
    return *Policy::Make(forward_model_, running_cost_.L1Weight(), lower_bounds_, upper_bounds_, table);
  }

  // A standard normal variate mapped to a uniform on (0, 1) through the normal CDF, so that the
  // one coordinate-addressed sampler supplies both kinds of randomness.
  [[nodiscard]] static auto UniformFromNormal(Scalar normal) noexcept -> Scalar {
    return Scalar{0.5} * std::erfc(-normal / std::numbers::sqrt2_v<Scalar>);
  }

  // One exploratory control: each coordinate drawn uniformly from {lower, 0, upper}, the set an
  // L1-optimal policy's values lie in.
  [[nodiscard]] auto ExploratoryControl(const CounterBasedNormalSampler& sampler,
                                        std::size_t sample,
                                        std::size_t stage) const noexcept -> Control {
    Control control;
    for (std::size_t coordinate = 0; coordinate < kControlDim; ++coordinate) {
      const auto index = static_cast<Eigen::Index>(coordinate);
      const Scalar pick = UniformFromNormal(sampler.Sample(sample, stage, kControlPickComponent + coordinate));
      if (pick < Scalar{1} / Scalar{3}) {
        control[index] = lower_bounds_[index];
      } else if (pick < Scalar{2} / Scalar{3}) {
        control[index] = Scalar{0};
      } else {
        control[index] = upper_bounds_[index];
      }
    }
    return control;
  }

  // The forward pass. Returns the mean cost the path columns incurred.
  //
  // The batch's columns are of two kinds. The first `PathCount()` are *paths*: trajectories from
  // the initial distribution under the current policy, each stage's control replaced by an
  // exploratory one with the scheduled probability. The rest are *probes*: at every stage, each
  // probe column holds a one-step branch off a randomly chosen path -- that path's state at the
  // previous stage, stepped once under an exploratory control. A probe's future is never
  // simulated, because nothing needs it: the backward pass's targets (4.37) depend on a sample's
  // state alone, so a probe is a collocation point and nothing more. That is what makes probes
  // cheap coverage. They put samples where the Q-function minimization will query the next
  // stage's value function -- one control step to either side of where the policy goes, which on
  // an underactuated system can be far outside the policy's own spread -- without letting that
  // exploration compound along a path and smear the whole batch across the state space.
  //
  // Sampler coordinates: stage coordinate k+1 drives the k -> k+1 transition, as in
  // TrajectoryBatch. Components 0..N-1 are the state noise, the next decides whether a path
  // explores, the next picks a probe's parent, and the last M pick an exploratory control.
  auto SampleForwardPass(Scalar exploration, std::uint64_t seed) noexcept -> Scalar {
    const CounterBasedNormalSampler sampler(seed);
    const Policy policy = MakePolicy(current_);
    const std::size_t path_count = PathCount();

    State noise;
    for (std::size_t sample = 0; sample < NumSamples; ++sample) {
      sampler.FillNoise(sample, 0, noise);
      states_[0].col(static_cast<Eigen::Index>(sample)) = initial_mean_ + (initial_covariance_factor_ * noise);
    }

    // The running cost each column has accrued, before and after the current stage: two buffers,
    // because a probe reads its parent path's total while the paths are updating theirs.
    ValueBlock accrued = ValueBlock::Zero();
    ValueBlock next_accrued = ValueBlock::Zero();
    for (std::size_t stage = 0; stage < kNumControlStages; ++stage) {
      accrued_before_[stage] = accrued;
      ParallelFor(NumSamples, [&](std::size_t sample) noexcept {
        const auto column = static_cast<Eigen::Index>(sample);
        const bool is_path = sample < path_count;

        // A path steps from its own state; a probe from a path's.
        std::size_t parent = sample;
        if (!is_path) {
          const Scalar pick = UniformFromNormal(sampler.Sample(sample, stage + 1, kParentPickComponent));
          parent = std::min(static_cast<std::size_t>(pick * static_cast<Scalar>(path_count)), path_count - 1);
        }
        const State state = states_[stage].col(static_cast<Eigen::Index>(parent));

        const bool explore =
            !is_path || UniformFromNormal(sampler.Sample(sample, stage + 1, kExploreComponent)) < exploration;
        const Control control = explore ? ExploratoryControl(sampler, sample, stage + 1) : policy(stage, state);

        State noise;
        sampler.FillNoise(sample, stage + 1, noise);
        states_[stage + 1].col(column) = forward_model_(stage, state, control, noise);
        controls_[stage].col(column) = control;
        next_accrued[column] = accrued[static_cast<Eigen::Index>(parent)] + running_cost_(stage, state, control);
      });
      accrued = next_accrued;
    }
    accrued_before_[kNumControlStages] = accrued;

    auto total = Scalar{0};
    for (std::size_t sample = 0; sample < path_count; ++sample) {
      const auto column = static_cast<Eigen::Index>(sample);
      total += accrued[column] + terminal_cost_(State(states_[kNumControlStages].col(column)));
    }
    return total / static_cast<Scalar>(path_count);
  }

  // The number of path columns; the rest of the batch is probes. At least one, whatever the
  // probe fraction.
  [[nodiscard]] auto PathCount() const noexcept -> std::size_t {
    const auto probes = static_cast<std::size_t>(options_.probe_fraction * static_cast<Scalar>(NumSamples));
    return std::max<std::size_t>(NumSamples - std::min(probes, NumSamples), 1);
  }

  // The backward pass, writing a freshly fitted value function for every stage into `fitted_`.
  //
  // The targets evaluate the *current* policy mu, as Section 4.6 prescribes: the control at each
  // sample is mu's, and only the continuation value is read from the table being fitted,
  //
  //   Y^_i = Q~_i(X_i, mu_i(X_i)) = l(X_i, mu_i) + V~_{i+1}(X_i + F_i(X_i, mu_i)) + 1/2 tr(M_{i+1}),
  //
  // the noiseless estimator (4.37) of V^mu. The improvement to argmin_u Q~ happens afterwards,
  // between iterations. Taking the minimum inside the pass instead -- value iteration -- looks
  // like a shortcut and is not: minimizing over a fitted V~_{i+1} selects whichever control lands
  // where the fit's error is most negative, every stage adds that optimism to the last, and over a
  // horizon of a hundred stages the estimated value from the start drains to nothing while the
  // realized cost stays where it was. With mu fixed, the controls are chosen independently of the
  // errors in the table being fitted, and the bias has nothing to select.
  auto BackwardPass() noexcept -> Result<> {
    if (options_.backward_target == DtFbsdeBackwardTarget::kDoubleGreedy) {
      return DoubleGreedyBackwardPass();
    }
    const Policy target_policy = MakePolicy(current_);
    const Policy continuation = MakePolicy(fitted_);

    for (std::size_t stage = kNumControlStages; stage-- > 0;) {
      const StateBlock& states = states_[stage];
      ValueBlock targets;
      ParallelFor(NumSamples, [&](std::size_t sample) noexcept {
        const auto column = static_cast<Eigen::Index>(sample);
        const State state = states.col(column);
        targets[column] =
            std::max(continuation.QValue(stage, state, target_policy(stage, state)), options_.target_floor);
      });
      fitted_[stage] = current_[stage];
      if (const Result<> fitted = FitStage<kSampleCount>(fitted_[stage], states, targets, accrued_before_[stage]);
          !fitted.has_value()) {
        return fitted;
      }
    }
    return SuccessResult();
  }

  // The backward pass with greedy targets and double estimation.
  //
  // Greedy targets, `min_u Q~_i(X_i, u)`, estimate the value of the improved policy directly, so
  // an iteration can improve by more than one policy-iteration step. Their hazard is the
  // selection bias described above: the minimizing control is chosen by the same fitted table
  // that then scores it. Double estimation (van Hasselt, "Double Q-learning", 2010) separates the
  // two roles. The batch is split into two halves, each fitted to its own table; a sample in one
  // half has its control chosen by its own table's Q-function and *scored* by the other's, whose
  // errors are independent of the choice. The table the policy uses is the average of the two.
  auto DoubleGreedyBackwardPass() noexcept -> Result<> {
    constexpr int kHalf = kSampleCount / 2;
    using HalfStates = ValueFunctionStateBatch<State, kHalf>;
    using HalfValues = ValueFunctionValueBatch<State, kHalf>;

    const std::array<Policy, 2> choosers{MakePolicy(halves_[0]), MakePolicy(halves_[1])};
    for (std::size_t stage = kNumControlStages; stage-- > 0;) {
      for (std::size_t half = 0; half < 2; ++half) {
        const Policy& chooser = choosers[half];
        const Policy& scorer = choosers[1 - half];
        HalfStates states;
        HalfValues targets;
        HalfValues accrued;
        ParallelFor(static_cast<std::size_t>(kHalf), [&](std::size_t slot) noexcept {
          const auto column = static_cast<Eigen::Index>(slot);
          const State state = states_[stage].col(static_cast<Eigen::Index>((2 * slot) + half));
          states.col(column) = state;
          accrued[column] = accrued_before_[stage][static_cast<Eigen::Index>((2 * slot) + half)];
          targets[column] = std::max(scorer.QValue(stage, state, chooser(stage, state)), options_.target_floor);
        });
        halves_[half][stage] = current_[stage];
        if (const Result<> fitted = FitStage<kHalf>(halves_[half][stage], states, targets, accrued);
            !fitted.has_value()) {
          return fitted;
        }
      }
      fitted_[stage] = current_[stage];
      if (const Result<> averaged = fitted_[stage].SetParameters(
              Scalar{0.5} * (halves_[0][stage].Parameters() + halves_[1][stage].Parameters()));
          !averaged.has_value()) {
        return averaged;
      }
    }
    return SuccessResult();
  }

  // Writes `(1 - step) V~_current + step V~_fitted` into `trial_`, coefficient by coefficient --
  // which blends the functions themselves, since every table shares one normalization.
  auto BlendIntoTrial(Scalar step) noexcept -> Result<> {
    for (std::size_t stage = 0; stage < NumStages; ++stage) {
      const typename ValueFunctionT::ParameterVector blended =
          ((Scalar{1} - step) * current_[stage].Parameters()) + (step * fitted_[stage].Parameters());
      if (const Result<> assigned = trial_[stage].SetParameters(blended); !assigned.has_value()) {
        return assigned;
      }
    }
    return SuccessResult();
  }

  // Fits one stage's representation to its targets by the weighted least squares of (5.12).
  //
  // With a positive entropy temperature the weights are the thesis's local-entropy weights,
  // `exp(-rho / lambda)`, on the path-integrated heuristic (5.15) -- here the running cost a
  // sample accrued to reach its state plus its target, the estimated cost from there on, so that
  // `rho` estimates the total cost of the best path through the sample. Why it matters on a
  // swing-up: the targets at a late stage run from a couple of units, near the goal, to hundreds,
  // on the probes that swung the wrong way, and unweighted least squares spends the fit's
  // accuracy on the latter. The policy needs it on the former -- the final braking that decides
  // whether the links stop upright is a difference of a unit or so in value. The heuristic is
  // standardized over the batch so that `lambda` is in units of its spread, and shifted by its
  // minimum so the largest weight is exactly one.
  template <int Count>
  auto FitStage(ValueFunctionT& value_function,
                const ValueFunctionStateBatch<State, Count>& states,
                const ValueFunctionValueBatch<State, Count>& targets,
                const ValueFunctionValueBatch<State, Count>& accrued) const noexcept -> Result<> {
    ValueFunctionValueBatch<State, Count> weights = ValueFunctionValueBatch<State, Count>::Ones();
    if (options_.entropy_temperature > Scalar{0}) {
      const ValueFunctionValueBatch<State, Count> heuristic = accrued + targets;
      const Scalar mean = heuristic.mean();
      const Scalar spread =
          std::sqrt((heuristic.array() - mean).square().mean()) + std::numeric_limits<Scalar>::epsilon();
      weights = (-(heuristic.array() - heuristic.minCoeff()) / (spread * options_.entropy_temperature)).exp();
    }
    return value_function.template FitWeightedLeastSquares<Count>(states, targets, weights, options_.ridge);
  }

  // The mean cost of the policy over `table`, on NumEvaluationSamples rollouts with no exploration
  // and a fixed noise seed.
  [[nodiscard]] auto Evaluate(const std::vector<ValueFunctionT>& table) const noexcept -> Costs {
    const CounterBasedNormalSampler sampler(seed_ ^ kEvaluationStream);
    const Policy policy = MakePolicy(table);

    std::vector<Scalar> running(NumEvaluationSamples, Scalar{0});
    std::vector<Scalar> terminal(NumEvaluationSamples, Scalar{0});
    ParallelFor(NumEvaluationSamples, [&](std::size_t sample) noexcept {
      State noise;
      sampler.FillNoise(sample, 0, noise);
      State state = initial_mean_ + (initial_covariance_factor_ * noise);
      for (std::size_t stage = 0; stage < kNumControlStages; ++stage) {
        const Control control = policy(stage, state);
        running[sample] += running_cost_(stage, state, control);
        sampler.FillNoise(sample, stage + 1, noise);
        state = forward_model_(stage, state, control, noise);
      }
      terminal[sample] = terminal_cost_(state);
    });

    Costs costs;
    for (std::size_t sample = 0; sample < NumEvaluationSamples; ++sample) {
      costs.running += running[sample];
      costs.terminal += terminal[sample];
    }
    costs.running /= static_cast<Scalar>(NumEvaluationSamples);
    costs.terminal /= static_cast<Scalar>(NumEvaluationSamples);
    costs.total = costs.running + costs.terminal;
    return costs;
  }

  ForwardModelT forward_model_;
  RunningCost running_cost_;
  TerminalCostT terminal_cost_;
  State initial_mean_;
  StateMat initial_covariance_factor_;
  Control lower_bounds_;
  Control upper_bounds_;
  Options options_;
  std::uint64_t seed_;

  // The value function tables, one entry per stage 0..K. `current_` defines the policy; `fitted_`
  // receives each backward pass; `trial_` holds the blend being scored.
  std::vector<ValueFunctionT> current_;
  std::vector<ValueFunctionT> fitted_;
  std::vector<ValueFunctionT> trial_;
  // The two independently fitted halves of a double-estimation backward pass.
  std::array<std::vector<ValueFunctionT>, 2> halves_;
  Scalar current_cost_ = std::numeric_limits<Scalar>::infinity();

  // The most recent forward pass: the states at every stage and the control that produced each
  // column's state at the next. On the heap: at the batch sizes this is used at they run to
  // megabytes.
  std::vector<StateBlock> states_;
  std::vector<ControlBlock> controls_;
  // The running cost each column had accrued on arriving at each stage.
  std::vector<ValueBlock> accrued_before_;

  std::size_t iteration_ = 0;
};

}  // namespace fbsde_traj_opt

#endif  // FBSDE_TRAJ_OPT_DT_FBSDE_ITERATIVE_SOLVER_HPP_
