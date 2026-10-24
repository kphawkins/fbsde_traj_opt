// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#ifndef FBSDE_TRAJ_OPT_DT_FBSDE_ITERATIVE_SOLVER_HPP_
#define FBSDE_TRAJ_OPT_DT_FBSDE_ITERATIVE_SOLVER_HPP_

#include <algorithm>
#include <array>
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
#include "fbsde_traj_opt/policies/taylor_q_function_l1_policy.hpp"
#include "fbsde_traj_opt/sampling/counter_based_normal_sampler.hpp"
#include "fbsde_traj_opt/sampling/l1_exploratory_control.hpp"
#include "fbsde_traj_opt/sampling/normal_distribution_concepts.hpp"
#include "fbsde_traj_opt/sde/diffusion_scaled_forward_sde_model.hpp"
#include "fbsde_traj_opt/sde/sde_term_concepts.hpp"
#include "fbsde_traj_opt/solvers/policy_rollout_evaluation.hpp"
#include "fbsde_traj_opt/solvers/taylor_gradient_target.hpp"
#include "fbsde_traj_opt/utils/parallel_for.hpp"
#include "fbsde_traj_opt/utils/result.hpp"
#include "fbsde_traj_opt/value_function/value_function_approx_concepts.hpp"
#include "fbsde_traj_opt/value_function/value_function_gauss_newton_fitter.hpp"
#include "fbsde_traj_opt/value_function/value_function_sgd_fitter.hpp"
#include "fbsde_traj_opt/value_function/value_function_slope_targets.hpp"

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
// Two options make a rejected update cheaper, which matters at small batches, where most updates
// are rejected. `stop_uphill_line_search` gives up on an update once two candidates show the cost
// rising along it from the current table: if halving the step only shrinks the excess in
// proportion, every smaller step is worse too, and the remaining halvings are wasted rollouts.
// `screening_rollouts` scores each candidate on the first few evaluation rollouts before the rest;
// every candidate and the current policy see the same noise, so a small paired comparison already
// rules most of them out, and only a candidate that passes is scored on all of them.
//
// Adaptive diffusion inflation is a second form of exploration, for when the iterations stall. Each
// iteration that fails to lower the cost multiplies the forward pass's noise by a growth factor, up
// to a cap, and an iteration that lowers it resets the factor to one. The batch then spreads out
// around the policy's trajectories just when the fit has nothing new to learn from them, and draws
// back in once an update lands. Inflation changes where the samples are, not what is fitted to them:
// every target is computed from its sample's state under the problem's own dynamics and noise, so
// an inflated batch still estimates the problem's value function, only accurately over a wider
// region. Nor can an inflated batch degrade the policy. The table the policy runs on is replaced
// only by a candidate that lowers the cost on the evaluation rollouts, and those are always
// simulated under the problem's own noise; the table therefore always holds the best fit found so far
// under that noise, whatever the batch that produced it.
//
// The inflation's base -- where it starts and what an improvement resets it to -- can be changed
// between iterations by SetExplorationInflation(), which is how a caller anneals it: start the
// batch far wider than the problem's noise and narrow it to that noise over a schedule. And
// `uninflated_path_fraction` keeps a share of the paths at the problem's own noise whatever the
// inflation, so the fit stays accurate where the policy's trajectories actually go.
//
// Diffusion continuation is a different thing from inflation: it changes the problem rather than
// the sampling. SetDiffusionScale(s) makes the solver work on the problem with `s` times the
// diffusion -- the forward pass, the targets' noise-curvature correction, the policy, and the line
// search's rollouts all see it (DiffusionScaledForwardSdeModel) -- so its value functions are those
// of a smoother problem, whose minima are wider and fewer. Walking `s` down to one then follows the
// smoothed solution to the original one, a homotopy. A cost measured under one scale says nothing
// about another, so a change of scale re-scores the current policy before the next line search.
// CurrentPolicy() always acts on the problem as posed, with the diffusion at scale one.
//
// Slope matching (`slope_weight`, off by default) fits each stage's value function to slopes as well
// as values. The policy compares the next stage's value function across the states its candidate
// controls land on, which differ along the control columns, so what decides a control is that
// function's slope along those columns -- which a fit to values alone leaves unconstrained. The
// slope target is the derivative of the noiseless target itself: with the control fixed at the one
// the target used (the envelope argument: the control's own derivative contributes nothing almost
// everywhere) and the noise-curvature term's third-derivative part dropped,
//
//   G^_i(x) = J(x, u)' grad V~_{i+1}(x + F_i(x, u)),     J = d (x + F_i(x, u)) / dx,
//
// the discrete costate recursion. Its projection on each control column `b` of the stage before is
// matched, weighted by `slope_weight`; `b' grad V` is the value difference across one full control
// step, in the units of the value function. The Jacobian `J` is taken by central differences of
// the forward model's own noise-free step, so no model has to supply derivatives; it costs 2N
// extra steps per sample. docs/derivations/gradient_target.html has the derivation.
//
// The value function representation is fitted by its own closed-form weighted least squares when
// it offers one (PolynomialValueFunctionApprox does), and otherwise -- for a model that is not
// linear in its parameters, such as SoftMinQuadraticValueFunctionApprox -- by `sgd_epochs` epochs
// of minibatch SGD warm-started from the current fit, which stands in for the same solve. Blending
// is coefficient by coefficient either way; for a linear model that blends the functions exactly,
// and for a nonlinear one it is a path in parameter space that the line search still checks.
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

  // The ridge weight of the closed-form least-squares fit; see PolynomialValueFunctionApprox.
  Scalar ridge = Scalar{1e-6};

  // How a representation without a closed-form fit is fitted instead, from the current value
  // function, every iteration: by `sgd_epochs` epochs of ValueFunctionSgdFitter with `sgd`, or by
  // ValueFunctionGaussNewtonFitter with `gauss_newton` -- whose default, no trust region and several
  // steps, is a near-exact nonlinear least-squares solve, the closest stand-in for the closed form.
  ValueFunctionFitMethod nonlinear_fit_method = ValueFunctionFitMethod::kSgd;
  ValueFunctionSgdOptions<Scalar> sgd{};
  std::size_t sgd_epochs = 50;
  ValueFunctionGaussNewtonOptions<Scalar> gauss_newton = [] {
    ValueFunctionGaussNewtonOptions<Scalar> defaults;
    defaults.max_steps = 8;
    return defaults;
  }();

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

  // Give up on an update once it looks uphill. With `e(s)` the excess of the candidate at step `s`
  // over the current cost, a quadratic through `e(0) = 0`, `e(s)`, and `e(2s)` has slope
  // `(4 e(s) - e(2s)) / (2s)` at zero; when that is positive, smaller steps are predicted to be worse
  // as well, and the line search stops.
  bool stop_uphill_line_search = false;

  // Score each candidate on the first this many evaluation rollouts first, and on all of them only
  // if it beats the current policy there. Zero scores every candidate on all of them.
  std::size_t screening_rollouts = 0;

  // Adaptive diffusion inflation; see the class comment. After `stall_patience` consecutive
  // iterations whose cost fell by no more than `stall_relative_improvement` of itself, the forward
  // pass's noise is multiplied by `diffusion_inflation_growth`, and again after every further such
  // iteration, up to `max_diffusion_inflation`; any iteration that lowers the cost by more resets it
  // to `base_diffusion_inflation`, where it also starts; SetExplorationInflation() changes that base.
  // A growth of one holds the inflation at the base, and a base of one as well disables it.
  Scalar base_diffusion_inflation = Scalar{1};
  Scalar diffusion_inflation_growth = Scalar{1};
  Scalar max_diffusion_inflation = Scalar{8};
  Scalar stall_relative_improvement = Scalar{0};
  std::size_t stall_patience = 1;

  // The weight of the slope residual against the value residual in every stage's fit; see the class
  // comment. Zero, the default, fits values alone. A closed-form representation must offer the
  // slope-matching overload of FitWeightedLeastSquares; the Gauss-Newton fitter does not support it.
  Scalar slope_weight = Scalar{0};

  // The fraction of the paths sampled at the problem's own noise whatever the inflation, in [0, 1].
  // They are the first paths, so both halves of a double-estimation pass get their share.
  Scalar uninflated_path_fraction = Scalar{0};
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

  // The factor the forward pass multiplied its noise by, on the paths it inflated.
  Scalar diffusion_inflation = Scalar{1};

  // The scale of the problem's diffusion the iteration worked on; see SetDiffusionScale().
  Scalar diffusion_scale = Scalar{1};

  // How many candidate steps were rolled out before one was accepted or the halvings ran out.
  std::size_t candidates_tried = 0;

  // Wall-clock time spent in each phase, in seconds: the forward pass, the backward pass, and the
  // line search's blends and rollouts.
  double forward_seconds = 0.0;
  double backward_seconds = 0.0;
  double line_search_seconds = 0.0;
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
  using Model = DiffusionScaledForwardSdeModel<ForwardModelT>;
  using Policy = TaylorQFunctionL1Policy<N, M, Model, ValueFunctionT, Scalar>;
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
    RESULT_ASSERT(options.sgd_epochs >= 1, "DtFbsdeIterativeSolver::Make: an SGD fit needs at least one epoch.");
    RESULT_ASSERT(options.probe_fraction >= Scalar{0} && options.probe_fraction < Scalar{1},
                  "DtFbsdeIterativeSolver::Make: the probe fraction must lie in [0, 1).");
    RESULT_ASSERT(std::isfinite(options.entropy_temperature) && options.entropy_temperature >= Scalar{0},
                  "DtFbsdeIterativeSolver::Make: the entropy temperature must be finite and non-negative.");
    RESULT_ASSERT(!std::isnan(options.target_floor), "DtFbsdeIterativeSolver::Make: the target floor may not be NaN.");
    RESULT_ASSERT(options.screening_rollouts <= NumEvaluationSamples,
                  "DtFbsdeIterativeSolver::Make: a candidate cannot be screened on more rollouts than are evaluated.");
    RESULT_ASSERT(std::isfinite(options.diffusion_inflation_growth) &&
                      options.diffusion_inflation_growth >= Scalar{1} &&
                      std::isfinite(options.base_diffusion_inflation) &&
                      options.base_diffusion_inflation >= Scalar{1} && std::isfinite(options.max_diffusion_inflation) &&
                      options.max_diffusion_inflation >= options.base_diffusion_inflation,
                  "DtFbsdeIterativeSolver::Make: the diffusion inflation's growth and base must be finite and at least "
                  "one, and its cap at least its base.");
    RESULT_ASSERT(std::isfinite(options.stall_relative_improvement) && options.stall_relative_improvement >= Scalar{0},
                  "DtFbsdeIterativeSolver::Make: the stall threshold must be finite and non-negative.");
    RESULT_ASSERT(options.stall_patience >= 1,
                  "DtFbsdeIterativeSolver::Make: the stall patience must be at least one.");
    RESULT_ASSERT(std::isfinite(options.slope_weight) && options.slope_weight >= Scalar{0},
                  "DtFbsdeIterativeSolver::Make: the slope weight must be finite and non-negative.");
    RESULT_ASSERT(
        options.slope_weight == Scalar{0} || options.nonlinear_fit_method != ValueFunctionFitMethod::kGaussNewton,
        "DtFbsdeIterativeSolver::Make: the Gauss-Newton fitter does not support slope matching.");
    RESULT_ASSERT(options.uninflated_path_fraction >= Scalar{0} && options.uninflated_path_fraction <= Scalar{1},
                  "DtFbsdeIterativeSolver::Make: the uninflated path fraction must lie in [0, 1].");
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
    solver.current_costs_ = solver.Evaluate(solver.current_);
    if (options.screening_rollouts > 0) {
      solver.current_screened_cost_ = solver.Evaluate(solver.current_, options.screening_rollouts).total;
    }
    return SuccessResult(std::move(solver));
  }

  // Runs one iteration: forward pass, backward pass, damped policy improvement.
  //
  // Fails if a stage's regression fails, which leaves the current value functions as they were.
  auto Iterate() noexcept -> Result<IterationReport> {
    using Clock = std::chrono::steady_clock;
    IterationReport report;
    report.iteration = iteration_;

    auto start = Clock::now();
    const auto lap = [&start]() noexcept {
      const auto now = Clock::now();
      const double seconds = std::chrono::duration<double>(now - start).count();
      start = now;
      return seconds;
    };

    report.exploration_probability = ExplorationProbability(iteration_);
    report.diffusion_inflation = ExplorationInflation();
    report.diffusion_scale = forward_model_.scale();
    report.sampled_mean_cost =
        SampleForwardPass(report.exploration_probability, report.diffusion_inflation, SamplingSeed(iteration_));
    report.forward_seconds = lap();

    if (const Result<> fitted = BackwardPass(); !fitted.has_value()) {
      return ErrorResult(fitted.error().message, fitted.error().location);
    }
    report.backward_seconds = lap();

    const Scalar previous_cost = current_costs_.total;
    const bool screening = options_.screening_rollouts > 0;
    // The excess of the previous candidate, at twice this one's step, over the current cost -- on
    // the screening rollouts when screening, so that every excess compared is on the same rollouts.
    auto previous_excess = Scalar{0};
    auto step = Scalar{1};
    for (std::size_t halving = 0; halving <= options_.max_step_halvings; ++halving, step *= Scalar{0.5}) {
      if (const Result<> blended = BlendIntoTrial(step); !blended.has_value()) {
        return ErrorResult(blended.error().message, blended.error().location);
      }
      ++report.candidates_tried;
      auto excess = Scalar{0};
      Scalar screened_cost = Scalar{0};
      bool confirm = true;
      if (screening) {
        screened_cost = Evaluate(trial_, options_.screening_rollouts).total;
        excess = screened_cost - current_screened_cost_;
        confirm = excess < Scalar{0};
      }
      if (confirm) {
        const Costs trial_costs = Evaluate(trial_);
        if (!screening) {
          excess = trial_costs.total - current_costs_.total;
        }
        if (trial_costs.total < current_costs_.total) {
          // Copied element by element rather than swapped, so that `current_` keeps its storage and
          // a policy handed out by CurrentPolicy() keeps reading the live table.
          std::ranges::copy(trial_, current_.begin());
          current_costs_ = trial_costs;
          current_screened_cost_ = screened_cost;
          report.accepted_step = step;
          break;
        }
      }
      if (options_.stop_uphill_line_search && halving > 0 && (Scalar{4} * excess) - previous_excess > Scalar{0}) {
        break;
      }
      previous_excess = excess;
    }
    report.line_search_seconds = lap();

    if (previous_cost - current_costs_.total > options_.stall_relative_improvement * std::abs(previous_cost)) {
      stall_factor_ = Scalar{1};
      stalled_iterations_ = 0;
    } else if (++stalled_iterations_ >= options_.stall_patience) {
      stall_factor_ = std::min(stall_factor_ * options_.diffusion_inflation_growth, options_.max_diffusion_inflation);
    }

    // The evaluation noise is fixed, so the accepted candidate's costs are the current policy's.
    report.mean_cost = current_costs_.total;
    report.mean_running_cost = current_costs_.running;
    report.mean_terminal_cost = current_costs_.terminal;
    ++iteration_;
    return SuccessResult(report);
  }

  // The current policy: the Taylor Q-function minimizer over the current value functions. It
  // reads them by reference, so it tracks every later iteration and must not outlive the solver.
  [[nodiscard]] auto CurrentPolicy() const noexcept -> Policy {
    return MakePolicy(current_, Model(forward_model_.unscaled()));
  }

  // Sets the exploration inflation's base from the next iteration on; see the class comment.
  //
  // Fails if `base` is not finite or is below one.
  auto SetExplorationInflation(Scalar base) noexcept -> Result<> {
    RESULT_ASSERT(std::isfinite(base) && base >= Scalar{1},
                  "DtFbsdeIterativeSolver::SetExplorationInflation: the base must be finite and at least one.");
    base_inflation_ = base;
    return SuccessResult();
  }

  // Sets the scale of the diffusion of the problem the solver works on from the next iteration on,
  // and re-scores the current policy under it; see the class comment.
  //
  // Fails if `scale` is not finite and positive.
  auto SetDiffusionScale(Scalar scale) noexcept -> Result<> {
    RESULT_ASSERT(std::isfinite(scale) && scale > Scalar{0},
                  "DtFbsdeIterativeSolver::SetDiffusionScale: the scale must be finite and positive.");
    if (scale == forward_model_.scale()) {
      return SuccessResult();
    }
    forward_model_.set_scale(scale);
    current_costs_ = Evaluate(current_);
    if (options_.screening_rollouts > 0) {
      current_screened_cost_ = Evaluate(current_, options_.screening_rollouts).total;
    }
    return SuccessResult();
  }

  [[nodiscard]] auto DiffusionScale() const noexcept -> Scalar { return forward_model_.scale(); }

  // The current value functions, one per stage 0..K.
  [[nodiscard]] auto ValueFunctions() const noexcept -> std::span<const ValueFunctionT> { return current_; }

  // The value functions the most recent backward pass fitted, before any blending: the method's
  // estimate of the value of the policy that is greedy with respect to them.
  [[nodiscard]] auto FittedValueFunctions() const noexcept -> std::span<const ValueFunctionT> { return fitted_; }

  // The mean cost of the current policy on the evaluation rollouts, under the current diffusion
  // scale.
  [[nodiscard]] auto CurrentCost() const noexcept -> Scalar { return current_costs_.total; }

  [[nodiscard]] auto IterationCount() const noexcept -> std::size_t { return iteration_; }

  // The states and controls of the most recent forward pass, one block per stage, for plotting
  // what the sampling distribution looked like.
  [[nodiscard]] auto SampledStates() const noexcept -> std::span<const StateBlock> { return states_; }

  [[nodiscard]] auto SampledControls() const noexcept -> std::span<const ControlBlock> { return controls_; }

 private:
  using Costs = PolicyRolloutCosts<Scalar>;

  // The sampler components the forward pass draws its non-noise randomness from, past the N used
  // for the state noise.
  static constexpr std::size_t kExploreComponent = static_cast<std::size_t>(N);
  static constexpr std::size_t kParentPickComponent = static_cast<std::size_t>(N) + 1;
  static constexpr std::size_t kControlPickComponent = static_cast<std::size_t>(N) + 2;

  // The minibatch of an SGD fit, and the accumulation block of a Gauss-Newton one, for a
  // representation without a closed-form fit.
  static constexpr int kSgdMinibatch = 64;

  // The relative half-width of the central differences the slope targets' Jacobian is taken by.
  static constexpr Scalar kJacobianStep = Scalar{1e-5};

  template <int Count>
  using SlopeTargets = ValueFunctionSlopeTargets<State, Count, M>;

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
    base_inflation_ = options_.base_diffusion_inflation;
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
    return MakePolicy(table, forward_model_);
  }

  [[nodiscard]] auto MakePolicy(const std::vector<ValueFunctionT>& table, Model model) const noexcept -> Policy {
    // Cannot fail: Make() checked the bounds and the table has NumStages >= 2 entries.
    return *Policy::Make(std::move(model), running_cost_.L1Weight(), lower_bounds_, upper_bounds_, table);
  }

  // The inflation the next forward pass applies: the base times the stall factor, capped.
  [[nodiscard]] auto ExplorationInflation() const noexcept -> Scalar {
    return std::min(base_inflation_ * stall_factor_, std::max(options_.max_diffusion_inflation, base_inflation_));
  }

  // The forward pass, with its noise multiplied by `inflation` beyond the problem's own, except on the
  // first `uninflated_path_fraction` of the paths. Returns the mean cost the path columns incurred.
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
  auto SampleForwardPass(Scalar exploration, Scalar inflation, std::uint64_t seed) noexcept -> Scalar {
    const CounterBasedNormalSampler sampler(seed);
    const Policy policy = MakePolicy(current_);
    const std::size_t path_count = PathCount();
    const auto uninflated_count =
        static_cast<std::size_t>(std::lround(options_.uninflated_path_fraction * static_cast<Scalar>(path_count)));

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
          const auto pick = SampleUniform<Scalar>(sampler, sample, stage + 1, kParentPickComponent);
          parent = std::min(static_cast<std::size_t>(pick * static_cast<Scalar>(path_count)), path_count - 1);
        }
        const State state = states_[stage].col(static_cast<Eigen::Index>(parent));

        const bool explore =
            !is_path || SampleUniform<Scalar>(sampler, sample, stage + 1, kExploreComponent) < exploration;
        const Control control =
            explore ? ExploratoryL1Control<M>(
                          sampler, sample, stage + 1, kControlPickComponent, lower_bounds_, upper_bounds_)
                    : policy(stage, state);

        State noise;
        sampler.FillNoise(sample, stage + 1, noise);
        const Scalar column_inflation = sample < uninflated_count ? Scalar{1} : inflation;
        states_[stage + 1].col(column) = forward_model_(stage, state, control, State(column_inflation * noise));
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

    const bool slopes_wanted = options_.slope_weight > Scalar{0};
    for (std::size_t stage = kNumControlStages; stage-- > 0;) {
      const StateBlock& states = states_[stage];
      ValueBlock targets;
      SlopeTargets<kSampleCount> slopes;
      ParallelFor(NumSamples, [&](std::size_t sample) noexcept {
        const auto column = static_cast<Eigen::Index>(sample);
        const State state = states.col(column);
        const Control control = target_policy(stage, state);
        const Scalar q_value = continuation.QValue(stage, state, control);
        targets[column] = std::max(q_value, options_.target_floor);
        if (slopes_wanted) {
          const State gradient = q_value < options_.target_floor
                                     ? State(State::Zero())
                                     : GradientTarget(stage, state, control, fitted_[stage + 1]);
          SetSlopeTargets<kSampleCount>(slopes, column, stage, state, gradient);
        }
      });
      fitted_[stage] = current_[stage];
      if (const Result<> fitted = FitStage<kSampleCount>(fitted_[stage],
                                                         states,
                                                         targets,
                                                         accrued_before_[stage],
                                                         FitSeed(stage, 0),
                                                         slopes_wanted ? &slopes : nullptr);
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
    const bool slopes_wanted = options_.slope_weight > Scalar{0};
    for (std::size_t stage = kNumControlStages; stage-- > 0;) {
      for (std::size_t half = 0; half < 2; ++half) {
        const Policy& chooser = choosers[half];
        const Policy& scorer = choosers[1 - half];
        HalfStates states;
        HalfValues targets;
        HalfValues accrued;
        SlopeTargets<kHalf> slopes;
        ParallelFor(static_cast<std::size_t>(kHalf), [&](std::size_t slot) noexcept {
          const auto column = static_cast<Eigen::Index>(slot);
          const State state = states_[stage].col(static_cast<Eigen::Index>((2 * slot) + half));
          states.col(column) = state;
          accrued[column] = accrued_before_[stage][static_cast<Eigen::Index>((2 * slot) + half)];
          const Control control = chooser(stage, state);
          const Scalar q_value = scorer.QValue(stage, state, control);
          targets[column] = std::max(q_value, options_.target_floor);
          if (slopes_wanted) {
            // Scored, like the value, by the other half's table.
            const State gradient = q_value < options_.target_floor
                                       ? State(State::Zero())
                                       : GradientTarget(stage, state, control, halves_[1 - half][stage + 1]);
            SetSlopeTargets<kHalf>(slopes, column, stage, state, gradient);
          }
        });
        halves_[half][stage] = current_[stage];
        if (const Result<> fitted = FitStage<kHalf>(halves_[half][stage],
                                                    states,
                                                    targets,
                                                    accrued,
                                                    FitSeed(stage, half + 1),
                                                    slopes_wanted ? &slopes : nullptr);
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
                const ValueFunctionValueBatch<State, Count>& accrued,
                std::uint64_t seed,
                const SlopeTargets<Count>* slopes) const noexcept -> Result<> {
    ValueFunctionValueBatch<State, Count> weights = ValueFunctionValueBatch<State, Count>::Ones();
    if (options_.entropy_temperature > Scalar{0}) {
      const ValueFunctionValueBatch<State, Count> heuristic = accrued + targets;
      const Scalar mean = heuristic.mean();
      const Scalar spread =
          std::sqrt((heuristic.array() - mean).square().mean()) + std::numeric_limits<Scalar>::epsilon();
      weights = (-(heuristic.array() - heuristic.minCoeff()) / (spread * options_.entropy_temperature)).exp();
    }
    if constexpr (requires {
                    value_function.template FitWeightedLeastSquares<Count>(states, targets, weights, Scalar{});
                  }) {
      if (slopes == nullptr) {
        return value_function.template FitWeightedLeastSquares<Count>(states, targets, weights, options_.ridge);
      }
      if constexpr (requires {
                      value_function.template FitWeightedLeastSquares<Count, M>(
                          states, targets, weights, *slopes, Scalar{}, Scalar{});
                    }) {
        return value_function.template FitWeightedLeastSquares<Count, M>(
            states, targets, weights, *slopes, options_.slope_weight, options_.ridge);
      } else {
        return ErrorResult("DtFbsdeIterativeSolver: this representation's closed-form fit cannot match slopes.");
      }
    } else if (options_.nonlinear_fit_method == ValueFunctionFitMethod::kGaussNewton) {
      using Fitter = ValueFunctionGaussNewtonFitter<ValueFunctionT, std::min(kSgdMinibatch, Count)>;
      RESULT_ASSIGN_OR_RETURN(const Fitter fitter, Fitter::Make(options_.gauss_newton));
      ValueFunctionGaussNewtonReport<Scalar> report;
      return fitter.template Fit<Count>(value_function, states, targets, weights, report);
    } else {
      // A minibatch no wider than the data: a wider one would only repeat each sample, which
      // leaves the mean gradient unchanged and multiplies its cost.
      using Fitter = ValueFunctionSgdFitter<ValueFunctionT, std::min(kSgdMinibatch, Count)>;
      RESULT_ASSIGN_OR_RETURN(Fitter fitter, Fitter::Make(options_.sgd, value_function.Parameters()));
      std::vector<typename Fitter::StepReport> history;
      if (slopes != nullptr) {
        return fitter.template FitEpochs<Count, M>(value_function,
                                                   states,
                                                   targets,
                                                   weights,
                                                   *slopes,
                                                   options_.slope_weight,
                                                   options_.sgd_epochs,
                                                   seed,
                                                   history);
      }
      return fitter.template FitEpochs<Count>(
          value_function, states, targets, weights, options_.sgd_epochs, seed, history);
    }
  }

  // The slope target at `state` for the value target taken with `control` against `next`, the
  // next stage's table; see TaylorGradientTarget().
  [[nodiscard]] auto GradientTarget(std::size_t stage,
                                    const State& state,
                                    const Control& control,
                                    const ValueFunctionT& next) const noexcept -> State {
    return TaylorGradientTarget(forward_model_, stage, state, control, next, kJacobianStep);
  }

  // Writes sample `column`'s slope targets from its gradient target: the gradient projected on each
  // control column of the stage before, along which that stage's policy compares this stage's
  // value function. Stage 0's table serves no policy; its columns are its own stage's.
  template <int Count>
  auto SetSlopeTargets(SlopeTargets<Count>& slopes,
                       Eigen::Index column,
                       std::size_t stage,
                       const State& state,
                       const State& gradient) const noexcept -> void {
    const Eigen::Matrix<Scalar, N, M> columns = forward_model_.control_drift_mat()(stage == 0 ? 0 : stage - 1, state);
    for (std::size_t index = 0; index < kControlDim; ++index) {
      const auto control_column = static_cast<Eigen::Index>(index);
      slopes.directions[index].col(column) = columns.col(control_column);
      slopes.slopes[index][column] = columns.col(control_column).dot(gradient);
    }
  }

  // The minibatch shuffling seed of one stage's SGD fit: distinct per iteration, stage, and half.
  [[nodiscard]] auto FitSeed(std::size_t stage, std::size_t half) const noexcept -> std::uint64_t {
    return SamplingSeed(iteration_) ^ ((static_cast<std::uint64_t>(stage) * 3U + half) * 0xBF58476D1CE4E5B9ULL);
  }

  // The mean cost of the policy over `table`, on the first `rollouts` of the NumEvaluationSamples
  // evaluation rollouts, with no exploration and a fixed noise seed.
  [[nodiscard]] auto Evaluate(const std::vector<ValueFunctionT>& table,
                              std::size_t rollouts = NumEvaluationSamples) const noexcept -> Costs {
    return EvaluatePolicyByRollouts(forward_model_,
                                    MakePolicy(table),
                                    running_cost_,
                                    terminal_cost_,
                                    initial_mean_,
                                    initial_covariance_factor_,
                                    rollouts,
                                    kNumControlStages,
                                    seed_ ^ kPolicyEvaluationStream);
  }

  // The model at the diffusion scale the solver is working on.
  Model forward_model_;
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
  Costs current_costs_{.total = std::numeric_limits<Scalar>::infinity()};
  // The current policy's cost on the screening rollouts, when screening.
  Scalar current_screened_cost_ = Scalar{0};

  // The exploration inflation's base, the factor the stalls since the last improvement have
  // multiplied it by, and how many iterations in a row have failed to lower the cost.
  Scalar base_inflation_ = Scalar{1};
  Scalar stall_factor_ = Scalar{1};
  std::size_t stalled_iterations_ = 0;

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
