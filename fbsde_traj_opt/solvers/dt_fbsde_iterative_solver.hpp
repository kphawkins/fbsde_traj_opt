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
#include <concepts>
#include <deque>
#include <limits>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <Eigen/QR>

#include "fbsde_traj_opt/costs/l1_control_sde_running_cost_term.hpp"
#include "fbsde_traj_opt/policies/taylor_q_function_l1_policy.hpp"
#include "fbsde_traj_opt/sampling/counter_based_normal_sampler.hpp"
#include "fbsde_traj_opt/sampling/l1_exploratory_control.hpp"
#include "fbsde_traj_opt/sampling/normal_distribution_concepts.hpp"
#include "fbsde_traj_opt/sampling/sigma_point_rules.hpp"
#include "fbsde_traj_opt/sde/diffusion_scaled_forward_sde_model.hpp"
#include "fbsde_traj_opt/sde/sde_term_concepts.hpp"
#include "fbsde_traj_opt/solvers/policy_rollout_evaluation.hpp"
#include "fbsde_traj_opt/solvers/taylor_gradient_target.hpp"
#include "fbsde_traj_opt/solvers/taylor_noiseless_backward_step_estimator.hpp"
#include "fbsde_traj_opt/utils/parallel_for.hpp"
#include "fbsde_traj_opt/utils/phase_profiler.hpp"
#include "fbsde_traj_opt/utils/result.hpp"
#include "fbsde_traj_opt/value_function/time_shared_linear_fit.hpp"
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
// Sigma-point sampling (`sigma_point_rule`, off by default) replaces the random forward pass with a
// deterministic one; see SampleSigmaPointForwardPass(). The batch at each stage is then a point set
// shaped by the stage's state covariance -- the unscented transform's or a higher-degree rule's (see
// sigma_point_rules.hpp) -- and the mean and covariance are carried from stage to stage through the
// closed-loop dynamics, as a sigma-point filter's prediction step carries them. Only where the
// samples sit changes. The targets, the fit, the policy, and the line search are as above, and the
// line search still scores every candidate on random rollouts.
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

// Which policy the Taylor noiseless target evaluates; see DtFbsdeIterativeOptions::taylor_target_policy.
enum class DtFbsdeTaylorTargetPolicy : std::uint8_t { kCurrent, kGreedy };

// How a backward pass with time sharing (DtFbsdeIterativeOptions::time_sharing) shares parameters
// stage by stage. Every pass ends with a joint solve over all stages' equations; the choice is what
// the targets *within* the pass read, since stage k's targets read stage k+1's fit.
enum class DtFbsdeTimeSharingPass : std::uint8_t {
  // Each stage is fitted on its own, as without time sharing; only the closing solve shares. The
  // table the line search sees is the shared one.
  kAfterPass,
  // Each stage re-solves the knots around it (block Gauss-Seidel) from the stages this pass has
  // fitted so far and, scaled by DtFbsdeIterativeOptions::time_sharing_stale_weight, the previous
  // pass's equations of the stages it has not reached yet. Counted fully, those dominate a knot's
  // solve until the pass reaches them, so each stage's fit lags a pass behind and value information
  // crawls back from the terminal stage; dropped, a knot first reached by one stage at a small
  // interpolation weight is nearly undetermined. On the 2-D benchmarks every weight tried (0, 0.1,
  // 1) stalls or diverges to a non-finite fit within a few hundred iterations
  // (time_shared_study/README.md); kAfterPass does not.
  kGaussSeidel,
};

// Which control the backward pass's targets are taken at; see BackwardPass().
enum class DtFbsdeBackwardTarget : std::uint8_t {
  // The current policy's control: the targets estimate V^mu, and improvement happens between
  // iterations. Section 4.6's scheme.
  kCurrentPolicy,
  // The greedy control, chosen and evaluated by two independently fitted halves of the batch.
  kDoubleGreedy,
  // The off-policy drifted Taylor noiseless estimator of Hawkins (2021), Chapter 3, Table 3.1
  // (TaylorNoiselessBackwardStepEstimator), for the current policy: a second-order model of the next
  // stage's value function built where each sample actually went -- its state plus the drift K the
  // forward pass used from it -- and evaluated where the current policy would go. The targets
  // estimate V^mu; see TaylorNoiselessBackwardPass().
  kTaylorNoiseless,
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

  // The blending step the line search tries first, in (0, 1]; each later candidate halves it. With
  // `adaptive_initial_step`, an accepted step `s` makes the next line search start at min(1, 2 s)
  // instead, and a rejected one leaves the start where it was -- a trust radius remembered across
  // iterations. Either way only the candidates proposed change; each is accepted by the same test.
  Scalar initial_step = Scalar{1};
  bool adaptive_initial_step = false;

  // Pool the forward batches since the last accepted update into each stage's fit, up to this many
  // batches including the current one; zero or one fits this pass's batch alone. The Taylor
  // noiseless target estimates the value of the current policy, which changes only on acceptance, so
  // every batch drawn since then is a sample for the same regression. Each kept batch's targets are
  // recomputed every pass from the stage after's fresh fit, so none is stale; the pool is emptied when
  // an update is accepted. Needs the Taylor noiseless target and a representation fitted in closed
  // form through its parameter gradient (PolynomialValueFunctionApprox); not with slope matching or
  // time sharing.
  std::size_t pool_batches_since_acceptance = 0;

  // A trust region in function space: every stage's fit also pulls the fitted function toward the
  // current one at the states the current policy visits on the first `anchor_rollouts` evaluation
  // rollouts, with total weight `anchor_weight` times the data's. Coefficient blending shrinks the
  // change equally everywhere, including where no sample constrains it; the anchors hold the function
  // where the current policy acts and leave it free elsewhere. Zero disables it. The same
  // restrictions as pooling apply.
  Scalar anchor_weight = Scalar{0};
  std::size_t anchor_rollouts = 32;

  // A trust region inside the fit, in place of (or as well as) coefficient blending: every stage's fit
  // adds a proximal term pulling the coefficients toward the current ones,
  //
  //   lambda * W * (theta - theta_current)' (G / w_G) (theta - theta_current),
  //
  // where W is this batch's total sample weight and G the Gram matrix of the earlier batches' samples,
  // with total weight w_G, decayed by `proximal_memory` per iteration. Blending shrinks every
  // coefficient equally; this holds the directions earlier data pinned down and lets the ones it never
  // constrained move to the new batch's fit -- a Kalman-style update. At `lambda = 1` the prior counts
  // as much as one batch. `lambda` starts at `proximal_weight` (zero disables all of this) and adapts
  // like a Levenberg-Marquardt damping: divided by `proximal_adaptation` after an update the line
  // search accepts at its first candidate, multiplied by it after one accepted at a later candidate,
  // and multiplied by `proximal_rejection_adaptation` after a rejected one, within
  // [`min_proximal_weight`, `max_proximal_weight`]. Most DT-FBSDE iterations are rejected because the
  // proposal is noisy, not because it went too far, so a rejection factor of one -- leave the weight
  // alone -- may suit it better than the classic doubling. With `proximal_line_search`, the line search's
  // candidates are no longer blends: candidate h is the fit refitted at `lambda * 4^h`, from the same
  // targets, and an accepted candidate h sets `lambda` to its own weight divided by the adaptation.
  // Either way a candidate is accepted by the same test. The same restrictions as pooling apply.
  Scalar proximal_weight = Scalar{0};
  Scalar proximal_memory = Scalar{0.9};
  Scalar proximal_adaptation = Scalar{2};
  Scalar proximal_rejection_adaptation = Scalar{2};
  Scalar min_proximal_weight = Scalar{1e-3};
  Scalar max_proximal_weight = Scalar{1e3};
  bool proximal_line_search = false;

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

  // Deterministic sigma-point sampling in place of random paths; kNone samples paths. See
  // SampleSigmaPointForwardPass(). The exploration probability, the probe fraction, and the sampling
  // seed play no part in it; the diffusion inflation and the uninflated path fraction decide its
  // point sets, and `sigma_point_probes` its probes.
  SigmaPointRule sigma_point_rule = SigmaPointRule::kNone;
  bool sigma_point_probes = true;
  // Give the second half of a double-estimation pass the same rule turned by a fixed rotation, so
  // that the two halves are fitted on different states, as their random batches would be.
  bool rotate_sigma_point_halves = true;
  // Turn each half's points by a fresh random rotation every iteration -- the randomized unscented
  // transform of the stochastic integration filter (Dunik, Straka and Simandl, IEEE TAC 2013) --
  // which keeps every iteration's set matched to the covariance but varies where the points sit from
  // one iteration to the next.
  bool randomize_sigma_point_rotation = false;

  // Recursive least squares across iterations, for a representation fitted in closed form that
  // exposes its normal equations (LinearBasisValueFunctionApprox does). With a positive memory `rho`,
  // each stage keeps its normal equations -- one set per half of a double-estimation pass -- and
  // every iteration multiplies them by `rho` before adding its own batch's rows, then solves them:
  // a fit to the last ~1 / (1 - rho) batches rather than to this one alone. Zero, the default,
  // fits each batch afresh. A batch is kept whether or not the line search accepts the update it
  // produced, so a rejected iteration still contributes its samples. With
  // `reset_normal_equations_on_acceptance`, the kept equations are cleared whenever an update is
  // accepted: the targets are computed from the current tables, which change only then, so between
  // acceptances even `rho = 1` accumulates targets that are all current.
  Scalar normal_equation_memory = Scalar{0};
  bool reset_normal_equations_on_acceptance = false;

  // Experience replay, for the same representations: each stage fits the states of its last
  // `replay_batches` forward passes (this one included) together, with every target and slope
  // recomputed from the current tables, so the fit sees more samples and none of them stale. One,
  // like zero, fits this pass's batch alone. Needs the double-estimation backward target, and cannot
  // be combined with a normal equation memory.
  std::size_t replay_batches = 0;

  // The Taylor noiseless target's policy: the current one, whose value the pass then estimates, or
  // the greedy policy of the value functions this pass is fitting -- the improved policy, whose
  // value the off-policy samples estimate instead.
  DtFbsdeTaylorTargetPolicy taylor_target_policy = DtFbsdeTaylorTargetPolicy::kCurrent;
  // With the Taylor noiseless target, multiply each sample's fit weight by exp(-kappa ||D||^2 / 2),
  // D = Sigma^-1 (F^mu - K): the estimator's bias bound grows as exp(||D||^2 / 2), so a sample
  // taken far off the target policy is trusted less. Zero leaves the weights alone.
  Scalar girsanov_weight = Scalar{0};
  // How many consecutive stages a path keeps an exploratory control once it draws one: one is a
  // fresh draw every stage; more holds a random bang-off-bang arc, reaching the states a genuinely
  // different policy would.
  std::size_t exploration_hold_stages = 1;
  // The fraction of paths that explore at all; the rest always follow the policy, so that however
  // large epsilon is, part of the batch still covers the policy's own state distribution. The
  // exploring paths are spread evenly over the path indices, and so over the inflated and
  // uninflated halves alike. Probes explore regardless.
  Scalar exploring_path_fraction = Scalar{1};
  // How many orders of the Taylor noiseless target's expansion in dF = F^mu - K are kept: two, the
  // estimator itself; one drops the curvature term 1/2 dF' H dF; zero drops the first-order term
  // g' dF too, leaving the value of where the sample actually went, uncorrected for the control it
  // took. An ablation of the off-policy correction, not a recommended setting.
  std::size_t taylor_correction_order = 2;
  // With the Taylor noiseless target, also measure how far each estimate is from the exact
  // re-evaluation of its own expansion, l + V~(x + F^mu) + 1/2 tr(Sigma' H(x + F^mu) Sigma) -- the
  // Taylor residual the bias bound multiplies. It costs a second evaluation of the next stage per
  // off-policy sample, so it is off unless a study asks for it.
  bool measure_taylor_residual = false;

  // Parameters shared across stages, for a representation that exposes its normal equations; see
  // TimeSharedLinearFit. The default, one coefficient vector per stage and no shared basis, is the
  // per-stage table. Every pass keeps each stage's normal equations and ends with a joint solve over
  // all of them -- with a shared basis of rank r, alternating between the basis and the stages'
  // mixing weights -- whose result is the table the line search blends toward. The first pass with
  // a shared basis fits stage by stage and starts the basis from those fits. Not with replay.
  //
  // How a pass shares parameters before its closing joint solve; see DtFbsdeTimeSharingPass.
  TimeSharedFitOptions time_sharing{};
  DtFbsdeTimeSharingPass time_sharing_pass = DtFbsdeTimeSharingPass::kAfterPass;
  Scalar time_sharing_stale_weight = Scalar{0.1};
};

// The normal equations a representation lets DtFbsdeIterativeSolver carry across iterations, if it
// has any; see DtFbsdeIterativeOptions::normal_equation_memory.
template <typename ValueFunctionT>
struct DtFbsdeNormalEquationsOf {
  static constexpr bool kSupported = false;
  using type = char;
};

template <typename ValueFunctionT>
  requires requires { typename ValueFunctionT::NormalEquations; }
struct DtFbsdeNormalEquationsOf<ValueFunctionT> {
  static constexpr bool kSupported = true;
  using type = typename ValueFunctionT::NormalEquations;
};

// The time-shared fit DtFbsdeIterativeSolver keeps for a representation, if it can; see
// DtFbsdeIterativeOptions::time_sharing.
template <typename ValueFunctionT>
struct DtFbsdeTimeSharedFitOf {
  using type = char;
};

template <typename ValueFunctionT>
  requires requires { typename ValueFunctionT::NormalEquations; }
struct DtFbsdeTimeSharedFitOf<ValueFunctionT> {
  using type = TimeSharedLinearFit<ValueFunctionT>;
};

[[nodiscard]] inline auto TimeSharingEnabled(const TimeSharedFitOptions& options) noexcept -> bool {
  return options.kind != TimeBasisKind::kPerStage || options.rank > 0;
}

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

  // With the Taylor noiseless target, the mean over every sample and stage of the Girsanov drift
  // norm || Sigma^-1 (F^mu - K) ||, the thesis's measure of how far off-policy the sampling was:
  // the estimator's bias is bounded by exp(|| D ||^2 / 2) times the Taylor residual. Zero otherwise.
  Scalar mean_girsanov_drift = Scalar{0};
  // With the Taylor noiseless target, the mean over samples and stages of the first-order correction
  // |g' (F^mu - K)| -- the term the off-policy sampling contributes -- and of the target itself, so the
  // correction can be read relative to it.
  Scalar mean_first_order_correction = Scalar{0};
  Scalar mean_target_magnitude = Scalar{0};
  // The same correction's mean signed value, g' (F^mu - K), and its curvature companion
  // |1/2 dF' H dF|: a negative mean says the samples' own drifts led somewhere the target policy's
  // value function rates higher, so the correction is lowering the targets toward them.
  Scalar mean_signed_first_order_correction = Scalar{0};
  Scalar mean_second_order_correction = Scalar{0};
  // The fraction of samples whose drift was not the target policy's, and the mean first-order
  // correction over those samples alone -- the correction's size where it is in play at all.
  Scalar off_policy_fraction = Scalar{0};
  Scalar mean_off_policy_correction = Scalar{0};
  // With `measure_taylor_residual`, the mean over off-policy samples of | Y^ - Y_exact |; zero
  // otherwise.
  Scalar mean_taylor_residual = Scalar{0};

  // How many candidate steps were rolled out before one was accepted or the halvings ran out.
  std::size_t candidates_tried = 0;

  // The proximal weight the next iteration's fit uses, after this iteration's adaptation; zero when
  // the proximal fit is off. See DtFbsdeIterativeOptions::proximal_weight.
  Scalar proximal_weight = Scalar{0};

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
    RESULT_ASSERT(options.exploring_path_fraction >= Scalar{0} && options.exploring_path_fraction <= Scalar{1},
                  "the exploring path fraction is in [0, 1].");
    RESULT_ASSERT(options.taylor_correction_order <= 2, "the Taylor correction order is zero, one, or two.");
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
    RESULT_ASSERT(options.normal_equation_memory >= Scalar{0} && options.normal_equation_memory <= Scalar{1},
                  "DtFbsdeIterativeSolver::Make: the normal equation memory must lie in [0, 1].");
    RESULT_ASSERT(options.normal_equation_memory == Scalar{0} || DtFbsdeNormalEquationsOf<ValueFunctionT>::kSupported,
                  "DtFbsdeIterativeSolver::Make: carrying normal equations across iterations needs a representation "
                  "that exposes them.");
    RESULT_ASSERT(options.replay_batches <= 1 ||
                      (DtFbsdeNormalEquationsOf<ValueFunctionT>::kSupported &&
                       options.backward_target == DtFbsdeBackwardTarget::kDoubleGreedy &&
                       options.normal_equation_memory == Scalar{0}),
                  "DtFbsdeIterativeSolver::Make: replay needs a representation that exposes its normal equations, the "
                  "double-estimation backward target, and no normal equation memory.");
    RESULT_ASSERT(std::isfinite(options.time_sharing_stale_weight) && options.time_sharing_stale_weight >= Scalar{0} &&
                      options.time_sharing_stale_weight <= Scalar{1},
                  "DtFbsdeIterativeSolver::Make: the time sharing's stale weight must lie in [0, 1].");
    RESULT_ASSERT(!TimeSharingEnabled(options.time_sharing) ||
                      (DtFbsdeNormalEquationsOf<ValueFunctionT>::kSupported && options.replay_batches <= 1),
                  "DtFbsdeIterativeSolver::Make: time sharing needs a representation that exposes its normal "
                  "equations, and no replay.");
    RESULT_ASSERT(std::isfinite(options.initial_step) && options.initial_step > Scalar{0} &&
                      options.initial_step <= Scalar{1},
                  "DtFbsdeIterativeSolver::Make: the initial line search step must lie in (0, 1].");
    RESULT_ASSERT(std::isfinite(options.anchor_weight) && options.anchor_weight >= Scalar{0},
                  "DtFbsdeIterativeSolver::Make: the anchor weight must be finite and non-negative.");
    RESULT_ASSERT(options.anchor_weight == Scalar{0} || options.anchor_rollouts >= 1,
                  "DtFbsdeIterativeSolver::Make: anchoring needs at least one anchor rollout.");
    RESULT_ASSERT(std::isfinite(options.proximal_weight) && options.proximal_weight >= Scalar{0} &&
                      options.proximal_memory >= Scalar{0} && options.proximal_memory <= Scalar{1} &&
                      std::isfinite(options.proximal_adaptation) && options.proximal_adaptation >= Scalar{1} &&
                      std::isfinite(options.proximal_rejection_adaptation) &&
                      options.proximal_rejection_adaptation >= Scalar{1} &&
                      std::isfinite(options.max_proximal_weight) && options.min_proximal_weight > Scalar{0} &&
                      options.min_proximal_weight <= options.max_proximal_weight,
                  "DtFbsdeIterativeSolver::Make: the proximal weight must be non-negative, its memory in [0, 1], its "
                  "adaptation at least one, and its bounds positive and ordered.");
    RESULT_ASSERT(!options.proximal_line_search || options.proximal_weight > Scalar{0},
                  "DtFbsdeIterativeSolver::Make: a line search over the proximal weight needs a proximal weight.");
    RESULT_ASSERT(!UsesPooledFit(options) ||
                      (kPooledFitSupported && options.backward_target == DtFbsdeBackwardTarget::kTaylorNoiseless &&
                       options.slope_weight == Scalar{0} && !TimeSharingEnabled(options.time_sharing) &&
                       options.normal_equation_memory == Scalar{0}),
                  "DtFbsdeIterativeSolver::Make: pooling and anchoring need the Taylor noiseless target, a "
                  "representation with a parameter gradient, and no slope matching, time sharing, or normal "
                  "equation memory.");
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
    if constexpr (DtFbsdeNormalEquationsOf<ValueFunctionT>::kSupported) {
      if (TimeSharingEnabled(options.time_sharing)) {
        for (std::size_t half = 0; half < 2; ++half) {
          RESULT_ASSIGN_OR_RETURN(SharedFit fit, SharedFit::Make(options.time_sharing, kNumControlStages));
          RESULT_RETURN_IF_ERROR(
              fit.Initialize(std::span<const ValueFunctionT>(solver.current_).first(kNumControlStages)));
          solver.shared_fits_.push_back(std::move(fit));
        }
      }
    }
    solver.current_costs_ = solver.Evaluate(solver.current_);
    solver.start_step_ = options.initial_step;
    if (options.proximal_weight > Scalar{0}) {
      solver.proximal_lambda_ =
          std::clamp(options.proximal_weight, options.min_proximal_weight, options.max_proximal_weight);
    }
    solver.OnCurrentChanged();
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
    report.mean_girsanov_drift = last_girsanov_drift_;
    report.mean_first_order_correction = last_first_order_correction_;
    report.mean_target_magnitude = last_target_magnitude_;
    report.mean_signed_first_order_correction = last_correction_stats_.signed_first_order;
    report.mean_second_order_correction = last_correction_stats_.second_order;
    report.off_policy_fraction = last_correction_stats_.off_policy_fraction;
    report.mean_off_policy_correction = last_correction_stats_.off_policy_first_order;
    report.mean_taylor_residual = last_correction_stats_.taylor_residual;

    const Scalar previous_cost = current_costs_.total;
    const bool screening = options_.screening_rollouts > 0;
    // The excess of the previous candidate, at twice this one's step, over the current cost -- on
    // the screening rollouts when screening, so that every excess compared is on the same rollouts.
    auto previous_excess = Scalar{0};
    // A line search over the proximal weight needs a prior to weigh; the first pass has none, and
    // blends instead.
    const bool proximal_candidates =
        options_.proximal_line_search && !proximal_.empty() && proximal_[0].prior.squaredNorm() > Scalar{0};
    auto step = proximal_candidates ? Scalar{1} : start_step_;
    std::optional<std::size_t> accepted_candidate;
    std::optional<profiling::ScopedPhase> line_search_phase(std::in_place, profiling::Phase::kLineSearch);
    for (std::size_t halving = 0; halving <= options_.max_step_halvings; ++halving, step *= Scalar{0.5}) {
      {
        const profiling::ScopedPhase blend_phase(profiling::Phase::kLineSearchBlend);
        const Result<> blended = proximal_candidates
                                     ? ProximalIntoTrial(proximal_lambda_ * std::pow(Scalar{4}, static_cast<Scalar>(halving)))
                                     : BlendIntoTrial(step);
        if (!blended.has_value()) {
          return ErrorResult(blended.error().message, blended.error().location);
        }
      }
      ++report.candidates_tried;
      auto excess = Scalar{0};
      Scalar screened_cost = Scalar{0};
      bool confirm = true;
      if (screening) {
        const profiling::ScopedPhase screen_phase(profiling::Phase::kLineSearchScreen);
        screened_cost = Evaluate(trial_, options_.screening_rollouts).total;
        excess = screened_cost - current_screened_cost_;
        confirm = excess < Scalar{0};
      }
      if (confirm) {
        const Costs trial_costs = [&] {
          const profiling::ScopedPhase full_phase(profiling::Phase::kLineSearchFull);
          return Evaluate(trial_);
        }();
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
          if (options_.adaptive_initial_step) {
            start_step_ = std::min(Scalar{1}, Scalar{2} * step);
          }
          accepted_candidate = halving;
          OnCurrentChanged();
          break;
        }
      }
      if (options_.stop_uphill_line_search && halving > 0 && (Scalar{4} * excess) - previous_excess > Scalar{0}) {
        break;
      }
      previous_excess = excess;
    }
    line_search_phase.reset();
    report.line_search_seconds = lap();
    if (options_.proximal_weight > Scalar{0}) {
      AdaptProximalWeight(accepted_candidate, proximal_candidates);
    }
    report.proximal_weight = proximal_lambda_;

    if (options_.reset_normal_equations_on_acceptance && report.accepted_step > Scalar{0}) {
      for (auto& halves : accumulated_) {
        for (auto& equations : halves) {
          equations = NormalEquations{};
        }
      }
    }

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
    OnCurrentChanged();
    if (options_.screening_rollouts > 0) {
      current_screened_cost_ = Evaluate(current_, options_.screening_rollouts).total;
    }
    return SuccessResult();
  }

  [[nodiscard]] auto DiffusionScale() const noexcept -> Scalar { return forward_model_.scale(); }

  // Replaces the value functions at every stage before the terminal one with `table`, one entry per
  // stage 0..K-1, and re-scores the current policy: a warm start from another method's solution,
  // such as iLQR's quadratic value expansions (examples/ilqr_seeded_table.hpp). The terminal entry
  // is the problem's own terminal cost and is kept. Both halves of a double-estimation pass start
  // from the new table, the stall counter is reset, and any normal equations or replay batches
  // carried over from earlier iterations are dropped, since their targets came from the old table.
  //
  // The new table is adopted whether or not its policy beats the one it replaces; CurrentCost()
  // before and after says which is better. Seeding sets no exploration schedule of its own: a seeded
  // solver should usually be built with `initial_exploration_probability` equal to the regular one,
  // so that the first forward pass samples around the seeded policy instead of exploring blindly.
  //
  // Fails, leaving the solver untouched, unless `table` has one entry per control stage.
  auto SeedValueFunctions(std::span<const ValueFunctionT> table) noexcept -> Result<> {
    RESULT_ASSERT(table.size() == kNumControlStages,
                  "DtFbsdeIterativeSolver::SeedValueFunctions: the table needs one entry per control stage.");
    for (std::size_t stage = 0; stage < kNumControlStages; ++stage) {
      current_[stage] = table[stage];
      fitted_[stage] = table[stage];
      trial_[stage] = table[stage];
      halves_[0][stage] = table[stage];
      halves_[1][stage] = table[stage];
    }
    for (auto& halves : accumulated_) {
      for (auto& equations : halves) {
        equations = NormalEquations{};
      }
    }
    for (auto& halves : replay_) {
      for (auto& kept : halves) {
        kept.clear();
      }
    }
    stall_factor_ = Scalar{1};
    stalled_iterations_ = 0;
    current_costs_ = Evaluate(current_);
    OnCurrentChanged();
    if (options_.screening_rollouts > 0) {
      current_screened_cost_ = Evaluate(current_, options_.screening_rollouts).total;
    }
    return SuccessResult();
  }

  // The cost of the policy over `table` -- with the terminal entry kept -- on the evaluation rollouts,
  // which is what SeedValueFunctions() would score it at, without adopting it: for choosing among
  // candidate seeds.
  //
  // Fails unless `table` has one entry per control stage.
  [[nodiscard]] auto SeededCost(std::span<const ValueFunctionT> table) const noexcept -> Result<Scalar> {
    RESULT_ASSERT(table.size() == kNumControlStages,
                  "DtFbsdeIterativeSolver::SeededCost: the table needs one entry per control stage.");
    std::vector<ValueFunctionT> candidate = current_;
    std::ranges::copy(table, candidate.begin());
    return SuccessResult(Evaluate(candidate).total);
  }

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

  using NormalEquations = typename DtFbsdeNormalEquationsOf<ValueFunctionT>::type;
  using SharedFit = typename DtFbsdeTimeSharedFitOf<ValueFunctionT>::type;

  // One half of a double-estimation batch.
  static constexpr int kHalf = kSampleCount / 2;
  using HalfStates = ValueFunctionStateBatch<State, kHalf>;
  using HalfValues = ValueFunctionValueBatch<State, kHalf>;

  // Whether the representation can be fitted by the pooled closed-form solve.
  static constexpr bool kPooledFitSupported = requires(ValueFunctionT value_function, const State& state) {
    { value_function.ParameterGradient(state) } -> std::convertible_to<typename ValueFunctionT::ParameterVector>;
    value_function.SetParameters(typename ValueFunctionT::ParameterVector{});
  };

  // A forward batch kept for pooling at one stage: its states, the controls its paths took from them,
  // the running cost each column had accrued, and how many leading columns are paths, not probes.
  struct PoolBlock {
    StateBlock states;
    ControlBlock controls;
    ValueBlock accrued;
    std::size_t path_count = 0;
  };

  using AnchorBlock = Eigen::Matrix<Scalar, N, Eigen::Dynamic>;
  using DynamicMat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  using DynamicVector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

  // One stage's proximal fit: its latest pass's normal equations (ridge included), the prior term at
  // unit weight that pass used, and the decayed Gram matrix of every batch so far with its total
  // sample weight.
  struct ProximalStage {
    DynamicMat normal_matrix;
    DynamicVector normal_vector;
    DynamicMat prior;
    DynamicMat information;
    Scalar information_weight = Scalar{0};
  };

  // A half-batch's states and accrued costs, kept for replay; targets are recomputed when replayed.
  struct ReplayBlock {
    HalfStates states;
    HalfValues accrued;
  };

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
    const profiling::ScopedPhase forward_phase(profiling::Phase::kForwardPass);
    if (options_.sigma_point_rule != SigmaPointRule::kNone) {
      return SampleSigmaPointForwardPass(inflation, seed);
    }
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
    std::ranges::fill(hold_remaining_, std::size_t{0});
    for (std::size_t stage = 0; stage < kNumControlStages; ++stage) {
      accrued_before_[stage] = accrued;
      ParallelFor(NumSamples, [&](std::size_t sample) noexcept {
        const profiling::ScopedPhase sample_phase(profiling::Phase::kForwardSample);
        const auto column = static_cast<Eigen::Index>(sample);
        const bool is_path = sample < path_count;

        // A path steps from its own state; a probe from a path's.
        std::size_t parent = sample;
        if (!is_path) {
          const auto pick = SampleUniform<Scalar>(sampler, sample, stage + 1, kParentPickComponent);
          parent = std::min(static_cast<std::size_t>(pick * static_cast<Scalar>(path_count)), path_count - 1);
        }
        const State state = states_[stage].col(static_cast<Eigen::Index>(parent));

        // A path still inside a held exploratory arc keeps its control; otherwise it explores with
        // the scheduled probability and, when it does, holds the drawn control for
        // `exploration_hold_stages` stages in all.
        const bool holding = is_path && hold_remaining_[sample] > 0;
        const auto index = static_cast<Scalar>(sample);
        const bool explorer = std::floor((index + Scalar{1}) * options_.exploring_path_fraction) >
                              std::floor(index * options_.exploring_path_fraction);
        const bool explore = !is_path || holding ||
                             (explorer && SampleUniform<Scalar>(sampler, sample, stage + 1, kExploreComponent) < exploration);
        Control control;
        if (holding) {
          control = held_controls_[sample];
          --hold_remaining_[sample];
        } else if (explore) {
          control = ExploratoryL1Control<M>(
              sampler, sample, stage + 1, kControlPickComponent, lower_bounds_, upper_bounds_);
          if (is_path && options_.exploration_hold_stages > 1) {
            held_controls_[sample] = control;
            hold_remaining_[sample] = options_.exploration_hold_stages - 1;
          }
        } else {
          const profiling::ScopedPhase policy_phase(profiling::Phase::kForwardPolicy);
          control = policy(stage, state);
        }

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

  // The deterministic forward pass. The state distribution at each stage is represented as a
  // Gaussian N(m, L L') -- or two, one at the problem's noise and one at `inflation` times it, as the
  // random pass's uninflated and inflated paths are -- and its states are the rule's points,
  // `m + L xi_j`. Stepping every point under the current policy without noise, the next stage's
  // mean and covariance are the points' weighted moments plus the (inflated) noise covariance:
  //
  //   m' = sum_j w_j x'_j,   P' = sum_j w_j (x'_j - m')(x'_j - m')' + c^2 S S',   x'_j = x_j + F(x_j, mu(x_j)),
  //
  // the prediction step of a sigma-point filter. With `sigma_point_probes`, every point is also
  // followed by one-step branches under each control of the L1 set {lower, 0, upper} per control,
  // stepped without noise from the point at the stage before -- the random pass's probes, made
  // deterministic. A point's accrued cost is its Gaussian's expected running cost so far, and a
  // probe's is that at the stage before plus its own step's cost.
  //
  // The columns cycle through these items, half by half: column s belongs to half s % 2, and both
  // halves see every item when the batch is at least twice their number. The second half's points
  // are the rule's turned by a fixed rotation (`rotate_sigma_point_halves`), so the two independent
  // fits of a double-estimation pass see different states. Returns the expected cost of the
  // problem-noise Gaussian under the current policy, as the rule estimates it.
  auto SampleSigmaPointForwardPass(Scalar inflation, std::uint64_t seed) noexcept -> Scalar {
    const Policy policy = MakePolicy(current_);
    const std::size_t point_count = sigma_points_.points.size();
    std::array<StateMat, 2> turns{identity_, options_.rotate_sigma_point_halves ? sigma_rotation_ : identity_};
    if (options_.randomize_sigma_point_rotation) {
      const CounterBasedNormalSampler sampler(seed);
      for (std::size_t half = 0; half < 2; ++half) {
        turns[half] = RandomRotation(sampler, half);
      }
    }

    std::vector<Scalar> component_inflations;
    if (options_.uninflated_path_fraction > Scalar{0} || inflation <= Scalar{1}) {
      component_inflations.push_back(Scalar{1});
    }
    if (inflation > Scalar{1} && options_.uninflated_path_fraction < Scalar{1}) {
      component_inflations.push_back(inflation);
    }
    const std::size_t component_count = component_inflations.size();

    std::vector<Control> branch_controls;
    if (options_.sigma_point_probes) {
      branch_controls.push_back(Control::Zero());
      for (std::size_t index = 0; index < kControlDim; ++index) {
        const auto axis = static_cast<Eigen::Index>(index);
        Control low = Control::Zero();
        Control high = Control::Zero();
        low[axis] = lower_bounds_[axis];
        high[axis] = upper_bounds_[axis];
        branch_controls.push_back(low);
        branch_controls.push_back(high);
      }
    }
    const std::size_t branches = 1 + branch_controls.size();
    const std::size_t items = component_count * point_count * branches;

    // Each Gaussian's mean, square-root covariance, and expected accrued cost.
    std::vector<State> means(component_count, initial_mean_);
    std::vector<StateMat> factors(component_count, initial_covariance_factor_);
    std::vector<Scalar> accrued(component_count, Scalar{0});
    // The points of the stage before, per half, component, and point: a probe's parent.
    std::array<std::vector<State>, 2> previous_points{std::vector<State>(component_count * point_count),
                                                      std::vector<State>(component_count * point_count)};
    std::vector<Scalar> previous_accrued(component_count, Scalar{0});

    Scalar expected_cost = Scalar{0};
    for (std::size_t stage = 0; stage < NumStages; ++stage) {
      std::array<std::vector<State>, 2> points{std::vector<State>(component_count * point_count),
                                               std::vector<State>(component_count * point_count)};
      for (std::size_t half = 0; half < 2; ++half) {
        const StateMat& turn = turns[half];
        for (std::size_t component = 0; component < component_count; ++component) {
          for (std::size_t point = 0; point < point_count; ++point) {
            points[half][(component * point_count) + point] =
                means[component] + (factors[component] * (turn * sigma_points_.points[point]));
          }
        }
      }

      // Fill the stage's columns.
      for (std::size_t sample = 0; sample < NumSamples; ++sample) {
        const auto column = static_cast<Eigen::Index>(sample);
        const std::size_t half = sample % 2;
        const std::size_t item = (sample / 2) % items;
        const std::size_t component = item / (point_count * branches);
        const std::size_t point = (item / branches) % point_count;
        const std::size_t branch = item % branches;
        const std::size_t index = (component * point_count) + point;
        if (branch == 0 || stage == 0) {
          states_[stage].col(column) = points[half][index];
          accrued_before_[stage][column] = accrued[component];
        } else {
          const State& parent = previous_points[half][index];
          const Control& control = branch_controls[branch - 1];
          states_[stage].col(column) = forward_model_(stage - 1, parent, control, State::Zero());
          accrued_before_[stage][column] = previous_accrued[component] + running_cost_(stage - 1, parent, control);
          controls_[stage - 1].col(column) = control;
        }
        if (branch == 0 && stage > 0) {
          // Points are drawn from a Gaussian, not stepped from a parent: no control produced them.
          controls_[stage - 1].col(column).setZero();
        }
      }
      if (stage == kNumControlStages) {
        for (std::size_t point = 0; point < point_count; ++point) {
          expected_cost += sigma_points_.weights[point] * terminal_cost_(points[0][point]);
        }
        expected_cost += accrued[0];
        break;
      }

      // Carry each Gaussian to the next stage through the closed loop.
      previous_points = points;
      previous_accrued = accrued;
      // Moments are carried with half 0's points, whatever their rotation: any rotation of a
      // symmetric rule matches the same moments.
      for (std::size_t component = 0; component < component_count; ++component) {
        std::vector<State> next(point_count);
        std::vector<Control> chosen(point_count);
        ParallelFor(point_count, [&](std::size_t point) noexcept {
          const State& x = points[0][(component * point_count) + point];
          chosen[point] = policy(stage, x);
          next[point] = forward_model_(stage, x, chosen[point], State::Zero());
        });
        State mean = State::Zero();
        Scalar running = Scalar{0};
        for (std::size_t point = 0; point < point_count; ++point) {
          mean += sigma_points_.weights[point] * next[point];
          running += sigma_points_.weights[point] *
                     running_cost_(stage, State(points[0][(component * point_count) + point]), chosen[point]);
        }
        StateMat covariance = StateMat::Zero();
        for (std::size_t point = 0; point < point_count; ++point) {
          const State offset = next[point] - mean;
          covariance += sigma_points_.weights[point] * (offset * offset.transpose());
        }
        const StateMat diffusion = forward_model_.diffusion()(stage, means[component]);
        const Scalar scale = component_inflations[component];
        covariance += (scale * scale) * (diffusion * diffusion.transpose());
        means[component] = mean;
        factors[component] = SquareRoot(covariance);
        accrued[component] += running;
      }
    }
    return expected_cost;
  }

  // A square root of a symmetric positive semidefinite matrix, robust to the slightly indefinite
  // covariance that negative weights or rounding can leave: negative eigenvalues are clipped to zero.
  [[nodiscard]] static auto SquareRoot(const StateMat& covariance) noexcept -> StateMat {
    const Eigen::SelfAdjointEigenSolver<StateMat> eigen(Scalar{0.5} * (covariance + covariance.transpose()));
    return eigen.eigenvectors() * eigen.eigenvalues().cwiseMax(Scalar{0}).cwiseSqrt().asDiagonal();
  }

  // A rotation drawn uniformly from the orthogonal group: the Q of a Gaussian matrix's QR
  // factorization, its columns' signs fixed by R's diagonal.
  [[nodiscard]] static auto RandomRotation(const CounterBasedNormalSampler& sampler, std::size_t half) noexcept
      -> StateMat {
    StateMat gaussian;
    for (int row = 0; row < N; ++row) {
      for (int column = 0; column < N; ++column) {
        gaussian(row, column) = static_cast<Scalar>(
            sampler.Sample(half, static_cast<std::size_t>(row), static_cast<std::size_t>(column) + 1000U));
      }
    }
    const Eigen::HouseholderQR<StateMat> qr(gaussian);
    StateMat q = qr.householderQ();
    const StateMat r = qr.matrixQR().template triangularView<Eigen::Upper>();
    for (int column = 0; column < N; ++column) {
      if (r(column, column) < Scalar{0}) {
        q.col(column) = -q.col(column);
      }
    }
    return q;
  }

  // The fixed rotation the second half's points are turned by: pi/8 in each plane of consecutive
  // axes, an angle that maps none of the rules' symmetric sets onto itself.
  [[nodiscard]] static auto SigmaRotation() noexcept -> StateMat {
    StateMat rotation = StateMat::Identity();
    const Scalar angle = Scalar{0.39269908169872414};
    for (int axis = 0; axis + 1 < N; ++axis) {
      StateMat plane = StateMat::Identity();
      plane(axis, axis) = std::cos(angle);
      plane(axis + 1, axis + 1) = std::cos(angle);
      plane(axis, axis + 1) = -std::sin(angle);
      plane(axis + 1, axis) = std::sin(angle);
      rotation = plane * rotation;
    }
    return rotation;
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
    const profiling::ScopedPhase backward_phase(profiling::Phase::kBackwardPass);
    if (options_.backward_target == DtFbsdeBackwardTarget::kDoubleGreedy) {
      return DoubleGreedyBackwardPass();
    }
    if (options_.backward_target == DtFbsdeBackwardTarget::kTaylorNoiseless) {
      return TaylorNoiselessBackwardPass();
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
      if (!shared_fits_.empty()) {
        RESULT_RETURN_IF_ERROR(FitSharedStage<kSampleCount>(fitted_,
                                                            stage,
                                                            0,
                                                            states,
                                                            targets,
                                                            accrued_before_[stage],
                                                            slopes_wanted ? &slopes : nullptr,
                                                            Accumulator(stage, 0)));
        continue;
      }
      const profiling::ScopedPhase fit_phase(profiling::Phase::kBackwardFit);
      if (const Result<> fitted = FitStage<kSampleCount>(fitted_[stage],
                                                         states,
                                                         targets,
                                                         accrued_before_[stage],
                                                         FitSeed(stage, 0),
                                                         slopes_wanted ? &slopes : nullptr,
                                                         Accumulator(stage, 0));
          !fitted.has_value()) {
        return fitted;
      }
    }
    return FinishSharedPass(fitted_, 0);
  }

  // The backward pass with the off-policy drifted Taylor noiseless estimator (Hawkins (2021),
  // Chapter 3, Table 3.1; TaylorNoiselessBackwardStepEstimator), evaluating the current policy mu:
  //
  //   Y^_i = l(X_i, mu(X_i)) + V~_{i+1}(xbar) + g' dF + 1/2 dF' H dF + 1/2 tr(Sigma' H Sigma),
  //
  // with xbar = X_i + K_i the mean of where the sample went, g and H the gradient and Hessian of
  // V~_{i+1} there, and dF = F^mu_i - K_i. A path sample's K_i is the drift f + B u of the control
  // the forward pass actually took from it, exploratory or not; a probe is never stepped onward, so
  // it has no drift of its own and is evaluated on-policy, K_i = F^mu_i. With slope matching, the
  // slope target is the gradient of the same second-order model at the policy's landing point,
  // carried back through the step: G = J' (g + H dF), J = d(x + F(x, mu(x)))/dx at fixed control.
  // V~_{i+1} is the table this pass has just fitted, as in BackwardPass(). The pass also measures
  // the Girsanov drift || Sigma^-1 dF || it ran at, which IterationReport::mean_girsanov_drift
  // reports.
  auto TaylorNoiselessBackwardPass() noexcept -> Result<> {
    // The target policy: the current one, or the greedy policy of the value functions this pass is
    // fitting, whose stage-k decision reads the stage k + 1 fit made just before it.
    const Policy target_policy = options_.taylor_target_policy == DtFbsdeTaylorTargetPolicy::kGreedy
                                     ? MakePolicy(fitted_)
                                     : MakePolicy(current_);
    using Estimator = TaylorNoiselessBackwardStepEstimator<N, M, Model, Policy, RunningCost, Scalar>;
    const Estimator estimator(forward_model_, target_policy, running_cost_);
    const std::size_t path_count =
        options_.sigma_point_rule == SigmaPointRule::kNone ? PathCount() : std::size_t{0};
    const bool slopes_wanted = options_.slope_weight > Scalar{0};
    Scalar drift_norm_sum = Scalar{0};
    Scalar correction_sum = Scalar{0};
    Scalar target_sum = Scalar{0};
    CorrectionStats stats;
    for (std::size_t stage = kNumControlStages; stage-- > 0;) {
      const StateBlock& states = states_[stage];
      const ValueFunctionT& next = fitted_[stage + 1];
      ValueBlock targets;
      ValueBlock norms;
      ValueBlock corrections;
      ValueBlock signed_corrections;
      ValueBlock curvature_corrections;
      ValueBlock off_policy;
      ValueBlock residuals = ValueBlock::Zero();
      ValueBlock girsanov_weights = ValueBlock::Ones();
      SlopeTargets<kSampleCount> slopes;
      ParallelFor(NumSamples, [&](std::size_t sample) noexcept {
        const auto column = static_cast<Eigen::Index>(sample);
        const State state = states.col(column);
        const State uncontrolled = forward_model_.state_drift()(stage, state);
        const Eigen::Matrix<Scalar, N, M> columns = forward_model_.control_drift_mat()(stage, state);
        const Control control = target_policy(stage, state);
        const State policy_drift = uncontrolled + (columns * control);
        // A path's drift is the one its sampled control produced; a probe has none of its own.
        const State drift =
            sample < path_count ? State(uncontrolled + (columns * controls_[stage].col(column))) : policy_drift;
        const State mean_next = state + drift;
        const State gradient = next.Gradient(mean_next);
        const StateMat hessian = next.Hessian(mean_next);
        const State difference = policy_drift - drift;
        const Scalar first_order = gradient.dot(difference);
        const Scalar second_order = Scalar{0.5} * difference.dot(hessian * difference);
        const Scalar estimate =
            estimator.EstimateFromParts(stage, state, drift, control, next(mean_next), gradient, hessian);
        const Scalar truncated = estimate - (options_.taylor_correction_order < 2 ? second_order : Scalar{0}) -
                                 (options_.taylor_correction_order < 1 ? first_order : Scalar{0});
        targets[column] = std::max(truncated, options_.target_floor);
        corrections[column] = std::abs(first_order);
        signed_corrections[column] = first_order;
        curvature_corrections[column] = std::abs(second_order);
        const bool is_off_policy = !difference.isZero();
        off_policy[column] = is_off_policy ? Scalar{1} : Scalar{0};
        if (options_.measure_taylor_residual && is_off_policy) {
          const State policy_next = state + policy_drift;
          const StateMat diffusion = forward_model_.diffusion()(stage, state);
          const Scalar exact = running_cost_(stage, state, control) + next(policy_next) +
                               (Scalar{0.5} * next.Hessian(policy_next).cwiseProduct(diffusion * diffusion.transpose()).sum());
          residuals[column] = std::abs(truncated - exact);
        }
        const StateMat diffusion = forward_model_.diffusion()(stage, state);
        norms[column] = diffusion.fullPivLu().solve(difference).norm();
        if (options_.girsanov_weight > Scalar{0}) {
          girsanov_weights[column] = std::exp(-Scalar{0.5} * options_.girsanov_weight * norms[column] * norms[column]);
        }
        if (slopes_wanted) {
          StateMat jacobian;
          for (Eigen::Index axis = 0; axis < N; ++axis) {
            const Scalar step = kJacobianStep * std::max(Scalar{1}, std::abs(state[axis]));
            const State offset = step * State::Unit(axis);
            jacobian.col(axis) = (forward_model_(stage, State(state + offset), control, State::Zero()) -
                                  forward_model_(stage, State(state - offset), control, State::Zero())) /
                                 (Scalar{2} * step);
          }
          const State slope_gradient = targets[column] <= options_.target_floor
                                           ? State(State::Zero())
                                           : State(jacobian.transpose() * (gradient + (hessian * difference)));
          SetSlopeTargets<kSampleCount>(slopes, column, stage, state, slope_gradient);
        }
      });
      drift_norm_sum += norms.mean();
      correction_sum += corrections.mean();
      target_sum += targets.cwiseAbs().mean();
      const Scalar off_policy_count = off_policy.sum();
      stats.signed_first_order += signed_corrections.mean();
      stats.second_order += curvature_corrections.mean();
      stats.off_policy_fraction += off_policy.mean();
      if (off_policy_count > Scalar{0}) {
        stats.off_policy_first_order += corrections.cwiseProduct(off_policy).sum() / off_policy_count;
        stats.taylor_residual += residuals.sum() / off_policy_count;
      }
      fitted_[stage] = current_[stage];
      if (UsesPooledFit(options_)) {
        RESULT_RETURN_IF_ERROR(FitPooledStage(stage, targets, girsanov_weights, target_policy, estimator, path_count));
        continue;
      }
      if (!shared_fits_.empty()) {
        RESULT_RETURN_IF_ERROR(
            FitSharedStage<kSampleCount>(fitted_,
                                         stage,
                                         0,
                                         states,
                                         targets,
                                         accrued_before_[stage],
                                         slopes_wanted ? &slopes : nullptr,
                                         Accumulator(stage, 0),
                                         options_.girsanov_weight > Scalar{0} ? &girsanov_weights : nullptr));
        continue;
      }
      const profiling::ScopedPhase fit_phase(profiling::Phase::kBackwardFit);
      if (const Result<> fitted = FitStage<kSampleCount>(fitted_[stage],
                                                         states,
                                                         targets,
                                                         accrued_before_[stage],
                                                         FitSeed(stage, 0),
                                                         slopes_wanted ? &slopes : nullptr,
                                                         Accumulator(stage, 0),
                                                         options_.girsanov_weight > Scalar{0} ? &girsanov_weights
                                                                                              : nullptr);
          !fitted.has_value()) {
        return fitted;
      }
    }
    RESULT_RETURN_IF_ERROR(FinishSharedPass(fitted_, 0));
    const auto stages = static_cast<Scalar>(kNumControlStages);
    last_girsanov_drift_ = drift_norm_sum / stages;
    last_first_order_correction_ = correction_sum / stages;
    last_target_magnitude_ = target_sum / stages;
    last_correction_stats_ = CorrectionStats{.signed_first_order = stats.signed_first_order / stages,
                                             .second_order = stats.second_order / stages,
                                             .off_policy_fraction = stats.off_policy_fraction / stages,
                                             .off_policy_first_order = stats.off_policy_first_order / stages,
                                             .taylor_residual = stats.taylor_residual / stages};
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

    const std::array<Policy, 2> choosers{MakePolicy(halves_[0]), MakePolicy(halves_[1])};
    const bool slopes_wanted = options_.slope_weight > Scalar{0};
    for (std::size_t stage = kNumControlStages; stage-- > 0;) {
      for (std::size_t half = 0; half < 2; ++half) {
        const Policy& chooser = choosers[half];
        const Policy& scorer = choosers[1 - half];
        // The targets and slopes of a block of states, from the current tables: the control chosen
        // by this half's table and scored -- value and slope -- by the other's.
        const auto targets_of = [&](const HalfStates& states, HalfValues& targets, SlopeTargets<kHalf>& slopes) {
          const profiling::ScopedPhase targets_phase(profiling::Phase::kBackwardTargets);
          ParallelFor(static_cast<std::size_t>(kHalf), [&](std::size_t slot) noexcept {
            const profiling::ScopedPhase sample_phase(profiling::Phase::kBackwardTargetSample);
            const auto column = static_cast<Eigen::Index>(slot);
            const State state = states.col(column);
            const Control control = [&] {
              const profiling::ScopedPhase argmin_phase(profiling::Phase::kBackwardArgmin);
              return chooser(stage, state);
            }();
            const Scalar q_value = [&] {
              const profiling::ScopedPhase q_value_phase(profiling::Phase::kBackwardQValue);
              return scorer.QValue(stage, state, control);
            }();
            targets[column] = std::max(q_value, options_.target_floor);
            if (slopes_wanted) {
              const profiling::ScopedPhase slope_phase(profiling::Phase::kBackwardSlopeTarget);
              const State gradient = q_value < options_.target_floor
                                         ? State(State::Zero())
                                         : GradientTarget(stage, state, control, halves_[1 - half][stage + 1]);
              SetSlopeTargets<kHalf>(slopes, column, stage, state, gradient);
            }
          });
        };
        HalfStates states;
        HalfValues targets;
        HalfValues accrued;
        SlopeTargets<kHalf> slopes;
        for (std::size_t slot = 0; slot < static_cast<std::size_t>(kHalf); ++slot) {
          const auto column = static_cast<Eigen::Index>(slot);
          states.col(column) = states_[stage].col(static_cast<Eigen::Index>((2 * slot) + half));
          accrued[column] = accrued_before_[stage][static_cast<Eigen::Index>((2 * slot) + half)];
        }
        halves_[half][stage] = current_[stage];
        if (options_.replay_batches > 1) {
          if constexpr (DtFbsdeNormalEquationsOf<ValueFunctionT>::kSupported) {
            // Replay: fit this pass's block and the kept ones together, every target recomputed now.
            std::deque<ReplayBlock>& kept = replay_[stage][half];
            kept.push_front(ReplayBlock{.states = states, .accrued = accrued});
            while (kept.size() > options_.replay_batches) {
              kept.pop_back();
            }
            NormalEquations equations;
            for (const ReplayBlock& block : kept) {
              targets_of(block.states, targets, slopes);
              const HalfValues weights = SampleWeights<kHalf>(targets, block.accrued);
              if (slopes_wanted) {
                RESULT_RETURN_IF_ERROR((halves_[half][stage].template AccumulateNormalEquations<kHalf, M>(
                    block.states, targets, weights, slopes, options_.slope_weight, equations)));
              } else {
                RESULT_RETURN_IF_ERROR(halves_[half][stage].template AccumulateNormalEquations<kHalf>(
                    block.states, targets, weights, equations));
              }
            }
            RESULT_RETURN_IF_ERROR(halves_[half][stage].FitNormalEquations(equations, options_.ridge));
            continue;
          }
        }
        targets_of(states, targets, slopes);
        const profiling::ScopedPhase fit_phase(profiling::Phase::kBackwardFit);
        if (!shared_fits_.empty()) {
          RESULT_RETURN_IF_ERROR(FitSharedStage<kHalf>(halves_[half],
                                                       stage,
                                                       half,
                                                       states,
                                                       targets,
                                                       accrued,
                                                       slopes_wanted ? &slopes : nullptr,
                                                       Accumulator(stage, half)));
          continue;
        }
        if (const Result<> fitted = FitStage<kHalf>(halves_[half][stage],
                                                    states,
                                                    targets,
                                                    accrued,
                                                    FitSeed(stage, half + 1),
                                                    slopes_wanted ? &slopes : nullptr,
                                                    Accumulator(stage, half));
            !fitted.has_value()) {
          return fitted;
        }
      }
      const profiling::ScopedPhase average_phase(profiling::Phase::kBackwardAverage);
      fitted_[stage] = current_[stage];
      if (const Result<> averaged = fitted_[stage].SetParameters(
              Scalar{0.5} * (halves_[0][stage].Parameters() + halves_[1][stage].Parameters()));
          !averaged.has_value()) {
        return averaged;
      }
    }
    if (shared_fits_.empty()) {
      return SuccessResult();
    }
    // The pass's closing joint solves move every stage, so the average is taken again.
    for (std::size_t half = 0; half < 2; ++half) {
      RESULT_RETURN_IF_ERROR(FinishSharedPass(halves_[half], half));
    }
    for (std::size_t stage = 0; stage < kNumControlStages; ++stage) {
      RESULT_RETURN_IF_ERROR(fitted_[stage].SetParameters(
          Scalar{0.5} * (halves_[0][stage].Parameters() + halves_[1][stage].Parameters())));
    }
    return SuccessResult();
  }

  // Fits stage `stage` of `table` through the time-shared fit of `half` (zero for the single-table
  // passes); see DtFbsdeIterativeOptions::time_sharing. The stage's normal equations -- carried
  // across iterations through `accumulator` when there is a memory -- replace its previous ones in
  // the shared fit, and the knots around the stage are re-solved and written back to every stage
  // they reach. Before a shared basis exists, the stage is fitted on its own.
  template <int Count>
  auto FitSharedStage(std::vector<ValueFunctionT>& table,
                      std::size_t stage,
                      std::size_t half,
                      const ValueFunctionStateBatch<State, Count>& states,
                      const ValueFunctionValueBatch<State, Count>& targets,
                      const ValueFunctionValueBatch<State, Count>& accrued,
                      const SlopeTargets<Count>* slopes,
                      NormalEquations* accumulator,
                      const ValueFunctionValueBatch<State, Count>* extra_weights = nullptr) noexcept -> Result<> {
    if constexpr (DtFbsdeNormalEquationsOf<ValueFunctionT>::kSupported) {
      ValueFunctionValueBatch<State, Count> weights = SampleWeights<Count>(targets, accrued);
      if (extra_weights != nullptr) {
        weights = weights.cwiseProduct(*extra_weights);
      }
      NormalEquations fresh;
      NormalEquations& equations = accumulator != nullptr ? *accumulator : fresh;
      if (accumulator != nullptr) {
        accumulator->Decay(options_.normal_equation_memory);
      }
      ValueFunctionT& value_function = table[stage];
      if (slopes == nullptr) {
        RESULT_RETURN_IF_ERROR(
            value_function.template AccumulateNormalEquations<Count>(states, targets, weights, equations));
      } else {
        RESULT_RETURN_IF_ERROR((value_function.template AccumulateNormalEquations<Count, M>(
            states, targets, weights, *slopes, options_.slope_weight, equations)));
      }
      const typename ValueFunctionT::ParameterVector anchor = current_[stage].Parameters();
      const bool anchored = [&] {
        if constexpr (requires { value_function.Anchor(); }) {
          return value_function.Anchor() == decltype(value_function.Anchor())::kCurrent;
        } else {
          return false;
        }
      }();
      SharedFit& fit = shared_fits_[half];
      Scalar stage_weight = Scalar{1};
      if (options_.time_sharing.balance_stages) {
        const Scalar mean = targets.mean();
        const Scalar variance = (targets.array() - mean).square().mean();
        stage_weight = Scalar{1} / (variance + (Scalar{1e-6} * (Scalar{1} + (mean * mean))));
      }
      if (stage + 1 == kNumControlStages) {
        fit.BeginPass();
      }
      RESULT_RETURN_IF_ERROR(
          fit.SetStage(stage, equations, options_.ridge, anchored ? &anchor : nullptr, stage_weight));
      if (options_.time_sharing_pass == DtFbsdeTimeSharingPass::kAfterPass ||
          (fit.Rank() > 0 && !fit.HasSharedBasis())) {
        return value_function.FitNormalEquations(equations, options_.ridge);
      }
      RESULT_RETURN_IF_ERROR(fit.SolveAroundStage(stage, options_.time_sharing_stale_weight));
      return fit.WriteAroundStage(stage, std::span<ValueFunctionT>(table));
    } else {
      (void)table, (void)stage, (void)half, (void)states, (void)targets, (void)accrued, (void)slopes, (void)accumulator,
          (void)extra_weights;
      return ErrorResult("DtFbsdeIterativeSolver: time sharing needs a representation with normal equations.");
    }
  }

  // Closes a pass of the time-shared fit of `half` and writes it into `table`: the shared basis
  // started from this pass's stage-by-stage fits, or refitted to every stage's equations, and the
  // coefficients solved jointly. Does nothing without time sharing.
  auto FinishSharedPass(std::vector<ValueFunctionT>& table, std::size_t half) noexcept -> Result<> {
    if constexpr (DtFbsdeNormalEquationsOf<ValueFunctionT>::kSupported) {
      if (shared_fits_.empty()) {
        return SuccessResult();
      }
      const profiling::ScopedPhase time_share_phase(profiling::Phase::kBackwardTimeShare);
      SharedFit& fit = shared_fits_[half];
      if (fit.Rank() > 0 && fit.HasSharedBasis() && options_.time_sharing_pass != DtFbsdeTimeSharingPass::kAfterPass) {
        for (std::size_t sweep = 0; sweep < options_.time_sharing.sweeps; ++sweep) {
          RESULT_RETURN_IF_ERROR(fit.SolveSharedBasis());
          RESULT_RETURN_IF_ERROR(fit.SolveCoefficients());
        }
      } else {
        RESULT_RETURN_IF_ERROR(fit.Fit());
      }
      return fit.WriteTable(std::span<ValueFunctionT>(table).first(kNumControlStages));
    } else {
      (void)table, (void)half;
      return SuccessResult();
    }
  }

  // The Taylor noiseless target of one kept sample, recomputed against this pass's fit of the stage
  // after, and the sample's Girsanov weight (one unless the options ask for it); the same estimate the
  // pass makes for its own batch. `control` is the one the forward pass took from `state`, or null
  // for a probe, which is evaluated on-policy.
  template <typename EstimatorT>
  [[nodiscard]] auto PooledTarget(std::size_t stage,
                                  const State& state,
                                  const Control* control,
                                  const Policy& target_policy,
                                  const EstimatorT& estimator) const noexcept -> std::pair<Scalar, Scalar> {
    const ValueFunctionT& next = fitted_[stage + 1];
    const State uncontrolled = forward_model_.state_drift()(stage, state);
    const Eigen::Matrix<Scalar, N, M> columns = forward_model_.control_drift_mat()(stage, state);
    const Control policy_control = target_policy(stage, state);
    const State policy_drift = uncontrolled + (columns * policy_control);
    const State drift = control != nullptr ? State(uncontrolled + (columns * (*control))) : policy_drift;
    const State mean_next = state + drift;
    const State gradient = next.Gradient(mean_next);
    const StateMat hessian = next.Hessian(mean_next);
    const State difference = policy_drift - drift;
    const Scalar estimate =
        estimator.EstimateFromParts(stage, state, drift, policy_control, next(mean_next), gradient, hessian);
    const Scalar truncated =
        estimate -
        (options_.taylor_correction_order < 2 ? Scalar{0.5} * difference.dot(hessian * difference) : Scalar{0}) -
        (options_.taylor_correction_order < 1 ? gradient.dot(difference) : Scalar{0});
    auto weight = Scalar{1};
    if (options_.girsanov_weight > Scalar{0}) {
      const Scalar norm = forward_model_.diffusion()(stage, state).fullPivLu().solve(difference).norm();
      weight = std::exp(-Scalar{0.5} * options_.girsanov_weight * norm * norm);
    }
    return {std::max(truncated, options_.target_floor), weight};
  }

  // Fits stage `stage` of `fitted_` by weighted least squares over this pass's batch, the batches
  // pooled since the last acceptance, and the trust region's anchors; see
  // DtFbsdeIterativeOptions::pool_batches_since_acceptance and ::anchor_weight. `targets` and
  // `girsanov_weights` are this pass's batch's, already computed; the batch then joins the pool.
  template <typename EstimatorT>
  auto FitPooledStage(std::size_t stage,
                      const ValueBlock& targets,
                      const ValueBlock& girsanov_weights,
                      const Policy& target_policy,
                      const EstimatorT& estimator,
                      std::size_t path_count) noexcept -> Result<> {
    if constexpr (kPooledFitSupported) {
      using ParameterVector = typename ValueFunctionT::ParameterVector;
      using ParameterMat = Eigen::Matrix<Scalar, ParameterVector::RowsAtCompileTime, ParameterVector::RowsAtCompileTime>;
      const profiling::ScopedPhase fit_phase(profiling::Phase::kBackwardFit);
      ValueFunctionT& value_function = fitted_[stage];
      ParameterMat normal_matrix = ParameterMat::Zero();
      ParameterVector normal_vector = ParameterVector::Zero();
      Scalar data_weight = Scalar{0};
      std::size_t rows = 0;
      const auto accumulate = [&](const StateBlock& states, const ValueBlock& block_targets, const ValueBlock& weights) {
        for (std::size_t sample = 0; sample < NumSamples; ++sample) {
          const auto column = static_cast<Eigen::Index>(sample);
          const ParameterVector features = value_function.ParameterGradient(State(states.col(column)));
          normal_matrix.template selfadjointView<Eigen::Lower>().rankUpdate(features, weights[column]);
          normal_vector += (weights[column] * block_targets[column]) * features;
          data_weight += weights[column];
        }
        rows += NumSamples;
      };
      accumulate(states_[stage],
                 targets,
                 ValueBlock(SampleWeights<kSampleCount>(targets, accrued_before_[stage]).cwiseProduct(girsanov_weights)));

      if (options_.pool_batches_since_acceptance > 1) {
        std::deque<PoolBlock>& pool = pool_[stage];
        for (const PoolBlock& block : pool) {
          ValueBlock block_targets;
          ValueBlock block_girsanov;
          ParallelFor(NumSamples, [&](std::size_t sample) noexcept {
            const auto column = static_cast<Eigen::Index>(sample);
            const Control control = block.controls.col(column);
            const auto [target, weight] = PooledTarget(stage,
                                                       State(block.states.col(column)),
                                                       sample < block.path_count ? &control : nullptr,
                                                       target_policy,
                                                       estimator);
            block_targets[column] = target;
            block_girsanov[column] = weight;
          });
          accumulate(block.states,
                     block_targets,
                     ValueBlock(SampleWeights<kSampleCount>(block_targets, block.accrued).cwiseProduct(block_girsanov)));
        }
        pool.push_front(PoolBlock{.states = states_[stage],
                                  .controls = controls_[stage],
                                  .accrued = accrued_before_[stage],
                                  .path_count = path_count});
        while (pool.size() + 1 > options_.pool_batches_since_acceptance) {
          pool.pop_back();
        }
      }

      if (options_.anchor_weight > Scalar{0} && stage < anchors_.size() && anchors_[stage].cols() > 0) {
        const AnchorBlock& anchors = anchors_[stage];
        const Scalar weight = options_.anchor_weight * data_weight / static_cast<Scalar>(anchors.cols());
        for (Eigen::Index column = 0; column < anchors.cols(); ++column) {
          const State state = anchors.col(column);
          const ParameterVector features = value_function.ParameterGradient(state);
          normal_matrix.template selfadjointView<Eigen::Lower>().rankUpdate(features, weight);
          normal_vector += (weight * current_[stage](state)) * features;
        }
      }

      // The ridge scales with the data rows, as the closed-form fit's does with its batch.
      normal_matrix.diagonal().array() += options_.ridge * static_cast<Scalar>(rows);
      if (options_.proximal_weight > Scalar{0}) {
        // Kept for a line search over the weight, which refits from the same equations.
        ProximalStage& kept = proximal_[stage];
        kept.normal_matrix = normal_matrix.template selfadjointView<Eigen::Lower>();
        kept.normal_vector = normal_vector;
        kept.prior = kept.information_weight > Scalar{0}
                         ? DynamicMat(data_weight * kept.information / kept.information_weight)
                         : DynamicMat(DynamicMat::Zero(normal_matrix.rows(), normal_matrix.cols()));
        DynamicMat data = kept.normal_matrix;
        data.diagonal().array() -= options_.ridge * static_cast<Scalar>(rows);
        if (kept.information.size() == 0) {
          kept.information = DynamicMat::Zero(data.rows(), data.cols());
        }
        kept.information = (options_.proximal_memory * kept.information) + data;
        kept.information_weight = (options_.proximal_memory * kept.information_weight) + data_weight;
        RESULT_ASSIGN_OR_RETURN(const DynamicVector solution, SolveProximal(stage, proximal_lambda_));
        return value_function.SetParameters(ParameterVector(solution));
      }
      const Eigen::LDLT<ParameterMat> factorization(normal_matrix.template selfadjointView<Eigen::Lower>());
      RESULT_ASSERT(factorization.info() == Eigen::Success,
                    "DtFbsdeIterativeSolver: the pooled normal equations are singular; add a ridge weight.");
      const ParameterVector solution = factorization.solve(normal_vector);
      RESULT_ASSERT(solution.allFinite(), "DtFbsdeIterativeSolver: the pooled fit produced non-finite coefficients.");
      return value_function.SetParameters(solution);
    } else {
      (void)stage, (void)targets, (void)girsanov_weights, (void)target_policy, (void)estimator, (void)path_count;
      return ErrorResult("DtFbsdeIterativeSolver: pooling needs a representation with a parameter gradient.");
    }
  }

  // Stage `stage`'s proximal fit at weight `lambda`, from the equations its backward pass kept.
  [[nodiscard]] auto SolveProximal(std::size_t stage, Scalar lambda) const noexcept -> Result<DynamicVector> {
    const ProximalStage& kept = proximal_[stage];
    const DynamicVector anchor = current_[stage].Parameters();
    const DynamicMat matrix = kept.normal_matrix + (lambda * kept.prior);
    const DynamicVector vector = kept.normal_vector + (lambda * (kept.prior * anchor));
    const Eigen::LDLT<DynamicMat> factorization(matrix);
    RESULT_ASSERT(factorization.info() == Eigen::Success,
                  "DtFbsdeIterativeSolver: the proximal normal equations are singular; add a ridge weight.");
    const DynamicVector solution = factorization.solve(vector);
    RESULT_ASSERT(solution.allFinite(), "DtFbsdeIterativeSolver: the proximal fit produced non-finite coefficients.");
    return SuccessResult(solution);
  }

  // Writes the proximal fit at weight `lambda` into every control stage of `trial_`.
  auto ProximalIntoTrial(Scalar lambda) noexcept -> Result<> {
    if constexpr (kPooledFitSupported) {
      for (std::size_t stage = 0; stage < kNumControlStages; ++stage) {
        RESULT_ASSIGN_OR_RETURN(const DynamicVector solution, SolveProximal(stage, lambda));
        RESULT_RETURN_IF_ERROR(trial_[stage].SetParameters(typename ValueFunctionT::ParameterVector(solution)));
      }
      return SuccessResult();
    } else {
      (void)lambda;
      return ErrorResult("DtFbsdeIterativeSolver: the proximal fit needs a representation with a parameter gradient.");
    }
  }

  // The Levenberg-Marquardt adaptation of the proximal weight after a line search that accepted
  // candidate `accepted`, or none; `proximal_candidates` says whether its candidates were refits at
  // growing weights rather than blends.
  auto AdaptProximalWeight(std::optional<std::size_t> accepted, bool proximal_candidates) noexcept -> void {
    const Scalar factor = options_.proximal_adaptation;
    if (proximal_candidates && accepted.has_value()) {
      proximal_lambda_ *= std::pow(Scalar{4}, static_cast<Scalar>(*accepted)) / factor;
    } else if (accepted.has_value() && *accepted == 0) {
      proximal_lambda_ /= factor;
    } else if (accepted.has_value()) {
      proximal_lambda_ *= factor;
    } else {
      proximal_lambda_ *= options_.proximal_rejection_adaptation;
    }
    proximal_lambda_ = std::clamp(proximal_lambda_, options_.min_proximal_weight, options_.max_proximal_weight);
  }

  // Called whenever the current table changes: empties the pool, whose batches estimated the old
  // policy's value, and records the new policy's anchor states.
  auto OnCurrentChanged() noexcept -> void {
    for (auto& pool : pool_) {
      pool.clear();
    }
    if (options_.anchor_weight <= Scalar{0}) {
      return;
    }
    // The first `anchor_rollouts` evaluation rollouts, with the evaluation's own noise, as
    // EvaluatePolicyByRollouts() draws them.
    const std::size_t rollouts = options_.anchor_rollouts;
    anchors_.assign(kNumControlStages, AnchorBlock(N, static_cast<Eigen::Index>(rollouts)));
    const CounterBasedNormalSampler sampler(seed_ ^ kPolicyEvaluationStream);
    const Policy policy = MakePolicy(current_);
    ParallelFor(rollouts, [&](std::size_t rollout) noexcept {
      State noise;
      sampler.FillNoise(rollout, 0, noise);
      State state = initial_mean_ + (initial_covariance_factor_ * noise);
      for (std::size_t stage = 0; stage < kNumControlStages; ++stage) {
        anchors_[stage].col(static_cast<Eigen::Index>(rollout)) = state;
        const Control control = policy(stage, state);
        sampler.FillNoise(rollout, stage + 1, noise);
        state = forward_model_(stage, state, control, noise);
      }
    });
  }

  [[nodiscard]] static constexpr auto UsesPooledFit(const Options& options) noexcept -> bool {
    return options.pool_batches_since_acceptance > 1 || options.anchor_weight > Scalar{0} ||
           options.proximal_weight > Scalar{0};
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

  // The fit's sample weights: the local-entropy weights described at FitStage(), or all ones.
  template <int Count>
  [[nodiscard]] auto SampleWeights(const ValueFunctionValueBatch<State, Count>& targets,
                                   const ValueFunctionValueBatch<State, Count>& accrued) const noexcept
      -> ValueFunctionValueBatch<State, Count> {
    if (options_.entropy_temperature <= Scalar{0}) {
      return ValueFunctionValueBatch<State, Count>::Ones();
    }
    const ValueFunctionValueBatch<State, Count> heuristic = accrued + targets;
    const Scalar mean = heuristic.mean();
    const Scalar spread =
        std::sqrt((heuristic.array() - mean).square().mean()) + std::numeric_limits<Scalar>::epsilon();
    return (-(heuristic.array() - heuristic.minCoeff()) / (spread * options_.entropy_temperature)).exp();
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
                const SlopeTargets<Count>* slopes,
                NormalEquations* accumulator,
                const ValueFunctionValueBatch<State, Count>* extra_weights = nullptr) const noexcept -> Result<> {
    ValueFunctionValueBatch<State, Count> weights = SampleWeights<Count>(targets, accrued);
    if (extra_weights != nullptr) {
      weights = weights.cwiseProduct(*extra_weights);
    }
    if constexpr (requires {
                    value_function.template FitWeightedLeastSquares<Count>(states, targets, weights, Scalar{});
                  }) {
      if constexpr (DtFbsdeNormalEquationsOf<ValueFunctionT>::kSupported) {
        if (accumulator != nullptr) {
          // Recursive least squares: forget, add this batch's rows, solve what has been kept.
          accumulator->Decay(options_.normal_equation_memory);
          if (slopes == nullptr) {
            RESULT_RETURN_IF_ERROR(
                value_function.template AccumulateNormalEquations<Count>(states, targets, weights, *accumulator));
          } else {
            RESULT_RETURN_IF_ERROR((value_function.template AccumulateNormalEquations<Count, M>(
                states, targets, weights, *slopes, options_.slope_weight, *accumulator)));
          }
          return value_function.FitNormalEquations(*accumulator, options_.ridge);
        }
      }
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

  // The kept normal equations of one stage and half, or null when the solver does not carry them.
  [[nodiscard]] auto Accumulator(std::size_t stage, std::size_t half) noexcept -> NormalEquations* {
    if (options_.normal_equation_memory <= Scalar{0} || accumulated_.empty()) {
      return nullptr;
    }
    return &accumulated_[stage][half];
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
  // The diagnostics of the most recent Taylor noiseless backward pass; see IterationReport.
  Scalar last_girsanov_drift_ = Scalar{0};
  Scalar last_first_order_correction_ = Scalar{0};
  Scalar last_target_magnitude_ = Scalar{0};
  // The rest of the Taylor noiseless pass's correction diagnostics, as IterationReport describes them.
  struct CorrectionStats {
    Scalar signed_first_order = Scalar{0};
    Scalar second_order = Scalar{0};
    Scalar off_policy_fraction = Scalar{0};
    Scalar off_policy_first_order = Scalar{0};
    Scalar taylor_residual = Scalar{0};
  };
  CorrectionStats last_correction_stats_{};
  // Per path: how many more stages its held exploratory control lasts, and that control.
  std::vector<std::size_t> hold_remaining_ = std::vector<std::size_t>(NumSamples, 0);
  std::vector<Control> held_controls_ = std::vector<Control>(NumSamples, Control::Zero());

  // The normal equations carried across iterations, per stage and half; empty unless the options ask
  // for them. See DtFbsdeIterativeOptions::normal_equation_memory.
  std::vector<std::array<NormalEquations, 2>> accumulated_ =
      options_.normal_equation_memory > Scalar{0} ? std::vector<std::array<NormalEquations, 2>>(NumStages)
                                                  : std::vector<std::array<NormalEquations, 2>>();

  // The time-shared fits, one per half of a double-estimation pass (the first serves the other
  // passes); empty unless the options ask for time sharing.
  std::vector<SharedFit> shared_fits_;

  // The half-batches kept for replay, newest first, per stage and half; empty unless replaying.
  std::vector<std::array<std::deque<ReplayBlock>, 2>> replay_ =
      options_.replay_batches > 1 ? std::vector<std::array<std::deque<ReplayBlock>, 2>>(NumStages)
                                  : std::vector<std::array<std::deque<ReplayBlock>, 2>>();

  // The batches pooled since the last acceptance, newest first, per stage; empty unless pooling.
  std::vector<std::deque<PoolBlock>> pool_ = options_.pool_batches_since_acceptance > 1
                                                 ? std::vector<std::deque<PoolBlock>>(NumStages)
                                                 : std::vector<std::deque<PoolBlock>>();
  // The current policy's states on the anchor rollouts, per control stage; empty unless anchoring.
  std::vector<AnchorBlock> anchors_;
  // The step the next line search starts at; see DtFbsdeIterativeOptions::initial_step.
  Scalar start_step_ = Scalar{1};
  // The proximal fit's kept equations per stage, and its current weight; see
  // DtFbsdeIterativeOptions::proximal_weight.
  std::vector<ProximalStage> proximal_ =
      options_.proximal_weight > Scalar{0} ? std::vector<ProximalStage>(NumStages) : std::vector<ProximalStage>();
  Scalar proximal_lambda_ = Scalar{0};

  // The sigma-point rule's points and weights, and the rotation the second half's points are turned by.
  SigmaPointSet<N, Scalar> sigma_points_ = MakeSigmaPoints<N, Scalar>(options_.sigma_point_rule);
  StateMat sigma_rotation_ = SigmaRotation();
  StateMat identity_ = StateMat::Identity();
};

}  // namespace fbsde_traj_opt

#endif  // FBSDE_TRAJ_OPT_DT_FBSDE_ITERATIVE_SOLVER_HPP_
