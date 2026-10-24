// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#ifndef FBSDE_TRAJ_OPT_SOLVERS_BOX_ILQR_SOLVER_HPP_
#define FBSDE_TRAJ_OPT_SOLVERS_BOX_ILQR_SOLVER_HPP_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "fbsde_traj_opt/utils/result.hpp"

namespace fbsde_traj_opt {

// Iterative LQR with a box on the control (Tassa, Mansard & Todorov, "Control-limited differential
// dynamic programming", ICRA 2014), for the L1 minimum-fuel problems: a baseline from outside the
// FBSDE family, to measure the FBSDE methods against.
//
// The problem is
//
//   min  sum_k c |u_k| + x_K' F x_K,   x_{k+1} = x_k + f(k, x_k) + B(k, x_k) u_k,   lower <= u <= upper,
//
// the forward model with its noise set to zero: iLQR is certainty-equivalent, and the noise reaches
// it only through the closed-loop policy it hands back, `u = clamp(u_k + K_k (x - x_k))`, which is
// what an experiment rolls out under noise.
//
// The L1 cost has no curvature, and iLQR needs some, so the solver minimizes the pseudo-Huber
// smoothing `c (sqrt(u^2 + e^2) - e)`, with `e` shrinking every accepted iteration toward
// `min_smoothing` -- continuation from a smooth problem to nearly the true one. Reported costs are
// always the true L1 cost of the nominal trajectory.
//
// The box is handled the way the paper does for one control: the feedforward step is clamped, and
// at a clamped stage the feedback gain is zero, since the control there no longer responds to the
// state to first order. Only a single control is supported, where that clamping is exact.
//
// Dynamics Jacobians come from central differences of the noiseless step, so any forward model works.
//
// Replan() restarts the problem from a later stage and state, keeping the planned controls as the
// warm start, for receding-horizon use: the remaining stages form the same problem, shorter.

template <typename Scalar = double>
struct BoxIlqrOptions {
  // The pseudo-Huber smoothing `e`: where it starts, the factor it shrinks by per accepted
  // iteration, and its floor.
  Scalar initial_smoothing = Scalar{0.1};
  Scalar smoothing_decay = Scalar{0.7};
  Scalar min_smoothing = Scalar{1e-3};

  // Levenberg-Marquardt regularization added to Q_uu, multiplied or divided by
  // `regularization_factor` on failure or success.
  Scalar initial_regularization = Scalar{1e-6};
  Scalar min_regularization = Scalar{1e-9};
  Scalar max_regularization = Scalar{1e10};
  Scalar regularization_factor = Scalar{10};

  // Backtracking line search: step sizes 1, 1/2, 1/4, ...
  std::size_t line_search_steps = 12;

  // Once the smoothing is at its floor, an accepted step that lowers the smoothed cost by less than
  // this fraction of it counts as convergence.
  Scalar convergence_tolerance = Scalar{1e-6};

  // Central-difference step for the dynamics Jacobians.
  Scalar finite_difference_step = Scalar{1e-6};
};

template <typename Scalar = double>
struct BoxIlqrIterationReport {
  // The true L1 cost of the nominal trajectory after the iteration.
  Scalar cost = Scalar{0};
  // The smoothed cost the iteration minimized, before and after.
  Scalar smoothed_cost_before = Scalar{0};
  Scalar smoothed_cost_after = Scalar{0};
  Scalar step = Scalar{0};
  Scalar smoothing = Scalar{0};
  Scalar regularization = Scalar{0};
  bool accepted = false;
  // At the smoothing's floor, either the regularization hit its ceiling without an accepted step or
  // the accepted step improved by less than the convergence tolerance: a local minimum.
  bool converged = false;
};

// The closed-loop policy: track the nominal trajectory with the time-varying gains, inside the box.
template <int N, typename Scalar = double>
class BoxIlqrPolicy {
 public:
  using State = Eigen::Matrix<Scalar, N, 1>;
  using Control = Eigen::Matrix<Scalar, 1, 1>;
  using Gain = Eigen::Matrix<Scalar, 1, N>;

  BoxIlqrPolicy(std::vector<State> states,
                std::vector<Scalar> controls,
                std::vector<Gain> gains,
                Scalar lower,
                Scalar upper) noexcept
      : states_(std::move(states)),
        controls_(std::move(controls)),
        gains_(std::move(gains)),
        lower_(lower),
        upper_(upper) {}

  auto operator()(std::size_t stage, const State& state) const noexcept -> Control {
    const Scalar control = controls_[stage] + (gains_[stage] * (state - states_[stage])).value();
    return Control(std::clamp(control, lower_, upper_));
  }

 private:
  std::vector<State> states_;
  std::vector<Scalar> controls_;
  std::vector<Gain> gains_;
  Scalar lower_;
  Scalar upper_;
};

template <int N, std::size_t NumStages, typename ForwardModelT, typename Scalar = double>
class BoxIlqrSolver {
 public:
  using State = Eigen::Matrix<Scalar, N, 1>;
  using Control = Eigen::Matrix<Scalar, 1, 1>;
  using StateMat = Eigen::Matrix<Scalar, N, N>;
  using Gain = Eigen::Matrix<Scalar, 1, N>;
  using Policy = BoxIlqrPolicy<N, Scalar>;
  using IterationReport = BoxIlqrIterationReport<Scalar>;

  static constexpr std::size_t kNumControlStages = NumStages - 1;
  static_assert(NumStages >= 2, "a problem needs an initial and a terminal stage.");

  // `initial_controls` holds one control per control stage, and is clamped into the box.
  //
  // Fails unless the box is nonempty, the L1 weight is non-negative, the control count matches the
  // horizon, and the options are sensible.
  [[nodiscard]] static auto Make(ForwardModelT forward_model,
                                 Scalar l1_weight,
                                 const StateMat& terminal_cost_mat,
                                 const State& initial_state,
                                 Scalar lower,
                                 Scalar upper,
                                 std::span<const Scalar> initial_controls,
                                 const BoxIlqrOptions<Scalar>& options) noexcept -> Result<BoxIlqrSolver> {
    RESULT_ASSERT(lower < upper, "BoxIlqrSolver::Make: the control box is empty.");
    RESULT_ASSERT(l1_weight >= Scalar{0}, "BoxIlqrSolver::Make: the L1 weight must be non-negative.");
    RESULT_ASSERT(initial_controls.size() == kNumControlStages,
                  "BoxIlqrSolver::Make: there must be one initial control per control stage.");
    RESULT_ASSERT(options.initial_smoothing > Scalar{0} && options.min_smoothing > Scalar{0} &&
                      options.smoothing_decay > Scalar{0} && options.smoothing_decay <= Scalar{1},
                  "BoxIlqrSolver::Make: the smoothing must be positive and decay by a factor in (0, 1].");
    RESULT_ASSERT(options.regularization_factor > Scalar{1} && options.line_search_steps >= 1,
                  "BoxIlqrSolver::Make: the regularization factor must exceed one and the line search take a step.");
    BoxIlqrSolver solver(std::move(forward_model), l1_weight, terminal_cost_mat, initial_state, lower, upper, options);
    for (std::size_t stage = 0; stage < kNumControlStages; ++stage) {
      solver.controls_[stage] = std::clamp(initial_controls[stage], lower, upper);
    }
    solver.Rollout(solver.controls_, solver.states_);
    return SuccessResult(std::move(solver));
  }

  // One iteration: linearize about the nominal trajectory, solve the box-constrained LQ subproblem
  // backward, and line-search the update forward. On failure the regularization grows and the
  // nominal trajectory stays; once it is at the ceiling the solver reports convergence.
  auto Iterate() noexcept -> IterationReport {
    IterationReport report;
    report.smoothing = smoothing_;
    report.smoothed_cost_before = SmoothedCost(controls_, states_);
    report.smoothed_cost_after = report.smoothed_cost_before;
    Linearize();
    while (regularization_ <= options_.max_regularization) {
      if (!BackwardPass()) {
        regularization_ *= options_.regularization_factor;
        continue;
      }
      Scalar step = Scalar{1};
      for (std::size_t attempt = 0; attempt < options_.line_search_steps; ++attempt, step *= Scalar{0.5}) {
        ForwardPass(step);
        const Scalar trial_cost = SmoothedCost(trial_controls_, trial_states_);
        if (trial_cost < report.smoothed_cost_before) {
          std::swap(controls_, trial_controls_);
          std::swap(states_, trial_states_);
          report.accepted = true;
          report.step = step;
          report.smoothed_cost_after = trial_cost;
          break;
        }
      }
      if (report.accepted) {
        regularization_ = std::max(options_.min_regularization, regularization_ / options_.regularization_factor);
        break;
      }
      regularization_ *= options_.regularization_factor;
    }
    report.regularization = regularization_;
    const bool at_floor = smoothing_ <= options_.min_smoothing;
    if (!report.accepted) {
      regularization_ = options_.initial_regularization;
      report.converged = at_floor;
    } else {
      report.converged = at_floor && (report.smoothed_cost_before - report.smoothed_cost_after) <
                                         options_.convergence_tolerance * std::abs(report.smoothed_cost_before);
    }
    // Tighten the smoothing whether or not the step was accepted: a stall at one smoothing is no
    // reason to stop short of the true problem.
    smoothing_ = std::max(options_.min_smoothing, smoothing_ * options_.smoothing_decay);
    report.cost = Cost();
    return report;
  }

  // Moves the start of the problem to `state` at `first_stage`, keeping the planned controls from
  // there on, and rolls them out from it. The smoothing and the regularization are left as they are.
  auto Replan(std::size_t first_stage, const State& state) noexcept -> void {
    first_stage_ = std::min(first_stage, kNumControlStages - 1);
    initial_state_ = state;
    Rollout(controls_, states_);
  }

  // The closed-loop policy about the current nominal trajectory, with gains from a backward pass at
  // `smoothing` (the current one if not given). A larger smoothing than the solve's gives feedback at
  // coasting stages, where the nearly-L1 cost's curvature at zero all but switches it off.
  [[nodiscard]] auto CurrentPolicy(Scalar smoothing = Scalar{0}) noexcept -> Policy {
    const Scalar saved = smoothing_;
    if (smoothing > Scalar{0}) {
      smoothing_ = smoothing;
    }
    Linearize();
    Scalar regularization = options_.initial_regularization;
    std::swap(regularization, regularization_);
    while (!BackwardPass() && regularization_ <= options_.max_regularization) {
      regularization_ *= options_.regularization_factor;
    }
    std::swap(regularization, regularization_);
    smoothing_ = saved;
    return Policy(std::vector<State>(states_.begin(), states_.end() - 1), controls_, gains_, lower_, upper_);
  }

  // The true L1 cost of the nominal trajectory.
  [[nodiscard]] auto Cost() const noexcept -> Scalar {
    Scalar cost = TerminalCost(states_.back());
    for (std::size_t stage = first_stage_; stage < kNumControlStages; ++stage) {
      cost += l1_weight_ * std::abs(controls_[stage]);
    }
    return cost;
  }

  [[nodiscard]] auto NominalControls() const noexcept -> std::span<const Scalar> { return controls_; }
  [[nodiscard]] auto NominalStates() const noexcept -> std::span<const State> { return states_; }
  [[nodiscard]] auto Smoothing() const noexcept -> Scalar { return smoothing_; }

 private:
  using StateVector = Eigen::Matrix<Scalar, N, 1>;

  BoxIlqrSolver(ForwardModelT forward_model,
                Scalar l1_weight,
                const StateMat& terminal_cost_mat,
                const State& initial_state,
                Scalar lower,
                Scalar upper,
                const BoxIlqrOptions<Scalar>& options) noexcept
      : forward_model_(std::move(forward_model)),
        l1_weight_(l1_weight),
        terminal_cost_mat_(terminal_cost_mat),
        initial_state_(initial_state),
        lower_(lower),
        upper_(upper),
        options_(options),
        smoothing_(options.initial_smoothing),
        regularization_(options.initial_regularization),
        controls_(kNumControlStages, Scalar{0}),
        trial_controls_(kNumControlStages, Scalar{0}),
        states_(NumStages, State::Zero()),
        trial_states_(NumStages, State::Zero()),
        state_jacobians_(kNumControlStages, StateMat::Zero()),
        control_jacobians_(kNumControlStages, State::Zero()),
        feedforward_(kNumControlStages, Scalar{0}),
        gains_(kNumControlStages, Gain::Zero()) {}

  [[nodiscard]] auto Step(std::size_t stage, const State& state, Scalar control) const noexcept -> State {
    return forward_model_(stage, state, Control(control), State::Zero());
  }

  [[nodiscard]] auto TerminalCost(const State& state) const noexcept -> Scalar {
    return state.dot(terminal_cost_mat_ * state);
  }

  [[nodiscard]] auto SmoothedL1(Scalar control) const noexcept -> Scalar {
    return l1_weight_ * (std::sqrt((control * control) + (smoothing_ * smoothing_)) - smoothing_);
  }

  [[nodiscard]] auto SmoothedCost(const std::vector<Scalar>& controls, const std::vector<State>& states) const noexcept
      -> Scalar {
    Scalar cost = TerminalCost(states.back());
    for (std::size_t stage = first_stage_; stage < kNumControlStages; ++stage) {
      cost += SmoothedL1(controls[stage]);
    }
    return cost;
  }

  auto Rollout(const std::vector<Scalar>& controls, std::vector<State>& states_out) const noexcept -> void {
    states_out[first_stage_] = initial_state_;
    for (std::size_t stage = first_stage_; stage < kNumControlStages; ++stage) {
      states_out[stage + 1] = Step(stage, states_out[stage], controls[stage]);
    }
  }

  auto Linearize() noexcept -> void {
    const Scalar h = options_.finite_difference_step;
    for (std::size_t stage = first_stage_; stage < kNumControlStages; ++stage) {
      const State& state = states_[stage];
      const Scalar control = controls_[stage];
      for (int column = 0; column < N; ++column) {
        State plus = state;
        State minus = state;
        plus[column] += h;
        minus[column] -= h;
        state_jacobians_[stage].col(column) = (Step(stage, plus, control) - Step(stage, minus, control)) / (2 * h);
      }
      control_jacobians_[stage] = (Step(stage, state, control + h) - Step(stage, state, control - h)) / (2 * h);
    }
  }

  // The backward Riccati recursion of the box-constrained LQ subproblem. Fails if a regularized
  // Q_uu is not positive.
  [[nodiscard]] auto BackwardPass() noexcept -> bool {
    StateVector value_gradient = 2 * (terminal_cost_mat_ * states_.back());
    StateMat value_hessian = 2 * terminal_cost_mat_;
    for (std::size_t stage = kNumControlStages; stage-- > first_stage_;) {
      const StateMat& a = state_jacobians_[stage];
      const State& b = control_jacobians_[stage];
      const Scalar control = controls_[stage];
      const Scalar radius = std::sqrt((control * control) + (smoothing_ * smoothing_));
      const Scalar cost_gradient = l1_weight_ * control / radius;
      const Scalar cost_curvature = l1_weight_ * smoothing_ * smoothing_ / (radius * radius * radius);

      const StateVector q_x = a.transpose() * value_gradient;
      const Scalar q_u = cost_gradient + b.dot(value_gradient);
      const StateMat q_xx = a.transpose() * value_hessian * a;
      const Gain q_ux = b.transpose() * value_hessian * a;
      const Scalar q_uu = cost_curvature + b.dot(value_hessian * b);
      const Scalar q_uu_regularized = q_uu + regularization_;
      if (!(q_uu_regularized > Scalar{0}) || !std::isfinite(q_uu_regularized)) {
        return false;
      }

      Scalar feedforward = -q_u / q_uu_regularized;
      Gain gain = -q_ux / q_uu_regularized;
      const Scalar target = control + feedforward;
      if (target < lower_ || target > upper_) {
        feedforward = std::clamp(target, lower_, upper_) - control;
        gain.setZero();
      }
      feedforward_[stage] = feedforward;
      gains_[stage] = gain;

      value_gradient = q_x + (gain.transpose() * q_uu * feedforward) + (gain.transpose() * q_u) +
                       (q_ux.transpose() * feedforward);
      value_hessian = q_xx + (gain.transpose() * q_uu * gain) + (gain.transpose() * q_ux) + (q_ux.transpose() * gain);
      value_hessian = Scalar{0.5} * (value_hessian + value_hessian.transpose());
    }
    return true;
  }

  auto ForwardPass(Scalar step) noexcept -> void {
    for (std::size_t stage = 0; stage < first_stage_; ++stage) {
      trial_states_[stage] = states_[stage];
      trial_controls_[stage] = controls_[stage];
    }
    trial_states_[first_stage_] = initial_state_;
    for (std::size_t stage = first_stage_; stage < kNumControlStages; ++stage) {
      const Scalar control = controls_[stage] + (step * feedforward_[stage]) +
                             (gains_[stage] * (trial_states_[stage] - states_[stage])).value();
      trial_controls_[stage] = std::clamp(control, lower_, upper_);
      trial_states_[stage + 1] = Step(stage, trial_states_[stage], trial_controls_[stage]);
    }
  }

  ForwardModelT forward_model_;
  Scalar l1_weight_;
  StateMat terminal_cost_mat_;
  State initial_state_;
  Scalar lower_;
  Scalar upper_;
  BoxIlqrOptions<Scalar> options_;
  Scalar smoothing_;
  Scalar regularization_;
  std::size_t first_stage_ = 0;

  std::vector<Scalar> controls_;
  std::vector<Scalar> trial_controls_;
  std::vector<State> states_;
  std::vector<State> trial_states_;
  std::vector<StateMat> state_jacobians_;
  std::vector<State> control_jacobians_;
  std::vector<Scalar> feedforward_;
  std::vector<Gain> gains_;
};

}  // namespace fbsde_traj_opt

#endif  // FBSDE_TRAJ_OPT_SOLVERS_BOX_ILQR_SOLVER_HPP_
