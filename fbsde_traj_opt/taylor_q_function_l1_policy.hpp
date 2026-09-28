// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#ifndef FBSDE_TRAJ_OPT_TAYLOR_Q_FUNCTION_L1_POLICY_HPP_
#define FBSDE_TRAJ_OPT_TAYLOR_Q_FUNCTION_L1_POLICY_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <span>
#include <utility>

#include <Eigen/Core>

#include "fbsde_traj_opt/sde_term_concepts.hpp"
#include "fbsde_traj_opt/utils/result.hpp"
#include "fbsde_traj_opt/value_function_approx_concepts.hpp"

namespace fbsde_traj_opt {

// The policy-improvement step of the DT-FBSDE method, for a minimum-fuel (L1) running cost and a
// box-constrained control: the control that minimizes the approximate Taylor Q-value function of
// Hawkins (2021), Section 4.4,
//
//   Q~_i(x, u) = c || u ||_1 + V~_{i+1}(x + F_i(x, u)) + 1/2 tr( Sigma_i' d2V~_{i+1}(x + F_i(x, u)) Sigma_i ),   (4.65)
//
//   mu_i(x) = argmin_{u in [lower, upper]} Q~_i(x, u),                                                        (4.67)
//
// where `F_i(x, u) = f(i, x) + B(i, x) u` is the drift of the forward model and `V~_{i+1}` the
// value function approximation at the next stage.
//
// Why the Q-function and not the Hamiltonian. The continuous-time alternative (4.61) minimizes
// `l(x, u) + f(x, u)' dV/dx (x)`: it reads the value function's gradient at the *current* state
// and extrapolates linearly along the drift. The Taylor Q-function instead evaluates the next
// stage's value function where the step actually lands, and the trace term charges the curvature
// the noise will sample around that point. When `V~_{i+1}` is quadratic the second-order expansion
// is exact and so is this Q-function, which is what makes it exact on LQR problems; the thesis's
// Figure 4.8 compares the two policies against the ground truth on a nonlinear problem.
//
// How the minimum is found. `Q~` is exactly the L1 term plus a smooth function of `u`, so along
// any one control coordinate `t` it is approximated near the current control by
//
//   phi(t) = c |t| + b (t - t0) + a/2 (t - t0)^2,     b = g' B_j,  a = B_j' H B_j,
//
// with `g`, `H` the gradient and Hessian of `V~_{i+1}` at the landing point and `B_j` the j-th
// column of B. The minimizer of that over [lower_j, upper_j] is one of at most five points: the
// two bounds, zero -- where the L1 term has its kink, and which is exactly why minimum-fuel
// controls coast -- and the stationary point on each side of zero when `a > 0`. Rather than trust
// the local model to choose among them, every candidate is scored by the *exact* `Q~`, so a
// value function that is far from quadratic cannot talk the policy into a candidate the model
// likes and the function does not. With one control the choice is then exact over the
// candidate set, and repeated sweeps refine the interior stationary point by Newton's method; with
// several controls the sweeps are also coordinate descent.
//
// The policy does not own the value functions. It reads them through a span over one
// representation per stage, 0..K, and stage i reads entry i + 1 -- so an iterative method can
// refit the table in place between iterations, and every copy of this policy sees the update. The
// caller keeps the table alive for as long as the policy is used.
//
// `N` is the compile-time state dimension, `M` the compile-time control dimension, and
// `ForwardModelT` a ComposedForwardSdeModel of those dimensions.
template <int N, int M, typename ForwardModelT, typename ValueFunctionT, typename Scalar = double>
  requires ValueFunctionApprox<ValueFunctionT, Eigen::Matrix<Scalar, N, 1>>
class TaylorQFunctionL1Policy {
 public:
  using State = Eigen::Matrix<Scalar, N, 1>;
  using Control = Eigen::Matrix<Scalar, M, 1>;
  using StateMat = Eigen::Matrix<Scalar, N, N>;

  // How many times the coordinates are swept. Each sweep rebuilds the local model at the control
  // the previous one chose, so with one control the sweeps are Newton iterations on the smooth
  // part of Q~ -- one is exact when V~ is quadratic, and a few converge quadratically when it is
  // not -- and with several they are also block coordinate descent.
  static constexpr std::size_t kCoordinateSweeps = 3;

  static constexpr auto kControlDim = static_cast<std::size_t>(M);

  // The chosen control and the Q-value it achieves, which is the backward pass's target for this
  // state: `min_u Q~_i(x, u)` is the noiseless Taylor estimate (4.37) of the value of the improved
  // policy.
  struct Decision {
    Control control = Control::Zero();
    Scalar q_value = Scalar{0};
  };

  // Builds the policy.
  //
  // `l1_weight` is the per-stage weight `c` of the running cost; `lower_bounds` and
  // `upper_bounds` bound each control coordinate; `value_functions` holds one representation per
  // stage 0..K, of which entries 1..K are read.
  //
  // Fails if the L1 weight is negative or not finite, if any bound is not finite, if a lower
  // bound exceeds its upper bound or zero lies outside them -- the L1 kink the minimizer relies on
  // has to be a feasible control -- or if the table has fewer than two stages.
  static auto Make(ForwardModelT forward_model,
                   Scalar l1_weight,
                   const Control& lower_bounds,
                   const Control& upper_bounds,
                   std::span<const ValueFunctionT> value_functions) noexcept -> Result<TaylorQFunctionL1Policy> {
    RESULT_ASSERT(std::isfinite(l1_weight) && l1_weight >= Scalar{0},
                  "TaylorQFunctionL1Policy::Make: the L1 weight must be finite and non-negative.");
    RESULT_ASSERT(lower_bounds.allFinite() && upper_bounds.allFinite(),
                  "TaylorQFunctionL1Policy::Make: the control bounds must be finite.");
    RESULT_ASSERT((lower_bounds.array() <= Scalar{0}).all() && (upper_bounds.array() >= Scalar{0}).all(),
                  "TaylorQFunctionL1Policy::Make: every control's bounds must contain zero.");
    RESULT_ASSERT(value_functions.size() >= 2,
                  "TaylorQFunctionL1Policy::Make: the value function table needs at least two stages.");
    return SuccessResult(
        TaylorQFunctionL1Policy(std::move(forward_model), l1_weight, lower_bounds, upper_bounds, value_functions));
  }

  // Returns `mu_i(x)`, the minimizer of the Taylor Q-function at `stage`.
  //
  // Precondition: `stage + 1` is a stage of the value function table.
  auto operator()(std::size_t stage, const State& state) const noexcept -> Control {
    return Decide(stage, state).control;
  }

  // Returns the minimizing control together with the Q-value it achieves.
  //
  // Precondition: `stage + 1` is a stage of the value function table.
  [[nodiscard]] auto Decide(std::size_t stage, const State& state) const noexcept -> Decision {
    const ValueFunctionT& next_value_function = value_functions_[stage + 1];
    const State uncontrolled_next = state + forward_model_.state_drift()(stage, state);
    const Eigen::Matrix<Scalar, N, M> control_matrix = forward_model_.control_drift_mat()(stage, state);
    const StateMat diffusion = forward_model_.diffusion()(stage, state);
    const StateMat noise_covariance = diffusion * diffusion.transpose();

    Decision decision;
    decision.q_value =
        QValue(next_value_function, uncontrolled_next, control_matrix, noise_covariance, decision.control);

    for (std::size_t sweep = 0; sweep < kCoordinateSweeps; ++sweep) {
      for (std::size_t index = 0; index < kControlDim; ++index) {
        const auto coordinate = static_cast<Eigen::Index>(index);
        const State landing = uncontrolled_next + (control_matrix * decision.control);
        const State gradient = next_value_function.Gradient(landing);
        const StateMat hessian = next_value_function.Hessian(landing);
        const State column = control_matrix.col(coordinate);
        const Scalar slope = gradient.dot(column);
        const Scalar curvature = column.dot(hessian * column);
        const Scalar current = decision.control[coordinate];

        std::array<Scalar, 5> candidates{
            lower_bounds_[coordinate], Scalar{0}, upper_bounds_[coordinate], current, current};
        if (curvature > Scalar{0}) {
          // The stationary point of phi on each side of the kink, kept only if it lands on that
          // side; otherwise that side's minimum is at the kink or a bound, already a candidate.
          for (std::size_t side = 0; side < 2; ++side) {
            const Scalar sign = side == 0 ? Scalar{1} : Scalar{-1};
            const Scalar stationary = current - ((slope + (sign * l1_weight_)) / curvature);
            if (sign * stationary > Scalar{0}) {
              candidates[3 + side] = std::clamp(stationary, lower_bounds_[coordinate], upper_bounds_[coordinate]);
            }
          }
        }

        for (const Scalar candidate : candidates) {
          Control trial = decision.control;
          trial[coordinate] = candidate;
          const Scalar q_value =
              QValue(next_value_function, uncontrolled_next, control_matrix, noise_covariance, trial);
          if (q_value < decision.q_value) {
            decision.q_value = q_value;
            decision.control = trial;
          }
        }
      }
    }
    return decision;
  }

  // Returns `Q~_i(x, u)` for an arbitrary control, for callers that want to inspect the function
  // being minimized.
  [[nodiscard]] auto QValue(std::size_t stage, const State& state, const Control& control) const noexcept -> Scalar {
    const StateMat diffusion = forward_model_.diffusion()(stage, state);
    return QValue(value_functions_[stage + 1],
                  state + forward_model_.state_drift()(stage, state),
                  forward_model_.control_drift_mat()(stage, state),
                  diffusion * diffusion.transpose(),
                  control);
  }

  [[nodiscard]] auto L1Weight() const noexcept -> Scalar { return l1_weight_; }

  [[nodiscard]] auto forward_model() const noexcept -> const ForwardModelT& { return forward_model_; }

 private:
  TaylorQFunctionL1Policy(ForwardModelT forward_model,
                          Scalar l1_weight,
                          Control lower_bounds,
                          Control upper_bounds,
                          std::span<const ValueFunctionT> value_functions) noexcept
      : forward_model_(std::move(forward_model)),
        l1_weight_(l1_weight),
        lower_bounds_(std::move(lower_bounds)),
        upper_bounds_(std::move(upper_bounds)),
        value_functions_(value_functions) {}

  // (4.65), with the trace written as the coefficient-wise product against the noise covariance,
  // which is valid because both are symmetric.
  [[nodiscard]] auto QValue(const ValueFunctionT& next_value_function,
                            const State& uncontrolled_next,
                            const Eigen::Matrix<Scalar, N, M>& control_matrix,
                            const StateMat& noise_covariance,
                            const Control& control) const noexcept -> Scalar {
    const State landing = uncontrolled_next + (control_matrix * control);
    return (l1_weight_ * control.template lpNorm<1>()) + next_value_function(landing) +
           (Scalar{0.5} * next_value_function.Hessian(landing).cwiseProduct(noise_covariance).sum());
  }

  ForwardModelT forward_model_;
  Scalar l1_weight_;
  Control lower_bounds_;
  Control upper_bounds_;
  std::span<const ValueFunctionT> value_functions_;
};

}  // namespace fbsde_traj_opt

#endif  // FBSDE_TRAJ_OPT_TAYLOR_Q_FUNCTION_L1_POLICY_HPP_
