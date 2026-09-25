// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#ifndef FBSDE_TRAJ_OPT_TAYLOR_NOISELESS_BACKWARD_STEP_ESTIMATOR_HPP_
#define FBSDE_TRAJ_OPT_TAYLOR_NOISELESS_BACKWARD_STEP_ESTIMATOR_HPP_

#include <cstddef>
#include <utility>

#include <Eigen/Core>
#include <Eigen/LU>

#include "fbsde_traj_opt/sde_term_concepts.hpp"
#include "fbsde_traj_opt/utils/result.hpp"
#include "fbsde_traj_opt/value_function_approx_concepts.hpp"

namespace fbsde_traj_opt {

// The off-policy drifted Taylor *noiseless* estimator of one backward step of a discrete-time
// FBSDE: given a state `X_i`, the drift `K_i` the forward pass actually used to reach the next
// stage, and an approximation `V~_{i+1}` of the next stage's value function, it produces an
// estimate `Y^_i` of `V^mu_i(X_i)` for the target policy `mu`.
//
// Following Hawkins (2021), Chapter 3, Table 3.1, the estimator is
//
//   Y^_i = L^mu_i + Ybar_{i+1} + Zbar_{i+1}' D_i + 1/2 tr( Mbar_{i+1} (I + D_i D_i') ),
//
// with, writing `Sigma_i` for the diffusion and `F^mu_i` for the drift the target policy would
// have produced,
//
//   xbar     = X_i + K_i,                       the conditional mean of X_{i+1} under sampling
//   Ybar     = V~_{i+1}(xbar),
//   Zbar     = Sigma_i' dV~_{i+1}/dx (xbar),
//   Mbar     = Sigma_i' d2V~_{i+1}/dx2 (xbar) Sigma_i,
//   D_i      = Sigma_i^-1 (F^mu_i - K_i),       the Girsanov drift between the two measures
//   F^mu_i   = f(i, X_i) + B(i, X_i) mu(i, X_i),
//   L^mu_i   = l(i, X_i, mu(i, X_i)).
//
// What is computed here is that expression with `Sigma_i D_i` substituted back in for
// `F^mu_i - K_i`, which removes the inverse entirely. Writing `dF_i := F^mu_i - K_i`, `g` for the
// gradient and `H` for the Hessian of `V~_{i+1}` at `xbar`:
//
//   Zbar' D = g' Sigma Sigma^-1 dF      = g' dF,
//   tr(Mbar D D')                       = dF' H dF,
//   tr(Mbar)                            = tr(Sigma' H Sigma),
//
//   Y^_i = L^mu_i + V~_{i+1}(xbar) + g' dF + 1/2 dF' H dF + 1/2 tr(Sigma' H Sigma).
//
// Besides costing one fewer linear solve per sample, that form says plainly what the estimator
// is: the first four terms are the second-order Taylor model of `V~_{i+1}` built at the sampled
// mean `X_i + K_i` and evaluated at the on-policy mean `X_i + F^mu_i`, and the last is the
// Gaussian correction for the noise around it. In other words, the Bellman equation with the
// conditional expectation taken through a quadratic model of the next stage's value function.
// Setting `K_i = F^mu_i` gives `dF_i = 0` and recovers the on-policy noiseless estimator.
//
// Two properties are worth keeping in view. The estimate depends on `X_i` and `K_i` alone and not
// on the noise that was drawn -- hence *noiseless*: it has zero variance, so the targets it
// produces are fixed labels and fitting to them is ordinary nonlinear least squares rather than a
// stochastic-target problem. And when the true value function is quadratic, as it is throughout
// an LQR problem, the second-order model is exact and so is the estimator, for any choice of
// `K_i` whatsoever.
//
// Choosing `K_i` is the caller's business and deliberately outside this class: `states` and
// `drifts` arrive as two independent batches, so any sampling scheme drops in. What the theory
// asks of the choice is that `|| D_i ||` stay small -- the bias of the estimator is bounded by
// `exp(||D_i||^2 / 2)` times the mean squared Taylor residual, so the difference `F^mu_i - K_i`
// should be on the order of the diffusion rather than of the drift. GirsanovDriftNorms() reports
// that quantity so a caller can watch it.
//
// `N` is the compile-time state dimension, `M` the compile-time control dimension, and
// `ForwardModelT` a ComposedForwardSdeModel of those dimensions. The three model types are
// template parameters held by value rather than type-erased members, for the same reason
// ComposedForwardSdeModel holds its terms that way: the estimator is called once per sample per
// stage, and nothing here should survive inlining.
template <int N, int M, typename ForwardModelT, typename PolicyT, typename RunningCostT, typename Scalar = double>
  requires SdeControlPolicyTerm<PolicyT, Eigen::Matrix<Scalar, N, 1>, Eigen::Matrix<Scalar, M, 1>> &&
           SdeRunningCostTerm<RunningCostT, Eigen::Matrix<Scalar, N, 1>, Eigen::Matrix<Scalar, M, 1>>
class TaylorNoiselessBackwardStepEstimator {
 public:
  using State = Eigen::Matrix<Scalar, N, 1>;
  using Control = Eigen::Matrix<Scalar, M, 1>;
  using StateMat = Eigen::Matrix<Scalar, N, N>;

  template <int BatchSize>
  using StateBatch = ValueFunctionStateBatch<State, BatchSize>;
  template <int BatchSize>
  using ValueBatch = ValueFunctionValueBatch<State, BatchSize>;

  // Builds the estimator from the forward SDE, the target policy `mu`, and the running cost.
  //
  // There is no Make() factory returning a Result: the composition has no invariant of its own,
  // and each of the three models enforces whatever invariant it has in its own factory. The one
  // requirement this class adds -- that the diffusion be nonsingular -- is already a stated
  // precondition of SdeDiffusionTerm, and the computation below never inverts it anyway.
  TaylorNoiselessBackwardStepEstimator(ForwardModelT forward_model, PolicyT policy, RunningCostT running_cost) noexcept
      : forward_model_(std::move(forward_model)), policy_(std::move(policy)), running_cost_(std::move(running_cost)) {}

  // Returns the drift `F^mu_i = f(i, x) + B(i, x) mu(i, x)` the target policy would produce.
  //
  // Exposed because a caller choosing `K_i` usually wants to choose it relative to this: the
  // on-policy case is `K_i = PolicyDrift(...)` exactly, and the recommended off-policy case is
  // that drift perturbed by something on the order of the diffusion.
  [[nodiscard]] auto PolicyDrift(std::size_t stage, const State& state) const noexcept -> State {
    const Control control = policy_(stage, state);
    return forward_model_.state_drift()(stage, state) + (forward_model_.control_drift_mat()(stage, state) * control);
  }

  // Returns the estimate of `V^mu_stage(state)` for a single sample reached with drift `drift`.
  template <typename NextValueFunctionT>
    requires ValueFunctionApprox<NextValueFunctionT, State>
  auto operator()(std::size_t stage,
                  const State& state,
                  const State& drift,
                  const NextValueFunctionT& next_value_function) const noexcept -> Scalar {
    const State mean_next_state = state + drift;
    return CombineTarget(RunningCost(stage, state),
                         next_value_function(mean_next_state),
                         next_value_function.Gradient(mean_next_state),
                         next_value_function.Hessian(mean_next_state),
                         PolicyDrift(stage, state) - drift,
                         Diffusion(stage, state));
  }

  // Writes the estimate for every sample of a batch into `targets_out`: one column of `states`
  // and the matching column of `drifts` per sample.
  //
  // The next-stage value function is evaluated once for the whole batch, which is the reason this
  // is not simply a loop over the single-sample form: that one call is a handful of matrix
  // products over the batch where the loop would be a few hundred matrix-vector products.
  template <int BatchSize, typename NextValueFunctionT>
    requires BatchedValueFunctionApprox<NextValueFunctionT, State, BatchSize>
  auto EstimateTargets(std::size_t stage,
                       const StateBatch<BatchSize>& states,
                       const StateBatch<BatchSize>& drifts,
                       const NextValueFunctionT& next_value_function,
                       ValueBatch<BatchSize>& targets_out) const noexcept -> void {
    const StateBatch<BatchSize> mean_next_states = states + drifts;

    ValueBatch<BatchSize> next_values;
    ValueFunctionGradientBatch<State, BatchSize> next_gradients;
    ValueFunctionHessianBatch<State, BatchSize> next_hessians;
    next_value_function.template ValuesGradientsAndHessians<BatchSize>(
        mean_next_states, next_values, next_gradients, next_hessians);

    constexpr auto kSampleCount = static_cast<std::size_t>(BatchSize);
    for (std::size_t sample = 0; sample < kSampleCount; ++sample) {
      const auto column = static_cast<Eigen::Index>(sample);
      const State state = states.col(column);
      targets_out[column] = CombineTarget(RunningCost(stage, state),
                                          next_values[column],
                                          next_gradients.col(column),
                                          next_hessians.template block<N, N>(0, column * N),
                                          PolicyDrift(stage, state) - drifts.col(column),
                                          Diffusion(stage, state));
    }
  }

  // Writes `|| D_i ||`, the norm of the Girsanov drift between the sampling measure and the
  // policy's, for every sample of a batch.
  //
  // This is the diagnostic the bias bound of the chapter's Proposition is stated in: the bias of
  // the estimator is at most `exp(||D_i||^2 / 2)` times the root mean squared Taylor residual, so
  // a sampling scheme that keeps `||D_i||` at or below one is paying at most a factor of about
  // 1.65 for being off-policy, and one that lets it grow is paying exponentially.
  //
  // Unlike the estimate itself this does invert the diffusion, and so it can fail: it returns an
  // error if the diffusion at any sample is numerically singular, which SdeDiffusionTerm forbids
  // but cannot check.
  template <int BatchSize>
  auto GirsanovDriftNorms(std::size_t stage,
                          const StateBatch<BatchSize>& states,
                          const StateBatch<BatchSize>& drifts,
                          ValueBatch<BatchSize>& norms_out) const noexcept -> Result<> {
    constexpr auto kSampleCount = static_cast<std::size_t>(BatchSize);
    for (std::size_t sample = 0; sample < kSampleCount; ++sample) {
      const auto column = static_cast<Eigen::Index>(sample);
      const State state = states.col(column);

      const Eigen::FullPivLU<StateMat> factorization(Diffusion(stage, state));
      RESULT_ASSERT(factorization.isInvertible(),
                    "TaylorNoiselessBackwardStepEstimator::GirsanovDriftNorms: the diffusion is numerically singular "
                    "at some sample, so the Girsanov drift it defines does not exist.");

      const State drift_difference = PolicyDrift(stage, state) - drifts.col(column);
      norms_out[column] = factorization.solve(drift_difference).norm();
    }
    return SuccessResult();
  }

  [[nodiscard]] auto forward_model() const noexcept -> const ForwardModelT& { return forward_model_; }

  [[nodiscard]] auto policy() const noexcept -> const PolicyT& { return policy_; }

  [[nodiscard]] auto running_cost() const noexcept -> const RunningCostT& { return running_cost_; }

 private:
  // The estimator's algebra, kept in one place so that the single-sample and batched entry points
  // cannot drift apart: everything else in this class is about gathering these six quantities.
  template <typename GradientT, typename HessianT, typename DriftT>
  [[nodiscard]] static auto CombineTarget(Scalar running_cost,
                                          Scalar next_value,
                                          const GradientT& next_gradient,
                                          const HessianT& next_hessian,
                                          const DriftT& drift_difference,
                                          const StateMat& diffusion) noexcept -> Scalar {
    const State difference = drift_difference;
    const StateMat hessian = next_hessian;

    // tr(Sigma' H Sigma) == tr(H Sigma Sigma'), and with Sigma Sigma' symmetric that trace is the
    // coefficient-wise product summed -- no N x N product beyond the one forming Sigma Sigma'.
    const StateMat noise_covariance = diffusion * diffusion.transpose();
    const Scalar trace_correction = hessian.cwiseProduct(noise_covariance).sum();

    return running_cost + next_value + next_gradient.dot(difference) +
           (Scalar{0.5} * difference.dot(hessian * difference)) + (Scalar{0.5} * trace_correction);
  }

  [[nodiscard]] auto RunningCost(std::size_t stage, const State& state) const noexcept -> Scalar {
    return running_cost_(stage, state, policy_(stage, state));
  }

  // The diffusion, materialized dense. SdeDiffusionTerm permits any N x N expression, including
  // the Eigen::DiagonalMatrix a structurally diagonal term returns, and the algebra below needs
  // something it can transpose and multiply coefficient-wise.
  [[nodiscard]] auto Diffusion(std::size_t stage, const State& state) const noexcept -> StateMat {
    return forward_model_.diffusion()(stage, state);
  }

  ForwardModelT forward_model_;
  PolicyT policy_;
  RunningCostT running_cost_;
};

}  // namespace fbsde_traj_opt

#endif  // FBSDE_TRAJ_OPT_TAYLOR_NOISELESS_BACKWARD_STEP_ESTIMATOR_HPP_
