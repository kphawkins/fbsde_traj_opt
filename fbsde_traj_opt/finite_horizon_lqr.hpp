// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#ifndef FBSDE_TRAJ_OPT_FINITE_HORIZON_LQR_HPP_
#define FBSDE_TRAJ_OPT_FINITE_HORIZON_LQR_HPP_

#include <cstddef>
#include <type_traits>

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include "fbsde_traj_opt/eigen_concepts.hpp"
#include "fbsde_traj_opt/stage_varying_linear_feedback_sde_control_policy_term.hpp"
#include "fbsde_traj_opt/utils/result.hpp"

namespace fbsde_traj_opt {

// The exact solution of the finite-horizon linear-quadratic regulator problem by backward Riccati
// recursion.
//
// The problem solved is the one posed by a forward SDE whose drift terms are constant and linear
// and a cost model whose terms are constant quadratic regulator forms:
//
//   x_{k+1} = x_k + A x_k + B u_k + Sigma(k, x_k) z_k,
//   minimize  sum_{k=0}^{K-1} [ x_k' Q x_k + u_k' R u_k + 2 x_k' N u_k ] + x_K' F x_K.
//
// Two things about this setup are worth stating outright.
//
// First, the forward step adds its drift to the state rather than replacing it, so the transition
// matrix of the equivalent `x_{k+1} = A_eff x_k + B u_k` problem is `A_eff = I + A`, not `A`.
// Reading `A` as the transition matrix is the easiest mistake to make here and produces a policy
// that is wrong but not obviously so.
//
// Second, the diffusion term plays no part in the recursion and is never read. Under additive
// noise -- noise whose covariance depends on neither the state nor the control -- the optimal
// policy of the stochastic problem is exactly the optimal policy of the deterministic one, and
// the noise only shifts the value function by a constant per stage. That is the certainty
// equivalence principle, and it is why an LQR gain computed with no reference to Sigma is
// nonetheless optimal for the sampled trajectory distribution that Sigma generates. It also means
// that a state-dependent Sigma breaks the guarantee: the gain returned here remains a reasonable
// policy but is no longer the optimal one.

// Concept for a ForwardSdeModel whose drift terms expose their constant matrices -- that is, one
// built from ConstLinearSdeStateDriftTerm and ConstLinearSdeControlDriftMatTerm, or from any
// other terms that offer the same `DriftMat()` accessors.
template <typename T>
concept ConstLinearDriftForwardSdeModel = requires(const T& forward_model) {
  { forward_model.StateDrift().DriftMat() } -> DecaysToEigenExpressionWithCompileTimeShape;
  { forward_model.ControlDriftMat().DriftMat() } -> DecaysToEigenExpressionWithCompileTimeShape;
};

// Concept for a CostSdeModel whose terms expose their constant quadratic forms -- that is, one
// built from QuadraticRegulatorSdeRunningCostTerm and QuadraticRegulatorSdeTerminalCostTerm.
template <typename T>
concept QuadraticRegulatorCostSdeModel = requires(const T& cost_model) {
  { cost_model.RunningCost().StateCostMat() } -> DecaysToEigenExpressionWithCompileTimeShape;
  { cost_model.RunningCost().ControlCostMat() } -> DecaysToEigenExpressionWithCompileTimeShape;
  { cost_model.RunningCost().CrossCostMat() } -> DecaysToEigenExpressionWithCompileTimeShape;
  { cost_model.TerminalCost().TerminalCostMat() } -> DecaysToEigenExpressionWithCompileTimeShape;
};

// The type of `A`, the constant state drift matrix of `ForwardModelT`. Exists so the state
// dimension and scalar type can be recovered from the model rather than restated by the caller.
template <ConstLinearDriftForwardSdeModel ForwardModelT>
using StateDriftMatOf = std::remove_cvref_t<decltype(std::declval<const ForwardModelT&>().StateDrift().DriftMat())>;

// The type of `B`, the constant control drift matrix of `ForwardModelT`.
template <ConstLinearDriftForwardSdeModel ForwardModelT>
using ControlDriftMatOf =
    std::remove_cvref_t<decltype(std::declval<const ForwardModelT&>().ControlDriftMat().DriftMat())>;

// The policy type SolveFiniteHorizonLqr() returns for `ForwardModelT` over `NumControlStages`
// control stages: a stage-varying linear feedback whose dimensions are those of the model.
template <ConstLinearDriftForwardSdeModel ForwardModelT, std::size_t NumControlStages>
using FiniteHorizonLqrPolicy =
    StageVaryingLinearFeedbackSdeControlPolicyTerm<StateDriftMatOf<ForwardModelT>::RowsAtCompileTime,
                                                   ControlDriftMatOf<ForwardModelT>::ColsAtCompileTime,
                                                   NumControlStages,
                                                   typename StateDriftMatOf<ForwardModelT>::Scalar>;

// Solves the finite-horizon LQR problem posed by `forward_model` and `cost_model` over
// `NumStages` stages, returning the optimal stage-varying feedback policy.
//
// The returned policy has `NumStages - 1` gains, one per stage at which a control is applied, and
// is directly consumable by TrajectoryBatch<..., NumStages, ...>::Make().
//
// The recursion is the standard one. With `A_eff = I + A` and starting from `P_K = F`, each step
// back from stage k+1 to stage k forms
//
//   H_k = R + B' P_{k+1} B,          G_k = B' P_{k+1} A_eff + N',
//   K_k = -H_k^{-1} G_k,             P_k = Q + A_eff' P_{k+1} A_eff - G_k' H_k^{-1} G_k,
//
// where `H_k` is the curvature of the stage cost in the control and `G_k` its mixed curvature in
// state and control, so that `K_k` is a Newton step: the control that exactly minimizes the
// stage-k cost-to-go given `P_{k+1}`.
//
// `Q`, `R`, and `F` are symmetrized on the way in. A quadratic form `x' Q x` depends only on the
// symmetric part of `Q`, so a caller who supplies a slightly asymmetric matrix means the
// symmetric one; the recursion, unlike the quadratic form, would not be indifferent to the
// difference. `P` is likewise resymmetrized at each step, since it is symmetric in exact
// arithmetic and only drifts out of symmetry through rounding.
//
// Fails if any `H_k` is not positive definite. That is not merely a numerical complaint: `H_k`
// positive definite is exactly the condition under which the stage-k problem has a unique finite
// minimizer, and it fails when `R` is singular or indefinite in a direction the dynamics do not
// penalize -- a problem statement for which no optimal gain exists.
//
// Allocates nothing: every matrix is fixed-size and the recursion is performed in place.
template <std::size_t NumStages,
          ConstLinearDriftForwardSdeModel ForwardModelT,
          QuadraticRegulatorCostSdeModel CostModelT>
auto SolveFiniteHorizonLqr(const ForwardModelT& forward_model, const CostModelT& cost_model) noexcept
    -> Result<FiniteHorizonLqrPolicy<ForwardModelT, NumStages - 1>> {
  static_assert(NumStages >= 2, "An LQR horizon needs at least an initial and a terminal stage.");

  using StateDriftMat = StateDriftMatOf<ForwardModelT>;
  using ControlDriftMat = ControlDriftMatOf<ForwardModelT>;
  using Scalar = typename StateDriftMat::Scalar;

  static constexpr int kStateDim = StateDriftMat::RowsAtCompileTime;
  static constexpr int kControlDim = ControlDriftMat::ColsAtCompileTime;

  static_assert(StateDriftMat::ColsAtCompileTime == kStateDim, "The state drift matrix A must be square.");
  static_assert(ControlDriftMat::RowsAtCompileTime == kStateDim,
                "The control drift matrix B must have one row per state dimension.");

  using StateMat = Eigen::Matrix<Scalar, kStateDim, kStateDim>;
  using ControlMat = Eigen::Matrix<Scalar, kControlDim, kControlDim>;
  using CrossMat = Eigen::Matrix<Scalar, kStateDim, kControlDim>;
  using Gain = Eigen::Matrix<Scalar, kControlDim, kStateDim>;
  using Policy = FiniteHorizonLqrPolicy<ForwardModelT, NumStages - 1>;

  // Symmetrizes `matrix`; see the note above on why the recursion needs this and the quadratic
  // form does not.
  const auto symmetrized = [](const auto& matrix) noexcept {
    return static_cast<Scalar>(0.5) * (matrix + matrix.transpose());
  };

  const StateMat transition_mat = StateMat::Identity() + forward_model.StateDrift().DriftMat();
  const CrossMat control_mat = forward_model.ControlDriftMat().DriftMat();

  const StateMat state_cost_mat = symmetrized(cost_model.RunningCost().StateCostMat());
  const ControlMat control_cost_mat = symmetrized(cost_model.RunningCost().ControlCostMat());
  const CrossMat cross_cost_mat = cost_model.RunningCost().CrossCostMat();

  // P, the Hessian of the cost-to-go, seeded at the terminal stage by the terminal cost.
  StateMat cost_to_go_hessian = symmetrized(cost_model.TerminalCost().TerminalCostMat());

  typename Policy::GainArray gains{};
  for (std::size_t stage = Policy::kNumControlStages; stage-- > 0;) {
    const CrossMat hessian_times_control_mat = cost_to_go_hessian * control_mat;

    // H_k, the curvature of the stage cost in the control.
    const ControlMat control_curvature = control_cost_mat + (control_mat.transpose() * hessian_times_control_mat);
    const Eigen::LLT<ControlMat> control_curvature_factorization(control_curvature);
    RESULT_ASSERT(control_curvature_factorization.info() == Eigen::Success,
                  "SolveFiniteHorizonLqr: R + B' P B is not positive definite at some stage, so the problem has no "
                  "unique finite minimizing control there.");

    // G_k, the mixed curvature in state and control.
    const Gain mixed_curvature = (hessian_times_control_mat.transpose() * transition_mat) + cross_cost_mat.transpose();

    const Gain curvature_step = control_curvature_factorization.solve(mixed_curvature);
    gains[stage] = -curvature_step;

    // Evaluated into a named matrix before symmetrizing: `symmetrized` reads its argument twice,
    // once directly and once transposed, and this expression is not something to compute twice.
    const StateMat next_cost_to_go_hessian = state_cost_mat +
                                             (transition_mat.transpose() * cost_to_go_hessian * transition_mat) -
                                             (mixed_curvature.transpose() * curvature_step);
    cost_to_go_hessian = symmetrized(next_cost_to_go_hessian);
  }

  return SuccessResult(Policy(gains));
}

}  // namespace fbsde_traj_opt

#endif  // FBSDE_TRAJ_OPT_FINITE_HORIZON_LQR_HPP_
