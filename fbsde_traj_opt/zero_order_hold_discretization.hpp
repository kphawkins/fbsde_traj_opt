// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#ifndef FBSDE_TRAJ_OPT_ZERO_ORDER_HOLD_DISCRETIZATION_HPP_
#define FBSDE_TRAJ_OPT_ZERO_ORDER_HOLD_DISCRETIZATION_HPP_

#include <cmath>

#include <Eigen/Core>
#include <unsupported/Eigen/MatrixFunctions>

#include "fbsde_traj_opt/utils/result.hpp"

namespace fbsde_traj_opt {

// The discrete-time drift matrices of a continuous linear system, in the form the forward SDE
// terms of this library take them.
//
// `state_drift_mat` is the *increment* `A_d - I`, not the transition matrix `A_d` itself, because
// the forward step is written `x_{k+1} = x_k + A x_k + B u_k + ...`: the identity is already
// there. Handing `A_d` to ConstLinearSdeStateDriftTerm instead would double the identity and
// silently model a different system.
template <int N, int M, typename Scalar = double>
struct DiscretizedLinearDrift {
  Eigen::Matrix<Scalar, N, N> state_drift_mat;
  Eigen::Matrix<Scalar, N, M> control_drift_mat;
};

// Exactly discretizes the continuous linear system
//
//   xdot(t) = A_c x(t) + B_c u(t)
//
// under a zero-order hold: the control is held constant across each step of length `time_step`.
// The result satisfies `x((k+1)h) = x(kh) + A x(kh) + B u_k` with no approximation beyond the
// hold itself.
//
// Why not an explicit Euler step, `A = h A_c` and `B = h B_c`? For a damped or unstable system
// Euler is merely inaccurate, but for an undamped oscillatory one it is qualitatively wrong: it
// adds energy every step, so a marginally stable flexible mode grows without bound. A model whose
// plant rings louder each period is not the model anyone meant to write down, and a controller
// evaluated against it is being blamed for the integrator's behavior. The exponential costs one
// matrix function at setup and removes the question.
//
// The computation uses the standard augmented-matrix identity
//
//   expm( [[A_c, B_c], [0, 0]] * h ) = [[A_d, B_d], [0, I]],
//
// which produces both blocks at once and, unlike the closed form `B_d = A_c^-1 (A_d - I) B_c`,
// stays valid when `A_c` is singular -- which it is for every system with an integrator in it,
// meaning most of them.
//
// Fails if `time_step` is not strictly positive and finite, or if either input matrix has a
// non-finite entry.
template <int N, int M, typename Scalar = double>
[[nodiscard]] auto ZeroOrderHoldDiscretization(const Eigen::Matrix<Scalar, N, N>& continuous_state_mat,
                                               const Eigen::Matrix<Scalar, N, M>& continuous_control_mat,
                                               Scalar time_step) noexcept
    -> Result<DiscretizedLinearDrift<N, M, Scalar>> {
  RESULT_ASSERT(time_step > Scalar{0} && std::isfinite(time_step),
                "ZeroOrderHoldDiscretization: the time step must be strictly positive and finite.");
  RESULT_ASSERT(continuous_state_mat.allFinite() && continuous_control_mat.allFinite(),
                "ZeroOrderHoldDiscretization: the continuous-time matrices must be finite.");

  constexpr int kAugmentedDim = N + M;
  using AugmentedMat = Eigen::Matrix<Scalar, kAugmentedDim, kAugmentedDim>;

  AugmentedMat augmented = AugmentedMat::Zero();
  augmented.template topLeftCorner<N, N>() = continuous_state_mat * time_step;
  augmented.template topRightCorner<N, M>() = continuous_control_mat * time_step;

  const AugmentedMat exponential = augmented.exp();
  RESULT_ASSERT(exponential.allFinite(),
                "ZeroOrderHoldDiscretization: the matrix exponential did not converge to a finite result; the "
                "system matrix and time step may be far out of scale.");

  return SuccessResult(DiscretizedLinearDrift<N, M, Scalar>{
      .state_drift_mat = exponential.template topLeftCorner<N, N>() - Eigen::Matrix<Scalar, N, N>::Identity(),
      .control_drift_mat = exponential.template topRightCorner<N, M>(),
  });
}

}  // namespace fbsde_traj_opt

#endif  // FBSDE_TRAJ_OPT_ZERO_ORDER_HOLD_DISCRETIZATION_HPP_
