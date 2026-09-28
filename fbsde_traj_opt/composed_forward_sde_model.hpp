// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#ifndef FBSDE_TRAJ_OPT_COMPOSED_FORWARD_SDE_MODEL_HPP_
#define FBSDE_TRAJ_OPT_COMPOSED_FORWARD_SDE_MODEL_HPP_

#include <cstddef>
#include <utility>

#include <Eigen/Core>

#include "fbsde_traj_opt/sde_term_concepts.hpp"

namespace fbsde_traj_opt {

// A forward SDE model that owns one term of each kind (see sde_term_concepts.hpp) and assembles
// them into the forward step
//
//   x_{k+1} = x_k + f(k, x_k) + B(k, x_k) * u_k + Sigma(k, x_k) * z_k,     z_k ~ N(0, I).
//
// This is the generic composition: it fixes no structure on the individual terms beyond the
// concepts they must satisfy, so a linear model, a nonlinear one, and a state-dependent-noise one
// all use this same class and differ only in the terms they are built from.
//
// `N` is the compile-time state dimension, `M` is the compile-time control dimension, and
// `Scalar` defaults to `double`. The three term types are template parameters rather than
// type-erased members so that the assembled step inlines and allocates nothing, which matters
// because sampling a trajectory batch calls it once per trajectory per stage.
template <int N,
          int M,
          typename StateDriftTermT,
          typename ControlDriftMatTermT,
          typename DiffusionTermT,
          typename Scalar = double>
  requires SdeStateDriftTerm<StateDriftTermT, Eigen::Matrix<Scalar, N, 1>> &&
           SdeControlDriftMatTerm<ControlDriftMatTermT, Eigen::Matrix<Scalar, N, 1>> &&
           SdeDiffusionTerm<DiffusionTermT, Eigen::Matrix<Scalar, N, 1>>
class ComposedForwardSdeModel {
 public:
  using State = Eigen::Matrix<Scalar, N, 1>;
  using Control = Eigen::Matrix<Scalar, M, 1>;

  using StateDriftTerm = StateDriftTermT;
  using ControlDriftMatTerm = ControlDriftMatTermT;
  using DiffusionTerm = DiffusionTermT;

  // Builds a ComposedForwardSdeModel from the three terms of the forward SDE.
  ComposedForwardSdeModel(StateDriftTerm state_drift_term,
                          ControlDriftMatTerm control_drift_mat_term,
                          DiffusionTerm diffusion_term) noexcept
      : state_drift_term_(std::move(state_drift_term)),
        control_drift_mat_term_(std::move(control_drift_mat_term)),
        diffusion_term_(std::move(diffusion_term)) {}

  // Returns the next state x_{k+1} = x_k + f(k, x_k) + B(k, x_k) * u_k + Sigma(k, x_k) * z_k.
  //
  // `noise` is the standard normal increment z_k, of the same dimension as the state, which
  // Sigma shapes into the injected noise. Drawing it is the caller's job, so that a caller
  // sampling many trajectories controls exactly which random numbers reach which trajectory.
  auto operator()(std::size_t stage, const State& state, const Control& control, const State& noise) const noexcept
      -> State {
    return state + state_drift_term_(stage, state) + (control_drift_mat_term_(stage, state) * control) +
           (diffusion_term_(stage, state) * noise);
  }

  [[nodiscard]] auto state_drift() const noexcept -> const StateDriftTerm& { return state_drift_term_; }

  [[nodiscard]] auto control_drift_mat() const noexcept -> const ControlDriftMatTerm& {
    return control_drift_mat_term_;
  }

  [[nodiscard]] auto diffusion() const noexcept -> const DiffusionTerm& { return diffusion_term_; }

 private:
  StateDriftTerm state_drift_term_;
  ControlDriftMatTerm control_drift_mat_term_;
  DiffusionTerm diffusion_term_;
};

}  // namespace fbsde_traj_opt

#endif  // FBSDE_TRAJ_OPT_COMPOSED_FORWARD_SDE_MODEL_HPP_
