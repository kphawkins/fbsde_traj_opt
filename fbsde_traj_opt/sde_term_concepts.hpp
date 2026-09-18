// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#ifndef FBSDE_TRAJ_OPT_SDE_TERM_CONCEPTS_HPP_
#define FBSDE_TRAJ_OPT_SDE_TERM_CONCEPTS_HPP_

#include <concepts>
#include <cstddef>

#include "fbsde_traj_opt/eigen_concepts.hpp"

namespace fbsde_traj_opt {

// Concepts for the functor types that make up a discrete-time, control-affine forward SDE
//
//   x_{k+1} = f(k, x_k) + B(k, x_k) * u_k + Sigma(k, x_k) * z_k,     z_k ~ N(0, I),
//
// where `f` is the uncontrolled drift (SdeStateDriftTerm), `B` is the control drift matrix
// (SdeControlDriftMatTerm), and `Sigma` shapes the injected Brownian noise (SdeDiffusionTerm).
// `State` is the fixed-size Eigen column vector type of `x_k`.
//
// Also included is ForwardSdeModel, which composes SdeStateDriftTerm, SdeControlDriftMatTerm, and
// SdeDiffusionTerm into a single functor that both computes the forward step `x_{k+1}` above and
// exposes its three component terms, so that other algorithms (e.g. linearization or covariance
// propagation) can access `f`, `B`, and `Sigma` directly rather than only the assembled step.
//
// Also included is the concept for a feedback control policy `u_k = pi(k, x_k)`
// (SdeControlPolicyTerm) that closes the loop by choosing the control applied at each stage from
// the current stage and state, and the concepts for the cost terms of the associated trajectory
// optimization problem over stages 0..K: a running cost `l(k, x_k, u_k)` (SdeRunningCostTerm)
// charged at every stage up to but not including the terminal stage, and a terminal cost
// `phi(x_K)` (SdeTerminalCostTerm) charged only at the terminal stage. The value function of the
// problem is the expected value, under the SDE dynamics above, of the sum of the running costs
// over stages 0..K-1 plus the terminal cost at stage K.

// Concept for a functor type `T` that supplies the uncontrolled drift `f(k, x_k)` of the forward
// SDE -- the component of the drift that is independent of both the control `u_k` and the noise
// `z_k`.
//
// A conforming `T` is callable as `state_drift_term(stage, state)`, where `stage` is a
// `std::size_t` and `state` is a `State`, and returns a fixed-size Eigen column vector of the
// same compile-time size as `State`.
template <typename T, typename State>
concept SdeStateDriftTerm =
    EigenFixedSizeColumnVector<State> && requires(const T& state_drift_term, std::size_t stage, const State& state) {
      { state_drift_term(stage, state) } -> EigenFixedSizeColumnVectorOfDimension<State::RowsAtCompileTime>;
    };

// Concept for a functor type `T` that supplies the control drift matrix `B(k, x_k)` of the
// forward SDE, for which `B(k, x_k) * u_k` is the component of the drift that is affected by the
// control `u_k` in a control-affine setup.
//
// A conforming `T` is callable as `control_drift_mat_term(stage, state)`, where `stage` is a
// `std::size_t` and `state` is a `State`, and returns a fixed-size Eigen matrix whose row count
// matches `State`'s compile-time size. The column count -- the control dimension -- is not
// constrained by this concept and may be any fixed (non-Dynamic) value.
template <typename T, typename State>
concept SdeControlDriftMatTerm = EigenFixedSizeColumnVector<State> &&
                                 requires(const T& control_drift_mat_term, std::size_t stage, const State& state) {
                                   {
                                     control_drift_mat_term(stage, state)
                                   } -> EigenFixedSizeMatrixWithRowsOfDimension<State::RowsAtCompileTime>;
                                 };

// Concept for a functor type `T` that supplies the diffusion coefficient (the "Sigma" matrix) of a
// discrete-time forward SDE
//
//   x_{k+1} = f(k, x_k) + Sigma(k, x_k) * z_k,     z_k ~ N(0, I),
//
// shaping the Brownian noise increment injected at each stage.
//
// A conforming `T` is callable as `diffusion_term(stage, state)`, where `stage` is a
// `std::size_t` and `state` is a `State`, and returns a fixed-size N x N Eigen matrix, where N is
// `State`'s compile-time size. The return type may be any kind of Eigen matrix expression -- a
// dense Eigen::Matrix, an Eigen::DiagonalMatrix, or any other type that exposes the usual
// compile-time Eigen shape interface -- so that an implementation whose Sigma is structurally
// diagonal (the common case) can return a lightweight Eigen::DiagonalMatrix rather than
// materializing a dense matrix.
//
// Sigma is required to be nonsingular for every stage and state a conforming type is invoked
// with; this is a semantic requirement on implementations that the concept itself cannot check.
template <typename T, typename State>
concept SdeDiffusionTerm =
    EigenFixedSizeColumnVector<State> && requires(const T& diffusion_term, std::size_t stage, const State& state) {
      { diffusion_term(stage, state) } -> EigenFixedSizeSquareMatrixOfDimension<State::RowsAtCompileTime>;
    };

// Concept for a functor type `T` that supplies a feedback control policy `pi(k, x_k)` of a
// discrete-time stochastic control problem, mapping the current stage and state to the control
// `u_k` applied at that stage.
//
// A conforming `T` is callable as `control_policy_term(stage, state)`, where `stage` is a
// `std::size_t` and `state` is a `State`, and returns a fixed-size Eigen column vector of the
// same compile-time size as `Control`.
template <typename T, typename State, typename Control>
concept SdeControlPolicyTerm = EigenFixedSizeColumnVector<State> && EigenFixedSizeColumnVector<Control> &&
                               std::same_as<typename State::Scalar, typename Control::Scalar> &&
                               requires(const T& control_policy_term, std::size_t stage, const State& state) {
                                 {
                                   control_policy_term(stage, state)
                                 } -> EigenFixedSizeColumnVectorOfDimension<Control::RowsAtCompileTime>;
                               };

// Concept for a functor type `T` that supplies the running cost `l(k, x_k, u_k)` charged at every
// stage up to, but not including, the terminal stage of a discrete-time trajectory optimization
// problem. The value function of the problem is the expected value of the sum of the running cost
// over every stage up to the terminal stage, plus the terminal cost (SdeTerminalCostTerm) at the
// terminal stage.
//
// A conforming `T` is callable as `running_cost_term(stage, state, control)`, where `stage` is a
// `std::size_t`, `state` is a `State`, and `control` is a `Control`, and returns the same `Scalar`
// type as `State` and `Control` (which must agree).
template <typename T, typename State, typename Control>
concept SdeRunningCostTerm =
    EigenFixedSizeColumnVector<State> && EigenFixedSizeColumnVector<Control> &&
    std::same_as<typename State::Scalar, typename Control::Scalar> &&
    requires(const T& running_cost_term, std::size_t stage, const State& state, const Control& control) {
      { running_cost_term(stage, state, control) } -> std::same_as<typename State::Scalar>;
    };

// Concept for a functor type `T` that supplies the terminal cost `phi(x_K)` charged at the
// terminal stage `K` of a discrete-time trajectory optimization problem (see SdeRunningCostTerm
// for how this combines with the running cost to form the value function).
//
// A conforming `T` is callable as `terminal_cost_term(state)`, where `state` is a `State`, and
// returns the same `Scalar` type as `State`.
template <typename T, typename State>
concept SdeTerminalCostTerm =
    EigenFixedSizeColumnVector<State> && requires(const T& terminal_cost_term, const State& state) {
      { terminal_cost_term(state) } -> std::same_as<typename State::Scalar>;
    };

// Concept for a functor type `T` that assembles the terms of a discrete-time, control-affine
// forward SDE
//
//   x_{k+1} = f(k, x_k) + B(k, x_k) * u_k + Sigma(k, x_k) * z_k,     z_k ~ N(0, I),
//
// into the single forward step above, while also exposing its three component terms so that other
// algorithms can access `f`, `B`, and `Sigma` directly instead of only the assembled step.
//
// A conforming `T` is callable as `model(stage, state, control, noise)`, where `stage` is a
// `std::size_t`, `state` is a `State`, `control` is a `Control`, and `noise` is a `State`
// representing the noise increment `z_k` (dimensioned to match Sigma's N x N shape), and returns
// the next state `x_{k+1}` as a fixed-size Eigen column vector of the same compile-time size as
// `State`.
//
// `T` must also define member type aliases `StateDriftTerm`, `ControlDriftMatTerm`, and
// `DiffusionTerm` that conform, respectively, to SdeStateDriftTerm, SdeControlDriftMatTerm, and
// SdeDiffusionTerm for `State`, and expose them through const accessors `StateDrift()`,
// `ControlDriftMat()`, and `Diffusion()`.
template <typename T, typename State, typename Control>
concept ForwardSdeModel =
    EigenFixedSizeColumnVector<State> && EigenFixedSizeColumnVector<Control> &&
    std::same_as<typename State::Scalar, typename Control::Scalar> &&
    requires {
      typename T::StateDriftTerm;
      typename T::ControlDriftMatTerm;
      typename T::DiffusionTerm;
    } && SdeStateDriftTerm<typename T::StateDriftTerm, State> &&
    SdeControlDriftMatTerm<typename T::ControlDriftMatTerm, State> &&
    SdeDiffusionTerm<typename T::DiffusionTerm, State> &&
    requires(const T& model, std::size_t stage, const State& state, const Control& control, const State& noise) {
      { model(stage, state, control, noise) } -> EigenFixedSizeColumnVectorOfDimension<State::RowsAtCompileTime>;
      { model.StateDrift() } -> std::convertible_to<const typename T::StateDriftTerm&>;
      { model.ControlDriftMat() } -> std::convertible_to<const typename T::ControlDriftMatTerm&>;
      { model.Diffusion() } -> std::convertible_to<const typename T::DiffusionTerm&>;
    };

}  // namespace fbsde_traj_opt

#endif  // FBSDE_TRAJ_OPT_SDE_TERM_CONCEPTS_HPP_
