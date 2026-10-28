// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#ifndef FBSDE_TRAJ_OPT_EXAMPLES_L1_BENCHMARK_MODELS_HPP_
#define FBSDE_TRAJ_OPT_EXAMPLES_L1_BENCHMARK_MODELS_HPP_

#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <string>
#include <utility>

#include <Eigen/Core>

#include "fbsde_traj_opt/costs/l1_control_sde_running_cost_term.hpp"
#include "fbsde_traj_opt/costs/quadratic_regulator_sde_terminal_cost_term.hpp"
#include "fbsde_traj_opt/dynamics/const_diagonal_sde_diffusion_term.hpp"
#include "fbsde_traj_opt/dynamics/const_linear_sde_control_drift_mat_term.hpp"
#include "fbsde_traj_opt/dynamics/const_linear_sde_state_drift_term.hpp"
#include "fbsde_traj_opt/dynamics/zero_order_hold_discretization.hpp"
#include "fbsde_traj_opt/sampling/diagonal_covariance_normal_distribution.hpp"
#include "fbsde_traj_opt/sde/composed_forward_sde_model.hpp"
#include "fbsde_traj_opt/utils/result.hpp"
#include "examples/double_inverted_pendulum_model.hpp"

namespace fbsde_traj_opt::examples {

// The three minimum-fuel benchmarks of Hawkins (2021), Section 4.5, on which FBRRT was compared
// with parallel-sampled FBSDE: the L1 double integrator, the L1 inverted pendulum, and the L1
// double inverted pendulum. Each minimizes
//
//   E[ integral of c0 |u| dt + sum_j c_j X_T[j]^2 ],     |u| <= 1,
//
// under a constant diagonal diffusion given in continuous time; the discrete step scales it by
// sqrt(dt), and the running cost is c0 dt |u| per step.
//
// Where the thesis gives a value it is used. It gives neither horizons nor cost weights for the
// first two problems, nor the pendulum's constants, so those are chosen here and documented; the
// double inverted pendulum is the one this repository's DT-FBSDE example already solves, with that
// example's constants, so the comparison starts from a configuration the baseline is known to
// handle.

// One benchmark: everything a solver needs, plus the boxes an experiment normalizes over.
template <int N, typename StateDriftTermT, typename ControlDriftMatTermT>
struct L1Benchmark {
  using State = Eigen::Matrix<double, N, 1>;
  using ForwardModel =
      ComposedForwardSdeModel<N, 1, StateDriftTermT, ControlDriftMatTermT, ConstDiagonalSdeDiffusionTerm<N>>;

  std::string name;
  std::string description;
  double time_step = 0.0;
  ForwardModel forward_model;
  L1ControlSdeRunningCostTerm<N, 1> running_cost;
  QuadraticRegulatorSdeTerminalCostTerm<N> terminal_cost;
  Eigen::Matrix<double, N, N> terminal_cost_mat;
  DiagonalCovarianceNormalDistribution<N> initial_distribution;

  // The box the value functions' polynomial basis is normalized to, as a center and half-widths.
  State normalization_center;
  State normalization_scale;

  // The region FBRRT draws its RRT targets from.
  State region_lower;
  State region_upper;

  std::array<std::string, static_cast<std::size_t>(N)> state_names;
};

// ---------------------------------------------------------------------------------------------
// 1. L1 double integrator
// ---------------------------------------------------------------------------------------------

inline constexpr std::size_t kL1DoubleIntegratorNumStages = 65;  // the thesis's 64 steps

using L1DoubleIntegrator = L1Benchmark<2, ConstLinearSdeStateDriftTerm<2>, ConstLinearSdeControlDriftMatTerm<2, 1>>;

// A unit mass on a frictionless line, pushed by a bounded force, to be brought to rest at the
// origin, with the thesis's diffusion diag(0.01, 0.1). Starting with positive position and
// velocity, the minimum-fuel policy brakes to a negative velocity, coasts, and brakes again --
// bang-off-bang. Horizon 4 s; fuel weight 1 per second; terminal weights (20, 10).
[[nodiscard]] inline auto MakeL1DoubleIntegrator() noexcept -> Result<L1DoubleIntegrator> {
  constexpr double kHorizon = 4.0;
  const double time_step = kHorizon / static_cast<double>(kL1DoubleIntegratorNumStages - 1);
  const Eigen::Matrix2d continuous_state_mat = (Eigen::Matrix2d() << 0.0, 1.0, 0.0, 0.0).finished();
  RESULT_ASSIGN_OR_RETURN(
      const auto drift,
      (ZeroOrderHoldDiscretization<2, 1>(continuous_state_mat, Eigen::Vector2d(0.0, 1.0), time_step)));
  RESULT_ASSIGN_OR_RETURN(auto diffusion,
                          ConstDiagonalSdeDiffusionTerm<2>::Make(Eigen::Vector2d(0.01, 0.1) * std::sqrt(time_step)));
  RESULT_ASSIGN_OR_RETURN(auto running_cost, (L1ControlSdeRunningCostTerm<2, 1>::Make(time_step)));
  RESULT_ASSIGN_OR_RETURN(
      auto initial,
      DiagonalCovarianceNormalDistribution<2>::Make(Eigen::Vector2d(2.0, 1.0), Eigen::Vector2d(1e-3, 1e-3)));
  const Eigen::Matrix2d terminal_cost_mat = Eigen::Vector2d(20.0, 10.0).asDiagonal();
  return SuccessResult(L1DoubleIntegrator{
      .name = "L1 double integrator",
      .description = "A unit mass driven by a force |u| <= 1 from (2 m, 1 m/s) to rest at the origin in 4 s, paying "
                     "for fuel: the minimum-fuel answer brakes, coasts, and brakes again.",
      .time_step = time_step,
      .forward_model = {ConstLinearSdeStateDriftTerm<2>(drift.state_drift_mat),
                        ConstLinearSdeControlDriftMatTerm<2, 1>(drift.control_drift_mat),
                        std::move(diffusion)},
      .running_cost = running_cost,
      .terminal_cost = QuadraticRegulatorSdeTerminalCostTerm<2>(terminal_cost_mat),
      .terminal_cost_mat = terminal_cost_mat,
      .initial_distribution = initial,
      .normalization_center = Eigen::Vector2d(1.0, -0.25),
      .normalization_scale = Eigen::Vector2d(2.5, 2.25),
      .region_lower = Eigen::Vector2d(-1.5, -2.5),
      .region_upper = Eigen::Vector2d(3.5, 2.0),
      .state_names = {"Position (m)", "Velocity (m/s)"},
  });
}

// ---------------------------------------------------------------------------------------------
// 2. L1 inverted pendulum
// ---------------------------------------------------------------------------------------------

inline constexpr std::size_t kL1PendulumNumStages = 65;  // the thesis's 64 steps

// The thesis's pendulum, `theta_ddot = a1 theta_dot + a2 sin(theta) + a3 u`, with `theta` measured
// from upright so that the goal is the origin and gravity destabilizes it. The thesis gives the
// form but not the constants; these make a swing-up that must pump: a2 = g / l for a 1 m pendulum,
// light damping, and a motor whose full torque, a3 = 4, is well under gravity's 9.81 -- it cannot
// lift the pendulum directly.
struct L1PendulumParameters {
  double damping = -0.1;  // a1, 1/s
  double gravity = 9.81;  // a2, 1/s^2
  double motor = 4.0;     // a3, 1/s^2 at |u| = 1
};

// The held-control flow map of the pendulum over one step, by RK4 on `substeps` substeps.
[[nodiscard]] inline auto IntegratePendulumStep(const L1PendulumParameters& parameters,
                                                const Eigen::Vector2d& state,
                                                double control,
                                                double time_step,
                                                std::size_t substeps) noexcept -> Eigen::Vector2d {
  const auto derivative = [&](const Eigen::Vector2d& x) noexcept -> Eigen::Vector2d {
    return {x[1], (parameters.damping * x[1]) + (parameters.gravity * std::sin(x[0])) + (parameters.motor * control)};
  };
  const double step = time_step / static_cast<double>(substeps);
  Eigen::Vector2d current = state;
  for (std::size_t substep = 0; substep < substeps; ++substep) {
    const Eigen::Vector2d k1 = derivative(current);
    const Eigen::Vector2d k2 = derivative(current + (0.5 * step * k1));
    const Eigen::Vector2d k3 = derivative(current + (0.5 * step * k2));
    const Eigen::Vector2d k4 = derivative(current + (step * k3));
    current += (step / 6.0) * (k1 + (2.0 * k2) + (2.0 * k3) + k4);
  }
  return current;
}

// The control-affine step `x + f(x) + B(x) u` built from the flow map exactly as the double
// inverted pendulum's is -- `f(x) = Phi(x, 0) - x`, `B(x) = (Phi(x, 1) - Phi(x, -1)) / 2` -- which
// is exact at u = 0 and at the bounds up to the map's curvature in u. The pendulum is
// control-affine in continuous time, so over a 62.5 ms step that curvature is tiny.
class L1PendulumStateDriftTerm {
 public:
  static constexpr std::size_t kSubsteps = 2;

  L1PendulumStateDriftTerm(L1PendulumParameters parameters, double time_step) noexcept
      : parameters_(parameters), time_step_(time_step) {}

  auto operator()([[maybe_unused]] std::size_t stage, const Eigen::Vector2d& state) const noexcept -> Eigen::Vector2d {
    return IntegratePendulumStep(parameters_, state, 0.0, time_step_, kSubsteps) - state;
  }

 private:
  L1PendulumParameters parameters_;
  double time_step_;
};

class L1PendulumControlDriftMatTerm {
 public:
  L1PendulumControlDriftMatTerm(L1PendulumParameters parameters, double time_step) noexcept
      : parameters_(parameters), time_step_(time_step) {}

  auto operator()([[maybe_unused]] std::size_t stage, const Eigen::Vector2d& state) const noexcept -> Eigen::Vector2d {
    constexpr std::size_t kSubsteps = L1PendulumStateDriftTerm::kSubsteps;
    return 0.5 * (IntegratePendulumStep(parameters_, state, 1.0, time_step_, kSubsteps) -
                  IntegratePendulumStep(parameters_, state, -1.0, time_step_, kSubsteps));
  }

 private:
  L1PendulumParameters parameters_;
  double time_step_;
};

using L1Pendulum = L1Benchmark<2, L1PendulumStateDriftTerm, L1PendulumControlDriftMatTerm>;

// Swing the pendulum up from hanging at rest and balance it, in 4 s, under the thesis's diffusion
// diag(0.04, 0.4). Fuel weight 1 per second; terminal weights (10, 1) on angle and rate.
[[nodiscard]] inline auto MakeL1Pendulum() noexcept -> Result<L1Pendulum> {
  constexpr double kHorizon = 4.0;
  const double time_step = kHorizon / static_cast<double>(kL1PendulumNumStages - 1);
  const L1PendulumParameters parameters;
  RESULT_ASSIGN_OR_RETURN(auto diffusion,
                          ConstDiagonalSdeDiffusionTerm<2>::Make(Eigen::Vector2d(0.04, 0.4) * std::sqrt(time_step)));
  RESULT_ASSIGN_OR_RETURN(auto running_cost, (L1ControlSdeRunningCostTerm<2, 1>::Make(time_step)));
  RESULT_ASSIGN_OR_RETURN(auto initial,
                          DiagonalCovarianceNormalDistribution<2>::Make(Eigen::Vector2d(std::numbers::pi, 0.0),
                                                                        Eigen::Vector2d(1e-4, 1e-4)));
  const Eigen::Matrix2d terminal_cost_mat = Eigen::Vector2d(10.0, 1.0).asDiagonal();
  return SuccessResult(L1Pendulum{
      .name = "L1 inverted pendulum",
      .description = "A 1 m pendulum swung up from hanging at rest and balanced upright in 4 s by a motor too weak "
                     "to lift it directly (4 against gravity's 9.81 rad/s^2), paying for fuel.",
      .time_step = time_step,
      .forward_model = {L1PendulumStateDriftTerm(parameters, time_step),
                        L1PendulumControlDriftMatTerm(parameters, time_step),
                        std::move(diffusion)},
      .running_cost = running_cost,
      .terminal_cost = QuadraticRegulatorSdeTerminalCostTerm<2>(terminal_cost_mat),
      .terminal_cost_mat = terminal_cost_mat,
      .initial_distribution = initial,
      .normalization_center = Eigen::Vector2d(1.5, 0.0),
      .normalization_scale = Eigen::Vector2d(3.0, 7.0),
      .region_lower = Eigen::Vector2d(-1.5, -7.0),
      .region_upper = Eigen::Vector2d(4.5, 7.0),
      .state_names = {"Angle from upright (rad)", "Rate (rad/s)"},
  });
}

// ---------------------------------------------------------------------------------------------
// 3. L1 double inverted pendulum
// ---------------------------------------------------------------------------------------------

inline constexpr std::size_t kL1DoubleInvertedPendulumNumStages = 101;  // 100 steps of 25 ms

using L1DoubleInvertedPendulum =
    L1Benchmark<4, DoubleInvertedPendulumStateDriftTerm, DoubleInvertedPendulumControlDriftMatTerm>;

// The swing-up of DoubleInvertedPendulumExperiment, with its constants: two 1 kg, 0.5 m rods with a
// 5 N m motor at the base, from hanging at rest to upright in 2.5 s, under the thesis's diffusion
// diag(0.03, 0.03, 0.18, 0.18); fuel weight 1 per second, terminal weights (10, 10, 1, 1).
[[nodiscard]] inline auto MakeL1DoubleInvertedPendulum() noexcept -> Result<L1DoubleInvertedPendulum> {
  constexpr double kHorizon = 2.5;
  const double time_step = kHorizon / static_cast<double>(kL1DoubleInvertedPendulumNumStages - 1);
  RESULT_ASSIGN_OR_RETURN(const DoubleInvertedPendulumDynamics dynamics,
                          DoubleInvertedPendulumDynamics::Make(DoubleInvertedPendulumPhysicalParameters{}));
  RESULT_ASSIGN_OR_RETURN(auto state_drift, DoubleInvertedPendulumStateDriftTerm::Make(dynamics, time_step));
  RESULT_ASSIGN_OR_RETURN(auto control_drift, DoubleInvertedPendulumControlDriftMatTerm::Make(dynamics, time_step));
  RESULT_ASSIGN_OR_RETURN(
      auto diffusion,
      ConstDiagonalSdeDiffusionTerm<4>::Make(Eigen::Vector4d(0.03, 0.03, 0.18, 0.18) * std::sqrt(time_step)));
  RESULT_ASSIGN_OR_RETURN(auto running_cost, (L1ControlSdeRunningCostTerm<4, 1>::Make(time_step)));
  RESULT_ASSIGN_OR_RETURN(auto initial,
                          DiagonalCovarianceNormalDistribution<4>::Make(DoubleInvertedPendulumHangingState(),
                                                                        Eigen::Vector4d::Constant(1e-4)));
  const Eigen::Matrix4d terminal_cost_mat = Eigen::Vector4d(10.0, 10.0, 1.0, 1.0).asDiagonal();
  const Eigen::Vector4d center(1.5, 0.0, 0.0, 0.0);
  const Eigen::Vector4d scale(2.5, 2.0, 8.0, 12.0);
  return SuccessResult(L1DoubleInvertedPendulum{
      .name = "L1 double inverted pendulum",
      .description = "Two 1 kg, 0.5 m links with a 5 N m motor at the base, swung up from hanging at rest and "
                     "balanced upright in 2.5 s, paying for fuel.",
      .time_step = time_step,
      .forward_model = {std::move(state_drift), std::move(control_drift), std::move(diffusion)},
      .running_cost = running_cost,
      .terminal_cost = QuadraticRegulatorSdeTerminalCostTerm<4>(terminal_cost_mat),
      .terminal_cost_mat = terminal_cost_mat,
      .initial_distribution = initial,
      .normalization_center = center,
      .normalization_scale = scale,
      .region_lower = center - scale,
      .region_upper = center + scale,
      .state_names =
          {"First link angle (rad)", "Second link angle (rad)", "First link rate (rad/s)", "Second link rate (rad/s)"},
  });
}

}  // namespace fbsde_traj_opt::examples

#endif  // FBSDE_TRAJ_OPT_EXAMPLES_L1_BENCHMARK_MODELS_HPP_
