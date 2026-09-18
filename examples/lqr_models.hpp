// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#ifndef FBSDE_TRAJ_OPT_EXAMPLES_LQR_MODELS_HPP_
#define FBSDE_TRAJ_OPT_EXAMPLES_LQR_MODELS_HPP_

#include <cstddef>

#include <Eigen/Core>

#include "examples/lqr_experiment.hpp"

namespace fbsde_traj_opt::examples {

// Three linear-quadratic regulator problems taken from the control literature, each chosen
// because a plausible fixed-gain controller fails on it in a different and recognizable way.
//
// Each model is written in continuous time, with its cost given as a continuous rate that the
// builder multiplies by the time step to get the stage cost. That is the discretization of the
// integral cost, and keeping it explicit means the cost of a model can be read off against the
// paper it came from without mentally dividing by a step size.
//
// The baselines are not straw men. Each is the controller an engineer would actually reach for
// first on the system in question, and each fails for the reason that system is famous for.

// Stage and batch sizes. One trajectory count for all three models so the Monte Carlo error is
// comparable across them.
inline constexpr std::size_t kNumTrajectories = 128;

// ---------------------------------------------------------------------------------------------
// 1. Double integrator
// ---------------------------------------------------------------------------------------------

inline constexpr std::size_t kDoubleIntegratorNumStages = 61;

// The minimum-energy regulator for a double integrator: the example every treatment of LQR opens
// with (Athans and Falb, "Optimal Control", 1966; Bryson and Ho, "Applied Optimal Control", 1975).
// A unit mass on a frictionless line, pushed by a force, to be brought to rest at the origin.
//
// The baseline is proportional feedback with no derivative term. On a plant that is two
// integrators in series this is marginally stable: the closed loop has its poles exactly on the
// imaginary axis, so the mass oscillates about the origin forever instead of settling. It is the
// textbook demonstration of why position feedback alone is not enough, and here it shows up as a
// cost-to-go curve that stops falling.
[[nodiscard]] inline auto MakeDoubleIntegratorSpec() noexcept -> LqrExperimentSpec<2, 1> {
  constexpr double kTimeStep = 0.05;

  LqrExperimentSpec<2, 1> spec;
  spec.name = "Double integrator";
  spec.description =
      "A unit mass on a frictionless line, driven by a force and asked to come to rest at the origin -- the "
      "example every treatment of LQR opens with. The baseline is proportional position feedback with no "
      "derivative term, which puts the closed-loop poles exactly on the imaginary axis: the mass oscillates "
      "about the origin forever rather than settling. Horizon 3 s at a 50 ms step.";
  spec.time_step = kTimeStep;

  spec.continuous_state_mat << 0.0, 1.0,  //
      0.0, 0.0;
  spec.continuous_control_mat << 0.0, 1.0;

  // Penalize position error and control effort; let velocity be whatever it needs to be on the
  // way. The terminal cost is what asks for "at rest at the origin" specifically.
  spec.state_cost_mat = Eigen::Vector2d(1.0, 0.0).asDiagonal() * kTimeStep;
  spec.control_cost_mat = Eigen::Matrix<double, 1, 1>::Constant(0.1 * kTimeStep);
  spec.terminal_cost_mat = Eigen::Vector2d(20.0, 2.0).asDiagonal();

  spec.initial_mean = Eigen::Vector2d(1.0, 0.0);
  spec.initial_covariance_diagonal = Eigen::Vector2d(0.01, 0.01);
  spec.diffusion_diagonal = Eigen::Vector2d(0.002, 0.02);

  // u = -4 p: a gain an engineer might pick for a crisp response, with the derivative term
  // forgotten.
  spec.baseline_gain << -4.0, 0.0;
  spec.baseline_name = "proportional feedback";

  spec.state_names = {"Position (m)", "Velocity (m/s)"};
  spec.plotted_state_indices = {0, 1};
  spec.plot_phase_portrait = true;
  spec.phase_horizontal_index = 0;
  spec.phase_vertical_index = 1;
  return spec;
}

// ---------------------------------------------------------------------------------------------
// 2. Inverted pendulum on a cart
// ---------------------------------------------------------------------------------------------

inline constexpr std::size_t kCartPoleNumStages = 151;

// A pendulum hinged on a motorized cart, linearized about the upright equilibrium -- the standard
// unstable-plant benchmark (Ogata, "Modern Control Engineering"; Franklin, Powell and
// Emami-Naeini, "Feedback Control of Dynamic Systems"), with the usual parameters: a 2 kg cart, a
// 0.1 kg pole of half-length 0.5 m.
//
// The baseline is the controller the problem invites: proportional-derivative feedback on the
// pole angle, with enough gain to beat gravity. It works, in the sense that the pole comes
// upright and stays there. But every newton that rights the pole also pushes the cart, and
// nothing in the baseline is watching the cart, so the cart accelerates away and never comes
// back. This is the classic demonstration that stabilizing the obvious variable is not the same
// as solving the problem.
[[nodiscard]] inline auto MakeCartPoleSpec() noexcept -> LqrExperimentSpec<4, 1> {
  constexpr double kTimeStep = 0.02;
  constexpr double kCartMass = 2.0;
  constexpr double kPoleMass = 0.1;
  constexpr double kPoleHalfLength = 0.5;
  constexpr double kGravity = 9.81;

  LqrExperimentSpec<4, 1> spec;
  spec.name = "Inverted pendulum on a cart";
  spec.description =
      "A pendulum hinged on a motorized cart, linearized about upright: the standard unstable-plant benchmark, "
      "with a 2 kg cart and a 0.1 kg pole of half-length 0.5 m. The baseline is proportional-derivative "
      "feedback on the pole angle with enough gain to beat gravity. It does right the pole -- but every newton "
      "that rights it also pushes the cart, and nothing in the baseline is watching the cart, so the cart "
      "accelerates away and never comes back. Horizon 3 s at a 20 ms step.";
  spec.time_step = kTimeStep;

  // p_ddot     = (m g / M) theta + (1 / M) u
  // theta_ddot = ((M + m) g / (M l)) theta - (1 / (M l)) u
  spec.continuous_state_mat << 0.0, 1.0, 0.0, 0.0,        //
      0.0, 0.0, (kPoleMass * kGravity) / kCartMass, 0.0,  //
      0.0, 0.0, 0.0, 1.0,                                 //
      0.0, 0.0, ((kCartMass + kPoleMass) * kGravity) / (kCartMass * kPoleHalfLength), 0.0;
  spec.continuous_control_mat << 0.0, 1.0 / kCartMass, 0.0, -1.0 / (kCartMass * kPoleHalfLength);

  // The cart's position matters as much as the pole's angle; that is the whole difficulty.
  spec.state_cost_mat = Eigen::Vector4d(1.0, 0.1, 10.0, 0.5).asDiagonal() * kTimeStep;
  spec.control_cost_mat = Eigen::Matrix<double, 1, 1>::Constant(0.1 * kTimeStep);
  spec.terminal_cost_mat = Eigen::Vector4d(10.0, 1.0, 100.0, 10.0).asDiagonal();

  // The pole starts about 11 degrees off vertical, the cart at the origin and at rest.
  spec.initial_mean = Eigen::Vector4d(0.0, 0.0, 0.2, 0.0);
  spec.initial_covariance_diagonal = Eigen::Vector4d(2.5e-3, 2.5e-3, 4.0e-4, 4.0e-4);
  spec.diffusion_diagonal = Eigen::Vector4d(1.0e-4, 2.0e-3, 1.0e-4, 2.0e-3);

  // u = 40 theta + 8 theta_dot. The sign is positive because the control enters the pole equation
  // negatively; 40 exceeds the 20.6 of gravity's destabilizing term, so the pole is stabilized
  // with damping ratio about 0.9.
  spec.baseline_gain << 0.0, 0.0, 40.0, 8.0;
  spec.baseline_name = "pole-angle PD";

  spec.state_names = {"Cart position (m)", "Cart velocity (m/s)", "Pole angle (rad)", "Pole rate (rad/s)"};
  spec.plotted_state_indices = {2, 0};
  spec.plot_phase_portrait = false;
  return spec;
}

// ---------------------------------------------------------------------------------------------
// 3. Two-mass spring
// ---------------------------------------------------------------------------------------------

inline constexpr std::size_t kTwoMassSpringNumStages = 151;

// Two unit masses joined by a unit spring, with the force applied to the first mass and the
// second mass the one that must be brought to rest -- the ACC benchmark of Wie and Bernstein,
// "Benchmark Problems for Robust Control Design", Journal of Guidance, Control, and Dynamics
// 15(5), 1992. The plant is undamped, so its flexible mode rings forever unless a controller
// takes energy out of it.
//
// The baseline is stiff proportional-derivative feedback on the mass the actuator is attached to
// -- colocated control, the safe and usual choice. Here it is exactly wrong. High colocated gain
// pins the first mass in place, and a pinned first mass is a wall: the second mass becomes an
// undamped oscillator against it and rings at its natural frequency for the entire horizon, with
// the controller doing nothing about it because the variable it watches is already at zero. The
// control must act *through* the flexible mode, which is what makes the benchmark hard and what
// the LQR solution does.
[[nodiscard]] inline auto MakeTwoMassSpringSpec() noexcept -> LqrExperimentSpec<4, 1> {
  constexpr double kTimeStep = 0.1;
  constexpr double kSpringConstant = 1.0;

  LqrExperimentSpec<4, 1> spec;
  spec.name = "Two-mass spring";
  spec.description =
      "Two unit masses joined by a unit spring, force applied to the first, with the second the one that must "
      "be brought to rest -- the ACC robust-control benchmark of Wie and Bernstein (1992). The plant is "
      "undamped, so its flexible mode rings forever unless a controller takes energy out of it. The baseline "
      "is stiff colocated PD on the driven mass, the safe and usual choice, and here exactly wrong: high "
      "colocated gain pins the first mass, and a pinned mass is a wall, so the second rings against it "
      "undamped for the whole horizon while the controller sees nothing to correct. Horizon 15 s at a 100 ms "
      "step.";
  spec.time_step = kTimeStep;

  spec.continuous_state_mat << 0.0, 0.0, 1.0, 0.0,  //
      0.0, 0.0, 0.0, 1.0,                           //
      -kSpringConstant, kSpringConstant, 0.0, 0.0,  //
      kSpringConstant, -kSpringConstant, 0.0, 0.0;
  spec.continuous_control_mat << 0.0, 0.0, 1.0, 0.0;

  // The benchmark's objective: regulate the second mass, cheaply.
  spec.state_cost_mat = Eigen::Vector4d(0.0, 1.0, 0.0, 0.0).asDiagonal() * kTimeStep;
  spec.control_cost_mat = Eigen::Matrix<double, 1, 1>::Constant(0.1 * kTimeStep);
  spec.terminal_cost_mat = Eigen::Vector4d(1.0, 10.0, 1.0, 1.0).asDiagonal();

  // The second mass starts displaced, the first at the origin, both at rest: the spring is
  // stretched and the ringing begins immediately.
  spec.initial_mean = Eigen::Vector4d(0.0, 1.0, 0.0, 0.0);
  spec.initial_covariance_diagonal = Eigen::Vector4d(4.0e-4, 4.0e-4, 4.0e-4, 4.0e-4);
  spec.diffusion_diagonal = Eigen::Vector4d(1.0e-4, 1.0e-4, 2.0e-3, 2.0e-3);

  // u = -10 x1 - 10 v1: stiff feedback on the driven mass alone.
  spec.baseline_gain << -10.0, 0.0, -10.0, 0.0;
  spec.baseline_name = "colocated PD on mass 1";

  spec.state_names = {"Mass 1 position (m)", "Mass 2 position (m)", "Mass 1 velocity (m/s)", "Mass 2 velocity (m/s)"};
  spec.plotted_state_indices = {1, 0};
  spec.plot_phase_portrait = true;
  spec.phase_horizontal_index = 1;
  spec.phase_vertical_index = 3;
  return spec;
}

}  // namespace fbsde_traj_opt::examples

#endif  // FBSDE_TRAJ_OPT_EXAMPLES_LQR_MODELS_HPP_
