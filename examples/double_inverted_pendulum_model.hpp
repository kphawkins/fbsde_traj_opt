// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#ifndef FBSDE_TRAJ_OPT_EXAMPLES_DOUBLE_INVERTED_PENDULUM_MODEL_HPP_
#define FBSDE_TRAJ_OPT_EXAMPLES_DOUBLE_INVERTED_PENDULUM_MODEL_HPP_

#include <cmath>
#include <cstddef>
#include <numbers>
#include <utility>

#include <Eigen/Core>

#include "fbsde_traj_opt/utils/result.hpp"

namespace fbsde_traj_opt::examples {

// The double inverted pendulum of Hawkins (2021), Section 5.5.3: two rigid links in a vertical
// plane, the first hinged to the ground and the second hinged to the end of the first, with a
// bounded torque motor at the ground joint only (the "pendubot" arrangement) and viscous friction
// at both joints. The task is to swing it up from hanging at rest to balanced upright.
//
// State and conventions. The state is
//
//   x = [alpha, beta, omega, psi]',
//
// the thesis's ordering: `alpha` the angle of the first link measured from *upright*, `beta` the
// angle of the second link relative to the first, and `omega`, `psi` their rates. Upright and
// straight is therefore the origin -- where the quadratic terminal cost of the thesis's problem
// (5.19) puts its minimum -- and hanging at rest is `[pi, 0, 0, 0]`. A positive control is a
// positive torque on `alpha`.
//
// Equations of motion. With `c = cos(beta)`, `s = sin(beta)`,
//
//   M(beta) [omega_dot; psi_dot] = [ f1 sin(alpha) + f2 sin(alpha + beta) + d2 s (2 omega psi + psi^2) - f3 omega + d0
//   u ;
//                                    f2 sin(alpha + beta) - d2 s omega^2 - f4 psi ],
//
//   M(beta) = [ d1 + 2 d2 c,  d3 + d2 c ;
//               d3 + d2 c,    d3        ],
//
// which is the Lagrangian of two uniform rods written in the lumped constants the thesis uses:
//
//   d0 = the torque at |u| = 1                     f1 = (m1 lc1 + m2 l1) g
//   d1 = I1 + m1 lc1^2 + I2 + m2 (l1^2 + lc2^2)    f2 = m2 lc2 g
//   d2 = m2 l1 lc2                                 f3, f4 = joint friction
//   d3 = I2 + m2 lc2^2
//
// with `l` a link's length, `lc` the distance to its center of mass, and `I` its inertia about
// that center. Gravity destabilizes upright -- `+ f1 sin(alpha)` -- because alpha is measured
// from the top.
//
// A note on the thesis. Equation (5.20) there shows this system solved for the accelerations, and
// as printed it is not quite self-consistent: its first-link gravity term enters as `- f1 sin
// alpha` while the second link's enters as `+ f2 sin(alpha + beta)`, which puts one link's
// equilibrium at the top and the other's at the bottom, and its determinant `d1 d3 + 2 d2 d3 c -
// d2^2 c^2` corresponds to an off-diagonal inertia of `d2 c` rather than `d3 + d2 c`. The form
// above is the one both terms of (5.20) are evidently transcribing, rederived from the Lagrangian
// so that energy is conserved when friction and control are off -- which the tests check.
//
// Discretization. The thesis's forward process is a discrete-time, control-affine step (4.10),
// and the library's forward SDE models exactly that form, `x + f(x) + B(x) u + Sigma z`. The
// control is held over each step (zero-order hold), and the step is built from the exact
// held-control flow map `Phi(x, u)` of the continuous dynamics, integrated by classical RK4 on a
// few substeps, as Appendix B.2 recommends for highly nonlinear systems:
//
//   f(x) = Phi(x, 0) - x,
//   B(x) = ( Phi(x, +1) - Phi(x, -1) ) / 2.
//
// That is the affine interpolation of the flow map across the control range: exact at u = 0, and
// off at the bounds only by the map's curvature in u, which over one 25 ms step is below 1e-4 rad
// in the angles and 0.03 rad/s in the rates -- under the process noise injected at every step.
// Affine in u is what keeps the Q-function policy's minimizer (Section 4.4) in closed form.
//
// Why not a cheaper step. A one-evaluation scheme -- explicit Euler as in (4.11), or the
// symplectic semi-implicit Euler -- is hopeless at this step size: during the swing the second
// link turns at over 20 rad/s, half a radian per step, and such a model drifts by radians from the
// true motion over the horizon and changes the frictionless pendulum's energy by tens of percent.
// An optimizer handed that model optimizes the model's artifacts, not the pendulum. The
// experiment report checks the step against a far finer RK4 reference and against energy
// conservation.

// The physical description of the pendulum: two uniform rods, a motor, and joint friction.
struct DoubleInvertedPendulumPhysicalParameters {
  double first_link_mass = 1.0;         // kg
  double second_link_mass = 1.0;        // kg
  double first_link_length = 0.5;       // m
  double second_link_length = 0.5;      // m
  double gravity = 9.81;                // m / s^2
  double first_joint_friction = 0.05;   // N m s / rad
  double second_joint_friction = 0.05;  // N m s / rad
  double max_torque = 5.0;              // N m, the torque at |u| = 1
};

// The lumped constants of the equations of motion, named as in the thesis.
struct DoubleInvertedPendulumConstants {
  double d0 = 0.0;
  double d1 = 0.0;
  double d2 = 0.0;
  double d3 = 0.0;
  double f1 = 0.0;
  double f2 = 0.0;
  double f3 = 0.0;
  double f4 = 0.0;
};

// The continuous-time dynamics, energy, and geometry of the pendulum. Everything else in this
// header -- the discrete step, the SDE terms -- is built on this.
class DoubleInvertedPendulumDynamics {
 public:
  using State = Eigen::Vector4d;
  using Control = Eigen::Matrix<double, 1, 1>;

  static constexpr Eigen::Index kAlpha = 0;
  static constexpr Eigen::Index kBeta = 1;
  static constexpr Eigen::Index kOmega = 2;
  static constexpr Eigen::Index kPsi = 3;

  // The drift and control columns of the solved joint accelerations: `qdd = drift + control * u`.
  struct Accelerations {
    Eigen::Vector2d drift = Eigen::Vector2d::Zero();
    Eigen::Vector2d control = Eigen::Vector2d::Zero();
  };

  // Where the two joints and the tip are, in the plane, for drawing: the ground joint at the
  // origin, `y` up.
  struct JointPositions {
    Eigen::Vector2d elbow = Eigen::Vector2d::Zero();
    Eigen::Vector2d tip = Eigen::Vector2d::Zero();
  };

  // Builds the dynamics of two uniform rods from their physical description.
  //
  // Fails if any mass, length, or gravity is not finite and positive, if either friction is
  // negative or not finite, or if the torque limit is not finite and positive.
  static auto Make(const DoubleInvertedPendulumPhysicalParameters& parameters) noexcept
      -> Result<DoubleInvertedPendulumDynamics> {
    const auto positive = [](double value) noexcept { return std::isfinite(value) && value > 0.0; };
    const auto non_negative = [](double value) noexcept { return std::isfinite(value) && value >= 0.0; };
    RESULT_ASSERT(positive(parameters.first_link_mass) && positive(parameters.second_link_mass),
                  "DoubleInvertedPendulumDynamics::Make: both link masses must be finite and positive.");
    RESULT_ASSERT(positive(parameters.first_link_length) && positive(parameters.second_link_length),
                  "DoubleInvertedPendulumDynamics::Make: both link lengths must be finite and positive.");
    RESULT_ASSERT(positive(parameters.gravity),
                  "DoubleInvertedPendulumDynamics::Make: gravity must be finite and positive.");
    RESULT_ASSERT(non_negative(parameters.first_joint_friction) && non_negative(parameters.second_joint_friction),
                  "DoubleInvertedPendulumDynamics::Make: joint friction must be finite and non-negative.");
    RESULT_ASSERT(positive(parameters.max_torque),
                  "DoubleInvertedPendulumDynamics::Make: the torque limit must be finite and positive.");

    const double m1 = parameters.first_link_mass;
    const double m2 = parameters.second_link_mass;
    const double l1 = parameters.first_link_length;
    const double l2 = parameters.second_link_length;
    const double lc1 = 0.5 * l1;
    const double lc2 = 0.5 * l2;
    const double inertia1 = (m1 * l1 * l1) / 12.0;
    const double inertia2 = (m2 * l2 * l2) / 12.0;

    DoubleInvertedPendulumConstants constants;
    constants.d0 = parameters.max_torque;
    constants.d1 = inertia1 + (m1 * lc1 * lc1) + inertia2 + (m2 * ((l1 * l1) + (lc2 * lc2)));
    constants.d2 = m2 * l1 * lc2;
    constants.d3 = inertia2 + (m2 * lc2 * lc2);
    constants.f1 = ((m1 * lc1) + (m2 * l1)) * parameters.gravity;
    constants.f2 = m2 * lc2 * parameters.gravity;
    constants.f3 = parameters.first_joint_friction;
    constants.f4 = parameters.second_joint_friction;
    return SuccessResult(DoubleInvertedPendulumDynamics(constants, l1, l2));
  }

  // The inertia matrix M(beta). Positive definite for every beta, since it is the kinetic energy
  // metric of a physical system.
  [[nodiscard]] auto InertiaMatrix(double beta) const noexcept -> Eigen::Matrix2d {
    const double coupling = constants_.d2 * std::cos(beta);
    Eigen::Matrix2d inertia;
    inertia << constants_.d1 + (2.0 * coupling), constants_.d3 + coupling,  //
        constants_.d3 + coupling, constants_.d3;
    return inertia;
  }

  // The joint accelerations at `state`, split into the part that does not depend on the control
  // and the part that multiplies it.
  [[nodiscard]] auto JointAccelerations(const State& state) const noexcept -> Accelerations {
    const double alpha = state[kAlpha];
    const double beta = state[kBeta];
    const double omega = state[kOmega];
    const double psi = state[kPsi];
    const double sin_beta = std::sin(beta);
    const double gravity_second = constants_.f2 * std::sin(alpha + beta);

    const Eigen::Vector2d generalized_forces(
        (constants_.f1 * std::sin(alpha)) + gravity_second +
            (constants_.d2 * sin_beta * ((2.0 * omega * psi) + (psi * psi))) - (constants_.f3 * omega),
        gravity_second - (constants_.d2 * sin_beta * omega * omega) - (constants_.f4 * psi));

    // The 2 x 2 inverse in closed form: this runs once per sample per stage per iteration, and a
    // general factorization would be most of its cost.
    const Eigen::Matrix2d inertia = InertiaMatrix(beta);
    const double determinant = (inertia(0, 0) * inertia(1, 1)) - (inertia(0, 1) * inertia(1, 0));
    Eigen::Matrix2d inverse;
    inverse << inertia(1, 1), -inertia(0, 1),  //
        -inertia(1, 0), inertia(0, 0);
    inverse /= determinant;

    Accelerations accelerations;
    accelerations.drift = inverse * generalized_forces;
    accelerations.control = inverse.col(0) * constants_.d0;
    return accelerations;
  }

  // The continuous-time state derivative under a held control `u`.
  [[nodiscard]] auto StateDerivative(const State& state, double control) const noexcept -> State {
    const Accelerations accelerations = JointAccelerations(state);
    State derivative;
    derivative.head<2>() = state.tail<2>();
    derivative.tail<2>() = accelerations.drift + (accelerations.control * control);
    return derivative;
  }

  // Kinetic plus potential energy, with the potential zero at the ground joint's height. Conserved
  // exactly by the continuous dynamics when friction and control are both zero.
  [[nodiscard]] auto TotalEnergy(const State& state) const noexcept -> double {
    const Eigen::Vector2d rates = state.tail<2>();
    const double kinetic = 0.5 * rates.dot(InertiaMatrix(state[kBeta]) * rates);
    const double potential =
        (constants_.f1 * std::cos(state[kAlpha])) + (constants_.f2 * std::cos(state[kAlpha] + state[kBeta]));
    return kinetic + potential;
  }

  [[nodiscard]] auto Joints(const State& state) const noexcept -> JointPositions {
    const double alpha = state[kAlpha];
    const double absolute_second = alpha + state[kBeta];
    JointPositions joints;
    joints.elbow = first_link_length_ * Eigen::Vector2d(std::sin(alpha), std::cos(alpha));
    joints.tip =
        joints.elbow + (second_link_length_ * Eigen::Vector2d(std::sin(absolute_second), std::cos(absolute_second)));
    return joints;
  }

  [[nodiscard]] auto Constants() const noexcept -> const DoubleInvertedPendulumConstants& { return constants_; }

  [[nodiscard]] auto FirstLinkLength() const noexcept -> double { return first_link_length_; }

  [[nodiscard]] auto SecondLinkLength() const noexcept -> double { return second_link_length_; }

 private:
  DoubleInvertedPendulumDynamics(DoubleInvertedPendulumConstants constants,
                                 double first_link_length,
                                 double second_link_length) noexcept
      : constants_(constants), first_link_length_(first_link_length), second_link_length_(second_link_length) {}

  DoubleInvertedPendulumConstants constants_;
  double first_link_length_;
  double second_link_length_;
};

// The held-control flow map `Phi(x, u)` over one step, by RK4 on `substeps` equal substeps.
[[nodiscard]] inline auto IntegrateReferenceStep(const DoubleInvertedPendulumDynamics& dynamics,
                                                 const DoubleInvertedPendulumDynamics::State& state,
                                                 double control,
                                                 double time_step,
                                                 std::size_t substeps) noexcept
    -> DoubleInvertedPendulumDynamics::State;

// The uncontrolled part `f(x) = Phi(x, 0) - x` of the zero-order-hold step, as an
// SdeStateDriftTerm.
class DoubleInvertedPendulumStateDriftTerm {
 public:
  using State = DoubleInvertedPendulumDynamics::State;

  // RK4 substeps per step: at the experiment's 25 ms step, 6.25 ms substeps agree with a 64-substep
  // integration to about 1e-5.
  static constexpr std::size_t kSubsteps = 4;

  // Fails unless `time_step` is finite and positive.
  static auto Make(DoubleInvertedPendulumDynamics dynamics, double time_step) noexcept
      -> Result<DoubleInvertedPendulumStateDriftTerm> {
    const bool valid_time_step = std::isfinite(time_step) && time_step > 0.0;
    RESULT_ASSERT(valid_time_step,
                  "DoubleInvertedPendulumStateDriftTerm::Make: the time step must be finite and positive.");
    return SuccessResult(DoubleInvertedPendulumStateDriftTerm(dynamics, time_step));
  }

  auto operator()([[maybe_unused]] std::size_t stage, const State& state) const noexcept -> State {
    return IntegrateReferenceStep(dynamics_, state, 0.0, time_step_, kSubsteps) - state;
  }

  [[nodiscard]] auto Dynamics() const noexcept -> const DoubleInvertedPendulumDynamics& { return dynamics_; }

  [[nodiscard]] auto TimeStep() const noexcept -> double { return time_step_; }

 private:
  DoubleInvertedPendulumStateDriftTerm(DoubleInvertedPendulumDynamics dynamics, double time_step) noexcept
      : dynamics_(dynamics), time_step_(time_step) {}

  DoubleInvertedPendulumDynamics dynamics_;
  double time_step_;
};

// The control matrix `B(x) = (Phi(x, +1) - Phi(x, -1)) / 2` of the zero-order-hold step, as an
// SdeControlDriftMatTerm.
class DoubleInvertedPendulumControlDriftMatTerm {
 public:
  using State = DoubleInvertedPendulumDynamics::State;

  // Fails unless `time_step` is finite and positive.
  static auto Make(DoubleInvertedPendulumDynamics dynamics, double time_step) noexcept
      -> Result<DoubleInvertedPendulumControlDriftMatTerm> {
    const bool valid_time_step = std::isfinite(time_step) && time_step > 0.0;
    RESULT_ASSERT(valid_time_step,
                  "DoubleInvertedPendulumControlDriftMatTerm::Make: the time step must be finite and positive.");
    return SuccessResult(DoubleInvertedPendulumControlDriftMatTerm(dynamics, time_step));
  }

  auto operator()([[maybe_unused]] std::size_t stage, const State& state) const noexcept
      -> Eigen::Matrix<double, 4, 1> {
    constexpr std::size_t kSubsteps = DoubleInvertedPendulumStateDriftTerm::kSubsteps;
    return 0.5 * (IntegrateReferenceStep(dynamics_, state, 1.0, time_step_, kSubsteps) -
                  IntegrateReferenceStep(dynamics_, state, -1.0, time_step_, kSubsteps));
  }

 private:
  DoubleInvertedPendulumControlDriftMatTerm(DoubleInvertedPendulumDynamics dynamics, double time_step) noexcept
      : dynamics_(dynamics), time_step_(time_step) {}

  DoubleInvertedPendulumDynamics dynamics_;
  double time_step_;
};

// Integrates the continuous dynamics over one step of length `time_step`, holding `control`, by
// classical fourth-order Runge-Kutta on `substeps` equal substeps. The model's step uses a few
// substeps; the experiment's accuracy checks use this same function on a far finer grid as the
// reference.
[[nodiscard]] inline auto IntegrateReferenceStep(const DoubleInvertedPendulumDynamics& dynamics,
                                                 const DoubleInvertedPendulumDynamics::State& state,
                                                 double control,
                                                 double time_step,
                                                 std::size_t substeps) noexcept
    -> DoubleInvertedPendulumDynamics::State {
  using State = DoubleInvertedPendulumDynamics::State;
  const double step = time_step / static_cast<double>(substeps);
  State current = state;
  for (std::size_t substep = 0; substep < substeps; ++substep) {
    const State k1 = dynamics.StateDerivative(current, control);
    const State k2 = dynamics.StateDerivative(current + (0.5 * step * k1), control);
    const State k3 = dynamics.StateDerivative(current + (0.5 * step * k2), control);
    const State k4 = dynamics.StateDerivative(current + (step * k3), control);
    current += (step / 6.0) * (k1 + (2.0 * k2) + (2.0 * k3) + k4);
  }
  return current;
}

// Hanging straight down, at rest: where the swing-up starts.
[[nodiscard]] inline auto DoubleInvertedPendulumHangingState() noexcept -> DoubleInvertedPendulumDynamics::State {
  return {std::numbers::pi, 0.0, 0.0, 0.0};
}

}  // namespace fbsde_traj_opt::examples

#endif  // FBSDE_TRAJ_OPT_EXAMPLES_DOUBLE_INVERTED_PENDULUM_MODEL_HPP_
