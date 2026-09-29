// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#include "examples/double_inverted_pendulum_model.hpp"

#include <cmath>
#include <cstddef>
#include <numbers>

#include <Eigen/Core>
#include <Eigen/LU>
#include <gtest/gtest.h>

#include "fbsde_traj_opt/sde/sde_term_concepts.hpp"
#include "fbsde_traj_opt/utils/result.hpp"

namespace fbsde_traj_opt::examples {
namespace {

using State = DoubleInvertedPendulumDynamics::State;

static_assert(SdeStateDriftTerm<DoubleInvertedPendulumStateDriftTerm, State>);
static_assert(SdeControlDriftMatTerm<DoubleInvertedPendulumControlDriftMatTerm, State>);

constexpr double kTimeStep = 0.025;

auto MakeDynamics(double friction = 0.05) -> DoubleInvertedPendulumDynamics {
  DoubleInvertedPendulumPhysicalParameters parameters;
  parameters.first_joint_friction = friction;
  parameters.second_joint_friction = friction;
  const Result<DoubleInvertedPendulumDynamics> dynamics = DoubleInvertedPendulumDynamics::Make(parameters);
  EXPECT_TRUE(dynamics.has_value());
  return *dynamics;
}

TEST(DoubleInvertedPendulumModelTest, LumpedConstantsMatchTwoUniformRods) {
  const DoubleInvertedPendulumConstants constants = MakeDynamics().Constants();
  // Two 1 kg, 0.5 m rods: lc = 0.25, I = 1/48.
  EXPECT_NEAR(constants.d1, (1.0 / 48.0) + 0.0625 + (1.0 / 48.0) + 0.25 + 0.0625, 1e-12);
  EXPECT_NEAR(constants.d2, 0.125, 1e-12);
  EXPECT_NEAR(constants.d3, (1.0 / 48.0) + 0.0625, 1e-12);
  EXPECT_NEAR(constants.f1, 0.75 * 9.81, 1e-12);
  EXPECT_NEAR(constants.f2, 0.25 * 9.81, 1e-12);
  EXPECT_DOUBLE_EQ(constants.d0, 5.0);
}

TEST(DoubleInvertedPendulumModelTest, HangingAndInvertedAreEquilibria) {
  const DoubleInvertedPendulumDynamics dynamics = MakeDynamics();
  for (const State& state : {DoubleInvertedPendulumHangingState(), State(State::Zero())}) {
    EXPECT_LT(dynamics.JointAccelerations(state).drift.norm(), 1e-12);
  }
}

TEST(DoubleInvertedPendulumModelTest, GravityDestabilizesInvertedAndRestoresHanging) {
  const DoubleInvertedPendulumDynamics dynamics = MakeDynamics();
  // Tilted slightly from inverted, the first link falls further the same way.
  EXPECT_GT(dynamics.JointAccelerations(State(0.05, 0.0, 0.0, 0.0)).drift[0], 0.0);
  // Displaced slightly from hanging, it is pulled back.
  EXPECT_LT(dynamics.JointAccelerations(State(std::numbers::pi + 0.05, 0.0, 0.0, 0.0)).drift[0], 0.0);
}

TEST(DoubleInvertedPendulumModelTest, PositiveTorqueAcceleratesTheFirstLinkPositively) {
  const DoubleInvertedPendulumDynamics dynamics = MakeDynamics();
  const auto accelerations = dynamics.JointAccelerations(DoubleInvertedPendulumHangingState());
  EXPECT_GT(accelerations.control[0], 0.0);
  // The reaction swings the second link back relative to the first.
  EXPECT_LT(accelerations.control[1], 0.0);
}

TEST(DoubleInvertedPendulumModelTest, InertiaMatrixIsSymmetricPositiveDefiniteAtEveryElbowAngle) {
  const DoubleInvertedPendulumDynamics dynamics = MakeDynamics();
  for (int index = 0; index < 64; ++index) {
    const double beta = 2.0 * std::numbers::pi * static_cast<double>(index) / 64.0;
    const Eigen::Matrix2d inertia = dynamics.InertiaMatrix(beta);
    EXPECT_DOUBLE_EQ(inertia(0, 1), inertia(1, 0));
    EXPECT_GT(inertia(0, 0), 0.0);
    EXPECT_GT(inertia.determinant(), 0.0);
  }
}

TEST(DoubleInvertedPendulumModelTest, FrictionlessUnforcedStepConservesEnergy) {
  // The sharpest single check on the equations of motion: a sign or inertia term out of place
  // makes the pendulum gain or lose energy on its own.
  const DoubleInvertedPendulumDynamics dynamics = MakeDynamics(0.0);
  const Result<DoubleInvertedPendulumStateDriftTerm> drift =
      DoubleInvertedPendulumStateDriftTerm::Make(dynamics, kTimeStep);
  ASSERT_TRUE(drift.has_value()) << drift.error();

  // Five seconds of chaotic motion, with the energy held to one part in 1e4 of its natural scale,
  // the potential energy swing between hanging and inverted.
  const double energy_scale = 2.0 * (dynamics.Constants().f1 + dynamics.Constants().f2);
  State state(std::numbers::pi / 2.0, 0.3, 0.0, 0.0);
  const double initial = dynamics.TotalEnergy(state);
  for (std::size_t step = 0; step < 200; ++step) {
    state += (*drift)(step, state);
    EXPECT_NEAR(dynamics.TotalEnergy(state), initial, 1e-4 * energy_scale) << "step " << step;
  }
}

TEST(DoubleInvertedPendulumModelTest, FrictionOnlyEverRemovesEnergy) {
  const DoubleInvertedPendulumDynamics dynamics = MakeDynamics(0.2);
  const Result<DoubleInvertedPendulumStateDriftTerm> drift =
      DoubleInvertedPendulumStateDriftTerm::Make(dynamics, kTimeStep);
  ASSERT_TRUE(drift.has_value()) << drift.error();

  State state(std::numbers::pi / 2.0, 0.3, 0.0, 0.0);
  double previous = dynamics.TotalEnergy(state);
  for (std::size_t step = 0; step < 200; ++step) {
    state += (*drift)(step, state);
    const double energy = dynamics.TotalEnergy(state);
    EXPECT_LE(energy, previous + 1e-9) << "step " << step;
    previous = energy;
  }
}

TEST(DoubleInvertedPendulumModelTest, AffineStepIsExactWithoutControlAndCloseAtTheBounds) {
  const DoubleInvertedPendulumDynamics dynamics = MakeDynamics();
  const Result<DoubleInvertedPendulumStateDriftTerm> drift =
      DoubleInvertedPendulumStateDriftTerm::Make(dynamics, kTimeStep);
  const Result<DoubleInvertedPendulumControlDriftMatTerm> control_matrix =
      DoubleInvertedPendulumControlDriftMatTerm::Make(dynamics, kTimeStep);
  ASSERT_TRUE(drift.has_value() && control_matrix.has_value());

  const State state(2.0, -0.7, 4.0, -9.0);
  const State fine_uncontrolled = IntegrateReferenceStep(dynamics, state, 0.0, kTimeStep, 256);
  EXPECT_LT((state + (*drift)(0, state) - fine_uncontrolled).norm(), 1e-5);

  for (const double control : {-1.0, 1.0}) {
    const State affine = state + (*drift)(0, state) + ((*control_matrix)(0, state) * control);
    const State exact = IntegrateReferenceStep(dynamics, state, control, kTimeStep, 256);
    // Below the per-step process noise of the experiment in every coordinate.
    EXPECT_LT((affine - exact).head<2>().cwiseAbs().maxCoeff(), 1e-3);
    EXPECT_LT((affine - exact).tail<2>().cwiseAbs().maxCoeff(), 3e-2);
  }
}

TEST(DoubleInvertedPendulumModelTest, JointsDrawHangingDownAndInvertedUp) {
  const DoubleInvertedPendulumDynamics dynamics = MakeDynamics();
  const auto hanging = dynamics.Joints(DoubleInvertedPendulumHangingState());
  EXPECT_NEAR(hanging.elbow.y(), -0.5, 1e-12);
  EXPECT_NEAR(hanging.tip.y(), -1.0, 1e-12);
  EXPECT_NEAR(hanging.tip.x(), 0.0, 1e-12);
  const auto inverted = dynamics.Joints(State::Zero());
  EXPECT_NEAR(inverted.tip.y(), 1.0, 1e-12);
}

TEST(DoubleInvertedPendulumModelTest, MakeRejectsNonPhysicalParameters) {
  DoubleInvertedPendulumPhysicalParameters parameters;
  parameters.first_link_mass = 0.0;
  EXPECT_FALSE(DoubleInvertedPendulumDynamics::Make(parameters).has_value());
  parameters = {};
  parameters.second_joint_friction = -0.1;
  EXPECT_FALSE(DoubleInvertedPendulumDynamics::Make(parameters).has_value());
  EXPECT_FALSE(DoubleInvertedPendulumStateDriftTerm::Make(MakeDynamics(), 0.0).has_value());
  EXPECT_FALSE(DoubleInvertedPendulumControlDriftMatTerm::Make(MakeDynamics(), -1.0).has_value());
}

}  // namespace
}  // namespace fbsde_traj_opt::examples
