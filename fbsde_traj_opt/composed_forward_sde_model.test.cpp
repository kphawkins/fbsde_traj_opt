// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#include "fbsde_traj_opt/composed_forward_sde_model.hpp"

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "fbsde_traj_opt/const_diagonal_sde_diffusion_term.hpp"
#include "fbsde_traj_opt/const_linear_sde_control_drift_mat_term.hpp"
#include "fbsde_traj_opt/const_linear_sde_state_drift_term.hpp"

namespace fbsde_traj_opt {
namespace {

using StateDrift3 = ConstLinearSdeStateDriftTerm<3>;
using ControlDrift3x2 = ConstLinearSdeControlDriftMatTerm<3, 2>;
using Diffusion3 = ConstDiagonalSdeDiffusionTerm<3>;
using Model3x2 = ComposedForwardSdeModel<3, 2, StateDrift3, ControlDrift3x2, Diffusion3>;

// Builds a model with drift A = 0.1 * I, control drift B = [[1, 0], [0, 1], [0, 0]], and
// Sigma = diag(2, 3, 4).
auto MakeTestModel() noexcept -> Model3x2 {
  const Eigen::Matrix3d a = Eigen::Matrix3d::Identity() * 0.1;

  Eigen::Matrix<double, 3, 2> b;
  b << 1.0, 0.0,  //
      0.0, 1.0,   //
      0.0, 0.0;

  const auto diffusion = Diffusion3::Make(Eigen::Vector3d(2.0, 3.0, 4.0));
  EXPECT_TRUE(diffusion.has_value());

  return {StateDrift3(a), ControlDrift3x2(b), *diffusion};
}

TEST(ComposedForwardSdeModelTest, StepSumsStateDriftControlAndNoise) {
  const Model3x2 model = MakeTestModel();

  const Eigen::Vector3d state(1.0, 2.0, 3.0);
  const Eigen::Vector2d control(10.0, 20.0);
  const Eigen::Vector3d noise(0.5, -0.5, 1.0);

  // x + A x = (1.1, 2.2, 3.3); + B u = (11.1, 22.2, 3.3); + Sigma z = (12.1, 20.7, 7.3).
  const Eigen::Vector3d next_state = model(0, state, control, noise);

  EXPECT_TRUE(next_state.isApprox(Eigen::Vector3d(12.1, 20.7, 7.3)));
}

TEST(ComposedForwardSdeModelTest, StepWithZeroControlAndNoiseIsDriftOnly) {
  const Model3x2 model = MakeTestModel();

  const Eigen::Vector3d state(1.0, 2.0, 3.0);

  const Eigen::Vector3d next_state = model(0, state, Eigen::Vector2d::Zero(), Eigen::Vector3d::Zero());

  EXPECT_TRUE(next_state.isApprox(Eigen::Vector3d(1.1, 2.2, 3.3)));
}

TEST(ComposedForwardSdeModelTest, AccessorsExposeTheComponentTerms) {
  const Model3x2 model = MakeTestModel();

  const Eigen::Vector3d state(1.0, 2.0, 3.0);

  EXPECT_TRUE(model.state_drift().drift_mat().isApprox(Eigen::Matrix3d::Identity() * 0.1));
  EXPECT_TRUE(model.control_drift_mat().drift_mat().col(0).isApprox(Eigen::Vector3d(1.0, 0.0, 0.0)));
  EXPECT_TRUE(Eigen::Matrix3d(model.diffusion()(0, state))
                  .isApprox(Eigen::Matrix3d(Eigen::Vector3d(2.0, 3.0, 4.0).asDiagonal())));
}

}  // namespace
}  // namespace fbsde_traj_opt
