// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#include "fbsde_traj_opt/sde/diffusion_scaled_forward_sde_model.hpp"

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "fbsde_traj_opt/dynamics/const_diagonal_sde_diffusion_term.hpp"
#include "fbsde_traj_opt/dynamics/const_linear_sde_control_drift_mat_term.hpp"
#include "fbsde_traj_opt/dynamics/const_linear_sde_state_drift_term.hpp"
#include "fbsde_traj_opt/sde/composed_forward_sde_model.hpp"

namespace fbsde_traj_opt {
namespace {

using Model = ComposedForwardSdeModel<2,
                                      1,
                                      ConstLinearSdeStateDriftTerm<2>,
                                      ConstLinearSdeControlDriftMatTerm<2, 1>,
                                      ConstDiagonalSdeDiffusionTerm<2>>;

auto MakeModel() -> Model {
  const Eigen::Matrix2d state_matrix = (Eigen::Matrix2d() << 0.0, 0.1, 0.0, 0.0).finished();
  return Model(ConstLinearSdeStateDriftTerm<2>(state_matrix),
               ConstLinearSdeControlDriftMatTerm<2, 1>(Eigen::Vector2d(0.005, 0.1)),
               *ConstDiagonalSdeDiffusionTerm<2>::Make(Eigen::Vector2d(0.01, 0.05)));
}

TEST(DiffusionScaledForwardSdeModelTest, AtScaleOneIsTheWrappedModelExactly) {
  const Model model = MakeModel();
  const DiffusionScaledForwardSdeModel<Model> scaled(model);
  const Eigen::Vector2d state(1.0, -0.5);
  const Eigen::Matrix<double, 1, 1> control(0.3);
  const Eigen::Vector2d noise(0.7, -1.2);
  EXPECT_EQ(scaled(3, state, control, noise), model(3, state, control, noise));
  EXPECT_EQ(scaled.diffusion()(3, state), Eigen::Matrix2d(model.diffusion()(3, state)));
}

TEST(DiffusionScaledForwardSdeModelTest, ScalesTheNoiseAndTheDiffusionItReports) {
  const Model model = MakeModel();
  DiffusionScaledForwardSdeModel<Model> scaled(model);
  scaled.set_scale(3.0);
  const Eigen::Vector2d state(1.0, -0.5);
  const Eigen::Matrix<double, 1, 1> control(0.3);
  const Eigen::Vector2d noise(0.7, -1.2);
  const Eigen::Vector2d noiseless = model(3, state, control, Eigen::Vector2d::Zero());
  EXPECT_TRUE(
      (scaled(3, state, control, noise) - noiseless).isApprox(3.0 * (model(3, state, control, noise) - noiseless)));
  EXPECT_TRUE(scaled.diffusion()(3, state).isApprox(3.0 * Eigen::Matrix2d(model.diffusion()(3, state))));
  // The drift is the wrapped model's.
  EXPECT_EQ(scaled(3, state, control, Eigen::Vector2d::Zero()), noiseless);
}

}  // namespace
}  // namespace fbsde_traj_opt
