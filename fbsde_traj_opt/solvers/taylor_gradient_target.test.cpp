// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#include "fbsde_traj_opt/solvers/taylor_gradient_target.hpp"

#include <array>
#include <cstddef>
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "fbsde_traj_opt/dynamics/const_diagonal_sde_diffusion_term.hpp"
#include "fbsde_traj_opt/dynamics/const_linear_sde_control_drift_mat_term.hpp"
#include "fbsde_traj_opt/dynamics/const_linear_sde_state_drift_term.hpp"
#include "fbsde_traj_opt/policies/taylor_q_function_l1_policy.hpp"
#include "fbsde_traj_opt/sde/composed_forward_sde_model.hpp"
#include "fbsde_traj_opt/value_function/polynomial_value_function_approx.hpp"

namespace fbsde_traj_opt {
namespace {

using Model = ComposedForwardSdeModel<2,
                                      1,
                                      ConstLinearSdeStateDriftTerm<2>,
                                      ConstLinearSdeControlDriftMatTerm<2, 1>,
                                      ConstDiagonalSdeDiffusionTerm<2>>;
using Quadratic = PolynomialValueFunctionApprox<2, 2>;
using Control = Eigen::Matrix<double, 1, 1>;

constexpr double kTimeStep = 0.1;

// The double integrator, stepped exactly.
auto MakeModel() -> Model {
  const Eigen::Matrix2d drift = (Eigen::Matrix2d() << 0.0, kTimeStep, 0.0, 0.0).finished();
  return Model(ConstLinearSdeStateDriftTerm<2>(drift),
               ConstLinearSdeControlDriftMatTerm<2, 1>(Eigen::Vector2d(0.5 * kTimeStep * kTimeStep, kTimeStep)),
               *ConstDiagonalSdeDiffusionTerm<2>::Make(Eigen::Vector2d(0.01, 0.05)));
}

// A quadratic with a linear part and a cross term, off-center, so no simplification hides a mistake.
auto MakeQuadratic() -> Quadratic {
  Quadratic::ParameterVector parameters;
  parameters << 0.2, -0.5, 0.8, 1.6, 0.4, 0.9;
  return *Quadratic::Make(parameters, Eigen::Vector2d(0.3, -0.2), Eigen::Vector2d(1.2, 0.8));
}

TEST(TaylorGradientTargetTest, IsTheTransposedStepJacobianTimesTheNextGradientExactlyForLinearDynamics) {
  const Model model = MakeModel();
  const Quadratic next = MakeQuadratic();
  const Eigen::Matrix2d step_jacobian = (Eigen::Matrix2d() << 1.0, kTimeStep, 0.0, 1.0).finished();
  const Eigen::Vector2d state(0.7, -0.4);
  const Control control(0.35);
  const Eigen::Vector2d landing = model(0, state, control, Eigen::Vector2d::Zero());
  const Eigen::Vector2d expected = step_jacobian.transpose() * next.Gradient(landing);
  EXPECT_TRUE(TaylorGradientTarget(model, 0, state, control, next).isApprox(expected, 1e-8));
}

// The envelope argument, checked: the gradient target at the policy's own control matches a
// finite difference of the policy's minimized Q-value, wherever the control does not switch
// between the two sides of the difference.
TEST(TaylorGradientTargetTest, MatchesTheDerivativeOfTheMinimizedQValueAwayFromSwitches) {
  const Model model = MakeModel();
  const std::vector<Quadratic> table(2, MakeQuadratic());
  const auto policy =
      *TaylorQFunctionL1Policy<2, 1, Model, Quadratic>::Make(model, 0.02, Control(-1.0), Control(1.0), table);
  constexpr double kStep = 1e-6;
  std::size_t checked = 0;
  for (int index = 0; index < 40; ++index) {
    const Eigen::Vector2d state(-1.5 + (0.075 * index), 1.0 - (0.05 * index));
    const auto decision = policy.Decide(0, state);
    const Eigen::Vector2d target = TaylorGradientTarget(model, 0, state, decision.control, table[1]);
    Eigen::Vector2d difference;
    bool switched = false;
    for (Eigen::Index axis = 0; axis < 2; ++axis) {
      const Eigen::Vector2d offset = kStep * Eigen::Vector2d::Unit(axis);
      const auto forward = policy.Decide(0, state + offset);
      const auto backward = policy.Decide(0, state - offset);
      // An interior control moves with the state; one at a bound or at zero switching does not count.
      const bool pinned = std::abs(decision.control[0]) == 1.0 || decision.control[0] == 0.0;
      switched = switched || (pinned && (forward.control != decision.control || backward.control != decision.control));
      difference[axis] = (forward.q_value - backward.q_value) / (2.0 * kStep);
    }
    if (switched) {
      continue;
    }
    ++checked;
    EXPECT_TRUE(target.isApprox(difference, 1e-5))
        << "at state " << state.transpose() << ", control " << decision.control[0] << ": " << target.transpose()
        << " vs " << difference.transpose();
  }
  EXPECT_GT(checked, 30U);
}

}  // namespace
}  // namespace fbsde_traj_opt
