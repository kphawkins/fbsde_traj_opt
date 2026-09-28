// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#include "fbsde_traj_opt/taylor_q_function_l1_policy.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "fbsde_traj_opt/composed_forward_sde_model.hpp"
#include "fbsde_traj_opt/const_diagonal_sde_diffusion_term.hpp"
#include "fbsde_traj_opt/const_linear_sde_control_drift_mat_term.hpp"
#include "fbsde_traj_opt/const_linear_sde_state_drift_term.hpp"
#include "fbsde_traj_opt/polynomial_value_function_approx.hpp"
#include "fbsde_traj_opt/sde_term_concepts.hpp"
#include "fbsde_traj_opt/utils/result.hpp"

namespace fbsde_traj_opt {
namespace {

template <int M>
using Model = ComposedForwardSdeModel<2,
                                      M,
                                      ConstLinearSdeStateDriftTerm<2>,
                                      ConstLinearSdeControlDriftMatTerm<2, M>,
                                      ConstDiagonalSdeDiffusionTerm<2>>;
using Cubic = PolynomialValueFunctionApprox<2, 3>;
using Quadratic = PolynomialValueFunctionApprox<2, 2>;

template <int M>
using Policy = TaylorQFunctionL1Policy<2, M, Model<M>, Cubic>;

static_assert(SdeControlPolicyTerm<Policy<1>, Eigen::Vector2d, Eigen::Matrix<double, 1, 1>>);
static_assert(SdeControlPolicyTerm<Policy<2>, Eigen::Vector2d, Eigen::Vector2d>);

template <int M>
auto MakeModel(const Eigen::Matrix<double, 2, M>& control_matrix) -> Model<M> {
  Eigen::Matrix2d state_matrix;
  state_matrix << 0.0, 0.1,  //
      -0.2, -0.05;
  const Result<ConstDiagonalSdeDiffusionTerm<2>> diffusion =
      ConstDiagonalSdeDiffusionTerm<2>::Make(Eigen::Vector2d(0.05, 0.1));
  EXPECT_TRUE(diffusion.has_value());
  return Model<M>(ConstLinearSdeStateDriftTerm<2>(state_matrix),
                  ConstLinearSdeControlDriftMatTerm<2, M>(control_matrix),
                  *diffusion);
}

// A cubic value function, so the Taylor model the minimizer proposes candidates from is *not*
// exact and the exact rescoring has to earn its keep.
auto MakeCubicTable() -> std::vector<Cubic> {
  Cubic::ParameterVector parameters;
  parameters << 0.3, 0.2, -0.1, 1.0, 0.25, 0.8, 0.05, -0.04, 0.03, 0.02;
  const Result<Cubic> cubic = Cubic::Make(parameters, Eigen::Vector2d(0.2, -0.1), Eigen::Vector2d(1.0, 1.2));
  EXPECT_TRUE(cubic.has_value());
  return {*cubic, *cubic};
}

TEST(TaylorQFunctionL1PolicyTest, SingleControlMatchesBruteForceMinimum) {
  const Model<1> model = MakeModel<1>(Eigen::Vector2d(0.0, 0.6));
  const std::vector<Cubic> table = MakeCubicTable();
  for (const double l1_weight : {0.0, 0.05, 0.3, 2.0}) {
    const Result<Policy<1>> policy = Policy<1>::Make(model,
                                                     l1_weight,
                                                     Eigen::Matrix<double, 1, 1>(-1.0),
                                                     Eigen::Matrix<double, 1, 1>(1.0),
                                                     std::span<const Cubic>(table));
    ASSERT_TRUE(policy.has_value()) << policy.error();

    for (const Eigen::Vector2d& state : {Eigen::Vector2d(1.0, 0.5),
                                         Eigen::Vector2d(-0.8, 0.2),
                                         Eigen::Vector2d(0.1, -1.5),
                                         Eigen::Vector2d(0.3, 0.1)}) {
      double brute_minimum = std::numeric_limits<double>::infinity();
      for (int step = -2000; step <= 2000; ++step) {
        const Eigen::Matrix<double, 1, 1> control(static_cast<double>(step) / 2000.0);
        brute_minimum = std::min(brute_minimum, policy->QValue(0, state, control));
      }
      const Policy<1>::Decision decision = policy->Decide(0, state);
      // The policy may do better than a grid; it must never do worse by more than the grid's
      // resolution could explain.
      EXPECT_LE(decision.q_value, brute_minimum + 1e-6) << "l1 " << l1_weight << " state " << state.transpose();
      EXPECT_NEAR(decision.q_value, policy->QValue(0, state, decision.control), 1e-12);
      EXPECT_LE(std::abs(decision.control[0]), 1.0);
    }
  }
}

TEST(TaylorQFunctionL1PolicyTest, LargeL1WeightMakesTheControlCoast) {
  const Model<1> model = MakeModel<1>(Eigen::Vector2d(0.0, 0.6));
  const std::vector<Cubic> table = MakeCubicTable();
  const Result<Policy<1>> policy = Policy<1>::Make(
      model, 100.0, Eigen::Matrix<double, 1, 1>(-1.0), Eigen::Matrix<double, 1, 1>(1.0), std::span<const Cubic>(table));
  ASSERT_TRUE(policy.has_value()) << policy.error();

  EXPECT_EQ(policy->Decide(0, Eigen::Vector2d(1.0, 0.5)).control[0], 0.0);
}

TEST(TaylorQFunctionL1PolicyTest, QuadraticValueGivesTheSoftThresholdedMinimizer) {
  // With a quadratic value function the Q-function in u is exactly c|u| + b u + a u^2 / 2 plus a
  // constant, whose minimizer on [-1, 1] is the soft threshold -sign(b) max(|b| - c, 0) / a,
  // clipped.
  const Eigen::Vector2d control_column(0.0, 0.5);
  const Model<1> model = MakeModel<1>(control_column);
  const Result<Quadratic> base =
      Quadratic::Make(Quadratic::ParameterVector::Zero(), Eigen::Vector2d::Zero(), Eigen::Vector2d::Ones());
  ASSERT_TRUE(base.has_value()) << base.error();
  Quadratic quadratic = *base;
  Eigen::Matrix2d metric;
  metric << 1.0, 0.2,  //
      0.2, 0.7;
  ASSERT_TRUE(quadratic.SetParameters(quadratic.QuadraticParameters(metric, 0.0)).has_value());
  const std::vector<Quadratic> table{quadratic, quadratic};

  using QuadraticPolicy = TaylorQFunctionL1Policy<2, 1, Model<1>, Quadratic>;
  constexpr double kL1Weight = 0.1;
  const Result<QuadraticPolicy> policy = QuadraticPolicy::Make(model,
                                                               kL1Weight,
                                                               Eigen::Matrix<double, 1, 1>(-1.0),
                                                               Eigen::Matrix<double, 1, 1>(1.0),
                                                               std::span<const Quadratic>(table));
  ASSERT_TRUE(policy.has_value()) << policy.error();

  const Eigen::Vector2d state(0.4, 0.3);
  const Eigen::Vector2d uncontrolled = state + (model.state_drift()(0, state));
  const double slope = (2.0 * metric * uncontrolled).dot(control_column);
  const double curvature = control_column.dot(2.0 * metric * control_column);
  const double expected =
      std::clamp(-std::copysign(std::max(std::abs(slope) - kL1Weight, 0.0), slope) / curvature, -1.0, 1.0);

  EXPECT_NEAR(policy->Decide(0, state).control[0], expected, 1e-12);
}

TEST(TaylorQFunctionL1PolicyTest, TwoControlsNeverDoWorseThanAnyVertexOrZero) {
  Eigen::Matrix2d control_matrix;
  control_matrix << 0.3, 0.0,  //
      0.2, 0.5;
  const Model<2> model = MakeModel<2>(control_matrix);
  const std::vector<Cubic> table = MakeCubicTable();
  const Result<Policy<2>> policy = Policy<2>::Make(
      model, 0.05, Eigen::Vector2d(-1.0, -1.0), Eigen::Vector2d(1.0, 1.0), std::span<const Cubic>(table));
  ASSERT_TRUE(policy.has_value()) << policy.error();

  const Eigen::Vector2d state(0.7, -0.4);
  const double chosen = policy->Decide(0, state).q_value;
  for (const double first : {-1.0, 0.0, 1.0}) {
    for (const double second : {-1.0, 0.0, 1.0}) {
      EXPECT_LE(chosen, policy->QValue(0, state, Eigen::Vector2d(first, second)) + 1e-12);
    }
  }
}

TEST(TaylorQFunctionL1PolicyTest, MakeRejectsBoundsExcludingZeroAndShortTables) {
  const Model<1> model = MakeModel<1>(Eigen::Vector2d(0.0, 1.0));
  const std::vector<Cubic> table = MakeCubicTable();
  EXPECT_FALSE(
      Policy<1>::Make(
          model, 0.1, Eigen::Matrix<double, 1, 1>(0.5), Eigen::Matrix<double, 1, 1>(1.0), std::span<const Cubic>(table))
          .has_value());
  EXPECT_FALSE(Policy<1>::Make(model,
                               -0.1,
                               Eigen::Matrix<double, 1, 1>(-1.0),
                               Eigen::Matrix<double, 1, 1>(1.0),
                               std::span<const Cubic>(table))
                   .has_value());
  EXPECT_FALSE(Policy<1>::Make(model,
                               0.1,
                               Eigen::Matrix<double, 1, 1>(-1.0),
                               Eigen::Matrix<double, 1, 1>(1.0),
                               std::span<const Cubic>(table).first(1))
                   .has_value());
}

}  // namespace
}  // namespace fbsde_traj_opt
