// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#include "fbsde_traj_opt/finite_horizon_lqr.hpp"

#include <array>
#include <cstddef>

#include <Eigen/Core>
#include <Eigen/LU>
#include <gtest/gtest.h>

#include "fbsde_traj_opt/composed_cost_sde_model.hpp"
#include "fbsde_traj_opt/composed_forward_sde_model.hpp"
#include "fbsde_traj_opt/const_diagonal_sde_diffusion_term.hpp"
#include "fbsde_traj_opt/const_linear_sde_control_drift_mat_term.hpp"
#include "fbsde_traj_opt/const_linear_sde_state_drift_term.hpp"
#include "fbsde_traj_opt/quadratic_regulator_sde_running_cost_term.hpp"
#include "fbsde_traj_opt/quadratic_regulator_sde_terminal_cost_term.hpp"
#include "fbsde_traj_opt/sde_term_concepts.hpp"

namespace fbsde_traj_opt {
namespace {

template <int N, int M>
using Model = ComposedForwardSdeModel<N,
                                      M,
                                      ConstLinearSdeStateDriftTerm<N>,
                                      ConstLinearSdeControlDriftMatTerm<N, M>,
                                      ConstDiagonalSdeDiffusionTerm<N>>;

template <int N, int M>
using CostModel =
    ComposedCostSdeModel<N, M, QuadraticRegulatorSdeRunningCostTerm<N, M>, QuadraticRegulatorSdeTerminalCostTerm<N>>;

static_assert(ConstLinearDriftForwardSdeModel<Model<2, 1>>);
static_assert(QuadraticRegulatorCostSdeModel<CostModel<2, 1>>);

template <int N, int M>
auto MakeModel(const Eigen::Matrix<double, N, N>& a, const Eigen::Matrix<double, N, M>& b) noexcept -> Model<N, M> {
  const auto diffusion = ConstDiagonalSdeDiffusionTerm<N>::Make(Eigen::Matrix<double, N, 1>::Constant(0.1));
  EXPECT_TRUE(diffusion.has_value());
  return {ConstLinearSdeStateDriftTerm<N>(a), ConstLinearSdeControlDriftMatTerm<N, M>(b), *diffusion};
}

template <int N, int M>
auto MakeCostModel(const Eigen::Matrix<double, N, N>& q,
                   const Eigen::Matrix<double, M, M>& r,
                   const Eigen::Matrix<double, N, M>& n,
                   const Eigen::Matrix<double, N, N>& f) noexcept -> CostModel<N, M> {
  return {QuadraticRegulatorSdeRunningCostTerm<N, M>(q, r, n), QuadraticRegulatorSdeTerminalCostTerm<N>(f)};
}

// The deterministic (noise-free) closed-loop cost of following `gains` from `initial_state`. The
// optimal-gain tests below compare this across gain sequences; the noise is irrelevant to them,
// because under certainty equivalence it shifts the cost of every policy by the same amount.
template <int N, int M, std::size_t NumControlStages>
auto ClosedLoopCost(const Eigen::Matrix<double, N, N>& transition_mat,
                    const Eigen::Matrix<double, N, M>& control_mat,
                    const Eigen::Matrix<double, N, N>& q,
                    const Eigen::Matrix<double, M, M>& r,
                    const Eigen::Matrix<double, N, M>& n,
                    const Eigen::Matrix<double, N, N>& f,
                    const std::array<Eigen::Matrix<double, M, N>, NumControlStages>& gains,
                    const Eigen::Matrix<double, N, 1>& initial_state) noexcept -> double {
  Eigen::Matrix<double, N, 1> state = initial_state;
  double cost = 0.0;
  for (std::size_t stage = 0; stage < NumControlStages; ++stage) {
    const Eigen::Matrix<double, M, 1> control = gains[stage] * state;
    cost += state.dot(q * state) + control.dot(r * control) + (2.0 * state.dot(n * control));
    state = (transition_mat * state) + (control_mat * control);
  }
  return cost + state.dot(f * state);
}

TEST(FiniteHorizonLqrTest, ScalarProblemMatchesTheHandCodedRiccatiRecursion) {
  constexpr std::size_t kNumStages = 6;
  constexpr double kDrift = 0.1;
  constexpr double kControlDrift = 0.5;
  constexpr double kStateCost = 2.0;
  constexpr double kControlCost = 1.0;
  constexpr double kTerminalCost = 3.0;

  const auto model = MakeModel<1, 1>(Eigen::Matrix<double, 1, 1>::Constant(kDrift),
                                     Eigen::Matrix<double, 1, 1>::Constant(kControlDrift));
  const auto cost_model = MakeCostModel<1, 1>(Eigen::Matrix<double, 1, 1>::Constant(kStateCost),
                                              Eigen::Matrix<double, 1, 1>::Constant(kControlCost),
                                              Eigen::Matrix<double, 1, 1>::Zero(),
                                              Eigen::Matrix<double, 1, 1>::Constant(kTerminalCost));

  const auto policy = SolveFiniteHorizonLqr<kNumStages>(model, cost_model);
  ASSERT_TRUE(policy.has_value()) << policy.error();

  // The same recursion written out in scalars. Note the transition is 1 + drift, not drift.
  const double transition = 1.0 + kDrift;
  double cost_to_go_hessian = kTerminalCost;
  std::array<double, kNumStages - 1> expected_gains{};
  for (std::size_t stage = kNumStages - 1; stage-- > 0;) {
    const double control_curvature = kControlCost + (kControlDrift * cost_to_go_hessian * kControlDrift);
    const double mixed_curvature = kControlDrift * cost_to_go_hessian * transition;
    expected_gains[stage] = -mixed_curvature / control_curvature;
    cost_to_go_hessian = kStateCost + (transition * cost_to_go_hessian * transition) -
                         ((mixed_curvature * mixed_curvature) / control_curvature);
  }

  for (std::size_t stage = 0; stage < kNumStages - 1; ++stage) {
    EXPECT_NEAR(policy->GainAtStage(stage)(0, 0), expected_gains[stage], 1e-12) << "at stage " << stage;
  }
}

TEST(FiniteHorizonLqrTest, ZeroStateAndTerminalCostsGiveZeroGains) {
  constexpr std::size_t kNumStages = 5;

  const auto model = MakeModel<2, 1>(Eigen::Matrix2d::Identity() * 0.1, Eigen::Vector2d(0.0, 0.1));
  const auto cost_model = MakeCostModel<2, 1>(Eigen::Matrix2d::Zero(),
                                              Eigen::Matrix<double, 1, 1>::Identity(),
                                              Eigen::Matrix<double, 2, 1>::Zero(),
                                              Eigen::Matrix2d::Zero());

  const auto policy = SolveFiniteHorizonLqr<kNumStages>(model, cost_model);
  ASSERT_TRUE(policy.has_value()) << policy.error();

  // Nothing penalizes the state, so the cheapest control is no control at all.
  for (std::size_t stage = 0; stage < kNumStages - 1; ++stage) {
    EXPECT_TRUE(policy->GainAtStage(stage).isZero(1e-14)) << "at stage " << stage;
  }
}

TEST(FiniteHorizonLqrTest, RejectsASingularControlCost) {
  constexpr std::size_t kNumStages = 4;

  const auto model = MakeModel<2, 1>(Eigen::Matrix2d::Zero(), Eigen::Vector2d(0.0, 0.1));
  // R = 0 with a zero terminal cost makes R + B' P B singular at the last step, so there is no
  // unique minimizing control.
  const auto cost_model = MakeCostModel<2, 1>(Eigen::Matrix2d::Identity(),
                                              Eigen::Matrix<double, 1, 1>::Zero(),
                                              Eigen::Matrix<double, 2, 1>::Zero(),
                                              Eigen::Matrix2d::Zero());

  const auto policy = SolveFiniteHorizonLqr<kNumStages>(model, cost_model);

  EXPECT_FALSE(policy.has_value());
}

// The defining property: no perturbation of any single stage's gain lowers the closed-loop cost.
TEST(FiniteHorizonLqrTest, NoPerturbationOfTheGainsLowersTheCost) {
  constexpr std::size_t kNumStages = 8;
  constexpr std::size_t kNumControlStages = kNumStages - 1;

  Eigen::Matrix2d a;
  a << 0.0, 0.1,  //
      0.0, 0.0;
  const Eigen::Vector2d b(0.0, 0.1);
  const Eigen::Matrix2d q = Eigen::Vector2d(1.0, 0.1).asDiagonal();
  const Eigen::Matrix<double, 1, 1> r = Eigen::Matrix<double, 1, 1>::Constant(0.05);
  const Eigen::Matrix<double, 2, 1> n = Eigen::Matrix<double, 2, 1>::Zero();
  const Eigen::Matrix2d f = Eigen::Vector2d(10.0, 1.0).asDiagonal();

  const auto policy = SolveFiniteHorizonLqr<kNumStages>(MakeModel<2, 1>(a, b), MakeCostModel<2, 1>(q, r, n, f));
  ASSERT_TRUE(policy.has_value()) << policy.error();

  const Eigen::Matrix2d transition_mat = Eigen::Matrix2d::Identity() + a;
  const Eigen::Vector2d initial_state(1.0, -0.5);

  std::array<Eigen::Matrix<double, 1, 2>, kNumControlStages> gains{};
  for (std::size_t stage = 0; stage < kNumControlStages; ++stage) {
    gains[stage] = policy->GainAtStage(stage);
  }

  const double optimal_cost =
      ClosedLoopCost<2, 1, kNumControlStages>(transition_mat, b, q, r, n, f, gains, initial_state);

  for (std::size_t stage = 0; stage < kNumControlStages; ++stage) {
    for (Eigen::Index component = 0; component < 2; ++component) {
      for (const double perturbation : {-0.05, 0.05}) {
        auto perturbed_gains = gains;
        perturbed_gains[stage](0, component) += perturbation;

        const double perturbed_cost =
            ClosedLoopCost<2, 1, kNumControlStages>(transition_mat, b, q, r, n, f, perturbed_gains, initial_state);

        EXPECT_GE(perturbed_cost, optimal_cost - 1e-12)
            << "stage " << stage << ", component " << component << ", perturbation " << perturbation;
      }
    }
  }
}

// Far from the terminal stage the gain settles onto the infinite-horizon gain; near it, it moves
// away, since there is no longer time left to recover the cost of an aggressive correction.
TEST(FiniteHorizonLqrTest, GainsSettleAwayFromTheTerminalStage) {
  constexpr std::size_t kNumStages = 120;

  Eigen::Matrix2d a;
  a << 0.0, 0.05,  //
      0.0, 0.0;
  const Eigen::Vector2d b(0.0, 0.05);

  const auto policy = SolveFiniteHorizonLqr<kNumStages>(MakeModel<2, 1>(a, b),
                                                        MakeCostModel<2, 1>(Eigen::Matrix2d::Identity(),
                                                                            Eigen::Matrix<double, 1, 1>::Identity(),
                                                                            Eigen::Matrix<double, 2, 1>::Zero(),
                                                                            Eigen::Matrix2d::Identity()));
  ASSERT_TRUE(policy.has_value()) << policy.error();

  // The recursion converges to the infinite-horizon gain geometrically rather than exactly, so
  // over this horizon the two earliest gains agree to about one part in 1e5, not to machine
  // precision. The terminal-adjacent gain, by contrast, differs from the settled one by most of
  // its own magnitude.
  EXPECT_TRUE(policy->GainAtStage(0).isApprox(policy->GainAtStage(1), 1e-4));
  EXPECT_FALSE(policy->GainAtStage(kNumStages - 2).isApprox(policy->GainAtStage(0), 0.5));
}

// A cross cost term is equivalent to a shifted control on a transformed problem: substituting
// u = v - R^{-1} N' x turns (A_eff, B, Q, R, N, F) into (A_eff - B R^{-1} N', B, Q - N R^{-1} N',
// R, 0, F), and the two optimal gains differ by exactly that shift.
TEST(FiniteHorizonLqrTest, CrossCostTermMatchesTheEquivalentShiftedProblem) {
  constexpr std::size_t kNumStages = 7;

  Eigen::Matrix2d a;
  a << 0.0, 0.1,  //
      -0.02, 0.0;
  const Eigen::Vector2d b(0.0, 0.1);
  const Eigen::Matrix2d q = Eigen::Vector2d(2.0, 1.0).asDiagonal();
  const Eigen::Matrix<double, 1, 1> r = Eigen::Matrix<double, 1, 1>::Constant(0.5);
  const Eigen::Matrix<double, 2, 1> n(0.3, -0.2);
  const Eigen::Matrix2d f = Eigen::Vector2d(4.0, 2.0).asDiagonal();

  const auto cross_policy = SolveFiniteHorizonLqr<kNumStages>(MakeModel<2, 1>(a, b), MakeCostModel<2, 1>(q, r, n, f));
  ASSERT_TRUE(cross_policy.has_value()) << cross_policy.error();

  const Eigen::Matrix<double, 1, 2> shift = r.inverse() * n.transpose();
  // The shifted problem's drift, again written as an increment on the state rather than as a
  // transition matrix, since that is what the model expects.
  const Eigen::Matrix2d shifted_a = a - (b * shift);
  const Eigen::Matrix2d shifted_q = q - (n * r.inverse() * n.transpose());

  const auto shifted_policy = SolveFiniteHorizonLqr<kNumStages>(
      MakeModel<2, 1>(shifted_a, b), MakeCostModel<2, 1>(shifted_q, r, Eigen::Matrix<double, 2, 1>::Zero(), f));
  ASSERT_TRUE(shifted_policy.has_value()) << shifted_policy.error();

  for (std::size_t stage = 0; stage < kNumStages - 1; ++stage) {
    const Eigen::Matrix<double, 1, 2> expected_gain = shifted_policy->GainAtStage(stage) - shift;
    EXPECT_TRUE(cross_policy->GainAtStage(stage).isApprox(expected_gain, 1e-10)) << "at stage " << stage;
  }
}

TEST(FiniteHorizonLqrTest, AsymmetricCostMatricesAreSymmetrized) {
  constexpr std::size_t kNumStages = 5;

  const Eigen::Vector2d b(0.0, 0.1);
  Eigen::Matrix2d asymmetric_q;
  asymmetric_q << 1.0, 0.6,  //
      -0.2, 1.0;
  const Eigen::Matrix2d symmetric_q = 0.5 * (asymmetric_q + asymmetric_q.transpose());

  const auto asymmetric_policy =
      SolveFiniteHorizonLqr<kNumStages>(MakeModel<2, 1>(Eigen::Matrix2d::Zero(), b),
                                        MakeCostModel<2, 1>(asymmetric_q,
                                                            Eigen::Matrix<double, 1, 1>::Identity(),
                                                            Eigen::Matrix<double, 2, 1>::Zero(),
                                                            Eigen::Matrix2d::Identity()));
  const auto symmetric_policy =
      SolveFiniteHorizonLqr<kNumStages>(MakeModel<2, 1>(Eigen::Matrix2d::Zero(), b),
                                        MakeCostModel<2, 1>(symmetric_q,
                                                            Eigen::Matrix<double, 1, 1>::Identity(),
                                                            Eigen::Matrix<double, 2, 1>::Zero(),
                                                            Eigen::Matrix2d::Identity()));
  ASSERT_TRUE(asymmetric_policy.has_value() && symmetric_policy.has_value());

  for (std::size_t stage = 0; stage < kNumStages - 1; ++stage) {
    EXPECT_TRUE(asymmetric_policy->GainAtStage(stage).isApprox(symmetric_policy->GainAtStage(stage), 1e-12))
        << "at stage " << stage;
  }
}

TEST(FiniteHorizonLqrTest, SolvedPolicySatisfiesTheControlPolicyConcept) {
  constexpr std::size_t kNumStages = 5;

  using Policy = FiniteHorizonLqrPolicy<Model<2, 1>, kNumStages - 1>;
  static_assert(SdeControlPolicyTerm<Policy, Eigen::Vector2d, Eigen::Matrix<double, 1, 1>>);
  static_assert(Policy::kNumControlStages == kNumStages - 1);

  SUCCEED();
}

}  // namespace
}  // namespace fbsde_traj_opt
