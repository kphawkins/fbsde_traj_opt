// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#include "fbsde_traj_opt/solvers/box_ilqr_solver.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

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

constexpr std::size_t kNumStages = 65;
constexpr double kTimeStep = 4.0 / 64.0;
constexpr double kL1Weight = kTimeStep;

// A unit mass on a line: the L1 double integrator's dynamics, by Euler.
auto MakeModel() noexcept -> Model {
  const Eigen::Matrix2d drift = (Eigen::Matrix2d() << 0.0, kTimeStep, 0.0, 0.0).finished();
  const Eigen::Vector2d control(0.5 * kTimeStep * kTimeStep, kTimeStep);
  const auto diffusion = ConstDiagonalSdeDiffusionTerm<2>::Make(Eigen::Vector2d::Constant(0.01));
  EXPECT_TRUE(diffusion.has_value());
  return {ConstLinearSdeStateDriftTerm<2>(drift), ConstLinearSdeControlDriftMatTerm<2, 1>(control), *diffusion};
}

auto TerminalCostMat() noexcept -> Eigen::Matrix2d { return Eigen::Vector2d(20.0, 10.0).asDiagonal(); }

using Solver = BoxIlqrSolver<2, kNumStages, Model>;

auto MakeSolver(const std::vector<double>& controls) noexcept -> Result<Solver> {
  return Solver::Make(
      MakeModel(), kL1Weight, TerminalCostMat(), Eigen::Vector2d(2.0, 1.0), -1.0, 1.0, controls, BoxIlqrOptions<double>{});
}

TEST(BoxIlqrSolverTest, RejectsAnEmptyBoxAndAWrongHorizon) {
  const std::vector<double> controls(kNumStages - 1, 0.0);
  EXPECT_FALSE(Solver::Make(MakeModel(),
                            kL1Weight,
                            TerminalCostMat(),
                            Eigen::Vector2d::Zero(),
                            1.0,
                            -1.0,
                            controls,
                            BoxIlqrOptions<double>{})
                   .has_value());
  EXPECT_FALSE(MakeSolver(std::vector<double>(kNumStages - 2, 0.0)).has_value());
}

// From rest at the initial controls of zero, the solver brings the mass most of the way home and
// lands on a bang-off-bang sequence inside the box.
TEST(BoxIlqrSolverTest, FindsABangOffBangMinimumFuelSequence) {
  Result<Solver> solver = MakeSolver(std::vector<double>(kNumStages - 1, 0.0));
  ASSERT_TRUE(solver.has_value());
  const double initial = solver->Cost();
  double previous = initial;
  for (std::size_t iteration = 0; iteration < 100; ++iteration) {
    const BoxIlqrIterationReport<double> report = solver->Iterate();
    EXPECT_LE(report.smoothed_cost_after, report.smoothed_cost_before);
    previous = report.cost;
    if (report.converged) {
      break;
    }
  }
  EXPECT_LT(previous, 0.25 * initial);
  std::size_t saturated_or_off = 0;
  for (const double control : solver->NominalControls()) {
    EXPECT_GE(control, -1.0);
    EXPECT_LE(control, 1.0);
    saturated_or_off += (std::abs(control) < 0.05 || std::abs(control) > 0.95) ? 1 : 0;
  }
  EXPECT_GE(saturated_or_off, (8 * (kNumStages - 1)) / 10);
}

// On the nominal trajectory the policy plays the nominal controls.
TEST(BoxIlqrSolverTest, PolicyReproducesTheNominalTrajectory) {
  Result<Solver> solver = MakeSolver(std::vector<double>(kNumStages - 1, 0.3));
  ASSERT_TRUE(solver.has_value());
  for (std::size_t iteration = 0; iteration < 5; ++iteration) {
    (void)solver->Iterate();
  }
  const Solver::Policy policy = solver->CurrentPolicy();
  for (std::size_t stage = 0; stage + 1 < kNumStages; ++stage) {
    EXPECT_NEAR(policy(stage, solver->NominalStates()[stage])[0], solver->NominalControls()[stage], 1e-12);
  }
}

// Replanning from a state on the nominal trajectory changes nothing: the remainder of an optimal
// plan is optimal for the remaining problem.
TEST(BoxIlqrSolverTest, ReplanningFromTheNominalTrajectoryKeepsThePlan) {
  Result<Solver> solver = MakeSolver(std::vector<double>(kNumStages - 1, 0.0));
  ASSERT_TRUE(solver.has_value());
  for (std::size_t iteration = 0; iteration < 200 && !solver->Iterate().converged; ++iteration) {
  }
  const std::vector<double> plan(solver->NominalControls().begin(), solver->NominalControls().end());
  constexpr std::size_t kStage = 20;
  solver->Replan(kStage, solver->NominalStates()[kStage]);
  for (std::size_t iteration = 0; iteration < 3; ++iteration) {
    (void)solver->Iterate();
  }
  for (std::size_t stage = kStage; stage + 1 < kNumStages; ++stage) {
    EXPECT_NEAR(solver->NominalControls()[stage], plan[stage], 1e-3);
  }
}

}  // namespace
}  // namespace fbsde_traj_opt
