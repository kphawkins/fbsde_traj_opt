// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#include "fbsde_traj_opt/solvers/mppi_solver.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "fbsde_traj_opt/costs/l1_control_sde_running_cost_term.hpp"
#include "fbsde_traj_opt/costs/quadratic_regulator_sde_terminal_cost_term.hpp"
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

auto MakeModel() noexcept -> Model {
  const Eigen::Matrix2d drift = (Eigen::Matrix2d() << 0.0, kTimeStep, 0.0, 0.0).finished();
  const Eigen::Vector2d control(0.5 * kTimeStep * kTimeStep, kTimeStep);
  const auto diffusion = ConstDiagonalSdeDiffusionTerm<2>::Make(Eigen::Vector2d(0.01, 0.1) * std::sqrt(kTimeStep));
  EXPECT_TRUE(diffusion.has_value());
  return {ConstLinearSdeStateDriftTerm<2>(drift), ConstLinearSdeControlDriftMatTerm<2, 1>(control), *diffusion};
}

using Solver = MppiSolver<2,
                          kNumStages,
                          Model,
                          L1ControlSdeRunningCostTerm<2, 1>,
                          QuadraticRegulatorSdeTerminalCostTerm<2>>;

auto MakeSolver(const MppiOptions<double>& options, std::uint64_t seed) noexcept -> Result<Solver> {
  const auto running_cost = L1ControlSdeRunningCostTerm<2, 1>::Make(kTimeStep);
  EXPECT_TRUE(running_cost.has_value());
  const std::vector<double> controls(kNumStages - 1, 0.0);
  return Solver::Make(MakeModel(),
                      *running_cost,
                      QuadraticRegulatorSdeTerminalCostTerm<2>(Eigen::Vector2d(20.0, 10.0).asDiagonal()),
                      Eigen::Vector2d(2.0, 1.0),
                      Eigen::Matrix2d::Identity() * 0.03,
                      -1.0,
                      1.0,
                      controls,
                      options,
                      seed);
}

TEST(MppiSolverTest, RejectsTooFewSamplesAndABadEffectiveSampleFraction) {
  MppiOptions<double> options;
  options.num_samples = 1;
  EXPECT_FALSE(MakeSolver(options, 1).has_value());
  options.num_samples = 16;
  options.effective_sample_fraction = 1.0;
  EXPECT_FALSE(MakeSolver(options, 1).has_value());
}

// The temperature is chosen so that the update averages the requested share of the samples.
TEST(MppiSolverTest, HitsTheTargetEffectiveSampleSize) {
  MppiOptions<double> options;
  options.num_samples = 512;
  options.effective_sample_fraction = 0.1;
  Result<Solver> solver = MakeSolver(options, 5);
  ASSERT_TRUE(solver.has_value());
  const MppiIterationReport<double> report = solver->Iterate();
  EXPECT_NEAR(report.effective_samples, 51.2, 0.5);
}

TEST(MppiSolverTest, BringsTheCostDownAndStaysInTheBox) {
  Result<Solver> solver = MakeSolver(MppiOptions<double>{}, 7);
  ASSERT_TRUE(solver.has_value());
  const double initial = solver->Iterate().nominal_sample_cost;
  double last = initial;
  for (std::size_t iteration = 0; iteration < 60; ++iteration) {
    last = solver->Iterate().mean_sample_cost;
  }
  EXPECT_LT(last, 0.25 * initial);
  for (const double control : solver->NominalControls()) {
    EXPECT_GE(control, -1.0);
    EXPECT_LE(control, 1.0);
  }
}

TEST(MppiSolverTest, IsReproducibleFromItsSeedWhetherOrNotItRunsInParallel) {
  MppiOptions<double> parallel;
  MppiOptions<double> serial;
  serial.parallel = false;
  Result<Solver> first = MakeSolver(parallel, 3);
  Result<Solver> second = MakeSolver(serial, 3);
  ASSERT_TRUE(first.has_value() && second.has_value());
  for (std::size_t iteration = 0; iteration < 5; ++iteration) {
    (void)first->Iterate();
    (void)second->Iterate();
  }
  for (std::size_t stage = 0; stage + 1 < kNumStages; ++stage) {
    EXPECT_DOUBLE_EQ(first->NominalControls()[stage], second->NominalControls()[stage]);
  }
}

}  // namespace
}  // namespace fbsde_traj_opt
