// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#include "examples/l1_grid_dp_reference.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "fbsde_traj_opt/costs/l1_control_sde_running_cost_term.hpp"
#include "examples/l1_benchmark_models.hpp"

namespace fbsde_traj_opt::examples {
namespace {

using Grid = L1GridDpReference<L1DoubleIntegrator>;

auto Benchmark() -> L1DoubleIntegrator {
  auto benchmark = MakeL1DoubleIntegrator();
  EXPECT_TRUE(benchmark.has_value());
  return *benchmark;
}

auto Options(std::size_t controls) -> Grid::Options {
  Grid::Options options;
  options.lower = Eigen::Vector2d(-2.5, -3.5);
  options.upper = Eigen::Vector2d(4.5, 2.5);
  options.cells_per_axis = 140;
  options.num_controls = controls;
  options.quadrature_order = 3;
  return options;
}

// The noise-free step's matrices and the noise covariance of the benchmark, read off the model.
struct Linear {
  Eigen::Matrix2d transition;
  Eigen::Vector2d column;
  Eigen::Matrix2d noise;
};

auto Linearize(const L1DoubleIntegrator& benchmark) -> Linear {
  using Control = Eigen::Matrix<double, 1, 1>;
  Linear linear;
  for (int axis = 0; axis < 2; ++axis) {
    linear.transition.col(axis) =
        benchmark.forward_model(0, Eigen::Vector2d::Unit(axis), Control::Zero(), Eigen::Vector2d::Zero());
  }
  linear.column = benchmark.forward_model(0, Eigen::Vector2d::Zero(), Control::Ones(), Eigen::Vector2d::Zero());
  Eigen::Matrix2d sigma;
  for (int axis = 0; axis < 2; ++axis) {
    sigma.col(axis) = benchmark.forward_model(0, Eigen::Vector2d::Zero(), Control::Zero(), Eigen::Vector2d::Unit(axis));
  }
  linear.noise = sigma * sigma.transpose();
  return linear;
}

// One Bellman step: V_0(x) = min over the control grid of c|u| + E[V_1(A x + b u + Sigma w)], with
// V_1 the quadratic terminal cost, whose Gaussian expectation is exact: V_1(mean) + tr(F S). Two
// bilinear interpolations stand between the grid and that: of V_1 inside the expectation, and of
// V_0 at a state between nodes. Each overestimates a convex function by at most sum_i h_i^2 / 8
// times its second derivative along axis i -- for V_1 that is 2F, and for V_0 at most 2 A'FA, since
// the minimum over controls can only lower its curvature.
TEST(L1GridDpReferenceTest, OneStepIsTheBellmanBackupOfTheTerminalCost) {
  const L1DoubleIntegrator benchmark = Benchmark();
  constexpr std::size_t kControls = 41;
  auto grid = Grid::Make(benchmark, 2, Options(kControls));
  ASSERT_TRUE(grid.has_value());
  const Linear linear = Linearize(benchmark);
  const Eigen::Matrix2d& f = benchmark.terminal_cost_mat;
  const Eigen::Vector2d spacing = Eigen::Vector2d(7.0, 6.0) / 140.0;
  const Eigen::Matrix2d stepped = linear.transition.transpose() * f * linear.transition;
  const auto bound = [&](const Eigen::Matrix2d& m) {
    return 0.125 * 2.0 * ((spacing[0] * spacing[0] * m(0, 0)) + (spacing[1] * spacing[1] * m(1, 1)));
  };
  const double tolerance = bound(f) + bound(stepped);
  for (const Eigen::Vector2d& state : {Eigen::Vector2d(2.0, 1.0), Eigen::Vector2d(0.5, -0.7), Eigen::Vector2d(-1.0, 0.3)}) {
    double expected = std::numeric_limits<double>::infinity();
    double expected_control = 0.0;
    for (std::size_t index = 0; index < kControls; ++index) {
      const double u = -1.0 + (2.0 * static_cast<double>(index) / static_cast<double>(kControls - 1));
      const Eigen::Vector2d mean = (linear.transition * state) + (u * linear.column);
      const double value = (benchmark.running_cost.L1Weight() * std::abs(u)) + mean.dot(f * mean) +
                           f.cwiseProduct(linear.noise).sum();
      if (value < expected) {
        expected = value;
        expected_control = u;
      }
    }
    EXPECT_NEAR(grid->Value(0, state), expected, tolerance + 1e-9) << state.transpose();
    EXPECT_NEAR((*grid)(0, state)[0], expected_control, 0.05 + 1e-12) << state.transpose();
  }
}

// The whole horizon with fuel so dear that the policy coasts: V_k(x) = x' P_k x + n_k with
// P_k = A' P_{k+1} A, P_K = F, and n_k = n_{k+1} + tr(P_{k+1} S). The grid compounds one bilinear
// interpolation per stage, each an overestimate of order h^2, so it is checked for what it promises:
// it lies above the truth, and halving the cell size cuts the error about fourfold. (Measured: 0.59
// at 140 cells, 0.17 at 280, 0.043 at 560, on a value of 6.78.)
TEST(L1GridDpReferenceTest, CoastingValueConvergesToTheLinearRecursionAtSecondOrder) {
  L1DoubleIntegrator benchmark = Benchmark();
  auto expensive = L1ControlSdeRunningCostTerm<2, 1>::Make(1e6);
  ASSERT_TRUE(expensive.has_value());
  benchmark.running_cost = *expensive;
  const Linear linear = Linearize(benchmark);
  Eigen::Matrix2d p = benchmark.terminal_cost_mat;
  double noise_total = 0.0;
  for (std::size_t stage = kL1DoubleIntegratorNumStages - 1; stage-- > 0;) {
    noise_total += p.cwiseProduct(linear.noise).sum();
    p = linear.transition.transpose() * p * linear.transition;
  }
  const Eigen::Vector2d state(0.5, -0.2);
  const double expected = state.dot(p * state) + noise_total;
  std::array<double, 2> errors{};
  for (std::size_t index = 0; index < 2; ++index) {
    Grid::Options options = Options(3);
    options.cells_per_axis = index == 0 ? 140 : 280;
    auto grid = Grid::Make(benchmark, kL1DoubleIntegratorNumStages, options);
    ASSERT_TRUE(grid.has_value());
    errors[index] = grid->Value(0, state) - expected;
    EXPECT_GT(errors[index], 0.0);
    EXPECT_EQ((*grid)(0, state)[0], 0.0);
  }
  EXPECT_LT(errors[1], errors[0] / 3.0);
  EXPECT_LT(errors[1], 0.03 * expected);
}

}  // namespace
}  // namespace fbsde_traj_opt::examples
