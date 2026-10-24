// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#ifndef FBSDE_TRAJ_OPT_EXAMPLES_ILQR_SEEDED_TABLE_HPP_
#define FBSDE_TRAJ_OPT_EXAMPLES_ILQR_SEEDED_TABLE_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <span>
#include <utility>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "fbsde_traj_opt/solvers/box_ilqr_solver.hpp"
#include "fbsde_traj_opt/utils/result.hpp"
#include "fbsde_traj_opt/value_function/polynomial_value_function_approx.hpp"
#include "fbsde_traj_opt/value_function/soft_min_quadratic_value_function_approx.hpp"

namespace fbsde_traj_opt::examples {

// Turns iLQR's quadratic value expansions (BoxIlqrSolver::ValueExpansions) into a DT-FBSDE value
// function table, one entry per control stage, for DtFbsdeIterativeSolver::SeedValueFunctions().
//
// At every stage the expansion is the quadratic
//
//   q(x) = v + g' (x - xbar) + 1/2 (x - xbar)' H (x - xbar)
//
// about the nominal state `xbar`, and each representation takes it as it can:
//
//   * PolynomialValueFunctionApprox of degree two or more holds any quadratic exactly. The
//     coefficients are fitted by least squares on a tensor grid over the basis's region of interest,
//     which is exact to rounding because q is in the span of the basis.
//   * SoftMinQuadraticValueFunctionApprox holds only convex bowls. H is positive semidefinite (see
//     BoxIlqrValueExpansion), but often barely: on a coasting arc of an L1 plan it is flat along the
//     directions the control moves the state in. Its eigenvalues are therefore floored at
//     `metric_floor`, absolute and relative to the largest, before the bowl is formed as
//
//       q(x) ~ (x - b)' A (x - b) + c,   A = H / 2,   b = xbar - A^-1 g / 2,   c = v - g' A^-1 g / 4,
//
//     which keeps the value and the slope at `xbar` exactly and changes only the curvature along the
//     floored directions. The components are that one bowl with its center moved along the flattest
//     direction of A, evenly over a value change of up to `center_spread`, so that a later fit can
//     move them apart: identical components receive identical gradients and would never separate.
struct IlqrSeedOptions {
  // The soft-min model's floor on A's eigenvalues: the larger of this and `relative_metric_floor`
  // times the largest one.
  double metric_floor = 1e-3;
  double relative_metric_floor = 1e-4;
  // How far apart the soft-min model's components start, as the value A assigns to the farthest
  // center's offset from the bowl's.
  double center_spread = 0.5;
};

template <int N, int Degree>
auto IlqrSeededTable(const PolynomialValueFunctionApprox<N, Degree>& basis,
                     std::span<const BoxIlqrValueExpansion<N>> expansions,
                     const IlqrSeedOptions& /*options*/) noexcept
    -> Result<std::vector<PolynomialValueFunctionApprox<N, Degree>>> {
  static_assert(Degree >= 2, "a polynomial seeded from iLQR must hold a quadratic.");
  using ValueFunction = PolynomialValueFunctionApprox<N, Degree>;
  using State = typename ValueFunction::State;
  // Degree + 1 points per axis determine a polynomial of that degree in each coordinate.
  constexpr int kPointsPerAxis = Degree + 1;
  constexpr int kPoints = [] {
    int points = 1;
    for (int axis = 0; axis < N; ++axis) {
      points *= kPointsPerAxis;
    }
    return points;
  }();
  RESULT_ASSERT(!expansions.empty(), "IlqrSeededTable: there are no expansions to seed from.");

  ValueFunctionStateBatch<State, kPoints> points;
  for (std::size_t point = 0; point < static_cast<std::size_t>(kPoints); ++point) {
    std::size_t remainder = point;
    State unit;
    for (std::size_t axis = 0; axis < static_cast<std::size_t>(N); ++axis) {
      const std::size_t index = remainder % static_cast<std::size_t>(kPointsPerAxis);
      remainder /= static_cast<std::size_t>(kPointsPerAxis);
      unit[static_cast<Eigen::Index>(axis)] =
          -1.0 + (2.0 * static_cast<double>(index) / static_cast<double>(kPointsPerAxis - 1));
    }
    points.col(static_cast<Eigen::Index>(point)) = basis.Center() + basis.Scale().cwiseProduct(unit);
  }
  const ValueFunctionValueBatch<State, kPoints> weights = ValueFunctionValueBatch<State, kPoints>::Ones();

  std::vector<ValueFunction> table(expansions.size() - 1, basis);
  for (std::size_t stage = 0; stage + 1 < expansions.size(); ++stage) {
    const BoxIlqrValueExpansion<N>& expansion = expansions[stage];
    ValueFunctionValueBatch<State, kPoints> targets;
    for (std::size_t point = 0; point < static_cast<std::size_t>(kPoints); ++point) {
      const auto column = static_cast<Eigen::Index>(point);
      const State offset = State(points.col(column)) - expansion.state;
      targets[column] =
          expansion.value + expansion.gradient.dot(offset) + (0.5 * offset.dot(expansion.hessian * offset));
    }
    RESULT_RETURN_IF_ERROR(table[stage].FitWeightedLeastSquares(points, targets, weights, 0.0));
  }
  return SuccessResult(std::move(table));
}

template <int N, int K>
auto IlqrSeededTable(const SoftMinQuadraticValueFunctionApprox<N, K>& basis,
                     std::span<const BoxIlqrValueExpansion<N>> expansions,
                     const IlqrSeedOptions& options) noexcept
    -> Result<std::vector<SoftMinQuadraticValueFunctionApprox<N, K>>> {
  using ValueFunction = SoftMinQuadraticValueFunctionApprox<N, K>;
  using State = typename ValueFunction::State;
  using StateMat = typename ValueFunction::StateMat;
  RESULT_ASSERT(!expansions.empty(), "IlqrSeededTable: there are no expansions to seed from.");
  RESULT_ASSERT(options.metric_floor > 0.0 && options.relative_metric_floor >= 0.0 && options.center_spread >= 0.0,
                "IlqrSeededTable: the metric floor must be positive and the spread non-negative.");

  std::vector<ValueFunction> table;
  table.reserve(expansions.size() - 1);
  for (std::size_t stage = 0; stage + 1 < expansions.size(); ++stage) {
    const BoxIlqrValueExpansion<N>& expansion = expansions[stage];
    const Eigen::SelfAdjointEigenSolver<StateMat> eigen(0.5 * (expansion.hessian + expansion.hessian.transpose()));
    RESULT_ASSERT(eigen.info() == Eigen::Success, "IlqrSeededTable: the Hessian's eigendecomposition failed.");
    const double largest = 0.5 * std::max(eigen.eigenvalues().maxCoeff(), 0.0);
    const double floor = std::max(options.metric_floor, options.relative_metric_floor * largest);
    const State eigenvalues = (0.5 * eigen.eigenvalues()).cwiseMax(floor);
    const StateMat metric = eigen.eigenvectors() * eigenvalues.asDiagonal() * eigen.eigenvectors().transpose();
    const State shift = 0.5 * metric.ldlt().solve(expansion.gradient);
    const State center = expansion.state - shift;
    const double offset = expansion.value - expansion.gradient.dot(shift) + shift.dot(metric * shift);

    // The flattest direction is the first eigenvector; the outermost centers sit where the bowl has
    // risen by `center_spread` along it.
    const State flattest = eigen.eigenvectors().col(0);
    const double reach = std::sqrt(options.center_spread / eigenvalues[0]);
    std::array<State, static_cast<std::size_t>(K)> centers{};
    for (std::size_t component = 0; component < centers.size(); ++component) {
      const double position = K == 1 ? 0.0 : (2.0 * static_cast<double>(component) / static_cast<double>(K - 1)) - 1.0;
      centers[component] = center + (position * reach * flattest);
    }
    RESULT_ASSIGN_OR_RETURN(
        const typename ValueFunction::ParameterVector parameters,
        ValueFunction::QuadraticParametersAt(metric, centers, offset, basis.Sharpness(), basis.DiagonalOffset()));
    RESULT_ASSIGN_OR_RETURN(ValueFunction value_function,
                            ValueFunction::Make(parameters, basis.Sharpness(), basis.DiagonalOffset()));
    table.push_back(value_function);
  }
  return SuccessResult(std::move(table));
}

}  // namespace fbsde_traj_opt::examples

#endif  // FBSDE_TRAJ_OPT_EXAMPLES_ILQR_SEEDED_TABLE_HPP_
