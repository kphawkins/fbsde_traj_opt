// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#ifndef FBSDE_TRAJ_OPT_COMPOSED_COST_SDE_MODEL_HPP_
#define FBSDE_TRAJ_OPT_COMPOSED_COST_SDE_MODEL_HPP_

#include <utility>

#include <Eigen/Core>

#include "fbsde_traj_opt/sde_term_concepts.hpp"

namespace fbsde_traj_opt {

// A CostSdeModel (see sde_term_concepts.hpp) that owns a running cost term and a terminal cost
// term, together forming the objective of a discrete-time trajectory optimization problem over
// stages 0..K:
//
//   sum_{k=0}^{K-1} l(k, x_k, u_k) + phi(x_K).
//
// This is the cost-side counterpart of ComposedForwardSdeModel, and holds its terms by value as
// template parameters for the same reason: evaluating a batch's cost-to-go calls them once per
// trajectory per stage.
//
// `N` is the compile-time state dimension, `M` is the compile-time control dimension, and
// `Scalar` defaults to `double`.
template <int N, int M, typename RunningCostTermT, typename TerminalCostTermT, typename Scalar = double>
  requires SdeRunningCostTerm<RunningCostTermT, Eigen::Matrix<Scalar, N, 1>, Eigen::Matrix<Scalar, M, 1>> &&
           SdeTerminalCostTerm<TerminalCostTermT, Eigen::Matrix<Scalar, N, 1>>
class ComposedCostSdeModel {
 public:
  using State = Eigen::Matrix<Scalar, N, 1>;
  using Control = Eigen::Matrix<Scalar, M, 1>;

  using RunningCostTerm = RunningCostTermT;
  using TerminalCostTerm = TerminalCostTermT;

  // Builds a ComposedCostSdeModel from the two cost terms.
  //
  // As with ComposedForwardSdeModel, there is no Make() factory returning a Result: the
  // composition has no invariant of its own to enforce, and each term enforces its own.
  ComposedCostSdeModel(RunningCostTerm running_cost_term, TerminalCostTerm terminal_cost_term) noexcept
      : running_cost_term_(std::move(running_cost_term)), terminal_cost_term_(std::move(terminal_cost_term)) {}

  [[nodiscard]] auto RunningCost() const noexcept -> const RunningCostTerm& { return running_cost_term_; }

  [[nodiscard]] auto TerminalCost() const noexcept -> const TerminalCostTerm& { return terminal_cost_term_; }

 private:
  RunningCostTerm running_cost_term_;
  TerminalCostTerm terminal_cost_term_;
};

}  // namespace fbsde_traj_opt

#endif  // FBSDE_TRAJ_OPT_COMPOSED_COST_SDE_MODEL_HPP_
