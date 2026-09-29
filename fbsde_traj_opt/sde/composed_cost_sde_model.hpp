// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#ifndef FBSDE_TRAJ_OPT_COMPOSED_COST_SDE_MODEL_HPP_
#define FBSDE_TRAJ_OPT_COMPOSED_COST_SDE_MODEL_HPP_

#include <utility>

#include <Eigen/Core>

#include "fbsde_traj_opt/sde/sde_term_concepts.hpp"

namespace fbsde_traj_opt {

// A cost model that owns a running cost term and a terminal cost term (see sde_term_concepts.hpp),
// together forming the objective of a discrete-time trajectory optimization problem over
// stages 0..K:
//
//   sum_{k=0}^{K-1} l(k, x_k, u_k) + phi(x_K).
//
// The two term types are template parameters held by value, so that evaluating them inlines and
// allocates nothing, which matters because evaluating a batch's cost-to-go calls them once per
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
  ComposedCostSdeModel(RunningCostTerm running_cost_term, TerminalCostTerm terminal_cost_term) noexcept
      : running_cost_term_(std::move(running_cost_term)), terminal_cost_term_(std::move(terminal_cost_term)) {}

  [[nodiscard]] auto running_cost() const noexcept -> const RunningCostTerm& { return running_cost_term_; }

  [[nodiscard]] auto terminal_cost() const noexcept -> const TerminalCostTerm& { return terminal_cost_term_; }

 private:
  RunningCostTerm running_cost_term_;
  TerminalCostTerm terminal_cost_term_;
};

}  // namespace fbsde_traj_opt

#endif  // FBSDE_TRAJ_OPT_COMPOSED_COST_SDE_MODEL_HPP_
