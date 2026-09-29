// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#ifndef FBSDE_TRAJ_OPT_EXAMPLES_LQR_EXPERIMENT_HPP_
#define FBSDE_TRAJ_OPT_EXAMPLES_LQR_EXPERIMENT_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "fbsde_traj_opt/costs/quadratic_regulator_sde_running_cost_term.hpp"
#include "fbsde_traj_opt/costs/quadratic_regulator_sde_terminal_cost_term.hpp"
#include "fbsde_traj_opt/dynamics/const_diagonal_sde_diffusion_term.hpp"
#include "fbsde_traj_opt/dynamics/const_linear_sde_control_drift_mat_term.hpp"
#include "fbsde_traj_opt/dynamics/const_linear_sde_state_drift_term.hpp"
#include "fbsde_traj_opt/dynamics/zero_order_hold_discretization.hpp"
#include "fbsde_traj_opt/policies/const_linear_feedback_sde_control_policy_term.hpp"
#include "fbsde_traj_opt/sampling/diagonal_covariance_normal_distribution.hpp"
#include "fbsde_traj_opt/sde/composed_cost_sde_model.hpp"
#include "fbsde_traj_opt/sde/composed_forward_sde_model.hpp"
#include "fbsde_traj_opt/sde/trajectory_batch.hpp"
#include "fbsde_traj_opt/solvers/finite_horizon_lqr.hpp"
#include "fbsde_traj_opt/utils/result.hpp"
#include "fbsde_traj_opt/viz/plotly_figure.hpp"
#include "fbsde_traj_opt/viz/trajectory_batch_plots.hpp"

namespace fbsde_traj_opt::examples {

// One end-to-end comparison: a continuous linear plant with a quadratic cost, regulated once by
// the LQR policy the Riccati recursion produces and once by a hand-chosen fixed-gain feedback,
// with both rolled out over the same sampled noise.
//
// Running both policies against identical noise is the point of the exercise. Because
// TrajectoryBatch draws its noise from a coordinate-addressed sampler, the two batches differ
// only in the policy, trajectory by trajectory -- so a difference between their cost-to-go curves
// is a difference between the policies and not a difference between two draws.

// The plant, cost, noise, and baseline controller of one experiment.
//
// `N` is the state dimension and `M` the control dimension. The plant is given in continuous time
// and discretized exactly under a zero-order hold; the cost matrices are stage costs of the
// discretized problem, so a spec approximating a continuous integral scales them by the time step
// itself, where the choice is visible.
template <int N, int M>
struct LqrExperimentSpec {
  std::string name{};

  // A sentence or two of prose: what the system is, where it comes from, and what the figures are
  // expected to show.
  std::string description{};

  double time_step = 0.0;

  Eigen::Matrix<double, N, N> continuous_state_mat{};
  Eigen::Matrix<double, N, M> continuous_control_mat{};

  Eigen::Matrix<double, N, N> state_cost_mat{};
  Eigen::Matrix<double, M, M> control_cost_mat{};
  Eigen::Matrix<double, N, N> terminal_cost_mat{};

  Eigen::Matrix<double, N, 1> initial_mean{};
  Eigen::Matrix<double, N, 1> initial_covariance_diagonal{};

  // The diagonal of Sigma, the per-step noise shaping. These are standard deviations of the
  // increment injected at each step, not variances.
  Eigen::Matrix<double, N, 1> diffusion_diagonal{};

  // The fixed gain of the comparison controller, applied as u = K x at every stage.
  Eigen::Matrix<double, M, N> baseline_gain{};
  std::string baseline_name{};

  std::array<std::string, N> state_names{};

  // Which state dimensions get a figure of their own, and which pair gets a phase portrait.
  std::vector<std::size_t> plotted_state_indices{};
  bool plot_phase_portrait = false;
  std::size_t phase_horizontal_index = 0;
  std::size_t phase_vertical_index = 1;
};

// What one experiment produced: its figures, and the numbers worth printing next to them.
struct LqrExperimentReport {
  std::string name{};
  std::string description{};
  std::vector<viz::PlotlyFigure> figures{};

  // Expected cost-to-go from stage 0, under each policy, over the same sampled initial states.
  double optimal_initial_cost = 0.0;
  double baseline_initial_cost = 0.0;
};

// Solves `spec`, rolls out both policies, and builds the figures.
//
// `NumTrajectories` and `NumStages` are the shape of the sampled batches; `seed` fixes the noise
// both policies see.
//
// Fails if the plant cannot be discretized, if the Riccati recursion fails, if either rollout
// fails, or if a plotted state index is not a dimension of the state.
template <int N, int M, std::size_t NumTrajectories, std::size_t NumStages>
[[nodiscard]] auto RunLqrExperiment(const LqrExperimentSpec<N, M>& spec, std::uint64_t seed) noexcept
    -> Result<LqrExperimentReport> {
  using ForwardModel = ComposedForwardSdeModel<N,
                                               M,
                                               ConstLinearSdeStateDriftTerm<N>,
                                               ConstLinearSdeControlDriftMatTerm<N, M>,
                                               ConstDiagonalSdeDiffusionTerm<N>>;
  using CostModel =
      ComposedCostSdeModel<N, M, QuadraticRegulatorSdeRunningCostTerm<N, M>, QuadraticRegulatorSdeTerminalCostTerm<N>>;
  using Batch = TrajectoryBatch<N, M, NumTrajectories, NumStages>;
  using BaselinePolicy = ConstLinearFeedbackSdeControlPolicyTerm<N, M>;

  RESULT_ASSIGN_OR_RETURN(
      const auto discretized,
      ZeroOrderHoldDiscretization<N, M>(spec.continuous_state_mat, spec.continuous_control_mat, spec.time_step));

  RESULT_ASSIGN_OR_RETURN(const auto diffusion, ConstDiagonalSdeDiffusionTerm<N>::Make(spec.diffusion_diagonal));

  const ForwardModel forward_model(ConstLinearSdeStateDriftTerm<N>(discretized.state_drift_mat),
                                   ConstLinearSdeControlDriftMatTerm<N, M>(discretized.control_drift_mat),
                                   diffusion);

  const CostModel cost_model(QuadraticRegulatorSdeRunningCostTerm<N, M>(
                                 spec.state_cost_mat, spec.control_cost_mat, Eigen::Matrix<double, N, M>::Zero()),
                             QuadraticRegulatorSdeTerminalCostTerm<N>(spec.terminal_cost_mat));

  RESULT_ASSIGN_OR_RETURN(
      const auto initial_distribution,
      DiagonalCovarianceNormalDistribution<N>::Make(spec.initial_mean, spec.initial_covariance_diagonal));

  RESULT_ASSIGN_OR_RETURN(const auto optimal_policy, SolveFiniteHorizonLqr<NumStages>(forward_model, cost_model));

  const BaselinePolicy baseline_policy(spec.baseline_gain, Eigen::Matrix<double, M, 1>::Zero());

  // Same model, same initial distribution, same seed: the two batches are the same experiment
  // run twice with one thing changed. Each batch is bound by reference rather than copied out of
  // its Result, since a batch is large.
  RESULT_ASSIGN_OR_RETURN(const Batch& optimal_batch,
                          Batch::Make(forward_model, optimal_policy, initial_distribution, seed));
  RESULT_ASSIGN_OR_RETURN(const Batch& baseline_batch,
                          Batch::Make(forward_model, baseline_policy, initial_distribution, seed));

  constexpr viz::PlotColorRole kOptimalRole = viz::PlotColorRole::kSeries1;
  constexpr viz::PlotColorRole kBaselineRole = viz::PlotColorRole::kSeries2;
  const std::string optimal_name = "LQR";

  LqrExperimentReport report;
  report.name = spec.name;
  report.description = spec.description;

  for (const std::size_t state_index : spec.plotted_state_indices) {
    RESULT_ASSERT(state_index < static_cast<std::size_t>(N),
                  "RunLqrExperiment: a plotted state index is not a state dimension.");

    viz::PlotlyFigure figure(
        spec.state_names.at(state_index) + " over the horizon", "Stage", spec.state_names.at(state_index));

    RESULT_RETURN_IF_ERROR(
        viz::AddStateTrajectories(figure,
                                  optimal_batch,
                                  state_index,
                                  {.series_name = optimal_name, .color_role = kOptimalRole, .sample_opacity = 0.10}));

    RESULT_RETURN_IF_ERROR(viz::AddStateTrajectories(
        figure,
        baseline_batch,
        state_index,
        {.series_name = spec.baseline_name, .color_role = kBaselineRole, .sample_opacity = 0.10}));

    report.figures.push_back(std::move(figure));
  }

  if (spec.plot_phase_portrait) {
    viz::PlotlyFigure figure("Phase portrait: " + spec.state_names.at(spec.phase_vertical_index) + " against " +
                                 spec.state_names.at(spec.phase_horizontal_index),
                             spec.state_names.at(spec.phase_horizontal_index),
                             spec.state_names.at(spec.phase_vertical_index));

    RESULT_RETURN_IF_ERROR(
        viz::AddPhasePortrait(figure,
                              optimal_batch,
                              spec.phase_horizontal_index,
                              spec.phase_vertical_index,
                              {.series_name = optimal_name, .color_role = kOptimalRole, .sample_opacity = 0.10}));

    RESULT_RETURN_IF_ERROR(viz::AddPhasePortrait(
        figure,
        baseline_batch,
        spec.phase_horizontal_index,
        spec.phase_vertical_index,
        {.series_name = spec.baseline_name, .color_role = kBaselineRole, .sample_opacity = 0.10}));

    report.figures.push_back(std::move(figure));
  }

  const typename Batch::StageValues optimal_cost_to_go = optimal_batch.ExpectedCostToGo(cost_model);
  const typename Batch::StageValues baseline_cost_to_go = baseline_batch.ExpectedCostToGo(cost_model);

  const std::array<viz::CostToGoSeries, 2> cost_series{
      viz::CostToGoSeries{.name = optimal_name, .color_role = kOptimalRole, .expected_cost_to_go = optimal_cost_to_go},
      viz::CostToGoSeries{
          .name = spec.baseline_name, .color_role = kBaselineRole, .expected_cost_to_go = baseline_cost_to_go},
  };

  RESULT_ASSIGN_OR_RETURN(viz::PlotlyFigure cost_figure, viz::MakeCostToGoFigure("Expected cost-to-go", cost_series));
  report.figures.push_back(std::move(cost_figure));

  report.optimal_initial_cost = optimal_cost_to_go.front();
  report.baseline_initial_cost = baseline_cost_to_go.front();

  return SuccessResult(std::move(report));
}

}  // namespace fbsde_traj_opt::examples

#endif  // FBSDE_TRAJ_OPT_EXAMPLES_LQR_EXPERIMENT_HPP_
