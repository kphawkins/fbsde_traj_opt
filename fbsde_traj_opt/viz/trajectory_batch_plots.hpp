// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#ifndef FBSDE_TRAJ_OPT_VIZ_TRAJECTORY_BATCH_PLOTS_HPP_
#define FBSDE_TRAJ_OPT_VIZ_TRAJECTORY_BATCH_PLOTS_HPP_

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <span>
#include <string>
#include <vector>

#include "fbsde_traj_opt/utils/result.hpp"
#include "fbsde_traj_opt/viz/plotly_figure.hpp"

namespace fbsde_traj_opt::viz {

// Figures over a sampled trajectory batch.
//
// A batch is a distribution, and the figures here draw it the way a distribution over paths is
// usually drawn: the individual sampled paths faintly, so the shape of the cloud reads, with the
// mean path over the top in full strength. The two together say what neither says alone -- the
// mean shows what the policy does, the cloud shows how reliably it does it.
//
// The "add to an existing figure" form is the primitive rather than the "make a figure" form,
// because the interesting comparison is two batches on one set of axes: the same noise driving
// two different policies, one hue each.

// Concept for the batch types these functions plot -- satisfied by TrajectoryBatch, and stated
// structurally so that this header need not depend on it.
template <typename T>
concept PlottableTrajectoryBatch = requires(const T& batch, std::size_t index) {
  { T::kNumStages } -> std::convertible_to<std::size_t>;
  { T::kNumTrajectories } -> std::convertible_to<std::size_t>;
  { T::kStateDim } -> std::convertible_to<std::size_t>;
  { batch.StateAt(index, index)[0] } -> std::convertible_to<double>;
  { batch.MeanStateAtStage(index)[0] } -> std::convertible_to<double>;
};

// How a batch is drawn.
struct TrajectoryPlotOptions {
  // Names the batch in the legend. The sample cloud and the mean get one entry each, derived
  // from this.
  std::string series_name{"trajectories"};

  // The hue the whole batch is drawn in -- cloud and mean alike, since they are one thing.
  PlotColorRole color_role = PlotColorRole::kSeries1;

  // How many sampled paths to draw. Past a few dozen overlapping faint lines the cloud stops
  // getting more informative and starts getting slower, so a large batch is drawn from a prefix
  // of its trajectories; they are independent and identically distributed, so any subset is as
  // representative as any other.
  std::size_t max_sampled_trajectories = 48;

  // Opacity of one sampled path. Low enough that a single line is barely visible and the density
  // of the cloud is what the eye reads.
  double sample_opacity = 0.14;

  bool show_samples = true;
  bool show_mean = true;
};

// One policy's expected cost-to-go curve, for MakeCostToGoFigure().
struct CostToGoSeries {
  std::string name{};
  PlotColorRole color_role = PlotColorRole::kSeries1;

  // The expected cost-to-go at each stage, as returned by TrajectoryBatch::ExpectedCostToGo().
  std::span<const double> expected_cost_to_go{};
};

// Builds a figure comparing the expected cost-to-go of one or more policies over the horizon.
//
// This is the figure that settles which policy is better, and it settles it at every stage rather
// than only at the start: a policy can look competitive from the initial distribution and then
// fall behind badly from the states it actually drives itself into.
//
// Fails if `series` is empty or if the series do not all cover the same number of stages, since
// plotting them against one stage axis would then be comparing different horizons.
[[nodiscard]] auto MakeCostToGoFigure(std::string title, std::span<const CostToGoSeries> series) noexcept
    -> Result<PlotlyFigure>;

// Adds one state dimension of `batch` to `figure`, as a faint cloud of sampled paths with the
// mean path over it.
//
// Fails if `state_index` is not a dimension of the batch's state.
template <PlottableTrajectoryBatch BatchT>
[[nodiscard]] auto AddStateTrajectories(PlotlyFigure& figure,
                                        const BatchT& batch,
                                        std::size_t state_index,
                                        const TrajectoryPlotOptions& options) noexcept -> Result<> {
  RESULT_ASSERT(state_index < BatchT::kStateDim, "AddStateTrajectories: state_index is not a state dimension.");

  std::vector<double> stages(BatchT::kNumStages);
  for (std::size_t stage = 0; stage < BatchT::kNumStages; ++stage) {
    stages[stage] = static_cast<double>(stage);
  }

  std::vector<double> values(BatchT::kNumStages);

  if (options.show_samples) {
    const std::size_t sample_count = std::min(options.max_sampled_trajectories, BatchT::kNumTrajectories);
    for (std::size_t trajectory = 0; trajectory < sample_count; ++trajectory) {
      for (std::size_t stage = 0; stage < BatchT::kNumStages; ++stage) {
        values[stage] = static_cast<double>(batch.StateAt(trajectory, stage)[static_cast<int>(state_index)]);
      }

      const LineStyle style{
          .name = options.series_name + " samples",
          .color_role = options.color_role,
          // One pixel rather than the two a data line gets: these are context for the mean, and
          // at this opacity a heavier line only muddies the cloud.
          .width = 1.0,
          .opacity = options.sample_opacity,
          // The whole cloud is named once, on its first path, so the legend has one entry for it
          // rather than one per sample.
          .show_in_legend = trajectory == 0,
          .legend_group = options.series_name + "-samples",
          // Hovering the cloud should report the mean, not whichever individual sample the
          // cursor happened to land on.
          .show_on_hover = false,
      };
      const Result<> added = figure.AddLine(stages, values, style);
      if (!added.has_value()) {
        return added;
      }
    }
  }

  if (options.show_mean) {
    for (std::size_t stage = 0; stage < BatchT::kNumStages; ++stage) {
      values[stage] = static_cast<double>(batch.MeanStateAtStage(stage)[static_cast<int>(state_index)]);
    }

    const LineStyle style{
        .name = options.series_name + " mean",
        .color_role = options.color_role,
        .width = 2.0,
        .opacity = 1.0,
        .show_in_legend = true,
        .legend_group = options.series_name + "-mean",
        .show_on_hover = true,
    };
    const Result<> added = figure.AddLine(stages, values, style);
    if (!added.has_value()) {
      return added;
    }
  }

  return SuccessResult();
}

// Adds a phase portrait of `batch` to `figure`: one state dimension against another, with stage
// running along each path rather than along an axis.
//
// Fails if either index is not a dimension of the batch's state.
template <PlottableTrajectoryBatch BatchT>
[[nodiscard]] auto AddPhasePortrait(PlotlyFigure& figure,
                                    const BatchT& batch,
                                    std::size_t horizontal_state_index,
                                    std::size_t vertical_state_index,
                                    const TrajectoryPlotOptions& options) noexcept -> Result<> {
  RESULT_ASSERT(horizontal_state_index < BatchT::kStateDim && vertical_state_index < BatchT::kStateDim,
                "AddPhasePortrait: a state index is not a state dimension.");

  std::vector<double> horizontal(BatchT::kNumStages);
  std::vector<double> vertical(BatchT::kNumStages);

  if (options.show_samples) {
    const std::size_t sample_count = std::min(options.max_sampled_trajectories, BatchT::kNumTrajectories);
    for (std::size_t trajectory = 0; trajectory < sample_count; ++trajectory) {
      for (std::size_t stage = 0; stage < BatchT::kNumStages; ++stage) {
        const auto state = batch.StateAt(trajectory, stage);
        horizontal[stage] = static_cast<double>(state[static_cast<int>(horizontal_state_index)]);
        vertical[stage] = static_cast<double>(state[static_cast<int>(vertical_state_index)]);
      }

      const LineStyle style{
          .name = options.series_name + " samples",
          .color_role = options.color_role,
          .width = 1.0,
          .opacity = options.sample_opacity,
          .show_in_legend = trajectory == 0,
          .legend_group = options.series_name + "-samples",
          .show_on_hover = false,
      };
      const Result<> added = figure.AddLine(horizontal, vertical, style);
      if (!added.has_value()) {
        return added;
      }
    }
  }

  if (options.show_mean) {
    for (std::size_t stage = 0; stage < BatchT::kNumStages; ++stage) {
      const auto state = batch.MeanStateAtStage(stage);
      horizontal[stage] = static_cast<double>(state[static_cast<int>(horizontal_state_index)]);
      vertical[stage] = static_cast<double>(state[static_cast<int>(vertical_state_index)]);
    }

    const LineStyle style{
        .name = options.series_name + " mean",
        .color_role = options.color_role,
        .width = 2.0,
        .opacity = 1.0,
        .show_in_legend = true,
        .legend_group = options.series_name + "-mean",
        .show_on_hover = true,
    };
    const Result<> added = figure.AddLine(horizontal, vertical, style);
    if (!added.has_value()) {
      return added;
    }
  }

  return SuccessResult();
}

}  // namespace fbsde_traj_opt::viz

#endif  // FBSDE_TRAJ_OPT_VIZ_TRAJECTORY_BATCH_PLOTS_HPP_
