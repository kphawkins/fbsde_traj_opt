// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#include "fbsde_traj_opt/viz/trajectory_batch_plots.hpp"

#include <algorithm>
#include <cstddef>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "fbsde_traj_opt/utils/result.hpp"
#include "fbsde_traj_opt/viz/plotly_figure.hpp"

namespace fbsde_traj_opt::viz {

auto MakeCostToGoFigure(std::string title,
                        std::span<const CostToGoSeries> series,
                        CostToGoAxisScale axis_scale) noexcept -> Result<PlotlyFigure> {
  RESULT_ASSERT(!series.empty(), "MakeCostToGoFigure: at least one series is required.");

  const std::size_t stage_count = series.front().expected_cost_to_go.size();
  for (const CostToGoSeries& one : series) {
    RESULT_ASSERT(one.expected_cost_to_go.size() == stage_count,
                  "MakeCostToGoFigure: every series must cover the same number of stages.");
  }

  std::vector<double> stages(stage_count);
  for (std::size_t stage = 0; stage < stage_count; ++stage) {
    stages[stage] = static_cast<double>(stage);
  }

  PlotlyFigure figure(std::move(title), "Stage", "Expected cost-to-go");
  for (const CostToGoSeries& one : series) {
    const LineStyle style{
        .name = one.name,
        .color_role = one.color_role,
        .width = 2.0,
        .opacity = 1.0,
        .show_in_legend = true,
        .legend_group = one.name,
        .show_on_hover = true,
    };
    RESULT_RETURN_IF_ERROR(figure.AddLine(stages, one.expected_cost_to_go, style));
  }

  if (axis_scale == CostToGoAxisScale::kLogarithmicWhenPositive) {
    const bool all_positive = std::ranges::all_of(series, [](const CostToGoSeries& one) noexcept {
      return std::ranges::all_of(one.expected_cost_to_go, [](double value) noexcept { return value > 0.0; });
    });
    if (all_positive) {
      figure.UseLogarithmicYAxis();
    }
  }

  return SuccessResult(std::move(figure));
}

}  // namespace fbsde_traj_opt::viz
