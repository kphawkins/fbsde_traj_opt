// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#include "fbsde_traj_opt/viz/plotly_figure.hpp"

#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "fbsde_traj_opt/utils/result.hpp"

namespace fbsde_traj_opt::viz {

auto PlotColorRoleName(PlotColorRole role) -> std::string_view {
  switch (role) {
    case PlotColorRole::kSeries1:
      return "series-1";
    case PlotColorRole::kSeries2:
      return "series-2";
    case PlotColorRole::kSeries3:
      return "series-3";
    case PlotColorRole::kSeries4:
      return "series-4";
    case PlotColorRole::kSeries5:
      return "series-5";
    case PlotColorRole::kSeries6:
      return "series-6";
    case PlotColorRole::kSeries7:
      return "series-7";
    case PlotColorRole::kSeries8:
      return "series-8";
    case PlotColorRole::kMuted:
      return "muted";
  }
  return "muted";
}

PlotlyFigure::PlotlyFigure(std::string title, std::string x_axis_title, std::string y_axis_title) noexcept
    : title_(std::move(title)) {
  // Axis titles, spacing, and interaction only. Every color in the layout is filled in by the
  // page from its own tokens; see the header for why.
  layout_ = {
      // A shared crosshair readout rather than one tooltip per line: on a stage-indexed figure the
      // question is almost always "what do all the series read at this stage", not "what does this
      // one line read here".
      {"hovermode", "x unified"},
      {"margin", {{"l", 64}, {"r", 24}, {"t", 16}, {"b", 56}}},
      {"xaxis",
       {{"title", {{"text", std::move(x_axis_title)}}},
        // Solid hairlines, never dashed: a dashed grid reads as a projection or a threshold when
        // it is only a grid.
        {"zeroline", false},
        {"showgrid", true},
        {"gridwidth", 1},
        {"ticks", "outside"},
        {"ticklen", 4}}},
      {"yaxis",
       {{"title", {{"text", std::move(y_axis_title)}}},
        {"zeroline", false},
        {"showgrid", true},
        {"gridwidth", 1},
        {"ticks", "outside"},
        {"ticklen", 4}}},
      {"legend", {{"orientation", "h"}, {"yanchor", "bottom"}, {"y", 1.02}, {"xanchor", "left"}, {"x", 0}}},
      {"showlegend", true},
  };
}

auto PlotlyFigure::AddLine(std::span<const double> x_values,
                           std::span<const double> y_values,
                           const LineStyle& style) noexcept -> Result<> {
  RESULT_ASSERT(x_values.size() == y_values.size(),
                "PlotlyFigure::AddLine: the x and y value ranges must have the same length.");

  traces_.push_back({
      // SVG traces, not WebGL: the figures here run to a few tens of thousands of points,
      // which SVG renders comfortably, and the WebGL trace type does not support the shared
      // crosshair readout the layout asks for.
      {"type", "scatter"},
      {"mode", "lines"},
      {"x", std::vector<double>(x_values.begin(), x_values.end())},
      {"y", std::vector<double>(y_values.begin(), y_values.end())},
      {"name", style.name},
      {"opacity", style.opacity},
      {"showlegend", style.show_in_legend},
      {"legendgroup", style.legend_group},
      {"hoverinfo", style.show_on_hover ? "all" : "skip"},
      {"line", {{"width", style.width}}},
      // Not a Plotly attribute: the page reads it, resolves it against its own palette, and
      // writes the result into `line.color` before plotting.
      {"colorRole", PlotColorRoleName(style.color_role)},
  });

  return SuccessResult();
}

auto PlotlyFigure::ToJson() const noexcept -> nlohmann::json {
  return {{"title", title_}, {"data", traces_}, {"layout", layout_}};
}

}  // namespace fbsde_traj_opt::viz
