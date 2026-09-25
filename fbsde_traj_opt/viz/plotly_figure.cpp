// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#include "fbsde_traj_opt/viz/plotly_figure.hpp"

#include <cstddef>
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

auto PlotColorscaleRoleName(PlotColorscaleRole role) -> std::string_view {
  switch (role) {
    case PlotColorscaleRole::kSequential:
      return "scale-sequential";
    case PlotColorscaleRole::kDiverging:
      return "scale-diverging";
  }
  return "scale-sequential";
}

auto PlotlyFigure::AddLine(std::span<const double> x_values,
                           std::span<const double> y_values,
                           const LineStyle& style) noexcept -> Result<> {
  RESULT_ASSERT(x_values.size() == y_values.size(),
                "PlotlyFigure::AddLine: the x and y value ranges must have the same length.");
  RESULT_ASSERT(!holds_heatmap_,
                "PlotlyFigure::AddLine: this figure already holds a heatmap, and a figure holds either lines or a "
                "heatmap but not both.");

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

auto PlotlyFigure::AddHeatmap(std::span<const double> x_values,
                              std::span<const double> y_values,
                              std::span<const double> z_values_row_major,
                              const HeatmapStyle& style) noexcept -> Result<> {
  RESULT_ASSERT(!x_values.empty(), "PlotlyFigure::AddHeatmap: the grid's x axis may not be empty.");
  RESULT_ASSERT(!y_values.empty(), "PlotlyFigure::AddHeatmap: the grid's y axis may not be empty.");
  RESULT_ASSERT(z_values_row_major.size() == x_values.size() * y_values.size(),
                "PlotlyFigure::AddHeatmap: the value range must hold exactly one value per grid cell, laid out row "
                "by row.");
  RESULT_ASSERT(traces_.empty(),
                "PlotlyFigure::AddHeatmap: this figure already holds a trace, and a figure holds either lines or a "
                "single heatmap.");

  // Plotly wants `z` as an array of rows, each row running along x.
  nlohmann::json rows = nlohmann::json::array();
  for (std::size_t row = 0; row < y_values.size(); ++row) {
    const std::span<const double> values = z_values_row_major.subspan(row * x_values.size(), x_values.size());
    rows.push_back(std::vector<double>(values.begin(), values.end()));
  }

  nlohmann::json trace = {
      {"type", "heatmap"},
      {"x", std::vector<double>(x_values.begin(), x_values.end())},
      {"y", std::vector<double>(y_values.begin(), y_values.end())},
      {"z", std::move(rows)},
      {"name", style.name},
      // Cells, not an interpolated wash: the grid is what was actually evaluated, and smoothing
      // it would show the reader detail between samples that was never computed.
      {"zsmooth", false},
      {"colorbar", {{"title", {{"text", style.value_label}}}}},
      // Not a Plotly attribute: the page reads it, assembles the stops from its own palette, and
      // writes the result into `colorscale` before plotting.
      {"colorscaleRole", PlotColorscaleRoleName(style.colorscale_role)},
  };
  if (style.centered_on_zero) {
    trace["zmid"] = 0.0;
  }
  traces_.push_back(std::move(trace));
  holds_heatmap_ = true;

  // The shared crosshair a line figure wants reads every series at one x. A heatmap has no
  // series to read across, so it reports the cell under the pointer instead.
  layout_["hovermode"] = "closest";
  layout_["showlegend"] = false;

  return SuccessResult();
}

auto PlotlyFigure::UseLogarithmicYAxis() noexcept -> void {
  layout_["yaxis"]["type"] = "log";
  layout_["yaxis"]["title"]["text"] = layout_["yaxis"]["title"]["text"].get<std::string>() + " (log scale)";
}

auto PlotlyFigure::ToJson() const noexcept -> nlohmann::json {
  return {{"title", title_}, {"data", traces_}, {"layout", layout_}};
}

}  // namespace fbsde_traj_opt::viz
