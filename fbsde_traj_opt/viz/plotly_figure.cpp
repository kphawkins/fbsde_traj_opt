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
  RESULT_ASSERT(frames_.empty(),
                "PlotlyFigure::AddLine: this figure is already animated, and its frames were sized against the "
                "traces it had at the time.");

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
  RESULT_ASSERT(frames_.empty(),
                "PlotlyFigure::AddHeatmap: this figure is already animated, and its frames were sized against the "
                "traces it had at the time.");

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

namespace {

// The number of values one frame must carry for `trace`: its point count for a line, its cell
// count for a heatmap.
auto TraceValueCount(const nlohmann::json& trace) noexcept -> std::size_t {
  if (trace.at("type") == "heatmap") {
    return trace.at("x").size() * trace.at("y").size();
  }
  return trace.at("y").size();
}

// Writes `values` into `frame_trace` in the shape that trace expects: a flat `y` for a line, `z`
// re-nested row by row for a heatmap.
auto AssignFrameValues(const nlohmann::json& trace,
                       const std::vector<double>& values,
                       nlohmann::json& frame_trace) noexcept -> void {
  if (trace.at("type") != "heatmap") {
    frame_trace["y"] = values;
    return;
  }

  const std::size_t width = trace.at("x").size();
  nlohmann::json rows = nlohmann::json::array();
  for (std::size_t row = 0; row * width < values.size(); ++row) {
    rows.push_back(std::vector<double>(values.begin() + static_cast<std::ptrdiff_t>(row * width),
                                       values.begin() + static_cast<std::ptrdiff_t>((row + 1) * width)));
  }
  frame_trace["z"] = std::move(rows);
}

}  // namespace

auto PlotlyFigure::Animate(std::span<const std::size_t> trace_indices,
                           std::span<const AnimationFrame> frames,
                           const AnimationStyle& style) noexcept -> Result<> {
  RESULT_ASSERT(!frames.empty(), "PlotlyFigure::Animate: an animation needs at least one frame.");
  RESULT_ASSERT(!trace_indices.empty(), "PlotlyFigure::Animate: an animation needs at least one trace to animate.");
  RESULT_ASSERT(frames_.empty(), "PlotlyFigure::Animate: this figure is already animated.");
  for (const std::size_t index : trace_indices) {
    RESULT_ASSERT(index < traces_.size(), "PlotlyFigure::Animate: a trace index is not a trace of this figure.");
  }
  for (const AnimationFrame& frame : frames) {
    RESULT_ASSERT(frame.trace_values.size() == trace_indices.size(),
                  "PlotlyFigure::Animate: every frame must carry exactly one set of values per animated trace.");
    for (std::size_t slot = 0; slot < trace_indices.size(); ++slot) {
      RESULT_ASSERT(frame.trace_values[slot].size() == TraceValueCount(traces_[trace_indices[slot]]),
                    "PlotlyFigure::Animate: a frame's values are not the length of the data the trace holds; a frame "
                    "replaces values and never reshapes a trace.");
    }
  }

  // The range every frame shares. Computed before anything is written, over the frames and over
  // whatever the figure already holds, so that a static reference curve stays on screen too.
  double lowest = std::numeric_limits<double>::infinity();
  double highest = -std::numeric_limits<double>::infinity();
  for (const AnimationFrame& frame : frames) {
    for (const std::vector<double>& values : frame.trace_values) {
      for (const double value : values) {
        if (std::isfinite(value)) {
          lowest = std::min(lowest, value);
          highest = std::max(highest, value);
        }
      }
    }
  }

  const bool animating_heatmap = traces_[trace_indices[0]].at("type") == "heatmap";
  if (!animating_heatmap) {
    for (const nlohmann::json& trace : traces_) {
      for (const auto& value : trace.at("y")) {
        const auto point = value.get<double>();
        if (std::isfinite(point)) {
          lowest = std::min(lowest, point);
          highest = std::max(highest, point);
        }
      }
    }
  }
  if (!std::isfinite(lowest) || !std::isfinite(highest)) {
    lowest = 0.0;
    highest = 1.0;
  }

  // The figure rests on the last frame; see the header for why.
  const AnimationFrame& resting = frames.back();
  for (std::size_t slot = 0; slot < trace_indices.size(); ++slot) {
    nlohmann::json& trace = traces_[trace_indices[slot]];
    AssignFrameValues(trace, resting.trace_values[slot], trace);
  }

  if (animating_heatmap) {
    for (const std::size_t index : trace_indices) {
      nlohmann::json& trace = traces_[index];
      if (trace.contains("zmid")) {
        // A scale centered on zero needs a symmetric range, or the neutral color drifts off the
        // baseline it is there to mark.
        const double reach = std::max(std::abs(lowest), std::abs(highest));
        trace["zmin"] = -reach;
        trace["zmax"] = reach;
      } else {
        trace["zmin"] = lowest;
        trace["zmax"] = highest;
      }
    }
  } else {
    const double margin = 0.05 * std::max(highest - lowest, 1e-12);
    const bool logarithmic = layout_["yaxis"].contains("type") && layout_["yaxis"]["type"] == "log";
    if (logarithmic) {
      // A log axis takes its range in decades, and a non-positive bound has no logarithm.
      const double floor_value = lowest > 0.0 ? lowest : highest / 1e6;
      if (floor_value > 0.0 && highest > 0.0) {
        layout_["yaxis"]["range"] = {std::log10(floor_value) - 0.1, std::log10(highest) + 0.1};
      }
    } else {
      layout_["yaxis"]["range"] = {lowest - margin, highest + margin};
    }
  }

  nlohmann::json steps = nlohmann::json::array();
  for (std::size_t index = 0; index < frames.size(); ++index) {
    const std::string name = std::to_string(index);
    nlohmann::json frame_traces = nlohmann::json::array();
    for (std::size_t slot = 0; slot < trace_indices.size(); ++slot) {
      nlohmann::json frame_trace = nlohmann::json::object();
      AssignFrameValues(traces_[trace_indices[slot]], frames[index].trace_values[slot], frame_trace);
      frame_traces.push_back(std::move(frame_trace));
    }
    frames_.push_back({{"name", name}, {"data", std::move(frame_traces)}, {"traces", trace_indices}});

    steps.push_back({{"label", frames[index].label},
                     {"method", "animate"},
                     {"args",
                      {nlohmann::json::array({name}),
                       {{"mode", "immediate"},
                        {"frame", {{"duration", 0}, {"redraw", true}}},
                        {"transition", {{"duration", 0}}}}}}});
  }

  // Play restarts from the first frame rather than resuming from the current one, because the
  // figure rests at the end: "from current" there would play nothing at all.
  layout_["updatemenus"] =
      nlohmann::json::array({{{"type", "buttons"},
                              {"direction", "left"},
                              {"showactive", false},
                              {"x", 0.0},
                              {"y", -0.30},
                              {"xanchor", "left"},
                              {"yanchor", "top"},
                              {"pad", {{"t", 0}, {"r", 10}}},
                              {"buttons",
                               {{{"label", "Play"},
                                 {"method", "animate"},
                                 {"args",
                                  {nullptr,
                                   {{"mode", "immediate"},
                                    {"fromcurrent", false},
                                    {"frame", {{"duration", style.frame_duration_ms}, {"redraw", true}}},
                                    {"transition", {{"duration", 0}}}}}}},
                                {{"label", "Pause"},
                                 {"method", "animate"},
                                 {"args",
                                  {nlohmann::json::array({nullptr}),
                                   {{"mode", "immediate"},
                                    {"frame", {{"duration", 0}, {"redraw", false}}},
                                    {"transition", {{"duration", 0}}}}}}}}}}});

  layout_["sliders"] = nlohmann::json::array(
      {{{"active", frames.size() - 1},
        {"x", 0.14},
        {"y", -0.24},
        {"len", 0.86},
        {"xanchor", "left"},
        {"yanchor", "top"},
        {"pad", {{"t", 0}, {"b", 0}}},
        {"currentvalue", {{"visible", true}, {"prefix", style.slider_prefix}, {"xanchor", "right"}, {"offset", 6}}},
        {"transition", {{"duration", 0}}},
        {"steps", std::move(steps)}}});

  // Room under the plot for the control row.
  layout_["margin"]["b"] = 128;

  return SuccessResult();
}

auto PlotlyFigure::UseLogarithmicYAxis() noexcept -> void {
  layout_["yaxis"]["type"] = "log";
  layout_["yaxis"]["title"]["text"] = layout_["yaxis"]["title"]["text"].get<std::string>() + " (log scale)";
}

auto PlotlyFigure::ToJson() const noexcept -> nlohmann::json {
  nlohmann::json figure = {{"title", title_}, {"data", traces_}, {"layout", layout_}};
  if (!frames_.empty()) {
    figure["frames"] = frames_;
  }
  return figure;
}

}  // namespace fbsde_traj_opt::viz
