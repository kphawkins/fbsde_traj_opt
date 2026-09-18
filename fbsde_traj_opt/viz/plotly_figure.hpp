// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#ifndef FBSDE_TRAJ_OPT_VIZ_PLOTLY_FIGURE_HPP_
#define FBSDE_TRAJ_OPT_VIZ_PLOTLY_FIGURE_HPP_

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

#include "fbsde_traj_opt/utils/result.hpp"

namespace fbsde_traj_opt::viz {

// A single Plotly figure, built as the JSON spec Plotly.newPlot() consumes.
//
// Plotly's API is a JSON document -- an array of traces plus a layout object -- so a figure is
// assembled here rather than drawn, and rendered later by plotly.js in a browser. See
// plotly_report.hpp for how one or more figures become a page.
//
// Colors are deliberately absent from the JSON this class produces. A trace carries a
// PlotColorRole instead, and the page resolves that role to a concrete color at render time by
// reading a CSS custom property. That indirection is what lets the page have a real dark mode:
// the dark palette is a separate set of colors selected for a dark surface, not a programmatic
// inversion of the light one, and switching between them re-resolves every role rather than
// rewriting every figure.
//
// Allocation: these operations allocate, and are `noexcept` per this project's convention, so an
// allocation failure terminates rather than unwinding. For a tool whose job is to write a plot
// file that is the intended behavior.

// The role a color plays in a figure, resolved to a concrete color by the page.
//
// The eight series slots are a fixed order, assigned in sequence and never cycled: a ninth series
// is not a ninth color but a sign that the figure should be split into small multiples. The order
// is chosen so that adjacent slots stay distinguishable under the common forms of color vision
// deficiency.
enum class PlotColorRole : std::uint8_t {
  kSeries1,
  kSeries2,
  kSeries3,
  kSeries4,
  kSeries5,
  kSeries6,
  kSeries7,
  kSeries8,
  // Recessive ink, for reference lines and other marks that are context rather than data.
  kMuted,
};

// Returns the CSS custom property name `role` resolves to, without the leading `--`. The returned
// view points at a string literal and outlives any caller.
auto PlotColorRoleName(PlotColorRole role) -> std::string_view;

// How one line is drawn.
struct LineStyle {
  // The name shown in the legend and the hover label.
  std::string name{};

  PlotColorRole color_role = PlotColorRole::kSeries1;

  // Line width in pixels. Data lines are thin -- 2px -- so that a dense figure stays legible;
  // a line drawn heavier than that is making a claim about importance, not about data.
  double width = 2.0;

  // Opacity in [0, 1]. Below 1 this is a de-emphasized line: one sample among many, drawn faintly
  // so that the shape of the whole cloud reads even though no single line does.
  double opacity = 1.0;

  // Whether this line gets its own legend entry. A group of faint sample lines sets this on the
  // first line only, so the group is named once rather than hundreds of times.
  bool show_in_legend = true;

  // Lines sharing a group name are shown and hidden together by the legend. Empty means the line
  // stands alone.
  std::string legend_group{};

  // Whether hovering reports this line. Off for the faint sample lines, whose individual values
  // are not what a reader is asking for when they hover over the cloud.
  bool show_on_hover = true;
};

class PlotlyFigure {
 public:
  // Builds an empty figure. `title` names it above the plot; the axis titles name the quantities.
  PlotlyFigure(std::string title, std::string x_axis_title, std::string y_axis_title) noexcept;

  // Adds a line through the given points.
  //
  // Fails if `x_values` and `y_values` differ in length, which would otherwise produce a figure
  // that plots the shorter of the two and silently discards the rest.
  auto AddLine(std::span<const double> x_values, std::span<const double> y_values, const LineStyle& style) noexcept
      -> Result<>;

  // Returns the figure's title.
  [[nodiscard]] auto Title() const noexcept -> const std::string& { return title_; }

  // Returns the number of lines added so far.
  [[nodiscard]] auto LineCount() const noexcept -> std::size_t { return traces_.size(); }

  // Returns the figure as `{"title": ..., "data": [...], "layout": {...}}`, the form the page's
  // renderer expects.
  [[nodiscard]] auto ToJson() const noexcept -> nlohmann::json;

 private:
  std::string title_;
  nlohmann::json layout_;
  nlohmann::json traces_ = nlohmann::json::array();
};

}  // namespace fbsde_traj_opt::viz

#endif  // FBSDE_TRAJ_OPT_VIZ_PLOTLY_FIGURE_HPP_
