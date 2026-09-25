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
// PlotColorRole -- or, for a heatmap, a PlotColorscaleRole -- instead, and the page resolves that
// role to a concrete color at render time by reading a CSS custom property. That indirection is
// what lets the page have a real dark mode: the dark palette is a separate set of colors selected
// for a dark surface, not a programmatic inversion of the light one, and switching between them
// re-resolves every role rather than rewriting every figure.
//
// A figure holds either lines or a single heatmap, never both. The two want different hover
// behavior, different axis treatment and different table twins, and a figure that mixed them
// would have to compromise on all three; two figures side by side do not.
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

// The job a continuous color scale does, resolved by the page to an ordered set of stops.
//
// The two jobs are not interchangeable. A sequential scale encodes magnitude and is a single hue
// running from near the surface to far from it -- which means its direction flips between light
// and dark mode, since "near the surface" is the light end on one and the dark end on the other.
// A diverging scale encodes polarity about a baseline and is two hues either side of a neutral
// gray midpoint, so that zero reads as nothing rather than as a color. Using a sequential scale
// for a signed quantity hides the sign; using a diverging one for an unsigned quantity invents a
// midpoint the data does not have.
enum class PlotColorscaleRole : std::uint8_t {
  kSequential,
  kDiverging,
};

// Returns the CSS custom property prefix `role` resolves to, without the leading `--`. The page
// appends `-0`, `-1`, ... for the individual stops. The returned view points at a string literal.
auto PlotColorscaleRoleName(PlotColorscaleRole role) -> std::string_view;

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

// How one heatmap is drawn.
struct HeatmapStyle {
  // The name shown in the hover label and the table twin.
  std::string name{};

  PlotColorscaleRole colorscale_role = PlotColorscaleRole::kSequential;

  // What the color encodes, named on the colorbar. A colorbar without it is a ramp of numbers
  // with no unit attached.
  std::string value_label{};

  // Whether to pin the scale's midpoint at zero. Meaningful only for a diverging scale, and
  // essentially always wanted there: a diverging scale whose midpoint floats with the data's
  // range puts the neutral color somewhere other than the baseline it is supposed to mark.
  bool centered_on_zero = false;
};

class PlotlyFigure {
 public:
  // Builds an empty figure. `title` names it above the plot; the axis titles name the quantities.
  PlotlyFigure(std::string title, std::string x_axis_title, std::string y_axis_title) noexcept;

  // Adds a line through the given points.
  //
  // Fails if `x_values` and `y_values` differ in length, which would otherwise produce a figure
  // that plots the shorter of the two and silently discards the rest, or if this figure already
  // holds a heatmap.
  auto AddLine(std::span<const double> x_values, std::span<const double> y_values, const LineStyle& style) noexcept
      -> Result<>;

  // Adds a heatmap over the grid `x_values` by `y_values`.
  //
  // `z_values_row_major` holds one value per grid cell, row by row: the value at
  // (`x_values[column]`, `y_values[row]`) is `z_values_row_major[row * x_values.size() + column]`.
  // A flat span rather than nested containers because that is the layout a caller sweeping a
  // function over a grid produces anyway, and it keeps the shape a single explicit statement
  // rather than an invariant across a container of containers.
  //
  // Adding a heatmap also switches the figure's hover mode from the shared crosshair the line
  // figures use to per-cell, since there is no column of series to read across.
  //
  // Fails if either axis is empty, if `z_values_row_major` is not exactly as long as the grid, or
  // if this figure already holds a line or a heatmap.
  auto AddHeatmap(std::span<const double> x_values,
                  std::span<const double> y_values,
                  std::span<const double> z_values_row_major,
                  const HeatmapStyle& style) noexcept -> Result<>;

  // Switches the y axis to a logarithmic scale and appends a note to its title saying so, since
  // a reader who misses that a scale is logarithmic misreads every distance on it.
  //
  // Only meaningful when every plotted value is strictly positive; Plotly drops non-positive
  // points from a log axis rather than reporting them, so the caller must establish that first.
  auto UseLogarithmicYAxis() noexcept -> void;

  // Returns the figure's title.
  [[nodiscard]] auto title() const noexcept -> const std::string& { return title_; }

  // Returns the number of traces added so far: the line count, or one for a heatmap figure.
  [[nodiscard]] auto LineCount() const noexcept -> std::size_t { return traces_.size(); }

  // Whether this figure holds a heatmap rather than lines.
  [[nodiscard]] auto HoldsHeatmap() const noexcept -> bool { return holds_heatmap_; }

  // Returns the figure as `{"title": ..., "data": [...], "layout": {...}}`, the form the page's
  // renderer expects.
  [[nodiscard]] auto ToJson() const noexcept -> nlohmann::json;

 private:
  std::string title_;
  nlohmann::json layout_;
  nlohmann::json traces_ = nlohmann::json::array();
  bool holds_heatmap_ = false;
};

}  // namespace fbsde_traj_opt::viz

#endif  // FBSDE_TRAJ_OPT_VIZ_PLOTLY_FIGURE_HPP_
