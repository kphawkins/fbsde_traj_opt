// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#ifndef FBSDE_TRAJ_OPT_VIZ_PLOTLY_REPORT_HPP_
#define FBSDE_TRAJ_OPT_VIZ_PLOTLY_REPORT_HPP_

#include <cstddef>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include "fbsde_traj_opt/utils/result.hpp"
#include "fbsde_traj_opt/viz/plotly_figure.hpp"

namespace fbsde_traj_opt::viz {

// A page of Plotly figures, written out as one self-contained HTML file.
//
// The file embeds every figure's JSON spec and loads plotly.js from a CDN, so viewing a report
// needs nothing but a browser and a network connection -- no local server, no build step, no
// notebook. That is the whole point: an experiment binary ends by writing one of these, and the
// plots are on screen as soon as the file is opened.
//
// The page owns everything about appearance that the figures deliberately leave out. It defines
// the palette as CSS custom properties in both a light and a dark set, resolves each trace's
// color role against whichever set is active, and re-resolves on a theme change. The dark set is
// a palette chosen for a dark surface rather than an inversion of the light one, which is why the
// page carries two sets of values rather than a filter.
//
// Every figure is also rendered as a table, collapsed beneath it. That is partly an accessibility
// obligation -- a value a reader can only obtain by hovering a colored line is a value some
// readers cannot obtain at all -- and partly just useful, since reading an exact cost off a plot
// is guesswork.
class PlotlyReport {
 public:
  // The pinned plotly.js the generated page loads. Pinned rather than tracking a floating
  // "latest" URL so that a report written today still renders the same way later.
  static constexpr std::string_view kPlotlyScriptUrl = "https://cdn.plot.ly/plotly-3.0.1.min.js";

  // Builds an empty report. `title` heads the page; `subtitle` is one line under it describing
  // what the reader is looking at, and may be empty.
  PlotlyReport(std::string title, std::string subtitle) noexcept;

  // Appends a figure. Figures appear in the order added, one card each.
  auto AddFigure(PlotlyFigure figure) noexcept -> void;

  // Returns the number of figures added so far.
  [[nodiscard]] auto FigureCount() const noexcept -> std::size_t { return figures_.size(); }

  // Renders the whole report as a single HTML document.
  [[nodiscard]] auto ToHtml() const noexcept -> std::string;

  // Writes ToHtml() to `path`, creating parent directories as needed.
  //
  // Fails if the parent directories cannot be created or the file cannot be written.
  [[nodiscard]] auto WriteHtml(const std::filesystem::path& path) const noexcept -> Result<>;

 private:
  std::string title_;
  std::string subtitle_;
  std::vector<PlotlyFigure> figures_;
};

}  // namespace fbsde_traj_opt::viz

#endif  // FBSDE_TRAJ_OPT_VIZ_PLOTLY_REPORT_HPP_
