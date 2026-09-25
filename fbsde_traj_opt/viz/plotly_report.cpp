// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#include "fbsde_traj_opt/viz/plotly_report.hpp"

#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

#include <nlohmann/json.hpp>

#include "fbsde_traj_opt/utils/result.hpp"
#include "fbsde_traj_opt/viz/plotly_figure.hpp"

namespace fbsde_traj_opt::viz {
namespace {

// The page template. `{{...}}` placeholders are substituted by ToHtml() below.
//
// The palette is the project's categorical set, defined once as CSS custom properties in a light
// and a dark variant. Traces name a role; ResolveToken() in the page script turns that name into
// whichever variant is live. The dark variant is its own set of steps chosen against the dark
// surface, not a programmatic lightening of the light one.
//
// The dark values are declared under two scopes on purpose. The media query follows the operating
// system's setting; the `[data-theme]` scope follows the page's own toggle, and has to win in both
// directions -- the `:not([data-theme="light"])` guard is what lets an explicit light choice beat
// an OS set to dark.
constexpr std::string_view kPageTemplate = R"HTML(<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{TITLE}}</title>
<script src="{{PLOTLY_SRC}}" charset="utf-8"></script>
<style>
:root {
  color-scheme: light;
  --page:           #f9f9f7;
  --surface-1:      #fcfcfb;
  --text-primary:   #0b0b0b;
  --text-secondary: #52514e;
  --muted:          #898781;
  --gridline:       #e1e0d9;
  --axis-line:      #c3c2b7;
  --border:         rgba(11, 11, 11, 0.10);
  --series-1:       #2a78d6;
  --series-2:       #eb6834;
  --series-3:       #1baf7a;
  --series-4:       #eda100;
  --series-5:       #e87ba4;
  --series-6:       #008300;
  --series-7:       #4a3aa7;
  --series-8:       #e34948;
  --scale-sequential-0: #cde2fb;
  --scale-sequential-1: #86b6ef;
  --scale-sequential-2: #3987e5;
  --scale-sequential-3: #1c5cab;
  --scale-sequential-4: #0d366b;
  --scale-diverging-0:  #1c5cab;
  --scale-diverging-1:  #f0efec;
  --scale-diverging-2:  #e34948;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --page:           #0d0d0d;
    --surface-1:      #1a1a19;
    --text-primary:   #ffffff;
    --text-secondary: #c3c2b7;
    --muted:          #898781;
    --gridline:       #2c2c2a;
    --axis-line:      #383835;
    --border:         rgba(255, 255, 255, 0.10);
    --series-1:       #3987e5;
    --series-2:       #d95926;
    --series-3:       #199e70;
    --series-4:       #c98500;
    --series-5:       #d55181;
    --series-6:       #008300;
    --series-7:       #9085e9;
    --series-8:       #e66767;
    --scale-sequential-0: #0d366b;
    --scale-sequential-1: #1c5cab;
    --scale-sequential-2: #3987e5;
    --scale-sequential-3: #86b6ef;
    --scale-sequential-4: #cde2fb;
    --scale-diverging-0:  #3987e5;
    --scale-diverging-1:  #383835;
    --scale-diverging-2:  #e66767;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page:           #0d0d0d;
  --surface-1:      #1a1a19;
  --text-primary:   #ffffff;
  --text-secondary: #c3c2b7;
  --muted:          #898781;
  --gridline:       #2c2c2a;
  --axis-line:      #383835;
  --border:         rgba(255, 255, 255, 0.10);
  --series-1:       #3987e5;
  --series-2:       #d95926;
  --series-3:       #199e70;
  --series-4:       #c98500;
  --series-5:       #d55181;
  --series-6:       #008300;
  --series-7:       #9085e9;
  --series-8:       #e66767;
  --scale-sequential-0: #0d366b;
  --scale-sequential-1: #1c5cab;
  --scale-sequential-2: #3987e5;
  --scale-sequential-3: #86b6ef;
  --scale-sequential-4: #cde2fb;
  --scale-diverging-0:  #3987e5;
  --scale-diverging-1:  #383835;
  --scale-diverging-2:  #e66767;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  padding: 32px 16px 64px;
  background: var(--page);
  color: var(--text-primary);
  font: 15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif;
}
.wrap { max-width: 1080px; margin: 0 auto; }
header { display: flex; align-items: flex-start; gap: 16px; margin-bottom: 28px; }
header .titles { flex: 1 1 auto; min-width: 0; }
h1 { margin: 0 0 6px; font-size: 26px; font-weight: 600; letter-spacing: -0.01em; }
.subtitle { margin: 0; color: var(--text-secondary); font-size: 14px; }
#theme-toggle {
  flex: 0 0 auto;
  padding: 7px 13px;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: var(--surface-1);
  color: var(--text-secondary);
  font: inherit;
  font-size: 13px;
  cursor: pointer;
}
#theme-toggle:hover { color: var(--text-primary); }
.card {
  background: var(--surface-1);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 20px 20px 12px;
  margin-bottom: 20px;
}
.card h2 { margin: 0 0 14px; font-size: 16px; font-weight: 600; }
.plot { width: 100%; height: 420px; }
details { margin-top: 4px; border-top: 1px solid var(--border); padding-top: 8px; }
summary { cursor: pointer; color: var(--text-secondary); font-size: 13px; padding: 4px 0; }
.table-scroll { max-height: 320px; overflow: auto; margin-top: 8px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; font-variant-numeric: tabular-nums; }
th, td { text-align: right; padding: 4px 10px; border-bottom: 1px solid var(--border); white-space: nowrap; }
th { position: sticky; top: 0; background: var(--surface-1); color: var(--text-secondary); font-weight: 600; }
td:first-child, th:first-child { text-align: left; }
.swatch { display: inline-block; width: 9px; height: 9px; border-radius: 2px; margin-right: 6px; vertical-align: baseline; }
@media (max-width: 640px) { .plot { height: 340px; } body { padding: 24px 16px 48px; } }
</style>
</head>
<body>
<div class="wrap">
<header>
  <div class="titles">
    <h1>{{TITLE}}</h1>
    <p class="subtitle">{{SUBTITLE}}</p>
  </div>
  <button id="theme-toggle" type="button" aria-live="polite">Theme: system</button>
</header>
<main id="figures"></main>
</div>
<script id="figure-data" type="application/json">{{FIGURES_JSON}}</script>
<script>
(function () {
  "use strict";

  var FIGURES = JSON.parse(document.getElementById("figure-data").textContent);
  var FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif';
  var PLOT_CONFIG = { responsive: true, displaylogo: false, modeBarButtonsToRemove: ["lasso2d", "select2d"] };

  // Reads a palette token off the document root, so every color the page draws comes from the
  // one place the theme is defined.
  function ResolveToken(name) {
    return getComputedStyle(document.documentElement).getPropertyValue("--" + name).trim();
  }

  // How many stops each continuous scale is defined with. The stop hexes themselves live in the
  // palette above as --<role>-0 .. --<role>-(n-1); only the count has to be agreed here.
  var COLORSCALE_STOPS = { "scale-sequential": 5, "scale-diverging": 3 };

  // Builds a Plotly colorscale -- [[position, color], ...] -- from a role's palette stops. The
  // sequential role's stops are themselves reversed between light and dark mode, so that the end
  // of the ramp meaning "least" always sits nearest the surface it is drawn on.
  function ResolveColorscale(role) {
    var count = COLORSCALE_STOPS[role] || 2;
    var stops = [];
    for (var index = 0; index < count; index += 1) {
      stops.push([index / (count - 1), ResolveToken(role + "-" + index)]);
    }
    return stops;
  }

  // Replaces each trace's color role with the color, or the color scale, that role currently
  // resolves to.
  function ThemedData(data) {
    var ink = ResolveToken("text-primary");
    var secondary = ResolveToken("text-secondary");
    var axis = ResolveToken("axis-line");

    return data.map(function (trace) {
      var themed = Object.assign({}, trace);
      if (trace.colorscaleRole) {
        delete themed.colorscaleRole;
        themed.colorscale = ResolveColorscale(trace.colorscaleRole);
        themed.colorbar = Object.assign({}, trace.colorbar, {
          outlinecolor: axis,
          tickcolor: axis,
          tickfont: { color: secondary, size: 12 },
          title: Object.assign({}, (trace.colorbar || {}).title, { font: { color: ink, size: 13 } })
        });
        return themed;
      }
      var color = ResolveToken(trace.colorRole || "series-1");
      delete themed.colorRole;
      themed.line = Object.assign({}, trace.line, { color: color });
      return themed;
    });
  }

  // Fills in every color the figure spec deliberately left out.
  function ThemedLayout(layout) {
    var surface = ResolveToken("surface-1");
    var ink = ResolveToken("text-primary");
    var secondary = ResolveToken("text-secondary");
    var grid = ResolveToken("gridline");
    var axis = ResolveToken("axis-line");

    function ThemedAxis(source) {
      return Object.assign({}, source, {
        gridcolor: grid,
        linecolor: axis,
        tickcolor: axis,
        tickfont: { color: secondary, size: 12 },
        title: Object.assign({}, source.title, { font: { color: secondary, size: 13 } })
      });
    }

    return Object.assign({}, layout, {
      paper_bgcolor: surface,
      plot_bgcolor: surface,
      font: { family: FONT, size: 13, color: secondary },
      xaxis: ThemedAxis(layout.xaxis || {}),
      yaxis: ThemedAxis(layout.yaxis || {}),
      legend: Object.assign({}, layout.legend, { font: { color: ink, size: 12 } }),
      hoverlabel: { bgcolor: surface, bordercolor: axis, font: { family: FONT, color: ink, size: 12 } }
    });
  }

  function FormatNumber(value) {
    if (typeof value !== "number" || !isFinite(value)) { return String(value); }
    if (value !== 0 && (Math.abs(value) < 1e-3 || Math.abs(value) >= 1e6)) { return value.toExponential(3); }
    return value.toFixed(4).replace(/\.?0+$/, "");
  }

  // The table twin of a figure. Only legended traces are tabulated: a cloud of hundreds of faint
  // sample lines is the one thing a table cannot usefully say, and the mean and the reference
  // lines are what a reader wants the numbers for anyway.
  // The table twin of a heatmap: the grid itself, x across the header and y down the first
  // column. A reader who cannot distinguish two steps of the ramp -- or who is reading this on
  // paper -- gets the numbers rather than nothing.
  function BuildGridTable(figure, trace) {
    var table = document.createElement("table");
    var head = document.createElement("tr");
    var body = document.createElement("tbody");

    function AddCell(row, text, header) {
      var cell = document.createElement(header ? "th" : "td");
      cell.textContent = text;
      row.appendChild(cell);
    }

    AddCell(head, figure.layout.yaxis.title.text + " \\ " + figure.layout.xaxis.title.text, true);
    trace.x.forEach(function (x) { AddCell(head, FormatNumber(x), true); });

    trace.z.forEach(function (row, index) {
      var element = document.createElement("tr");
      AddCell(element, FormatNumber(trace.y[index]), true);
      row.forEach(function (value) { AddCell(element, FormatNumber(value), false); });
      body.appendChild(element);
    });

    var header = document.createElement("thead");
    header.appendChild(head);
    table.appendChild(header);
    table.appendChild(body);
    return table;
  }

  function BuildTable(figure) {
    var heatmap = figure.data.find(function (trace) { return trace.type === "heatmap"; });
    if (heatmap) { return BuildGridTable(figure, heatmap); }

    var series = figure.data.filter(function (trace) { return trace.showlegend !== false; });
    if (series.length === 0) { return null; }

    var sharedX = series.every(function (trace) {
      return trace.x.length === series[0].x.length &&
             trace.x.every(function (value, index) { return value === series[0].x[index]; });
    });

    var table = document.createElement("table");
    var head = document.createElement("tr");
    var body = document.createElement("tbody");

    function AddHeaderCell(text, colorRole) {
      var cell = document.createElement("th");
      if (colorRole) {
        var swatch = document.createElement("span");
        swatch.className = "swatch";
        swatch.dataset.role = colorRole;
        swatch.style.background = ResolveToken(colorRole);
        cell.appendChild(swatch);
      }
      cell.appendChild(document.createTextNode(text));
      head.appendChild(cell);
    }

    function AddRow(cells) {
      var row = document.createElement("tr");
      cells.forEach(function (text) {
        var cell = document.createElement("td");
        cell.textContent = text;
        row.appendChild(cell);
      });
      body.appendChild(row);
    }

    if (sharedX) {
      AddHeaderCell(figure.layout.xaxis.title.text);
      series.forEach(function (trace) { AddHeaderCell(trace.name, trace.colorRole); });
      series[0].x.forEach(function (x, index) {
        AddRow([FormatNumber(x)].concat(series.map(function (trace) { return FormatNumber(trace.y[index]); })));
      });
    } else {
      // Traces on different x grids -- a phase portrait, say -- tabulate long rather than wide.
      AddHeaderCell("Series");
      AddHeaderCell(figure.layout.xaxis.title.text);
      AddHeaderCell(figure.layout.yaxis.title.text);
      series.forEach(function (trace) {
        trace.x.forEach(function (x, index) {
          AddRow([trace.name, FormatNumber(x), FormatNumber(trace.y[index])]);
        });
      });
    }

    var header = document.createElement("thead");
    header.appendChild(head);
    table.appendChild(header);
    table.appendChild(body);
    return table;
  }

  var plotNodes = [];

  function Render() {
    var container = document.getElementById("figures");
    FIGURES.forEach(function (figure, index) {
      var card = document.createElement("section");
      card.className = "card";

      var heading = document.createElement("h2");
      heading.textContent = figure.title;
      card.appendChild(heading);

      var plot = document.createElement("div");
      plot.className = "plot";
      plot.id = "plot-" + index;
      card.appendChild(plot);

      var table = BuildTable(figure);
      if (table) {
        var details = document.createElement("details");
        var summary = document.createElement("summary");
        summary.textContent = "Table view";
        var scroll = document.createElement("div");
        scroll.className = "table-scroll";
        scroll.appendChild(table);
        details.appendChild(summary);
        details.appendChild(scroll);
        card.appendChild(details);
      }

      container.appendChild(card);
      Plotly.newPlot(plot, ThemedData(figure.data), ThemedLayout(figure.layout), PLOT_CONFIG);
      plotNodes.push(plot);
    });
  }

  // Re-resolves every role against the now-current theme. Plotly.react diffs against what is
  // already drawn, so this recolors in place rather than rebuilding each figure.
  function Retheme() {
    plotNodes.forEach(function (plot, index) {
      Plotly.react(plot, ThemedData(FIGURES[index].data), ThemedLayout(FIGURES[index].layout), PLOT_CONFIG);
    });
    document.querySelectorAll(".swatch").forEach(function (swatch) {
      swatch.style.background = ResolveToken(swatch.dataset.role);
    });
  }

  var MODES = ["system", "light", "dark"];

  function ApplyMode(mode) {
    if (mode === "system") {
      document.documentElement.removeAttribute("data-theme");
    } else {
      document.documentElement.setAttribute("data-theme", mode);
    }
    document.getElementById("theme-toggle").textContent = "Theme: " + mode;
  }

  function StoredMode() {
    try {
      var stored = window.localStorage.getItem("fbsde-report-theme");
      return MODES.indexOf(stored) >= 0 ? stored : "system";
    } catch (error) {
      // Private windows and blocked site data both throw here; the system theme is a fine
      // fallback and the page must not fail to render over a preference.
      return "system";
    }
  }

  var mode = StoredMode();
  ApplyMode(mode);
  Render();

  document.getElementById("theme-toggle").addEventListener("click", function () {
    mode = MODES[(MODES.indexOf(mode) + 1) % MODES.length];
    ApplyMode(mode);
    try {
      window.localStorage.setItem("fbsde-report-theme", mode);
    } catch (error) {
      // Nothing to do: the choice simply will not survive a reload.
    }
    Retheme();
  });

  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", function () {
    if (mode === "system") { Retheme(); }
  });
}());
</script>
</body>
</html>
)HTML";

// Replaces every occurrence of `placeholder` in `text` with `value`.
auto Substitute(std::string& text, std::string_view placeholder, std::string_view value) noexcept -> void {
  std::size_t position = text.find(placeholder);
  while (position != std::string::npos) {
    text.replace(position, placeholder.size(), value);
    position = text.find(placeholder, position + value.size());
  }
}

// Escapes the five characters that would otherwise be markup. Applied to the page's own title and
// subtitle; figure titles and series names travel inside the JSON block and are written into the
// DOM as text, which needs no escaping.
auto EscapeHtml(std::string_view text) noexcept -> std::string {
  std::string escaped;
  escaped.reserve(text.size());
  for (const char character : text) {
    switch (character) {
      case '&':
        escaped += "&amp;";
        break;
      case '<':
        escaped += "&lt;";
        break;
      case '>':
        escaped += "&gt;";
        break;
      case '"':
        escaped += "&quot;";
        break;
      case '\'':
        escaped += "&#39;";
        break;
      default:
        escaped += character;
        break;
    }
  }
  return escaped;
}

}  // namespace

PlotlyReport::PlotlyReport(std::string title, std::string subtitle) noexcept
    : title_(std::move(title)), subtitle_(std::move(subtitle)) {}

auto PlotlyReport::AddFigure(PlotlyFigure figure) noexcept -> void {
  figures_.push_back(std::move(figure));
}

auto PlotlyReport::ToHtml() const noexcept -> std::string {
  nlohmann::json figures = nlohmann::json::array();
  for (const PlotlyFigure& figure : figures_) {
    figures.push_back(figure.ToJson());
  }

  // Inside a <script> element the HTML tokenizer stops at the first `</script`, wherever it
  // appears -- including in the middle of a JSON string. Every `</` in the serialized data is
  // therefore written `<\/`, which JSON parses back to `</` but which contains no sequence the
  // tokenizer can act on. The slash is escaped rather than the `<` because the alternative spells
  // the escape `\uXXXX`, a form that tooling which touches this source is liable to interpret
  // rather than pass through.
  std::string serialized = figures.dump();
  Substitute(serialized, "</", R"(<\/)");

  std::string html(kPageTemplate);
  Substitute(html, "{{PLOTLY_SRC}}", kPlotlyScriptUrl);
  Substitute(html, "{{TITLE}}", EscapeHtml(title_));
  Substitute(html, "{{SUBTITLE}}", EscapeHtml(subtitle_));
  Substitute(html, "{{FIGURES_JSON}}", serialized);
  return html;
}

auto PlotlyReport::WriteHtml(const std::filesystem::path& path) const noexcept -> Result<> {
  const std::filesystem::path parent = path.parent_path();
  if (!parent.empty()) {
    std::error_code error;
    std::filesystem::create_directories(parent, error);
    RESULT_ASSERT(!error, "PlotlyReport::WriteHtml: could not create the output directory.");
  }

  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  RESULT_ASSERT(stream.is_open(), "PlotlyReport::WriteHtml: could not open the output file for writing.");

  const std::string html = ToHtml();
  stream.write(html.data(), static_cast<std::streamsize>(html.size()));
  RESULT_ASSERT(stream.good(), "PlotlyReport::WriteHtml: writing the output file failed.");

  return SuccessResult();
}

}  // namespace fbsde_traj_opt::viz
