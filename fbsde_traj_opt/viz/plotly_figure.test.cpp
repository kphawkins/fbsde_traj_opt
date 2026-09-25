// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#include "fbsde_traj_opt/viz/plotly_figure.hpp"

#include <array>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

namespace fbsde_traj_opt::viz {
namespace {

constexpr std::array<double, 4> kStages{0.0, 1.0, 2.0, 3.0};
constexpr std::array<double, 4> kValues{1.0, 0.5, 0.25, 0.125};

TEST(PlotlyFigureTest, NewFigureHasNoLinesAndKeepsItsTitles) {
  const PlotlyFigure figure("Expected cost-to-go", "Stage", "Cost");

  EXPECT_EQ(figure.title(), "Expected cost-to-go");
  EXPECT_EQ(figure.LineCount(), 0U);

  const nlohmann::json json = figure.ToJson();
  EXPECT_EQ(json.at("title"), "Expected cost-to-go");
  EXPECT_TRUE(json.at("data").empty());
  EXPECT_EQ(json.at("layout").at("xaxis").at("title").at("text"), "Stage");
  EXPECT_EQ(json.at("layout").at("yaxis").at("title").at("text"), "Cost");
}

TEST(PlotlyFigureTest, LogarithmicYAxisIsMarkedInTheLayoutAndTheAxisTitle) {
  PlotlyFigure figure("Expected cost-to-go", "Stage", "Cost");

  figure.UseLogarithmicYAxis();

  const nlohmann::json layout = figure.ToJson().at("layout");
  EXPECT_EQ(layout.at("yaxis").at("type"), "log");
  // A reader who misses that a scale is logarithmic misreads every distance on it, so the axis
  // title says so rather than leaving it to the tick labels.
  EXPECT_EQ(layout.at("yaxis").at("title").at("text"), "Cost (log scale)");
  EXPECT_FALSE(layout.at("xaxis").contains("type"));
}

TEST(PlotlyFigureTest, AddLineStoresThePointsAndTheStyle) {
  PlotlyFigure figure("Trajectories", "Stage", "Position");

  const LineStyle style{.name = "mean trajectory",
                        .color_role = PlotColorRole::kSeries2,
                        .width = 2.0,
                        .opacity = 1.0,
                        .show_in_legend = true,
                        .legend_group = "lqr",
                        .show_on_hover = true};
  ASSERT_TRUE(figure.AddLine(kStages, kValues, style).has_value());

  EXPECT_EQ(figure.LineCount(), 1U);

  const nlohmann::json trace = figure.ToJson().at("data").at(0);
  EXPECT_EQ(trace.at("type"), "scatter");
  EXPECT_EQ(trace.at("mode"), "lines");
  EXPECT_EQ(trace.at("x").get<std::vector<double>>(), std::vector<double>(kStages.begin(), kStages.end()));
  EXPECT_EQ(trace.at("y").get<std::vector<double>>(), std::vector<double>(kValues.begin(), kValues.end()));
  EXPECT_EQ(trace.at("name"), "mean trajectory");
  EXPECT_EQ(trace.at("legendgroup"), "lqr");
  EXPECT_EQ(trace.at("colorRole"), "series-2");
  EXPECT_DOUBLE_EQ(trace.at("line").at("width").get<double>(), 2.0);
  EXPECT_EQ(trace.at("hoverinfo"), "all");
}

TEST(PlotlyFigureTest, FaintSampleLinesAreExcludedFromTheLegendAndHover) {
  PlotlyFigure figure("Trajectories", "Stage", "Position");

  const LineStyle style{.name = "sampled trajectories",
                        .color_role = PlotColorRole::kSeries1,
                        .width = 1.0,
                        .opacity = 0.12,
                        .show_in_legend = false,
                        .legend_group = "samples",
                        .show_on_hover = false};
  ASSERT_TRUE(figure.AddLine(kStages, kValues, style).has_value());

  const nlohmann::json trace = figure.ToJson().at("data").at(0);
  EXPECT_FALSE(trace.at("showlegend").get<bool>());
  EXPECT_EQ(trace.at("hoverinfo"), "skip");
  EXPECT_DOUBLE_EQ(trace.at("opacity").get<double>(), 0.12);
}

TEST(PlotlyFigureTest, AddLineRejectsMismatchedPointCounts) {
  PlotlyFigure figure("Trajectories", "Stage", "Position");

  constexpr std::array<double, 2> kShortValues{1.0, 2.0};

  EXPECT_FALSE(figure.AddLine(kStages, kShortValues, LineStyle{}).has_value());
  EXPECT_EQ(figure.LineCount(), 0U);
}

TEST(PlotlyFigureTest, AddLineAcceptsEmptyRanges) {
  PlotlyFigure figure("Trajectories", "Stage", "Position");

  EXPECT_TRUE(figure.AddLine({}, {}, LineStyle{}).has_value());
  EXPECT_EQ(figure.LineCount(), 1U);
}

// The design invariant: the figure spec carries color roles, never colors, so the page can
// resolve a role differently in light and dark mode.
TEST(PlotlyFigureTest, FigureJsonContainsNoConcreteColors) {
  PlotlyFigure figure("Trajectories", "Stage", "Position");
  ASSERT_TRUE(figure.AddLine(kStages, kValues, LineStyle{.name = "one"}).has_value());
  ASSERT_TRUE(
      figure.AddLine(kStages, kValues, LineStyle{.name = "two", .color_role = PlotColorRole::kMuted}).has_value());

  const std::string serialized = figure.ToJson().dump();

  EXPECT_EQ(serialized.find('#'), std::string::npos) << "a hex color leaked into the figure spec";
  EXPECT_EQ(serialized.find("rgb"), std::string::npos) << "an rgb color leaked into the figure spec";
  EXPECT_EQ(serialized.find("color\""), std::string::npos) << "a color attribute leaked into the figure spec";
}

TEST(PlotlyFigureTest, EveryColorRoleHasADistinctName) {
  constexpr std::array<PlotColorRole, 9> kRoles{
      PlotColorRole::kSeries1,
      PlotColorRole::kSeries2,
      PlotColorRole::kSeries3,
      PlotColorRole::kSeries4,
      PlotColorRole::kSeries5,
      PlotColorRole::kSeries6,
      PlotColorRole::kSeries7,
      PlotColorRole::kSeries8,
      PlotColorRole::kMuted,
  };

  std::set<std::string_view> names;
  for (const PlotColorRole role : kRoles) {
    names.insert(PlotColorRoleName(role));
  }

  EXPECT_EQ(names.size(), kRoles.size());
}

TEST(PlotlyFigureTest, AddHeatmapStoresTheGridRowByRow) {
  constexpr std::array<double, 3> kGridX{0.0, 1.0, 2.0};
  constexpr std::array<double, 2> kGridY{10.0, 20.0};
  // Row-major: the first three values are the row at y = 10.
  constexpr std::array<double, 6> kGridZ{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

  PlotlyFigure figure("Value function", "Position", "Velocity");
  ASSERT_TRUE(
      figure.AddHeatmap(kGridX, kGridY, kGridZ, HeatmapStyle{.name = "V", .value_label = "Cost-to-go"}).has_value());

  EXPECT_TRUE(figure.HoldsHeatmap());
  EXPECT_EQ(figure.LineCount(), 1U);

  const nlohmann::json trace = figure.ToJson().at("data").at(0);
  EXPECT_EQ(trace.at("type"), "heatmap");
  EXPECT_EQ(trace.at("x"), nlohmann::json({0.0, 1.0, 2.0}));
  EXPECT_EQ(trace.at("y"), nlohmann::json({10.0, 20.0}));
  EXPECT_EQ(trace.at("z"), nlohmann::json({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}));
  EXPECT_EQ(trace.at("colorscaleRole"), "scale-sequential");
  EXPECT_EQ(trace.at("colorbar").at("title").at("text"), "Cost-to-go");
  EXPECT_FALSE(trace.contains("zmid"));

  // A heatmap reports the cell under the pointer; the shared crosshair has no series to read.
  EXPECT_EQ(figure.ToJson().at("layout").at("hovermode"), "closest");
}

TEST(PlotlyFigureTest, ADivergingHeatmapCanBePinnedToZero) {
  constexpr std::array<double, 2> kGridX{0.0, 1.0};
  constexpr std::array<double, 2> kGridY{0.0, 1.0};
  constexpr std::array<double, 4> kGridZ{-1.0, 0.5, 2.0, -3.0};

  PlotlyFigure figure("Error", "Position", "Velocity");
  ASSERT_TRUE(figure
                  .AddHeatmap(kGridX,
                              kGridY,
                              kGridZ,
                              HeatmapStyle{.name = "error",
                                           .colorscale_role = PlotColorscaleRole::kDiverging,
                                           .value_label = "Signed error",
                                           .centered_on_zero = true})
                  .has_value());

  const nlohmann::json trace = figure.ToJson().at("data").at(0);
  EXPECT_EQ(trace.at("colorscaleRole"), "scale-diverging");
  EXPECT_EQ(trace.at("zmid"), 0.0);
}

TEST(PlotlyFigureTest, AddHeatmapRejectsAGridThatDoesNotMatchItsAxes) {
  constexpr std::array<double, 3> kGridX{0.0, 1.0, 2.0};
  constexpr std::array<double, 2> kGridY{0.0, 1.0};
  constexpr std::array<double, 5> kTooFew{1.0, 2.0, 3.0, 4.0, 5.0};
  constexpr std::array<double, 0> kEmpty{};

  PlotlyFigure figure("Value function", "Position", "Velocity");

  EXPECT_FALSE(figure.AddHeatmap(kGridX, kGridY, kTooFew, HeatmapStyle{}).has_value());
  EXPECT_FALSE(figure.AddHeatmap(kEmpty, kGridY, kEmpty, HeatmapStyle{}).has_value());
  EXPECT_FALSE(figure.HoldsHeatmap());
}

TEST(PlotlyFigureTest, AFigureHoldsEitherLinesOrAHeatmapButNotBoth) {
  constexpr std::array<double, 2> kGridX{0.0, 1.0};
  constexpr std::array<double, 2> kGridY{0.0, 1.0};
  constexpr std::array<double, 4> kGridZ{1.0, 2.0, 3.0, 4.0};

  PlotlyFigure with_line("Mixed", "Stage", "Cost");
  ASSERT_TRUE(with_line.AddLine(kStages, kValues, LineStyle{.name = "a"}).has_value());
  EXPECT_FALSE(with_line.AddHeatmap(kGridX, kGridY, kGridZ, HeatmapStyle{}).has_value());

  PlotlyFigure with_heatmap("Mixed", "Stage", "Cost");
  ASSERT_TRUE(with_heatmap.AddHeatmap(kGridX, kGridY, kGridZ, HeatmapStyle{}).has_value());
  EXPECT_FALSE(with_heatmap.AddLine(kStages, kValues, LineStyle{.name = "a"}).has_value());
  EXPECT_EQ(with_heatmap.LineCount(), 1U);
}

TEST(PlotlyFigureTest, AHeatmapSpecCarriesNoConcreteColorsEither) {
  constexpr std::array<double, 2> kGridX{0.0, 1.0};
  constexpr std::array<double, 2> kGridY{0.0, 1.0};
  constexpr std::array<double, 4> kGridZ{1.0, 2.0, 3.0, 4.0};

  PlotlyFigure figure("Value function", "Position", "Velocity");
  ASSERT_TRUE(figure.AddHeatmap(kGridX, kGridY, kGridZ, HeatmapStyle{.value_label = "Cost"}).has_value());

  const std::string serialized = figure.ToJson().dump();

  EXPECT_EQ(serialized.find('#'), std::string::npos) << "a hex color leaked into the figure spec";
  EXPECT_EQ(serialized.find("colorscale\":"), std::string::npos) << "a resolved colorscale leaked into the spec";
}

TEST(PlotlyFigureTest, EveryColorscaleRoleHasADistinctName) {
  EXPECT_NE(PlotColorscaleRoleName(PlotColorscaleRole::kSequential),
            PlotColorscaleRoleName(PlotColorscaleRole::kDiverging));
}

}  // namespace
}  // namespace fbsde_traj_opt::viz
