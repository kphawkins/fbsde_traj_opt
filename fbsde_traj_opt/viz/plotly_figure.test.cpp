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

TEST(PlotlyFigureTest, AnimatingALineRestsOnTheLastFrameAndPinsTheAxis) {
  PlotlyFigure figure("Approach", "State", "Value");
  ASSERT_TRUE(figure.AddLine(kStages, kValues, LineStyle{.name = "Target"}).has_value());
  ASSERT_TRUE(figure
                  .AddLine(kStages,
                           std::array<double, 4>{0.0, 0.0, 0.0, 0.0},
                           LineStyle{.name = "Fitted", .color_role = PlotColorRole::kSeries2})
                  .has_value());

  const std::array<AnimationFrame, 3> frames{
      AnimationFrame{.label = "start", .trace_values = {{0.0, 0.0, 0.0, 0.0}}},
      AnimationFrame{.label = "middle", .trace_values = {{0.5, 0.3, 0.2, 0.1}}},
      AnimationFrame{.label = "end", .trace_values = {{1.0, 0.5, 0.25, 0.125}}},
  };
  const std::array<std::size_t, 1> animated{1};

  ASSERT_TRUE(figure.Animate(animated, frames, AnimationStyle{.slider_prefix = "After "}).has_value());
  EXPECT_EQ(figure.FrameCount(), 3U);

  const nlohmann::json json = figure.ToJson();

  // The figure rests on the last frame, so a reader who never presses play sees the result.
  EXPECT_EQ(json.at("data").at(1).at("y"), nlohmann::json({1.0, 0.5, 0.25, 0.125}));

  // Every frame is present, in order, and names the traces it replaces.
  ASSERT_EQ(json.at("frames").size(), 3U);
  EXPECT_EQ(json.at("frames").at(0).at("data").at(0).at("y"), nlohmann::json({0.0, 0.0, 0.0, 0.0}));
  EXPECT_EQ(json.at("frames").at(0).at("traces"), nlohmann::json({1}));
  EXPECT_EQ(json.at("frames").at(2).at("data").at(0).at("y"), nlohmann::json({1.0, 0.5, 0.25, 0.125}));

  // The y range is pinned across the frames and the static trace, so the animation shows the
  // curve moving rather than the axis chasing it.
  const nlohmann::json range = json.at("layout").at("yaxis").at("range");
  ASSERT_EQ(range.size(), 2U);
  EXPECT_LE(range.at(0).get<double>(), 0.0);
  EXPECT_GE(range.at(1).get<double>(), 1.0);

  // The controls are there, and the slider rests at the end with the labels it was given.
  const nlohmann::json slider = json.at("layout").at("sliders").at(0);
  EXPECT_EQ(slider.at("active"), 2);
  EXPECT_EQ(slider.at("currentvalue").at("prefix"), "After ");
  ASSERT_EQ(slider.at("steps").size(), 3U);
  EXPECT_EQ(slider.at("steps").at(1).at("label"), "middle");
  EXPECT_EQ(json.at("layout").at("updatemenus").at(0).at("buttons").at(0).at("label"), "Play");
}

TEST(PlotlyFigureTest, AnimatingAHeatmapPinsTheColorRangeAndReNestsEachFrame) {
  constexpr std::array<double, 3> kGridX{0.0, 1.0, 2.0};
  constexpr std::array<double, 2> kGridY{0.0, 1.0};
  constexpr std::array<double, 6> kGridZ{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

  PlotlyFigure figure("Error", "x1", "x2");
  ASSERT_TRUE(figure.AddHeatmap(kGridX, kGridY, kGridZ, HeatmapStyle{.name = "Error"}).has_value());

  const std::array<AnimationFrame, 2> frames{
      AnimationFrame{.label = "start", .trace_values = {{1.0, 2.0, 3.0, 4.0, 5.0, 9.0}}},
      AnimationFrame{.label = "end", .trace_values = {{0.1, 0.2, 0.3, 0.4, 0.5, 0.6}}},
  };
  const std::array<std::size_t, 1> animated{0};

  ASSERT_TRUE(figure.Animate(animated, frames, AnimationStyle{}).has_value());

  const nlohmann::json json = figure.ToJson();
  const nlohmann::json trace = json.at("data").at(0);

  // Rests on the last frame, re-nested row by row.
  EXPECT_EQ(trace.at("z"), nlohmann::json({{0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}}));

  // The color range spans every frame, so the first frame's 9 and the last frame's 0.1 are read
  // against the same scale and the error is seen to shrink.
  EXPECT_EQ(trace.at("zmin"), 0.1);
  EXPECT_EQ(trace.at("zmax"), 9.0);

  EXPECT_EQ(json.at("frames").at(0).at("data").at(0).at("z"), nlohmann::json({{1.0, 2.0, 3.0}, {4.0, 5.0, 9.0}}));
}

TEST(PlotlyFigureTest, AnimatingAHeatmapCenteredOnZeroKeepsTheRangeSymmetric) {
  constexpr std::array<double, 2> kGridX{0.0, 1.0};
  constexpr std::array<double, 2> kGridY{0.0, 1.0};
  constexpr std::array<double, 4> kGridZ{0.0, 0.0, 0.0, 0.0};

  PlotlyFigure figure("Error", "x1", "x2");
  ASSERT_TRUE(figure
                  .AddHeatmap(kGridX,
                              kGridY,
                              kGridZ,
                              HeatmapStyle{.colorscale_role = PlotColorscaleRole::kDiverging, .centered_on_zero = true})
                  .has_value());

  const std::array<AnimationFrame, 1> frames{AnimationFrame{.label = "only", .trace_values = {{-1.0, 0.25, 4.0, 0.5}}}};
  const std::array<std::size_t, 1> animated{0};

  ASSERT_TRUE(figure.Animate(animated, frames, AnimationStyle{}).has_value());

  // Symmetric, not [-1, 4]: the neutral color has to keep meaning zero.
  const nlohmann::json trace = figure.ToJson().at("data").at(0);
  EXPECT_EQ(trace.at("zmin"), -4.0);
  EXPECT_EQ(trace.at("zmax"), 4.0);
}

TEST(PlotlyFigureTest, AnimateRejectsFramesThatDoNotMatchTheTracesTheyReplace) {
  PlotlyFigure figure("Approach", "State", "Value");
  ASSERT_TRUE(figure.AddLine(kStages, kValues, LineStyle{.name = "Fitted"}).has_value());

  const std::array<std::size_t, 1> animated{0};
  const std::array<std::size_t, 1> missing{3};
  const std::array<AnimationFrame, 1> good{AnimationFrame{.label = "a", .trace_values = {{1.0, 2.0, 3.0, 4.0}}}};
  const std::array<AnimationFrame, 1> wrong_length{AnimationFrame{.label = "a", .trace_values = {{1.0, 2.0}}}};
  const std::array<AnimationFrame, 1> wrong_count{
      AnimationFrame{.label = "a", .trace_values = {{1.0, 2.0, 3.0, 4.0}, {1.0, 2.0, 3.0, 4.0}}}};
  const std::array<AnimationFrame, 0> none{};

  EXPECT_FALSE(figure.Animate(animated, none, AnimationStyle{}).has_value());
  EXPECT_FALSE(figure.Animate(missing, good, AnimationStyle{}).has_value());
  EXPECT_FALSE(figure.Animate(animated, wrong_length, AnimationStyle{}).has_value());
  EXPECT_FALSE(figure.Animate(animated, wrong_count, AnimationStyle{}).has_value());
  EXPECT_EQ(figure.FrameCount(), 0U);

  // And a figure is animated once: its frames were sized against the traces it had at the time.
  ASSERT_TRUE(figure.Animate(animated, good, AnimationStyle{}).has_value());
  EXPECT_FALSE(figure.Animate(animated, good, AnimationStyle{}).has_value());
  EXPECT_FALSE(figure.AddLine(kStages, kValues, LineStyle{.name = "Late"}).has_value());
}

TEST(PlotlyFigureTest, AnUnanimatedFigureCarriesNoFrames) {
  PlotlyFigure figure("Plain", "State", "Value");
  ASSERT_TRUE(figure.AddLine(kStages, kValues, LineStyle{.name = "a"}).has_value());

  EXPECT_EQ(figure.FrameCount(), 0U);
  EXPECT_FALSE(figure.ToJson().contains("frames"));
  EXPECT_FALSE(figure.ToJson().at("layout").contains("sliders"));
}

}  // namespace
}  // namespace fbsde_traj_opt::viz
