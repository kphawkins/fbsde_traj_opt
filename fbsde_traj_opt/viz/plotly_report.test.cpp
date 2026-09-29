// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#include "fbsde_traj_opt/viz/plotly_report.hpp"

#include <array>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "fbsde_traj_opt/viz/plotly_figure.hpp"

namespace fbsde_traj_opt::viz {
namespace {

constexpr std::array<double, 3> kStages{0.0, 1.0, 2.0};
constexpr std::array<double, 3> kValues{3.0, 2.0, 1.0};

auto MakeFigure(std::string title) noexcept -> PlotlyFigure {
  PlotlyFigure figure(std::move(title), "Stage", "Cost");
  EXPECT_TRUE(figure.AddLine(kStages, kValues, LineStyle{.name = "optimal"}).has_value());
  return figure;
}

// Extracts the embedded figure JSON from a rendered page, undoing the `<` escaping ToHtml()
// applies on the way out.
auto EmbeddedFigureJson(const std::string& html) -> nlohmann::json {
  const std::string open_tag = R"(<script id="figure-data" type="application/json">)";
  const std::size_t start = html.find(open_tag);
  EXPECT_NE(start, std::string::npos);
  const std::size_t content_start = start + open_tag.size();
  const std::size_t content_end = html.find("</script>", content_start);
  EXPECT_NE(content_end, std::string::npos);
  return nlohmann::json::parse(html.substr(content_start, content_end - content_start));
}

TEST(PlotlyReportTest, EmptyReportStillRendersAWholePage) {
  const PlotlyReport report("Trajectory report", "A subtitle");

  EXPECT_EQ(report.FigureCount(), 0U);

  const std::string html = report.ToHtml();
  EXPECT_TRUE(html.starts_with("<!DOCTYPE html>"));
  EXPECT_NE(html.find("</html>"), std::string::npos);
  EXPECT_NE(html.find("Trajectory report"), std::string::npos);
  EXPECT_NE(html.find("A subtitle"), std::string::npos);
  EXPECT_TRUE(EmbeddedFigureJson(html).empty());
}

TEST(PlotlyReportTest, NoPlaceholderSurvivesRendering) {
  PlotlyReport report("Title", "Subtitle");
  report.AddFigure(MakeFigure("A figure"));

  const std::string html = report.ToHtml();

  // Only the template's own placeholders: `{{` and `}}` on their own occur in the page's CSS and
  // script, which are not substituted.
  for (const std::string_view placeholder : {"{{TITLE}}", "{{SUBTITLE}}", "{{PLOTLY_SRC}}", "{{FIGURES_JSON}}"}) {
    EXPECT_EQ(html.find(placeholder), std::string::npos) << "unsubstituted placeholder " << placeholder;
  }
}

TEST(PlotlyReportTest, FiguresAppearInTheOrderAdded) {
  PlotlyReport report("Title", "");
  report.AddFigure(MakeFigure("First"));
  report.AddFigure(MakeFigure("Second"));

  EXPECT_EQ(report.FigureCount(), 2U);

  const nlohmann::json figures = EmbeddedFigureJson(report.ToHtml());
  ASSERT_EQ(figures.size(), 2U);
  EXPECT_EQ(figures.at(0).at("title"), "First");
  EXPECT_EQ(figures.at(1).at("title"), "Second");
  EXPECT_EQ(figures.at(0).at("data").at(0).at("colorRole"), "series-1");
}

TEST(PlotlyReportTest, PageLoadsThePinnedPlotlyBuild) {
  const PlotlyReport report("Title", "");

  const std::string html = report.ToHtml();

  EXPECT_NE(html.find(std::string(PlotlyReport::kPlotlyScriptUrl)), std::string::npos);
  // A floating "latest" URL would silently change how an archived report renders.
  EXPECT_EQ(html.find("plotly-latest"), std::string::npos);
}

TEST(PlotlyReportTest, TitleAndSubtitleAreHtmlEscaped) {
  const PlotlyReport report("Cost <b>&</b> value", R"(a "quoted" phrase)");

  const std::string html = report.ToHtml();

  EXPECT_NE(html.find("Cost &lt;b&gt;&amp;&lt;/b&gt; value"), std::string::npos);
  EXPECT_EQ(html.find("<b>&</b>"), std::string::npos);
  EXPECT_NE(html.find("&quot;quoted&quot;"), std::string::npos);
}

// A figure title containing `</script>` would otherwise end the embedded JSON block early and
// spill the rest of the data into the document as markup. Note that the `<img>` in this title is
// harmless in itself: inside a script element nothing but `</script` ends the block, so the
// escaping has one job and this test checks exactly that job.
TEST(PlotlyReportTest, FigureDataCannotBreakOutOfTheScriptBlock) {
  PlotlyReport report("Title", "");
  report.AddFigure(MakeFigure("</script><img src=x onerror=alert(1)>"));

  const std::string html = report.ToHtml();

  const std::string open_tag = R"(<script id="figure-data" type="application/json">)";
  const std::size_t content_start = html.find(open_tag) + open_tag.size();
  ASSERT_NE(content_start, std::string::npos);

  // The title's own closing tag is nowhere in the data: the first `</script>` after the block
  // begins is the template's.
  EXPECT_EQ(html.find(R"(</script><img)", content_start), std::string::npos);

  // And the title survives intact once the JSON is parsed; only its transport is escaped. Had the
  // escaping failed, the block would have been cut short and this parse would have thrown.
  EXPECT_EQ(EmbeddedFigureJson(html).at(0).at("title"), "</script><img src=x onerror=alert(1)>");
}

TEST(PlotlyReportTest, EveryColorRoleUsedIsDefinedByThePagePalette) {
  PlotlyReport report("Title", "");
  PlotlyFigure figure("A figure", "Stage", "Cost");
  ASSERT_TRUE(
      figure.AddLine(kStages, kValues, LineStyle{.name = "a", .color_role = PlotColorRole::kSeries8}).has_value());
  ASSERT_TRUE(
      figure.AddLine(kStages, kValues, LineStyle{.name = "b", .color_role = PlotColorRole::kMuted}).has_value());
  report.AddFigure(std::move(figure));

  const std::string html = report.ToHtml();

  EXPECT_NE(html.find("--series-8:"), std::string::npos);
  EXPECT_NE(html.find("--muted:"), std::string::npos);
}

// The dark palette must be its own set of values, not a filter or an inversion applied to the
// light one.
TEST(PlotlyReportTest, PageDefinesADistinctDarkPalette) {
  const PlotlyReport report("Title", "");

  const std::string html = report.ToHtml();

  EXPECT_NE(html.find("prefers-color-scheme: dark"), std::string::npos);
  EXPECT_NE(html.find(R"(:root[data-theme="dark"])"), std::string::npos);
  // The light and dark steps of the first categorical slot differ.
  EXPECT_NE(html.find("#2a78d6"), std::string::npos);
  EXPECT_NE(html.find("#3987e5"), std::string::npos);
  EXPECT_EQ(html.find("filter: invert"), std::string::npos);
}

TEST(PlotlyReportTest, EveryColorscaleRoleUsedIsDefinedByThePagePalette) {
  constexpr std::array<double, 2> kGridX{0.0, 1.0};
  constexpr std::array<double, 2> kGridY{0.0, 1.0};
  constexpr std::array<double, 4> kGridZ{1.0, 2.0, 3.0, 4.0};

  PlotlyReport report("Title", "");
  PlotlyFigure sequential("Value", "Position", "Velocity");
  ASSERT_TRUE(sequential.AddHeatmap(kGridX, kGridY, kGridZ, HeatmapStyle{.value_label = "Cost"}).has_value());
  report.AddFigure(std::move(sequential));

  PlotlyFigure diverging("Error", "Position", "Velocity");
  ASSERT_TRUE(
      diverging.AddHeatmap(kGridX, kGridY, kGridZ, HeatmapStyle{.colorscale_role = PlotColorscaleRole::kDiverging})
          .has_value());
  report.AddFigure(std::move(diverging));

  const std::string html = report.ToHtml();

  // Every stop the page's resolver will ask for must be defined, in both modes.
  for (const std::string_view stop : {"--scale-sequential-0:",
                                      "--scale-sequential-1:",
                                      "--scale-sequential-2:",
                                      "--scale-sequential-3:",
                                      "--scale-sequential-4:",
                                      "--scale-diverging-0:",
                                      "--scale-diverging-1:",
                                      "--scale-diverging-2:"}) {
    EXPECT_NE(html.find(stop), std::string::npos) << stop;
  }

  // The sequential ramp is reversed rather than repeated in dark mode, so that its "least" end
  // stays nearest whichever surface it is drawn on.
  EXPECT_NE(html.find("--scale-sequential-0: #cde2fb;"), std::string::npos);
  EXPECT_NE(html.find("--scale-sequential-0: #0d366b;"), std::string::npos);

  // The diverging midpoint is neutral gray in both modes: zero must read as nothing.
  EXPECT_NE(html.find("--scale-diverging-1:  #f0efec;"), std::string::npos);
  EXPECT_NE(html.find("--scale-diverging-1:  #383835;"), std::string::npos);
}

TEST(PlotlyReportTest, WriteHtmlCreatesMissingParentDirectories) {
  const std::filesystem::path directory =
      std::filesystem::temp_directory_path() / "fbsde_plotly_report_test" / "nested";
  std::filesystem::remove_all(directory.parent_path());

  PlotlyReport report("Title", "");
  report.AddFigure(MakeFigure("A figure"));

  const std::filesystem::path path = directory / "report.html";
  ASSERT_TRUE(report.WriteHtml(path).has_value());
  ASSERT_TRUE(std::filesystem::exists(path));

  std::ifstream stream(path, std::ios::binary);
  std::ostringstream contents;
  contents << stream.rdbuf();
  EXPECT_EQ(contents.str(), report.ToHtml());

  std::filesystem::remove_all(directory.parent_path());
}

TEST(PlotlyReportTest, WriteHtmlFailsWhenThePathIsNotWritable) {
  const PlotlyReport report("Title", "");

  // A path whose parent is an existing regular file cannot be created as a directory.
  const std::filesystem::path file = std::filesystem::temp_directory_path() / "fbsde_plotly_report_blocker";
  {
    const std::ofstream blocker(file);
  }

  EXPECT_FALSE(report.WriteHtml(file / "nested" / "report.html").has_value());

  std::filesystem::remove(file);
}

}  // namespace
}  // namespace fbsde_traj_opt::viz
