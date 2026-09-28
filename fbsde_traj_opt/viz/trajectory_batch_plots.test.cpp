// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#include "fbsde_traj_opt/viz/trajectory_batch_plots.hpp"

#include <array>
#include <cstddef>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "fbsde_traj_opt/composed_forward_sde_model.hpp"
#include "fbsde_traj_opt/const_diagonal_sde_diffusion_term.hpp"
#include "fbsde_traj_opt/const_linear_feedback_sde_control_policy_term.hpp"
#include "fbsde_traj_opt/const_linear_sde_control_drift_mat_term.hpp"
#include "fbsde_traj_opt/const_linear_sde_state_drift_term.hpp"
#include "fbsde_traj_opt/diagonal_covariance_normal_distribution.hpp"
#include "fbsde_traj_opt/trajectory_batch.hpp"
#include "fbsde_traj_opt/viz/plotly_figure.hpp"

namespace fbsde_traj_opt::viz {
namespace {

constexpr std::size_t kNumTrajectories = 100;
constexpr std::size_t kNumStages = 5;

using Model = ComposedForwardSdeModel<2,
                                      1,
                                      ConstLinearSdeStateDriftTerm<2>,
                                      ConstLinearSdeControlDriftMatTerm<2, 1>,
                                      ConstDiagonalSdeDiffusionTerm<2>>;
using Batch = TrajectoryBatch<2, 1, kNumTrajectories, kNumStages>;

static_assert(PlottableTrajectoryBatch<Batch>);

auto MakeBatch() noexcept -> Batch {
  const auto diffusion = ConstDiagonalSdeDiffusionTerm<2>::Make(Eigen::Vector2d(0.05, 0.05));
  EXPECT_TRUE(diffusion.has_value());

  Eigen::Matrix2d a;
  a << 0.0, 0.1,  //
      0.0, 0.0;
  const Model model(ConstLinearSdeStateDriftTerm<2>(a),
                    ConstLinearSdeControlDriftMatTerm<2, 1>(Eigen::Vector2d(0.0, 0.1)),
                    *diffusion);

  Eigen::Matrix<double, 1, 2> gain;
  gain << -1.0, -2.0;
  const ConstLinearFeedbackSdeControlPolicyTerm<2, 1> policy(gain, Eigen::Matrix<double, 1, 1>::Zero());

  const auto distribution =
      DiagonalCovarianceNormalDistribution<2>::Make(Eigen::Vector2d(1.0, 0.0), Eigen::Vector2d(0.01, 0.01));
  EXPECT_TRUE(distribution.has_value());

  const auto batch = Batch::Make(model, policy, *distribution, 4242);
  EXPECT_TRUE(batch.has_value());
  return *batch;
}

TEST(AddStateTrajectoriesTest, SampleCountIsCappedButNeverExceedsTheBatch) {
  const Batch batch = MakeBatch();

  PlotlyFigure capped("Position", "Stage", "Position");
  ASSERT_TRUE(AddStateTrajectories(capped, batch, 0, {.max_sampled_trajectories = 5}).has_value());
  EXPECT_EQ(capped.LineCount(), 6U);

  PlotlyFigure uncapped("Position", "Stage", "Position");
  ASSERT_TRUE(AddStateTrajectories(uncapped, batch, 0, {.max_sampled_trajectories = 10000}).has_value());
  EXPECT_EQ(uncapped.LineCount(), kNumTrajectories + 1);
}

TEST(AddStateTrajectoriesTest, TheMeanLineCarriesTheBatchMean) {
  const Batch batch = MakeBatch();
  PlotlyFigure figure("Position", "Stage", "Position");

  ASSERT_TRUE(AddStateTrajectories(figure, batch, 1, {.max_sampled_trajectories = 3}).has_value());

  const nlohmann::json mean_trace = figure.ToJson().at("data").back();
  const std::vector<double> mean_values = mean_trace.at("y").get<std::vector<double>>();

  ASSERT_EQ(mean_values.size(), kNumStages);
  for (std::size_t stage = 0; stage < kNumStages; ++stage) {
    EXPECT_DOUBLE_EQ(mean_values[stage], batch.MeanStateAtStage(stage)[1]) << "at stage " << stage;
  }
}

TEST(AddStateTrajectoriesTest, SamplesAndMeanCanEachBeSuppressed) {
  const Batch batch = MakeBatch();

  PlotlyFigure mean_only("Position", "Stage", "Position");
  ASSERT_TRUE(AddStateTrajectories(mean_only, batch, 0, {.show_samples = false}).has_value());
  EXPECT_EQ(mean_only.LineCount(), 1U);

  PlotlyFigure samples_only("Position", "Stage", "Position");
  ASSERT_TRUE(
      AddStateTrajectories(samples_only, batch, 0, {.max_sampled_trajectories = 6, .show_mean = false}).has_value());
  EXPECT_EQ(samples_only.LineCount(), 6U);
}

TEST(AddStateTrajectoriesTest, RejectsAStateIndexOutsideTheState) {
  const Batch batch = MakeBatch();
  PlotlyFigure figure("Position", "Stage", "Position");

  EXPECT_FALSE(AddStateTrajectories(figure, batch, 2, {}).has_value());
  EXPECT_EQ(figure.LineCount(), 0U);
}

TEST(AddPhasePortraitTest, PlotsOneStateDimensionAgainstAnother) {
  const Batch batch = MakeBatch();
  PlotlyFigure figure("Phase portrait", "Position", "Velocity");

  ASSERT_TRUE(AddPhasePortrait(figure, batch, 0, 1, {.max_sampled_trajectories = 4}).has_value());

  EXPECT_EQ(figure.LineCount(), 5U);

  const nlohmann::json mean_trace = figure.ToJson().at("data").back();
  const std::vector<double> horizontal = mean_trace.at("x").get<std::vector<double>>();
  const std::vector<double> vertical = mean_trace.at("y").get<std::vector<double>>();
  for (std::size_t stage = 0; stage < kNumStages; ++stage) {
    EXPECT_DOUBLE_EQ(horizontal[stage], batch.MeanStateAtStage(stage)[0]);
    EXPECT_DOUBLE_EQ(vertical[stage], batch.MeanStateAtStage(stage)[1]);
  }
}

TEST(AddPhasePortraitTest, RejectsAStateIndexOutsideTheState) {
  const Batch batch = MakeBatch();
  PlotlyFigure figure("Phase portrait", "Position", "Velocity");

  EXPECT_FALSE(AddPhasePortrait(figure, batch, 0, 7, {}).has_value());
}

TEST(MakeCostToGoFigureTest, PlotsOneLinePerPolicyAgainstTheStageAxis) {
  constexpr std::array<double, 4> kOptimal{9.0, 6.0, 3.0, 1.0};
  constexpr std::array<double, 4> kNaive{14.0, 10.0, 6.0, 2.0};

  const std::array<CostToGoSeries, 2> series{
      CostToGoSeries{.name = "LQR", .color_role = PlotColorRole::kSeries1, .expected_cost_to_go = kOptimal},
      CostToGoSeries{.name = "proportional", .color_role = PlotColorRole::kSeries2, .expected_cost_to_go = kNaive},
  };

  const auto figure = MakeCostToGoFigure("Expected cost-to-go", series);
  ASSERT_TRUE(figure.has_value()) << figure.error();

  const nlohmann::json json = figure->ToJson();
  ASSERT_EQ(json.at("data").size(), 2U);
  EXPECT_EQ(json.at("data").at(0).at("y").get<std::vector<double>>(),
            std::vector<double>(kOptimal.begin(), kOptimal.end()));
  EXPECT_EQ(json.at("data").at(0).at("x").get<std::vector<double>>(), (std::vector<double>{0.0, 1.0, 2.0, 3.0}));
  EXPECT_EQ(json.at("data").at(1).at("colorRole"), "series-2");
  EXPECT_EQ(json.at("layout").at("yaxis").at("title").at("text"), "Expected cost-to-go (log scale)");
}

// Costs routinely differ between policies by more than an order of magnitude, and on a linear
// axis the cheaper policy is then a line pressed flat against zero.
TEST(MakeCostToGoFigureTest, PositiveCostsGetALogarithmicAxis) {
  constexpr std::array<double, 3> kCheap{0.01, 0.005, 0.001};
  constexpr std::array<double, 3> kExpensive{30.0, 29.0, 28.0};

  const std::array<CostToGoSeries, 2> series{
      CostToGoSeries{.name = "LQR", .expected_cost_to_go = kCheap},
      CostToGoSeries{.name = "baseline", .color_role = PlotColorRole::kSeries2, .expected_cost_to_go = kExpensive},
  };

  const auto figure = MakeCostToGoFigure("Expected cost-to-go", series);
  ASSERT_TRUE(figure.has_value()) << figure.error();

  EXPECT_EQ(figure->ToJson().at("layout").at("yaxis").at("type"), "log");
}

// Plotly silently drops non-positive points from a log axis, so a series that touches zero keeps
// the linear one.
TEST(MakeCostToGoFigureTest, ACostThatReachesZeroKeepsTheLinearAxis) {
  constexpr std::array<double, 3> kTouchesZero{2.0, 1.0, 0.0};

  const std::array<CostToGoSeries, 1> series{CostToGoSeries{.name = "LQR", .expected_cost_to_go = kTouchesZero}};

  const auto figure = MakeCostToGoFigure("Expected cost-to-go", series);
  ASSERT_TRUE(figure.has_value()) << figure.error();

  const nlohmann::json layout = figure->ToJson().at("layout");
  EXPECT_FALSE(layout.at("yaxis").contains("type"));
  EXPECT_EQ(layout.at("yaxis").at("title").at("text"), "Expected cost-to-go");
}

TEST(MakeCostToGoFigureTest, LinearScaleCanBeAskedForOutright) {
  constexpr std::array<double, 3> kPositive{3.0, 2.0, 1.0};

  const std::array<CostToGoSeries, 1> series{CostToGoSeries{.name = "LQR", .expected_cost_to_go = kPositive}};

  const auto figure = MakeCostToGoFigure("Expected cost-to-go", series, CostToGoAxisScale::kLinear);
  ASSERT_TRUE(figure.has_value()) << figure.error();

  EXPECT_FALSE(figure->ToJson().at("layout").at("yaxis").contains("type"));
}

TEST(MakeCostToGoFigureTest, RejectsNoSeries) {
  EXPECT_FALSE(MakeCostToGoFigure("Expected cost-to-go", {}).has_value());
}

// Two series over different horizons cannot share one stage axis.
TEST(MakeCostToGoFigureTest, RejectsSeriesOverDifferentHorizons) {
  constexpr std::array<double, 3> kShort{3.0, 2.0, 1.0};
  constexpr std::array<double, 4> kLong{4.0, 3.0, 2.0, 1.0};

  const std::array<CostToGoSeries, 2> series{
      CostToGoSeries{.name = "short", .expected_cost_to_go = kShort},
      CostToGoSeries{.name = "long", .expected_cost_to_go = kLong},
  };

  EXPECT_FALSE(MakeCostToGoFigure("Expected cost-to-go", series).has_value());
}

}  // namespace
}  // namespace fbsde_traj_opt::viz
