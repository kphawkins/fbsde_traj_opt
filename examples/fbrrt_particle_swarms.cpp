// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#include "examples/fbrrt_particle_swarms.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <format>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "fbsde_traj_opt/solvers/fbrrt_solver.hpp"
#include "fbsde_traj_opt/utils/result.hpp"
#include "fbsde_traj_opt/value_function/soft_min_quadratic_value_function_approx.hpp"
#include "fbsde_traj_opt/viz/plotly_figure.hpp"
#include "examples/fbrrt_comparison_setup.hpp"
#include "examples/l1_benchmark_models.hpp"

namespace fbsde_traj_opt::examples {
namespace {

// The share of the heuristic's values the over-time pictures' color scale spans. The heuristic of
// a particle that swung the wrong way runs to hundreds; letting those set the scale would leave
// every particle near a good trajectory the same color.
constexpr double kHeuristicColorQuantile = 0.9;

// Values are written to the page to four decimal places, which is finer than a screen can show
// and keeps the page a few megabytes rather than tens.
auto Rounded(double value) noexcept -> double {
  return std::round(value * 1e4) / 1e4;
}

// One picture's worth of points.
struct Cloud {
  std::vector<double> x{};
  std::vector<double> y{};
  std::vector<double> color{};

  auto Add(double x_value, double y_value, double color_value) noexcept -> void {
    x.push_back(Rounded(x_value));
    y.push_back(Rounded(y_value));
    color.push_back(Rounded(color_value));
  }
};

// The value below which `quantile` of all the clouds' colors fall.
auto ColorQuantile(const std::vector<Cloud>& clouds, double quantile) noexcept -> double {
  std::vector<double> values;
  for (const Cloud& cloud : clouds) {
    values.insert(values.end(), cloud.color.begin(), cloud.color.end());
  }
  if (values.empty()) {
    return 0.0;
  }
  const auto index = static_cast<std::size_t>(quantile * static_cast<double>(values.size() - 1));
  std::ranges::nth_element(values, values.begin() + static_cast<std::ptrdiff_t>(index));
  return values[index];
}

// A scatter animated over `clouds`, one frame each, with a policy rollout drawn through it: one
// path per frame, or a single path for every frame. Rests on the last frame. With `clip_colors`,
// colors above kHeuristicColorQuantile of all the frames' colors are clipped to it.
auto AnimatedSwarm(std::string title,
                   const std::array<std::string, 2>& axis_titles,
                   std::vector<Cloud> clouds,
                   const std::vector<std::string>& labels,
                   const std::vector<Cloud>& paths,
                   const std::string& path_name,
                   const std::string& value_label,
                   bool clip_colors,
                   const std::string& slider_prefix) noexcept -> Result<viz::PlotlyFigure> {
  RESULT_ASSERT(!clouds.empty() && labels.size() == clouds.size(), "AnimatedSwarm: one label per frame.");
  RESULT_ASSERT(paths.size() == 1 || paths.size() == clouds.size(), "AnimatedSwarm: one path, or one per frame.");
  viz::PlotlyFigure figure(std::move(title), axis_titles[0], axis_titles[1]);
  if (clip_colors) {
    const double ceiling = ColorQuantile(clouds, kHeuristicColorQuantile);
    for (Cloud& cloud : clouds) {
      std::ranges::transform(
          cloud.color, cloud.color.begin(), [ceiling](double value) { return std::min(value, ceiling); });
    }
  }

  const Cloud& resting = clouds.back();
  RESULT_RETURN_IF_ERROR(figure.AddScatter(
      resting.x,
      resting.y,
      resting.color,
      viz::ScatterStyle{.name = "particles", .marker_size = 4.0, .opacity = 0.55, .value_label = value_label}));
  RESULT_RETURN_IF_ERROR(figure.AddLine(
      paths.back().x,
      paths.back().y,
      viz::LineStyle{
          .name = path_name, .color_role = viz::PlotColorRole::kSeries2, .width = 2.0, .show_markers = false}));

  std::vector<viz::AnimationFrame> frames;
  frames.reserve(clouds.size());
  for (std::size_t index = 0; index < clouds.size(); ++index) {
    const Cloud& path = paths.size() == 1 ? paths.front() : paths[index];
    frames.push_back({.label = labels[index],
                      .trace_values = {clouds[index].y, path.y},
                      .trace_x_values = {clouds[index].x, path.x},
                      .trace_color_values = {clouds[index].color, {}}});
  }
  constexpr std::array<std::size_t, 2> kAnimated{0, 1};
  RESULT_RETURN_IF_ERROR(figure.Animate(
      kAnimated, frames, viz::AnimationStyle{.frame_duration_ms = 150.0, .slider_prefix = slider_prefix}));
  return SuccessResult(std::move(figure));
}

template <typename SetupT, typename BenchmarkT>
auto Draw(const BenchmarkT& benchmark, const FbrrtParticleSwarmOptions& options) noexcept
    -> Result<FbrrtParticleSwarmReport> {
  using Solver = typename SetupT::Fbrrt;
  using State = typename Solver::State;
  using Control = typename Solver::Control;
  constexpr auto kCapacity = static_cast<std::size_t>(Solver::Particles::kCapacity);
  constexpr std::size_t kNumStages = Solver::Particles::kNumStages;

  const FbrrtComparisonOptions& comparison = options.comparison;
  RESULT_ASSIGN_OR_RETURN(
      const StartingTable<typename SetupT::ValueFunction> table,
      MakeStartingTable(benchmark, comparison, static_cast<const typename SetupT::ValueFunction*>(nullptr)));
  FbrrtOptions<double> fbrrt_options = comparison.fbrrt;
  fbrrt_options.keep_pre_erosion_snapshot = true;
  RESULT_ASSIGN_OR_RETURN(const std::unique_ptr<Solver> solver,
                          MakeComparisonFbrrtSolver<SetupT>(
                              benchmark, table, comparison, fbrrt_options, comparison.keep_fraction, options.seed));

  const auto time = [&](std::size_t stage) noexcept { return static_cast<double>(stage) * benchmark.time_step; };

  // The current policy's noise-free rollout from the initial mean.
  const auto rollout = [&]() noexcept {
    Cloud path;
    const auto policy = solver->CurrentPolicy();
    State state = benchmark.initial_distribution.Mean();
    path.Add(state[0], state[1], 0.0);
    for (std::size_t stage = 0; stage + 1 < kNumStages; ++stage) {
      const Control control = policy(stage, state);
      state = benchmark.forward_model(stage, state, control, State::Zero());
      path.Add(state[0], state[1], 0.0);
    }
    return path;
  };

  // Every particle of every stage past the first, one frame per stage, colored by the heuristic
  // erosion ranked it by.
  const auto stage_clouds = [&]() noexcept {
    const auto& particles = solver->pre_erosion_particles();
    const auto value_functions = solver->working_value_functions();
    std::vector<Cloud> clouds;
    for (std::size_t stage = 1; stage < kNumStages; ++stage) {
      Cloud cloud;
      for (std::size_t slot = 0; slot < particles.count(stage); ++slot) {
        const auto column = static_cast<Eigen::Index>(slot);
        const State state = particles.states(stage).col(column);
        cloud.Add(state[0], state[1], particles.accrued_costs(stage)[column] + value_functions[stage](state));
      }
      clouds.push_back(std::move(cloud));
    }
    return clouds;
  };

  const std::size_t stride =
      std::max<std::size_t>(kCapacity / std::max<std::size_t>(options.particles_per_stage_shown, 1), 1);
  std::vector<Cloud> iteration_clouds;
  std::vector<Cloud> iteration_paths;
  std::vector<std::string> iteration_labels;
  std::vector<double> costs{solver->Evaluate().total};
  std::vector<Cloud> first_stage_clouds;
  Cloud first_path;
  for (std::size_t iteration = 0; iteration < comparison.iterations; ++iteration) {
    RESULT_ASSIGN_OR_RETURN(const FbrrtIterationReport<double> report, solver->Iterate());
    costs.push_back(report.mean_cost);

    const auto& particles = solver->pre_erosion_particles();
    Cloud cloud;
    for (std::size_t stage = 0; stage < kNumStages; ++stage) {
      for (std::size_t slot = 0; slot < particles.count(stage); slot += stride) {
        const State state = particles.states(stage).col(static_cast<Eigen::Index>(slot));
        cloud.Add(state[0], state[1], time(stage));
      }
    }
    iteration_clouds.push_back(std::move(cloud));
    iteration_paths.push_back(rollout());
    iteration_labels.push_back(std::format("{}  (cost {:.2f})", iteration + 1, report.mean_cost));
    if (iteration == 0) {
      first_stage_clouds = stage_clouds();
      first_path = iteration_paths.back();
    }
  }

  std::vector<std::string> stage_labels;
  for (std::size_t stage = 1; stage < kNumStages; ++stage) {
    stage_labels.push_back(std::format("{:.3f} s", time(stage)));
  }
  const std::array<std::string, 2> axes{benchmark.state_names[0], benchmark.state_names[1]};
  const std::string rho_label = "rho = L + V~ (clipped at 90th percentile)";

  FbrrtParticleSwarmReport report;
  report.name = benchmark.name;
  report.description = benchmark.description;

  RESULT_ASSIGN_OR_RETURN(viz::PlotlyFigure over_iterations,
                          AnimatedSwarm(benchmark.name + ": the swarm over iterations",
                                        axes,
                                        iteration_clouds,
                                        iteration_labels,
                                        iteration_paths,
                                        "policy after the iteration, noise-free rollout",
                                        "Time (s)",
                                        false,
                                        "Iteration "));
  report.figures.push_back(std::move(over_iterations));

  RESULT_ASSIGN_OR_RETURN(viz::PlotlyFigure first,
                          AnimatedSwarm(benchmark.name + ": the swarm over time, first iteration",
                                        axes,
                                        first_stage_clouds,
                                        stage_labels,
                                        {first_path},
                                        "policy after iteration 1, noise-free rollout",
                                        rho_label,
                                        true,
                                        "t = "));
  report.figures.push_back(std::move(first));

  RESULT_ASSIGN_OR_RETURN(viz::PlotlyFigure last,
                          AnimatedSwarm(benchmark.name + ": the swarm over time, last iteration",
                                        axes,
                                        stage_clouds(),
                                        stage_labels,
                                        {iteration_paths.back()},
                                        "final policy, noise-free rollout",
                                        rho_label,
                                        true,
                                        "t = "));
  report.figures.push_back(std::move(last));

  viz::PlotlyFigure cost(benchmark.name + ": policy cost over iterations", "Iteration", "Mean cost");
  std::vector<double> iterations(costs.size());
  for (std::size_t index = 0; index < costs.size(); ++index) {
    iterations[index] = static_cast<double>(index);
  }
  RESULT_RETURN_IF_ERROR(cost.AddLine(iterations, costs, viz::LineStyle{.name = "FBRRT", .show_markers = true}));
  cost.UseLogarithmicYAxis();
  report.figures.push_back(std::move(cost));
  return SuccessResult(std::move(report));
}

}  // namespace

auto DrawFbrrtParticleSwarms(const FbrrtParticleSwarmOptions& options) noexcept
    -> Result<std::vector<FbrrtParticleSwarmReport>> {
  RESULT_ASSERT(options.comparison.iterations >= 1, "DrawFbrrtParticleSwarms: the run needs at least one iteration.");
  std::vector<FbrrtParticleSwarmReport> reports;
  if (options.comparison.model == ComparisonValueFunctionModel::kPolynomial) {
    RESULT_ASSIGN_OR_RETURN(const L1DoubleIntegrator integrator, MakeL1DoubleIntegrator());
    RESULT_ASSIGN_OR_RETURN(
        FbrrtParticleSwarmReport integrator_report,
        (Draw<Setup<L1DoubleIntegrator, 1024, kL1DoubleIntegratorNumStages, PolynomialValueFunctionApprox<2, 2>>>(
            integrator, options)));
    reports.push_back(std::move(integrator_report));
    RESULT_ASSIGN_OR_RETURN(const L1Pendulum pendulum, MakeL1Pendulum());
    RESULT_ASSIGN_OR_RETURN(
        FbrrtParticleSwarmReport pendulum_report,
        (Draw<Setup<L1Pendulum, 1024, kL1PendulumNumStages, PolynomialValueFunctionApprox<2, 4>>>(pendulum, options)));
    reports.push_back(std::move(pendulum_report));
    return SuccessResult(std::move(reports));
  }
  if (options.comparison.model == ComparisonValueFunctionModel::kNonnegativeSoftMin) {
    using Nonnegative = NonnegativeSoftMinQuadraticValueFunctionApprox<2, kSoftMinComponents, double>;
    RESULT_ASSIGN_OR_RETURN(const L1DoubleIntegrator integrator, MakeL1DoubleIntegrator());
    RESULT_ASSIGN_OR_RETURN(
        FbrrtParticleSwarmReport integrator_report,
        (Draw<Setup<L1DoubleIntegrator, 1024, kL1DoubleIntegratorNumStages, Nonnegative>>(integrator, options)));
    reports.push_back(std::move(integrator_report));
    RESULT_ASSIGN_OR_RETURN(const L1Pendulum pendulum, MakeL1Pendulum());
    RESULT_ASSIGN_OR_RETURN(FbrrtParticleSwarmReport pendulum_report,
                            (Draw<Setup<L1Pendulum, 1024, kL1PendulumNumStages, Nonnegative>>(pendulum, options)));
    reports.push_back(std::move(pendulum_report));
    return SuccessResult(std::move(reports));
  }
  using SoftMin = SoftMinQuadraticValueFunctionApprox<2, kSoftMinComponents>;
  RESULT_ASSIGN_OR_RETURN(const L1DoubleIntegrator integrator, MakeL1DoubleIntegrator());
  RESULT_ASSIGN_OR_RETURN(
      FbrrtParticleSwarmReport integrator_report,
      (Draw<Setup<L1DoubleIntegrator, 1024, kL1DoubleIntegratorNumStages, SoftMin>>(integrator, options)));
  reports.push_back(std::move(integrator_report));
  RESULT_ASSIGN_OR_RETURN(const L1Pendulum pendulum, MakeL1Pendulum());
  RESULT_ASSIGN_OR_RETURN(FbrrtParticleSwarmReport pendulum_report,
                          (Draw<Setup<L1Pendulum, 1024, kL1PendulumNumStages, SoftMin>>(pendulum, options)));
  reports.push_back(std::move(pendulum_report));
  return SuccessResult(std::move(reports));
}

}  // namespace fbsde_traj_opt::examples
