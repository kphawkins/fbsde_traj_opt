// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

// Draws the FBRRT particle swarm on the L1 double integrator and pendulum, over time and over
// iterations, and writes one HTML report per benchmark.
//
//   bazel run -c opt //examples:fbrrt_particle_swarms
//   bazel run -c opt //examples:fbrrt_particle_swarms -- --iterations 15 --output-dir /tmp/swarms
//
// Build it optimized: an unoptimized build runs the solver a hundred times slower.

#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "fbsde_traj_opt/utils/result.hpp"
#include "fbsde_traj_opt/viz/plotly_figure.hpp"
#include "fbsde_traj_opt/viz/plotly_report.hpp"
#include "examples/fbrrt_particle_swarms.hpp"

namespace fbsde_traj_opt::examples {
namespace {

struct Options {
  std::filesystem::path output_dir = "fbrrt_particle_swarms_report";
  FbrrtParticleSwarmOptions swarms{};
};

auto PrintUsage() noexcept -> void {
  std::cout << "usage: fbrrt_particle_swarms [options]\n"
            << "  --output-dir DIR    where the HTML reports are written (default: fbrrt_particle_swarms_report)\n"
            << "  --iterations N      FBRRT iterations (default 30)\n"
            << "  --seed N            the seed of the run\n"
            << "  --model M           soft-min (default) or polynomial value functions\n"
            << "  --line-search S     on (default), off, or working-copy\n"
            << "  --per-stage N       particles per stage in the over-iterations picture (default 64)\n";
}

auto ParseOptions(int argc, char** argv, Options& options_out) noexcept -> Result<> {
  const std::vector<std::string_view> arguments(argv + 1, argv + argc);
  for (std::size_t index = 0; index < arguments.size(); ++index) {
    const std::string_view argument = arguments[index];
    RESULT_ASSERT(index + 1 < arguments.size(), "every option needs a value; run with --help for the options.");
    const std::string value(arguments[++index]);
    const auto count = static_cast<std::size_t>(std::strtoull(value.c_str(), nullptr, 10));
    FbrrtComparisonOptions& comparison = options_out.swarms.comparison;
    if (argument == "--output-dir") {
      options_out.output_dir = value;
    } else if (argument == "--iterations") {
      RESULT_ASSERT(count >= 1, "--iterations needs a positive count.");
      comparison.iterations = count;
    } else if (argument == "--seed") {
      options_out.swarms.seed = std::strtoull(value.c_str(), nullptr, 10);
    } else if (argument == "--model") {
      RESULT_ASSERT(value == "soft-min" || value == "polynomial", "--model is soft-min or polynomial.");
      comparison.model = value == "soft-min" ? ComparisonValueFunctionModel::kSoftMinQuadratic
                                             : ComparisonValueFunctionModel::kPolynomial;
    } else if (argument == "--line-search") {
      RESULT_ASSERT(value == "on" || value == "off" || value == "working-copy",
                    "--line-search is on, off, or working-copy.");
      comparison.fbrrt.line_search = value != "off";
      comparison.fbrrt.keep_working_copy = value == "working-copy";
    } else if (argument == "--per-stage") {
      RESULT_ASSERT(count >= 1, "--per-stage needs a positive count.");
      options_out.swarms.particles_per_stage_shown = count;
    } else {
      return ErrorResult("unrecognized argument; run with --help for the options.");
    }
  }
  return SuccessResult();
}

auto Run(int argc, char** argv) noexcept -> Result<> {
  Options options;
  RESULT_RETURN_IF_ERROR(ParseOptions(argc, argv, options));
  RESULT_ASSIGN_OR_RETURN(const std::vector<FbrrtParticleSwarmReport> reports, DrawFbrrtParticleSwarms(options.swarms));
  for (const FbrrtParticleSwarmReport& swarm : reports) {
    viz::PlotlyReport report("FBRRT particle swarm: " + swarm.name, swarm.description);
    for (const viz::PlotlyFigure& figure : swarm.figures) {
      report.AddFigure(figure);
    }
    std::string file = swarm.name;
    for (char& character : file) {
      character = character == ' ' ? '-' : character;
    }
    const std::filesystem::path path = options.output_dir / (file + ".html");
    RESULT_RETURN_IF_ERROR(report.WriteHtml(path));
    std::cout << swarm.name << ": " << std::filesystem::absolute(path).string() << '\n';
  }
  return SuccessResult();
}

}  // namespace
}  // namespace fbsde_traj_opt::examples

auto main(int argc, char** argv) -> int {
  for (int index = 1; index < argc; ++index) {
    if (std::string_view(argv[index]) == "--help") {
      fbsde_traj_opt::examples::PrintUsage();
      return EXIT_SUCCESS;
    }
  }
  const fbsde_traj_opt::Result<> result = fbsde_traj_opt::examples::Run(argc, argv);
  return RESULT_REPORT_RESULT(result) ? EXIT_SUCCESS : EXIT_FAILURE;
}
