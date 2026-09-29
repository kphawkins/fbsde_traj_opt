// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

// Swings up the L1 double inverted pendulum with the DT-FBSDE iterative method and writes the
// report.
//
//   bazel run -c opt //examples:double_inverted_pendulum
//   bazel run -c opt //examples:double_inverted_pendulum -- --iterations 30 --output-dir /tmp/dip
//
// Build it optimized: the method samples a few thousand trajectories per iteration, and an
// unoptimized build runs roughly twenty times slower.

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "fbsde_traj_opt/solvers/dt_fbsde_iterative_solver.hpp"
#include "fbsde_traj_opt/utils/result.hpp"
#include "fbsde_traj_opt/viz/plotly_report.hpp"
#include "examples/double_inverted_pendulum_experiment.hpp"

namespace fbsde_traj_opt::examples {
namespace {

struct Options {
  std::filesystem::path output_dir = "double_inverted_pendulum_report";
  DoubleInvertedPendulumExperimentOptions experiment{};
};

auto PrintUsage() noexcept -> void {
  std::cout << "usage: double_inverted_pendulum [--output-dir DIR] [--seed N] [--iterations N]\n"
            << "  --output-dir DIR  where the HTML report is written (default: double_inverted_pendulum_report)\n"
            << "  --seed N          the seed all sampling derives from (default: 20260926)\n"
            << "  --iterations N    iterations of the method (default: 150)\n";
}

auto ParseOptions(int argc, char** argv, Options& options_out) noexcept -> Result<> {
  const std::vector<std::string_view> arguments(argv + 1, argv + argc);
  for (std::size_t index = 0; index < arguments.size(); ++index) {
    const std::string_view argument = arguments[index];
    if (argument == "--output-dir") {
      RESULT_ASSERT(index + 1 < arguments.size(), "--output-dir needs a directory.");
      options_out.output_dir = arguments[++index];
    } else if (argument == "--seed") {
      RESULT_ASSERT(index + 1 < arguments.size(), "--seed needs a value.");
      options_out.experiment.seed = std::strtoull(std::string(arguments[++index]).c_str(), nullptr, 10);
    } else if (argument == "--iterations") {
      RESULT_ASSERT(index + 1 < arguments.size(), "--iterations needs a value.");
      options_out.experiment.iterations = std::strtoull(std::string(arguments[++index]).c_str(), nullptr, 10);
      RESULT_ASSERT(options_out.experiment.iterations >= 1, "--iterations needs a positive count.");
    } else {
      return ErrorResult("unrecognized argument; run with no arguments for the usual behavior.");
    }
  }
  return SuccessResult();
}

auto Run(int argc, char** argv) noexcept -> Result<> {
  Options options;
  if (const Result<> parsed = ParseOptions(argc, argv, options); !parsed.has_value()) {
    PrintUsage();
    return parsed;
  }

  RESULT_ASSIGN_OR_RETURN(
      const DoubleInvertedPendulumReport experiment,
      RunDoubleInvertedPendulumExperiment(
          options.experiment, [](const DtFbsdeIterationReport<double>& iteration) noexcept {
            std::cout << std::format("iteration {:3d}: cost {:8.3f} (fuel {:.3f}, terminal {:8.3f})  step {:.4f}\n",
                                     iteration.iteration + 1,
                                     iteration.mean_cost,
                                     iteration.mean_running_cost,
                                     iteration.mean_terminal_cost,
                                     iteration.accepted_step)
                      << std::flush;
          }));

  viz::PlotlyReport report(experiment.name, experiment.description);
  for (const viz::PlotlyFigure& figure : experiment.figures) {
    report.AddFigure(figure);
  }
  const std::filesystem::path path = options.output_dir / "double-inverted-pendulum.html";
  RESULT_RETURN_IF_ERROR(report.WriteHtml(path));

  std::cout << experiment.name << "\n  " << experiment.summary
            << "\n  report: " << std::filesystem::absolute(path).string() << '\n';
  return SuccessResult();
}

}  // namespace
}  // namespace fbsde_traj_opt::examples

auto main(int argc, char** argv) -> int {
  const fbsde_traj_opt::Result<> result = fbsde_traj_opt::examples::Run(argc, argv);
  return RESULT_REPORT_RESULT(result) ? 0 : 1;
}
