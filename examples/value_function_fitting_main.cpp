// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

// Runs every value function fitting experiment and writes one HTML report per experiment.
//
//   bazel run //examples:value_function_fitting_experiments
//   bazel run //examples:value_function_fitting_experiments -- --output-dir /tmp/vf --seed 9
//
// The three reports, in order: the model and the fitter alone on a generic one-dimensional
// target; the Taylor Noiseless estimator swept backward over an LQR problem, against the exact
// Riccati value function; and the same model over a two-dimensional state, drawn as heatmaps.

#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "fbsde_traj_opt/utils/result.hpp"
#include "fbsde_traj_opt/viz/plotly_report.hpp"
#include "examples/value_function_fitting.hpp"

namespace fbsde_traj_opt::examples {
namespace {

struct Options {
  std::filesystem::path output_dir = "value_function_reports";
  std::uint64_t seed = 20260924;
};

auto PrintUsage() noexcept -> void {
  std::cout << "usage: value_function_fitting_experiments [--output-dir DIR] [--seed N]\n"
            << "  --output-dir DIR  where the HTML reports are written (default: value_function_reports)\n"
            << "  --seed N          the seed every experiment's sampling and shuffling share (default: 20260924)\n";
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
      options_out.seed = std::strtoull(std::string(arguments[++index]).c_str(), nullptr, 10);
    } else {
      return ErrorResult("unrecognized argument; run with no arguments for the usual behavior.");
    }
  }
  return SuccessResult();
}

// Turns an experiment name into something usable as a filename: lowercase, with runs of
// non-alphanumeric characters collapsed to single dashes.
auto FileSlug(std::string_view name) noexcept -> std::string {
  std::string slug;
  slug.reserve(name.size());
  for (const char character : name) {
    const auto unsigned_character = static_cast<unsigned char>(character);
    if (std::isalnum(unsigned_character) != 0) {
      slug += static_cast<char>(std::tolower(unsigned_character));
    } else if (!slug.empty() && slug.back() != '-') {
      slug += '-';
    }
  }
  while (!slug.empty() && slug.back() == '-') {
    slug.pop_back();
  }
  return slug;
}

// Writes one experiment's figures as a report and prints its headline numbers.
auto PublishReport(const ValueFunctionFittingReport& experiment, const Options& options) noexcept -> Result<> {
  viz::PlotlyReport report(experiment.name, experiment.description);
  for (const viz::PlotlyFigure& figure : experiment.figures) {
    report.AddFigure(figure);
  }

  const std::filesystem::path path = options.output_dir / (FileSlug(experiment.name) + ".html");
  RESULT_RETURN_IF_ERROR(report.WriteHtml(path));

  std::cout << experiment.name << '\n'
            << "  " << experiment.summary << '\n'
            << "  final RMS residual:   " << experiment.final_root_mean_squared_residual << '\n'
            << "  worst relative error: " << experiment.worst_relative_error << '\n'
            << "  report: " << std::filesystem::absolute(path).string() << '\n';

  return SuccessResult();
}

auto Run(int argc, char** argv) noexcept -> Result<> {
  Options options;
  if (const Result<> parsed = ParseOptions(argc, argv, options); !parsed.has_value()) {
    PrintUsage();
    return parsed;
  }

  RESULT_ASSIGN_OR_RETURN(const ValueFunctionFittingReport generic, RunGenericFunctionFittingExperiment(options.seed));
  RESULT_RETURN_IF_ERROR(PublishReport(generic, options));

  RESULT_ASSIGN_OR_RETURN(const ValueFunctionFittingReport backward, RunLqrBackwardPassExperiment(options.seed));
  RESULT_RETURN_IF_ERROR(PublishReport(backward, options));

  RESULT_ASSIGN_OR_RETURN(const ValueFunctionFittingReport plane, RunTwoDimensionalFittingExperiment(options.seed));
  return PublishReport(plane, options);
}

}  // namespace
}  // namespace fbsde_traj_opt::examples

auto main(int argc, char** argv) -> int {
  const fbsde_traj_opt::Result<> result = fbsde_traj_opt::examples::Run(argc, argv);
  return RESULT_REPORT_RESULT(result) ? 0 : 1;
}
