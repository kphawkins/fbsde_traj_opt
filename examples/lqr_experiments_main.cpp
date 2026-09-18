// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

// Runs every LQR experiment in lqr_models.hpp, writes one HTML report per model, and opens them.
//
//   bazel run //examples:lqr_experiments
//   bazel run //examples:lqr_experiments -- --output-dir /tmp/lqr --no-open --seed 9
//
// Each report holds, for one model, the sampled state distributions under the optimal policy and
// under the baseline, a phase portrait where one is informative, and the expected cost-to-go of
// both policies over the horizon. The cost-to-go figure is the one that settles the comparison.

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
#include "fbsde_traj_opt/viz/open_in_browser.hpp"
#include "fbsde_traj_opt/viz/plotly_report.hpp"
#include "examples/lqr_experiment.hpp"
#include "examples/lqr_models.hpp"

namespace fbsde_traj_opt::examples {
namespace {

struct Options {
  std::filesystem::path output_dir = "lqr_reports";
  std::uint64_t seed = 20260918;
  bool open = true;
};

auto PrintUsage() noexcept -> void {
  std::cout << "usage: lqr_experiments [--output-dir DIR] [--seed N] [--no-open]\n"
            << "  --output-dir DIR  where the HTML reports are written (default: lqr_reports)\n"
            << "  --seed N          the RNG seed every model's rollouts share (default: 20260918)\n"
            << "  --no-open         write the reports without opening them in a browser\n";
}

auto ParseOptions(int argc, char** argv, Options& options_out) noexcept -> Result<> {
  const std::vector<std::string_view> arguments(argv + 1, argv + argc);
  for (std::size_t index = 0; index < arguments.size(); ++index) {
    const std::string_view argument = arguments[index];
    if (argument == "--no-open") {
      options_out.open = false;
    } else if (argument == "--output-dir") {
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

// Turns a model name into something usable as a filename: lowercase, with runs of non-alphanumeric
// characters collapsed to single dashes.
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

// Writes one experiment's figures as a report, opens it if asked, and prints its headline numbers.
auto PublishReport(const LqrExperimentReport& experiment, const Options& options) noexcept -> Result<> {
  viz::PlotlyReport report(experiment.name, experiment.description);
  for (const viz::PlotlyFigure& figure : experiment.figures) {
    report.AddFigure(figure);
  }

  const std::filesystem::path path = options.output_dir / (FileSlug(experiment.name) + ".html");
  const Result<> written = report.WriteHtml(path);
  if (!written.has_value()) {
    return written;
  }

  const double ratio =
      experiment.optimal_initial_cost > 0.0 ? experiment.baseline_initial_cost / experiment.optimal_initial_cost : 0.0;
  std::cout << experiment.name << '\n'
            << "  expected cost-to-go from stage 0, LQR:      " << experiment.optimal_initial_cost << '\n'
            << "  expected cost-to-go from stage 0, baseline: " << experiment.baseline_initial_cost << '\n'
            << "  the baseline costs " << ratio << " times as much\n"
            << "  report: " << std::filesystem::absolute(path).string() << '\n';

  if (options.open) {
    // A browser that refuses to open is not a reason to fail the run: the report is written and
    // its path is on screen either way.
    const Result<> opened = viz::OpenInBrowser(path);
    if (!opened.has_value()) {
      RESULT_REPORT_RESULT(opened);
    }
  }

  return SuccessResult();
}

// Runs one model end to end. Templated on the model's dimensions and horizon, which are
// compile-time throughout the library.
template <int N, int M, std::size_t NumStages>
auto RunAndPublish(const LqrExperimentSpec<N, M>& spec, const Options& options) noexcept -> Result<> {
  const auto experiment = RunLqrExperiment<N, M, kNumTrajectories, NumStages>(spec, options.seed);
  if (!experiment.has_value()) {
    return std::unexpected(experiment.error());
  }
  return PublishReport(*experiment, options);
}

auto Run(int argc, char** argv) noexcept -> Result<> {
  Options options;
  const Result<> parsed = ParseOptions(argc, argv, options);
  if (!parsed.has_value()) {
    PrintUsage();
    return parsed;
  }

  const Result<> double_integrator =
      RunAndPublish<2, 1, kDoubleIntegratorNumStages>(MakeDoubleIntegratorSpec(), options);
  if (!double_integrator.has_value()) {
    return double_integrator;
  }

  const Result<> cart_pole = RunAndPublish<4, 1, kCartPoleNumStages>(MakeCartPoleSpec(), options);
  if (!cart_pole.has_value()) {
    return cart_pole;
  }

  const Result<> two_mass_spring = RunAndPublish<4, 1, kTwoMassSpringNumStages>(MakeTwoMassSpringSpec(), options);
  if (!two_mass_spring.has_value()) {
    return two_mass_spring;
  }

  return SuccessResult();
}

}  // namespace
}  // namespace fbsde_traj_opt::examples

auto main(int argc, char** argv) -> int {
  return RESULT_REPORT_RESULT(fbsde_traj_opt::examples::Run(argc, argv)) ? EXIT_SUCCESS : EXIT_FAILURE;
}
