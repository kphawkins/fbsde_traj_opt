// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

// Compares FBRRT with the DT-FBSDE iterative method on the thesis's three L1 benchmarks and prints
// how fast each converges.
//
//   bazel run -c opt //examples:fbrrt_comparison
//   bazel run -c opt //examples:fbrrt_comparison -- --seeds 3 --skip-double-inverted-pendulum
//   bazel run -c opt //examples:fbrrt_comparison -- --rrt 0.8 --exploit 0.3 --learning-rate 0.1
//
// Build it optimized: an unoptimized build runs the solvers a hundred times slower.

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <format>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "fbsde_traj_opt/utils/result.hpp"
#include "examples/fbrrt_comparison_experiment.hpp"

namespace fbsde_traj_opt::examples {
namespace {

auto PrintUsage() noexcept -> void {
  std::cout
      << "usage: fbrrt_comparison [options]\n"
      << "  --seed N                          the seed all sampling derives from\n"
      << "  --seeds N                         runs per method per benchmark (default 10)\n"
      << "  --iterations N                    iterations on the 2-D benchmarks (default 30)\n"
      << "  --double-pendulum-iterations N    iterations on the double inverted pendulum (default 60)\n"
      << "  --skip-double-inverted-pendulum   run the 2-D benchmarks only\n"
      << "  --trace                           print every iteration's cost\n"
      << "  --model M                         soft-min (default) or polynomial value functions, for both methods\n"
      << "  --center-spread F                 how far toward the start the soft-min components begin (default 1)\n"
      << "  --sharpness B                     the soft-min model's beta (default 1)\n"
      << "  --dt-learning-rate R              DT-FBSDE's SGD learning rate on the soft-min model (default 0.05)\n"
      << "  --dt-epochs N                     DT-FBSDE's SGD epochs per fit on the soft-min model (default 50)\n"
      << "FBRRT design knobs:\n"
      << "  --rrt P             probability a parent is drawn by RRT (default 0.2)\n"
      << "  --exploit P         probability a control is the policy's (default 0.7)\n"
      << "  --keep F            fraction of each stage erosion keeps on the 2-D benchmarks (default 0.75)\n"
      << "  --learning-rate R   SGD learning rate of the backward pass (default 0.2)\n"
      << "  --epochs N          SGD epochs per stage per iteration (default 8)\n"
      << "  --line-search S     on (default), off, or working-copy: FBRRT adopts an update only if it\n"
      << "                      lowers the cost; working-copy keeps training the rejected updates\n"
      << "  --entropy-temperature L  weight FBRRT's SGD fit by the trajectory heuristic (default 0: off)\n"
      << "  --forward-noise S   off (default) or on: add the SDE's noise to FBRRT's new particles\n"
      << "  --target-floor F    floor under the backward pass's targets (default 0)\n"
      << "  --max-value-change D  SGD trust region on the RMS value change per step (default off)\n";
}

auto ParseOptions(int argc, char** argv, FbrrtComparisonOptions& options_out, bool& trace_out) noexcept -> Result<> {
  const std::vector<std::string_view> arguments(argv + 1, argv + argc);
  for (std::size_t index = 0; index < arguments.size(); ++index) {
    const std::string_view argument = arguments[index];
    if (argument == "--skip-double-inverted-pendulum") {
      options_out.include_double_inverted_pendulum = false;
      continue;
    }
    if (argument == "--trace") {
      trace_out = true;
      continue;
    }
    RESULT_ASSERT(index + 1 < arguments.size(), "every option but the two flags needs a value.");
    const std::string value(arguments[++index]);
    const double number = std::strtod(value.c_str(), nullptr);
    const auto count = static_cast<std::size_t>(std::strtoull(value.c_str(), nullptr, 10));
    if (argument == "--seed") {
      options_out.seed = std::strtoull(value.c_str(), nullptr, 10);
    } else if (argument == "--seeds") {
      options_out.num_seeds = count;
    } else if (argument == "--iterations") {
      options_out.iterations = count;
    } else if (argument == "--double-pendulum-iterations") {
      options_out.double_inverted_pendulum_iterations = count;
    } else if (argument == "--rrt") {
      options_out.fbrrt.rrt_probability = number;
    } else if (argument == "--exploit") {
      options_out.fbrrt.exploit_probability = number;
    } else if (argument == "--keep") {
      options_out.keep_fraction = number;
    } else if (argument == "--learning-rate") {
      options_out.backward.sgd.learning_rate = number;
    } else if (argument == "--epochs") {
      options_out.backward.epochs_per_stage = count;
    } else if (argument == "--model") {
      RESULT_ASSERT(value == "soft-min" || value == "polynomial", "--model is soft-min or polynomial.");
      options_out.model = value == "soft-min" ? ComparisonValueFunctionModel::kSoftMinQuadratic
                                              : ComparisonValueFunctionModel::kPolynomial;
    } else if (argument == "--center-spread") {
      options_out.soft_min_center_spread = number;
    } else if (argument == "--sharpness") {
      options_out.soft_min_sharpness = number;
    } else if (argument == "--dt-learning-rate") {
      options_out.dt_fbsde_sgd.learning_rate = number;
    } else if (argument == "--dt-epochs") {
      options_out.dt_fbsde_sgd_epochs = count;
    } else if (argument == "--line-search") {
      RESULT_ASSERT(value == "on" || value == "off" || value == "working-copy",
                    "--line-search is on, off, or working-copy.");
      options_out.fbrrt.line_search = value != "off";
      options_out.fbrrt.keep_working_copy = value == "working-copy";
    } else if (argument == "--entropy-temperature") {
      options_out.backward.entropy_temperature = number;
    } else if (argument == "--forward-noise") {
      RESULT_ASSERT(value == "on" || value == "off", "--forward-noise is on or off.");
      options_out.fbrrt.sample_forward_noise = value == "on";
    } else if (argument == "--target-floor") {
      options_out.backward.target_floor = number;
    } else if (argument == "--max-value-change") {
      options_out.backward.sgd.max_rms_value_change = number;
    } else {
      return ErrorResult("unrecognized argument; run with --help for the options.");
    }
  }
  return SuccessResult();
}

auto Run(int argc, char** argv) noexcept -> Result<> {
  FbrrtComparisonOptions options;
  bool trace = false;
  RESULT_RETURN_IF_ERROR(ParseOptions(argc, argv, options, trace));

  const auto progress =
      [trace](
          const std::string& benchmark, const std::string& method, std::size_t seed, const MethodRun& run) noexcept {
        std::cout << std::format("{:<28} {:<8} seed {:>2}: cost {:9.3f} -> {:9.3f} (best {:9.3f}) in {:7.2f} s\n",
                                 benchmark,
                                 method,
                                 seed,
                                 run.costs.front(),
                                 run.costs.back(),
                                 *std::ranges::min_element(run.costs),
                                 run.seconds.back());
        if (trace) {
          std::string costs;
          for (const double cost : run.costs) {
            costs += std::format(" {:.2f}", cost);
          }
          std::cout << "    costs:" << costs << '\n';
        }
      };
  RESULT_ASSIGN_OR_RETURN(const std::vector<BenchmarkComparison> comparisons, RunFbrrtComparison(options, progress));
  std::cout << '\n';
  for (const BenchmarkComparison& comparison : comparisons) {
    std::cout << SummarizeFbrrtComparison(comparison);
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
  if (!result.has_value()) {
    RESULT_REPORT_RESULT(result);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
