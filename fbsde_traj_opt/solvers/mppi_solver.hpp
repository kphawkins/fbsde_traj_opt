// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#ifndef FBSDE_TRAJ_OPT_SOLVERS_MPPI_SOLVER_HPP_
#define FBSDE_TRAJ_OPT_SOLVERS_MPPI_SOLVER_HPP_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "fbsde_traj_opt/sampling/counter_based_normal_sampler.hpp"
#include "fbsde_traj_opt/utils/parallel_for.hpp"
#include "fbsde_traj_opt/utils/result.hpp"

namespace fbsde_traj_opt {

// Model predictive path integral control (Williams et al., "Information theoretic MPC for
// model-based reinforcement learning", ICRA 2017) over a fixed horizon, for one bounded control: a
// sampling-based baseline from outside the FBSDE family, and the closest relative of the
// path-integral side of it.
//
// Each iteration perturbs the nominal control sequence `U` with Gaussian noise of standard
// deviation `control_noise`, clamps each perturbed control into the box, rolls every sample out
// through the forward model -- with the problem's own noise if `sample_process_noise` -- and moves
// `U` to the exponentially weighted average of the applied controls,
//
//   w_i  proportional to  exp(-(J_i - min J) / lambda).
//
// The temperature `lambda` is set every iteration, by bisection, so that the weights' effective
// sample size `1 / sum w_i^2` is `effective_sample_fraction` of the samples: a scale-free choice
// that one value serves across problems whose costs differ by orders of magnitude, and that a few
// outlying samples cannot wash out the way a temperature tied to the cost range is. Sample zero is
// always the unperturbed nominal sequence, so an iteration that finds nothing better keeps what it
// has. The control noise shrinks by `noise_decay` per iteration down to `min_control_noise`, so the
// search narrows as it converges.
//
// The running cost is evaluated on the clamped control, as the problem defines it; nothing is
// smoothed. The solver plans over the stages from `first_stage` to the horizon from a given state,
// which is the whole problem offline and the shrinking remainder of it in receding-horizon use.

template <typename Scalar = double>
struct MppiOptions {
  std::size_t num_samples = 1024;
  Scalar control_noise = Scalar{0.5};
  Scalar noise_decay = Scalar{1};
  Scalar min_control_noise = Scalar{0};
  Scalar effective_sample_fraction = Scalar{0.05};
  bool sample_process_noise = true;
  // Roll the samples out across threads. Off when many solvers run at once, as in receding-horizon
  // evaluation, which parallelizes over the solvers instead.
  bool parallel = true;
};

template <typename Scalar = double>
struct MppiIterationReport {
  // The mean and the minimum of the samples' costs.
  Scalar mean_sample_cost = Scalar{0};
  Scalar min_sample_cost = Scalar{0};
  // The cost of the unperturbed sample: the nominal sequence before the update.
  Scalar nominal_sample_cost = Scalar{0};
  // 1 / sum w_i^2: how many samples the update effectively averaged.
  Scalar effective_samples = Scalar{0};
  Scalar temperature = Scalar{0};
  Scalar control_noise = Scalar{0};
};

// The open-loop policy: the planned control at each stage, whatever the state.
template <int N, typename Scalar = double>
class OpenLoopPolicy {
 public:
  using State = Eigen::Matrix<Scalar, N, 1>;
  using Control = Eigen::Matrix<Scalar, 1, 1>;

  explicit OpenLoopPolicy(std::vector<Scalar> controls) noexcept : controls_(std::move(controls)) {}

  auto operator()(std::size_t stage, [[maybe_unused]] const State& state) const noexcept -> Control {
    return Control(controls_[stage]);
  }

 private:
  std::vector<Scalar> controls_;
};

template <int N,
          std::size_t NumStages,
          typename ForwardModelT,
          typename RunningCostT,
          typename TerminalCostT,
          typename Scalar = double>
class MppiSolver {
 public:
  using State = Eigen::Matrix<Scalar, N, 1>;
  using Control = Eigen::Matrix<Scalar, 1, 1>;
  using StateMat = Eigen::Matrix<Scalar, N, N>;
  using IterationReport = MppiIterationReport<Scalar>;
  using Policy = OpenLoopPolicy<N, Scalar>;

  static constexpr std::size_t kNumControlStages = NumStages - 1;
  static_assert(NumStages >= 2, "a problem needs an initial and a terminal stage.");

  // `initial_controls` holds one control per control stage and is clamped into the box. Offline
  // iterations start from `initial_mean + initial_covariance_factor * w`.
  //
  // Fails unless the box is nonempty, the control count matches the horizon, there are at least two
  // samples, the noise is positive and decays by a factor in (0, 1], and the effective sample
  // fraction is in (0, 1).
  [[nodiscard]] static auto Make(ForwardModelT forward_model,
                                 RunningCostT running_cost,
                                 TerminalCostT terminal_cost,
                                 const State& initial_mean,
                                 const StateMat& initial_covariance_factor,
                                 Scalar lower,
                                 Scalar upper,
                                 std::span<const Scalar> initial_controls,
                                 const MppiOptions<Scalar>& options,
                                 std::uint64_t seed) noexcept -> Result<MppiSolver> {
    RESULT_ASSERT(lower < upper, "MppiSolver::Make: the control box is empty.");
    RESULT_ASSERT(initial_controls.size() == kNumControlStages,
                  "MppiSolver::Make: there must be one initial control per control stage.");
    RESULT_ASSERT(options.num_samples >= 2, "MppiSolver::Make: MPPI needs at least two samples.");
    RESULT_ASSERT(options.control_noise > Scalar{0} && options.noise_decay > Scalar{0} &&
                      options.noise_decay <= Scalar{1},
                  "MppiSolver::Make: the control noise must be positive and decay by a factor in (0, 1].");
    RESULT_ASSERT(options.effective_sample_fraction > Scalar{0} && options.effective_sample_fraction < Scalar{1},
                  "MppiSolver::Make: the effective sample fraction must be in (0, 1).");
    MppiSolver solver(std::move(forward_model),
                      std::move(running_cost),
                      std::move(terminal_cost),
                      initial_mean,
                      initial_covariance_factor,
                      lower,
                      upper,
                      options,
                      seed);
    for (std::size_t stage = 0; stage < kNumControlStages; ++stage) {
      solver.controls_[stage] = std::clamp(initial_controls[stage], lower, upper);
    }
    return SuccessResult(std::move(solver));
  }

  // One offline iteration over the whole horizon.
  auto Iterate() noexcept -> IterationReport { return Update(0, nullptr); }

  // One iteration over the stages from `first_stage` on, every sample starting at `state`.
  auto IterateFrom(std::size_t first_stage, const State& state) noexcept -> IterationReport {
    return Update(first_stage, &state);
  }

  [[nodiscard]] auto CurrentPolicy() const noexcept -> Policy { return Policy(controls_); }
  [[nodiscard]] auto NominalControls() const noexcept -> std::span<const Scalar> { return controls_; }
  [[nodiscard]] auto IterationCount() const noexcept -> std::size_t { return iteration_; }

 private:
  static constexpr std::uint64_t kControlNoiseStream = 0x6A09E667F3BCC909ULL;
  static constexpr std::uint64_t kProcessNoiseStream = 0xBB67AE8584CAA73BULL;

  MppiSolver(ForwardModelT forward_model,
             RunningCostT running_cost,
             TerminalCostT terminal_cost,
             const State& initial_mean,
             const StateMat& initial_covariance_factor,
             Scalar lower,
             Scalar upper,
             const MppiOptions<Scalar>& options,
             std::uint64_t seed) noexcept
      : forward_model_(std::move(forward_model)),
        running_cost_(std::move(running_cost)),
        terminal_cost_(std::move(terminal_cost)),
        initial_mean_(initial_mean),
        initial_covariance_factor_(initial_covariance_factor),
        lower_(lower),
        upper_(upper),
        options_(options),
        control_noise_(options.control_noise),
        control_sampler_(seed ^ kControlNoiseStream),
        process_sampler_(seed ^ kProcessNoiseStream),
        controls_(kNumControlStages, Scalar{0}),
        applied_(options.num_samples * kNumControlStages, Scalar{0}),
        costs_(options.num_samples, Scalar{0}),
        weights_(options.num_samples, Scalar{0}) {}

  // `start` is null offline, where each sample draws its own initial state.
  auto Update(std::size_t first_stage, const State* start) noexcept -> IterationReport {
    const std::size_t samples = options_.num_samples;
    const std::size_t base = iteration_ * samples;
    const auto rollout = [&](std::size_t sample) noexcept {
      const std::size_t trajectory = base + sample;
      State noise;
      State state;
      if (start != nullptr) {
        state = *start;
      } else {
        process_sampler_.FillNoise(trajectory, 0, noise);
        state = initial_mean_ + (initial_covariance_factor_ * noise);
      }
      Scalar cost = Scalar{0};
      Scalar* applied = applied_.data() + (sample * kNumControlStages);
      for (std::size_t stage = first_stage; stage < kNumControlStages; ++stage) {
        const Scalar perturbation =
            sample == 0 ? Scalar{0}
                        : control_noise_ * static_cast<Scalar>(control_sampler_.Sample(trajectory, stage, 0));
        const Control control(std::clamp(controls_[stage] + perturbation, lower_, upper_));
        applied[stage] = control[0];
        cost += running_cost_(stage, state, control);
        if (options_.sample_process_noise) {
          process_sampler_.FillNoise(trajectory, stage + 1, noise);
        } else {
          noise.setZero();
        }
        state = forward_model_(stage, state, control, noise);
      }
      costs_[sample] = cost + terminal_cost_(state);
    };
    if (options_.parallel) {
      ParallelFor(samples, rollout);
    } else {
      for (std::size_t sample = 0; sample < samples; ++sample) {
        rollout(sample);
      }
    }

    IterationReport report;
    report.nominal_sample_cost = costs_[0];
    Scalar min_cost = std::numeric_limits<Scalar>::infinity();
    Scalar max_cost = -std::numeric_limits<Scalar>::infinity();
    Scalar sum = Scalar{0};
    for (const Scalar cost : costs_) {
      min_cost = std::min(min_cost, cost);
      max_cost = std::max(max_cost, cost);
      sum += cost;
    }
    report.mean_sample_cost = sum / static_cast<Scalar>(samples);
    report.min_sample_cost = min_cost;
    // Bisect the log-temperature between a millionth and a million times the cost range for the
    // target effective sample size, which grows monotonically with the temperature.
    const Scalar range = std::max(max_cost - min_cost, std::numeric_limits<Scalar>::min());
    const Scalar target = options_.effective_sample_fraction * static_cast<Scalar>(samples);
    Scalar low = std::log(range * Scalar{1e-6});
    Scalar high = std::log(range * Scalar{1e6});
    for (std::size_t step = 0; step < 50; ++step) {
      const Scalar middle = Scalar{0.5} * (low + high);
      (EffectiveSamples(min_cost, std::exp(middle)) < target ? low : high) = middle;
    }
    report.temperature = std::exp(Scalar{0.5} * (low + high));
    report.effective_samples = EffectiveSamples(min_cost, report.temperature);
    report.control_noise = control_noise_;
    for (std::size_t stage = first_stage; stage < kNumControlStages; ++stage) {
      Scalar control = Scalar{0};
      for (std::size_t sample = 0; sample < samples; ++sample) {
        control += weights_[sample] * applied_[(sample * kNumControlStages) + stage];
      }
      controls_[stage] = control;
    }
    control_noise_ = std::max(options_.min_control_noise, control_noise_ * options_.noise_decay);
    ++iteration_;
    return report;
  }

  // Sets the normalized weights at `temperature` and returns their effective sample size.
  auto EffectiveSamples(Scalar min_cost, Scalar temperature) noexcept -> Scalar {
    Scalar total = Scalar{0};
    for (std::size_t sample = 0; sample < costs_.size(); ++sample) {
      weights_[sample] = std::exp(-(costs_[sample] - min_cost) / temperature);
      total += weights_[sample];
    }
    Scalar square_sum = Scalar{0};
    for (Scalar& weight : weights_) {
      weight /= total;
      square_sum += weight * weight;
    }
    return Scalar{1} / square_sum;
  }

  ForwardModelT forward_model_;
  RunningCostT running_cost_;
  TerminalCostT terminal_cost_;
  State initial_mean_;
  StateMat initial_covariance_factor_;
  Scalar lower_;
  Scalar upper_;
  MppiOptions<Scalar> options_;
  Scalar control_noise_;
  CounterBasedNormalSampler control_sampler_;
  CounterBasedNormalSampler process_sampler_;
  std::size_t iteration_ = 0;

  std::vector<Scalar> controls_;
  std::vector<Scalar> applied_;
  std::vector<Scalar> costs_;
  std::vector<Scalar> weights_;
};

}  // namespace fbsde_traj_opt

#endif  // FBSDE_TRAJ_OPT_SOLVERS_MPPI_SOLVER_HPP_
