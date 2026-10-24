// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#ifndef FBSDE_TRAJ_OPT_FBRRT_PATH_INTEGRAL_ERODE_HPP_
#define FBSDE_TRAJ_OPT_FBRRT_PATH_INTEGRAL_ERODE_HPP_

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "fbsde_traj_opt/sampling/counter_based_normal_sampler.hpp"
#include "fbsde_traj_opt/sampling/l1_exploratory_control.hpp"
#include "fbsde_traj_opt/utils/result.hpp"

namespace fbsde_traj_opt {

// Concept for the rule that thins the particle set after a backward pass, so that the next
// forward pass has room to add new particles: given the set, the value function table, and a seed
// for any randomness it uses, remove particles in place.
template <typename T, typename ParticleSetT, typename ValueFunctionT>
concept FbrrtEroder =
    requires(const T& eroder, ParticleSetT& particles, const std::vector<ValueFunctionT>& table, std::uint64_t seed) {
      { eroder(particles, table, seed) } -> std::same_as<Result<>>;
    };

// The knobs of PathIntegralErode beyond its keep count. Every one is off at its default, which is
// the nominal erosion: each stage keeps its lowest-rho particles, on its own.
template <typename Scalar = double>
struct PathIntegralErodeOptions {
  // Zero keeps exactly the `keep_count` lowest-rho particles. A positive `lambda` instead draws the
  // survivors at random, without replacement, with probabilities proportional to
  // `exp(-rho / lambda)` -- erosion as a draw from the local-entropy distribution of the thesis
  // rather than a cut, so that a region whose particles are slightly worse is thinned rather than
  // emptied. `lambda` is in units of the heuristic's spread over the stage, so it transfers between
  // problems.
  Scalar temperature = Scalar{0};

  // A finite radius removes every particle whose edge left from a state farther than this from all
  // the particles that survive at the stage before -- a population whose parents have all been
  // eroded, and so which nothing at the previous stage can reach. Distances are in the metric
  // `metric_scale` gives. With it, stages are eroded forward in time, so that each stage is checked
  // against the survivors of the one before.
  Scalar orphan_radius = std::numeric_limits<Scalar>::infinity();

  // A positive weight `alpha` ranks a particle by `rho - alpha log(d / median d)` in units of the
  // heuristic's spread, `d` its distance to the nearest other particle at its stage, so that a dense
  // cluster loses particles before an isolated one does.
  Scalar density_weight = Scalar{0};

  // Particles that have survived more than this many erosions are removed first.
  std::size_t max_age = std::numeric_limits<std::size_t>::max();
};

// The erosion step of Hawkins (2021), Algorithm 4.4, without the tree constraint.
//
// Every particle at every stage past the first is scored by the optimal trajectory distribution
// heuristic of the thesis's Chapter 4,
//
//   rho_i = L_{0:i-1} + V~_i(x_i),
//
// the running cost it took to reach its state plus the estimated cost from there: an estimate of
// the cost of the best trajectory through it. Each stage then keeps its `keep_count` lowest-rho
// particles. The survivors are the particles on or near good trajectories, and the next forward
// pass refills the freed slots around them.
//
// The thesis removed a particle only if it had no children, which keeps the samples a connected
// tree -- and made long single branches hard to erode, since a whole branch had to stay as long as
// one leaf of it was worth keeping. Here each stage is thinned on its own. A particle carries its
// own edge and accrued cost (FbrrtParticleSet), so its children lose nothing when it is removed,
// and the sample set becomes a sequence of particle distributions rather than a tree. What makes
// the method RRT-like is the Voronoi-biased expansion, not the tree shape.
//
// Thinned on its own, a stage can also come apart from its neighbors: a population whose parents
// were eroded long ago survives as long as its heuristic stays low, and when the value function is
// wrong in its favor that can be indefinitely. The options counter that: resampling instead of a
// hard cut, removing particles nothing at the previous stage can reach, ranking by density as well
// as by the heuristic, and a limit on age; see PathIntegralErodeOptions.
//
// That heuristic is also where the thesis's weighting of the regression went: rather than
// weighting each sample's residual by exp(-rho / lambda) in the backward pass, the heuristic
// decides which samples are there at all.
class PathIntegralErode {
 public:
  using Options = PathIntegralErodeOptions<double>;

  // `metric_scale` weights each state coordinate in the distances the orphan and density options
  // measure -- the inverse half-widths of the region of interest, say -- and may be empty for the
  // plain Euclidean distance.
  //
  // Fails if `keep_count` is zero, if the temperature or density weight is negative or not finite,
  // if the orphan radius is not positive, or if a metric weight is not finite and positive.
  static auto Make(std::size_t keep_count, const Options& options = {}, std::vector<double> metric_scale = {}) noexcept
      -> Result<PathIntegralErode> {
    RESULT_ASSERT(keep_count >= 1, "PathIntegralErode::Make: every stage must keep at least one particle.");
    const bool valid_temperature = std::isfinite(options.temperature) && options.temperature >= 0.0;
    const bool valid_density_weight = std::isfinite(options.density_weight) && options.density_weight >= 0.0;
    RESULT_ASSERT(valid_temperature, "PathIntegralErode::Make: the temperature must be finite and non-negative.");
    RESULT_ASSERT(valid_density_weight, "PathIntegralErode::Make: the density weight must be finite and non-negative.");
    RESULT_ASSERT(options.orphan_radius > 0.0, "PathIntegralErode::Make: the orphan radius must be positive.");
    for (const double weight : metric_scale) {
      const bool valid_weight = std::isfinite(weight) && weight > 0.0;
      RESULT_ASSERT(valid_weight, "PathIntegralErode::Make: every metric weight must be finite and positive.");
    }
    return SuccessResult(PathIntegralErode(keep_count, options, std::move(metric_scale)));
  }

  // Scores every live particle of every stage past the first, records the scores as the stage's
  // heuristics, thins each stage to at most `keep_count` particles, and ages the survivors. `seed`
  // drives the resampling draws.
  //
  // Fails if the table does not have one value function per stage, or the metric is not of the
  // state's dimension.
  template <typename ParticleSetT, typename ValueFunctionT>
  auto operator()(ParticleSetT& particles, const std::vector<ValueFunctionT>& table, std::uint64_t seed) const noexcept
      -> Result<> {
    using State = typename ParticleSetT::State;
    constexpr std::size_t kNumStages = ParticleSetT::kNumStages;
    RESULT_ASSERT(table.size() == kNumStages, "PathIntegralErode: the table must hold one value function per stage.");
    const bool metric_matches =
        metric_scale_.empty() || metric_scale_.size() == static_cast<std::size_t>(State::RowsAtCompileTime);
    RESULT_ASSERT(metric_matches, "PathIntegralErode: the metric must have one weight per state coordinate.");
    State scale = State::Ones();
    for (std::size_t axis = 0; axis < metric_scale_.size(); ++axis) {
      scale[static_cast<Eigen::Index>(axis)] = metric_scale_[axis];
    }
    const CounterBasedNormalSampler sampler(seed ^ kErosionStream);

    const bool forward = std::isfinite(options_.orphan_radius);
    for (std::size_t step = 1; step < kNumStages; ++step) {
      const std::size_t stage = forward ? step : kNumStages - step;
      typename ParticleSetT::ValueBlock values;
      table[stage].template Values<ParticleSetT::kCapacity>(particles.states(stage), values);
      const typename ParticleSetT::ValueBlock heuristics = particles.accrued_costs(stage) + values;
      particles.SetHeuristics(stage, heuristics);
      typename ParticleSetT::ValueBlock keys = heuristics;
      const std::size_t keep = Rank(particles, stage, scale, sampler, keys);
      RESULT_RETURN_IF_ERROR(particles.KeepLowest(stage, keys, keep));
    }
    particles.IncrementAges();
    return SuccessResult();
  }

  [[nodiscard]] auto keep_count() const noexcept -> std::size_t { return keep_count_; }

  [[nodiscard]] auto options() const noexcept -> const Options& { return options_; }

 private:
  static constexpr std::uint64_t kErosionStream = 0x1F83D9ABFB41BD6BULL;

  // The key of a particle that must go.
  static constexpr double kRemoved = std::numeric_limits<double>::infinity();

  PathIntegralErode(std::size_t keep_count, const Options& options, std::vector<double> metric_scale) noexcept
      : keep_count_(keep_count), options_(options), metric_scale_(std::move(metric_scale)) {}

  // Turns `keys`, holding the stage's heuristics, into the keys KeepLowest() ranks by, and returns how
  // many particles to keep: no more than `keep_count`, and none whose key marks it removed.
  template <typename ParticleSetT>
  auto Rank(const ParticleSetT& particles,
            std::size_t stage,
            const typename ParticleSetT::State& scale,
            const CounterBasedNormalSampler& sampler,
            typename ParticleSetT::ValueBlock& keys) const noexcept -> std::size_t {
    using State = typename ParticleSetT::State;
    const std::size_t count = particles.count(stage);
    if (count == 0) {
      return 0;
    }
    const auto live = static_cast<Eigen::Index>(count);
    const auto distance = [&scale](const State& a, const State& b) noexcept {
      return (scale.cwiseProduct(a - b)).norm();
    };

    // The heuristic, standardized over the stage.
    const double lowest = keys.leftCols(live).minCoeff();
    const double mean = keys.leftCols(live).mean();
    const double spread =
        std::sqrt((keys.leftCols(live).array() - mean).square().mean()) + std::numeric_limits<double>::epsilon();
    keys.leftCols(live) = (keys.leftCols(live).array() - lowest) / spread;

    if (options_.density_weight > 0.0 && count > 1) {
      std::vector<double> nearest(count, std::numeric_limits<double>::infinity());
      for (std::size_t slot = 0; slot < count; ++slot) {
        const State x = particles.states(stage).col(static_cast<Eigen::Index>(slot));
        for (std::size_t other = 0; other < count; ++other) {
          if (other != slot) {
            nearest[slot] =
                std::min(nearest[slot], distance(x, particles.states(stage).col(static_cast<Eigen::Index>(other))));
          }
        }
      }
      std::vector<double> sorted = nearest;
      std::ranges::nth_element(sorted, sorted.begin() + static_cast<std::ptrdiff_t>(count / 2));
      const double median = std::max(sorted[count / 2], std::numeric_limits<double>::min());
      const double floor = 1e-6 * median;
      for (std::size_t slot = 0; slot < count; ++slot) {
        keys[static_cast<Eigen::Index>(slot)] -=
            options_.density_weight * std::log(std::max(nearest[slot], floor) / median);
      }
    }

    if (options_.temperature > 0.0) {
      // Gumbel-top-k: the `keep` lowest of `z / lambda - G`, G standard Gumbel, are a draw without
      // replacement with probabilities proportional to `exp(-z / lambda)`.
      for (std::size_t slot = 0; slot < count; ++slot) {
        const double uniform = std::clamp(SampleUniform<double>(sampler, slot, stage, 0), 1e-300, 1.0 - 1e-16);
        const auto column = static_cast<Eigen::Index>(slot);
        keys[column] = (keys[column] / options_.temperature) + std::log(-std::log(uniform));
      }
    }

    std::size_t available = count;
    for (std::size_t slot = 0; slot < count; ++slot) {
      const auto column = static_cast<Eigen::Index>(slot);
      bool removed = particles.ages(stage)[slot] > options_.max_age;
      if (!removed && std::isfinite(options_.orphan_radius) && stage >= 2) {
        const State source = particles.source_states(stage).col(column);
        removed = true;
        for (std::size_t parent = 0; parent < particles.count(stage - 1) && removed; ++parent) {
          removed = distance(source, particles.states(stage - 1).col(static_cast<Eigen::Index>(parent))) >
                    options_.orphan_radius;
        }
      }
      if (removed) {
        keys[column] = kRemoved;
        --available;
      }
    }
    return std::min(keep_count_, available);
  }

  std::size_t keep_count_;
  Options options_;
  std::vector<double> metric_scale_;
};

}  // namespace fbsde_traj_opt

#endif  // FBSDE_TRAJ_OPT_FBRRT_PATH_INTEGRAL_ERODE_HPP_
