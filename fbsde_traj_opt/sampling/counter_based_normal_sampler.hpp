// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#ifndef FBSDE_TRAJ_OPT_COUNTER_BASED_NORMAL_SAMPLER_HPP_
#define FBSDE_TRAJ_OPT_COUNTER_BASED_NORMAL_SAMPLER_HPP_

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>

#include <Eigen/Core>

#include "fbsde_traj_opt/utils/eigen_concepts.hpp"

namespace fbsde_traj_opt {

// A stateless, counter-based source of standard normal variates.
//
// Unlike a conventional generator, this one holds no mutable state: the variate at a given
// coordinate is a pure function of the seed and that coordinate,
//
//   z(trajectory, stage, component) = g(seed, trajectory, stage, component),
//
// so it can be evaluated in any order, any number of times, from any thread, always giving the
// same answer. This is the property that makes a sampled trajectory batch comparable across
// models: two batches drawn with the same seed, trajectory count, stage count, and state
// dimension see exactly the same noise, so any difference between them is attributable to the
// dynamics, the cost, or the policy rather than to the random numbers. A conventional generator
// could not promise this -- advancing it depends on how many variates have been drawn so far,
// which in turn depends on how the sampling loop happens to be written.
//
// The hash is the SplitMix64 finalizer, chosen because it passes the usual statistical batteries,
// needs no state beyond its input, and is a handful of instructions. The variate itself comes
// from a Box-Muller transform of two independently hashed uniforms. Box-Muller normally produces
// variates in pairs and a stateful sampler would cache the second one; here the second is simply
// discarded, since paying one extra hash and one extra transcendental is the price of the purity
// above.
class CounterBasedNormalSampler {
 public:
  // Builds a sampler for the given RNG seed. Any seed value is valid, including zero: the
  // SplitMix64 finalizer has no weak seeds, because the seed is mixed rather than used directly
  // as generator state.
  explicit constexpr CounterBasedNormalSampler(std::uint64_t seed) noexcept : seed_(seed) {}

  // Returns the standard normal variate at coordinate (`trajectory`, `stage`, `component`).
  //
  // The three coordinates are hashed rather than folded into a single linear index, so a caller
  // need not know the extent of any axis in advance, and enlarging a batch along one axis leaves
  // every already-drawn variate unchanged.
  [[nodiscard]] auto Sample(std::size_t trajectory, std::size_t stage, std::size_t component) const noexcept -> double {
    std::uint64_t hash = HashCombine(seed_, static_cast<std::uint64_t>(trajectory));
    hash = HashCombine(hash, static_cast<std::uint64_t>(stage));
    hash = HashCombine(hash, static_cast<std::uint64_t>(component));

    // Two uniforms from one coordinate hash, separated by distinct stream constants so that the
    // radius and the angle below are drawn independently.
    const double radius_uniform = UnitUniform(HashCombine(hash, kRadiusStream));
    const double angle_uniform = UnitUniform(HashCombine(hash, kAngleStream));

    // Box-Muller. `radius_uniform` lies in (0, 1], never 0, so the logarithm is finite.
    return std::sqrt(-2.0 * std::log(radius_uniform)) * std::cos(2.0 * std::numbers::pi * angle_uniform);
  }

  // Fills `noise_out` with the standard normal increment for (`trajectory`, `stage`), drawing one
  // variate per component.
  //
  // `Vector` is any fixed-size Eigen column vector; its scalar type may be narrower than double,
  // in which case each variate is drawn in double precision and then converted.
  template <EigenFixedSizeColumnVector Vector>
  auto FillNoise(std::size_t trajectory, std::size_t stage, Vector& noise_out) const noexcept -> void {
    using Scalar = typename Vector::Scalar;
    for (std::size_t component = 0; component < static_cast<std::size_t>(Vector::RowsAtCompileTime); ++component) {
      noise_out[static_cast<Eigen::Index>(component)] = static_cast<Scalar>(Sample(trajectory, stage, component));
    }
  }

  // Returns the seed this sampler was built with.
  [[nodiscard]] constexpr auto seed() const noexcept -> std::uint64_t { return seed_; }

 private:
  // Stream constants distinguishing the two uniforms drawn per coordinate. The values are
  // arbitrary; they only need to differ from each other and from any plausible coordinate.
  static constexpr std::uint64_t kRadiusStream = 0x9E3779B97F4A7C15ULL;
  static constexpr std::uint64_t kAngleStream = 0xBF58476D1CE4E5B9ULL;

  // The SplitMix64 finalizer: an avalanching bijection on 64 bits.
  [[nodiscard]] static constexpr auto Mix(std::uint64_t value) noexcept -> std::uint64_t {
    value ^= value >> 30U;
    value *= 0xBF58476D1CE4E5B9ULL;
    value ^= value >> 27U;
    value *= 0x94D049BB133111EBULL;
    value ^= value >> 31U;
    return value;
  }

  // Folds `value` into `hash`. The golden-ratio increment before mixing keeps successive small
  // values (a stage index counting up by one, say) from landing in correlated parts of the
  // output, which is the same trick SplitMix64 uses to advance its own state.
  [[nodiscard]] static constexpr auto HashCombine(std::uint64_t hash, std::uint64_t value) noexcept -> std::uint64_t {
    return Mix(hash + kRadiusStream + Mix(value));
  }

  // Maps a hashed word onto a uniform in (0, 1].
  //
  // The top 53 bits are used because that is exactly the precision of a double's significand.
  // Adding one before scaling shifts the range from [0, 1) to (0, 1], which matters because the
  // caller takes its logarithm: the excluded endpoint must be 0, not 1.
  [[nodiscard]] static auto UnitUniform(std::uint64_t hash) noexcept -> double {
    return static_cast<double>((hash >> 11U) + 1U) * 0x1.0p-53;
  }

  std::uint64_t seed_;
};

}  // namespace fbsde_traj_opt

#endif  // FBSDE_TRAJ_OPT_COUNTER_BASED_NORMAL_SAMPLER_HPP_
