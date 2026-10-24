// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#ifndef FBSDE_TRAJ_OPT_FBRRT_RRT_EXPANSION_SELECTOR_HPP_
#define FBSDE_TRAJ_OPT_FBRRT_RRT_EXPANSION_SELECTOR_HPP_

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <utility>

#include <Eigen/Core>

#include "fbsde_traj_opt/fbrrt/brute_force_nearest_neighbor.hpp"
#include "fbsde_traj_opt/sampling/counter_based_normal_sampler.hpp"
#include "fbsde_traj_opt/sampling/l1_exploratory_control.hpp"
#include "fbsde_traj_opt/utils/result.hpp"

namespace fbsde_traj_opt {

// Concept for the rule that chooses which particle at a stage a new edge branches from: given the
// stage's particle block, its live count, the probability of an RRT draw, and the sampler
// coordinate (`sample`, `stage`) of the new particle, return the parent's slot.
//
// Each component of the FBRRT method draws from a sampler of its own stream, so a selector owns
// every component of its coordinate and may use as many as it likes.
template <typename T, typename StateBlock, typename Scalar>
concept FbrrtExpansionSelector = requires(const T& selector,
                                          const StateBlock& states,
                                          std::size_t count,
                                          Scalar rrt_probability,
                                          const CounterBasedNormalSampler& sampler,
                                          std::size_t sample,
                                          std::size_t stage) {
  { selector(states, count, rrt_probability, sampler, sample, stage) } -> std::same_as<std::size_t>;
};

// The expansion rule of Hawkins (2021), Algorithm 4.2, lines 4-8: with probability `eps_rrt`, draw
// a target `x_rand` uniformly from a region of interest and branch from the particle nearest to
// it; otherwise branch from a particle chosen uniformly.
//
// The RRT draw is what spreads the samples. A particle's chance of being nearest to a uniform
// target is the volume of its Voronoi cell within the region, so isolated particles on the
// frontier are chosen far more often than those inside a cluster, and the particle distribution
// is pushed outward to cover the reachable set. The uniform draw does the opposite -- a dense
// cluster gets attention in proportion to its density -- and `eps_rrt` trades the two.
//
// Sampler components of the new particle's coordinate: 0 decides between the two draws, 1 picks
// the uniform parent, and 2..N+1 are the coordinates of the RRT target.
template <int N, typename NearestT = BruteForceNearestNeighbor<N>, typename Scalar = double>
class RrtExpansionSelector {
 public:
  using State = Eigen::Matrix<Scalar, N, 1>;

  // Fails unless the region is finite with every upper bound above its lower bound.
  static auto Make(const State& region_lower, const State& region_upper, NearestT nearest) noexcept
      -> Result<RrtExpansionSelector> {
    RESULT_ASSERT(region_lower.allFinite() && region_upper.allFinite(),
                  "RrtExpansionSelector::Make: the region of interest must be finite.");
    RESULT_ASSERT((region_upper.array() > region_lower.array()).all(),
                  "RrtExpansionSelector::Make: every upper bound of the region must exceed its lower bound.");
    return SuccessResult(RrtExpansionSelector(region_lower, region_upper, std::move(nearest)));
  }

  // Returns the parent slot for a new particle.
  //
  // Precondition: `count >= 1` and `rrt_probability` lies in [0, 1].
  template <int Columns>
  auto operator()(const Eigen::Matrix<Scalar, N, Columns>& states,
                  std::size_t count,
                  Scalar rrt_probability,
                  const CounterBasedNormalSampler& sampler,
                  std::size_t sample,
                  std::size_t stage) const noexcept -> std::size_t {
    if (SampleUniform<Scalar>(sampler, sample, stage, kDecisionComponent) < rrt_probability) {
      State target;
      for (std::size_t axis = 0; axis < kStateDim; ++axis) {
        const auto index = static_cast<Eigen::Index>(axis);
        const auto fraction = SampleUniform<Scalar>(sampler, sample, stage, kTargetComponent + axis);
        target[index] = region_lower_[index] + (fraction * (region_upper_[index] - region_lower_[index]));
      }
      return nearest_(states, count, target);
    }
    const auto pick = SampleUniform<Scalar>(sampler, sample, stage, kUniformComponent);
    return std::min(static_cast<std::size_t>(pick * static_cast<Scalar>(count)), count - 1);
  }

  [[nodiscard]] auto region_lower() const noexcept -> const State& { return region_lower_; }

  [[nodiscard]] auto region_upper() const noexcept -> const State& { return region_upper_; }

 private:
  static constexpr auto kStateDim = static_cast<std::size_t>(N);
  static constexpr std::size_t kDecisionComponent = 0;
  static constexpr std::size_t kUniformComponent = 1;
  static constexpr std::size_t kTargetComponent = 2;

  RrtExpansionSelector(State region_lower, State region_upper, NearestT nearest) noexcept
      : region_lower_(std::move(region_lower)), region_upper_(std::move(region_upper)), nearest_(std::move(nearest)) {}

  State region_lower_;
  State region_upper_;
  NearestT nearest_;
};

// An RrtExpansionSelector with brute-force search weighted by the region's inverse half-widths,
// so that distance is measured in fractions of the region along every axis.
//
// Fails if the region is rejected.
template <int N, typename Scalar = double>
[[nodiscard]] auto MakeBruteForceRrtExpansionSelector(const Eigen::Matrix<Scalar, N, 1>& region_lower,
                                                      const Eigen::Matrix<Scalar, N, 1>& region_upper) noexcept
    -> Result<RrtExpansionSelector<N, BruteForceNearestNeighbor<N, Scalar>, Scalar>> {
  using Search = BruteForceNearestNeighbor<N, Scalar>;
  using Selector = RrtExpansionSelector<N, Search, Scalar>;
  RESULT_ASSERT((region_upper.array() > region_lower.array()).all(),
                "MakeBruteForceRrtExpansionSelector: every upper bound of the region must exceed its lower bound.");
  const Eigen::Matrix<Scalar, N, 1> inverse_half_widths = (Scalar{2} / (region_upper - region_lower).array()).matrix();
  RESULT_ASSIGN_OR_RETURN(Search nearest, Search::Make(inverse_half_widths));
  return Selector::Make(region_lower, region_upper, std::move(nearest));
}

}  // namespace fbsde_traj_opt

#endif  // FBSDE_TRAJ_OPT_FBRRT_RRT_EXPANSION_SELECTOR_HPP_
