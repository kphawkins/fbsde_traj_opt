// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#ifndef FBSDE_TRAJ_OPT_FBRRT_PATH_INTEGRAL_ERODE_HPP_
#define FBSDE_TRAJ_OPT_FBRRT_PATH_INTEGRAL_ERODE_HPP_

#include <concepts>
#include <cstddef>
#include <vector>

#include "fbsde_traj_opt/utils/result.hpp"

namespace fbsde_traj_opt {

// Concept for the rule that thins the particle set after a backward pass, so that the next
// forward pass has room to add new particles: given the set and the value function table, remove
// particles in place.
template <typename T, typename ParticleSetT, typename ValueFunctionT>
concept FbrrtEroder = requires(const T& eroder, ParticleSetT& particles, const std::vector<ValueFunctionT>& table) {
  { eroder(particles, table) } -> std::same_as<Result<>>;
};

// The erosion step of Hawkins (2021), Algorithm 4.4, without the tree constraint.
//
// Every particle at every stage past the first is scored by the optimal trajectory distribution
// heuristic of Section 4.3.2,
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
// That heuristic is also where the thesis's weighting of the regression went: rather than
// weighting each sample's residual by exp(-rho / lambda) in the backward pass, the heuristic
// decides which samples are there at all.
class PathIntegralErode {
 public:
  // Fails if `keep_count` is zero: a stage with no particles has nothing to expand from.
  static auto Make(std::size_t keep_count) noexcept -> Result<PathIntegralErode> {
    RESULT_ASSERT(keep_count >= 1, "PathIntegralErode::Make: every stage must keep at least one particle.");
    return SuccessResult(PathIntegralErode(keep_count));
  }

  // Scores every live particle of every stage past the first, records the scores as the stage's
  // heuristics, and keeps the `keep_count` lowest at each stage.
  //
  // Fails if the table does not have one value function per stage.
  template <typename ParticleSetT, typename ValueFunctionT>
  auto operator()(ParticleSetT& particles, const std::vector<ValueFunctionT>& table) const noexcept -> Result<> {
    RESULT_ASSERT(table.size() == ParticleSetT::kNumStages,
                  "PathIntegralErode: the table must hold one value function per stage.");
    for (std::size_t stage = ParticleSetT::kNumStages; stage-- > 1;) {
      typename ParticleSetT::ValueBlock values;
      table[stage].template Values<ParticleSetT::kCapacity>(particles.states(stage), values);
      const typename ParticleSetT::ValueBlock heuristics = particles.accrued_costs(stage) + values;
      particles.SetHeuristics(stage, heuristics);
      RESULT_RETURN_IF_ERROR(particles.KeepLowest(stage, heuristics, keep_count_));
    }
    return SuccessResult();
  }

  [[nodiscard]] auto keep_count() const noexcept -> std::size_t { return keep_count_; }

 private:
  explicit PathIntegralErode(std::size_t keep_count) noexcept : keep_count_(keep_count) {}

  std::size_t keep_count_;
};

}  // namespace fbsde_traj_opt

#endif  // FBSDE_TRAJ_OPT_FBRRT_PATH_INTEGRAL_ERODE_HPP_
