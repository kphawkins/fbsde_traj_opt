// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#ifndef FBSDE_TRAJ_OPT_FBRRT_FBRRT_PARTICLE_SET_HPP_
#define FBSDE_TRAJ_OPT_FBRRT_FBRRT_PARTICLE_SET_HPP_

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <numeric>
#include <span>
#include <vector>

#include <Eigen/Core>

#include "fbsde_traj_opt/utils/result.hpp"

namespace fbsde_traj_opt {

// The sample store of the FBRRT method: at every stage, up to `Capacity` particles, each carrying
// the edge that produced it.
//
// In Hawkins (2021), Section 4.3, the forward samples form a tree -- every node knows its parent,
// and the path back to the root is what the path measure of a stage is made of. What the method
// actually reads from that path is less than the path itself. The backward pass at stage `i` needs
// only the edges into stage `i + 1`: the state `x_i` an edge left from and the drift `K_i` it took.
// The path-integral heuristic needs only the running cost accrued along the path, which the
// forward pass computes once and can store on the particle. So each particle here keeps exactly
// that -- its state, its edge's source state, drift and control, and its accrued cost -- and a
// stage can be thinned without looking at any other stage. Removing a particle whose children
// survive does not strand them: they still hold everything they need. The set is then a sequence
// of per-stage particle distributions, linked by the edges each particle remembers, rather than a
// tree that must stay connected.
//
// Each particle also records the slot of its parent at the time it was drawn, and thinning a stage
// remaps the parent slots of the next stage (to kDetachedParent when the parent is gone). Nothing
// in the nominal method reads them; they are kept so that a tree-preserving variant -- one that
// only removes particles without children, as the thesis did -- remains a drop-in experiment.
//
// Stage 0 holds the initial particles and has no edges. Storage is fixed-size Eigen blocks, one
// column per slot, held on the heap: at the widths the method runs at they come to megabytes.
template <int N, int M, std::size_t Capacity, std::size_t NumStages, typename Scalar = double>
class FbrrtParticleSet {
  static_assert(Capacity >= 1, "A stage needs room for at least one particle.");
  static_assert(NumStages >= 2, "A problem needs at least an initial and a terminal stage.");

 public:
  using State = Eigen::Matrix<Scalar, N, 1>;
  using Control = Eigen::Matrix<Scalar, M, 1>;

  static constexpr int kCapacity = static_cast<int>(Capacity);
  static constexpr std::size_t kNumStages = NumStages;
  static constexpr std::size_t kDetachedParent = std::numeric_limits<std::size_t>::max();

  using StateBlock = Eigen::Matrix<Scalar, N, kCapacity>;
  using ControlBlock = Eigen::Matrix<Scalar, M, kCapacity>;
  using ValueBlock = Eigen::Matrix<Scalar, 1, kCapacity>;

  // One particle at a stage past the first, with the edge that produced it.
  struct Particle {
    State state = State::Zero();
    State source_state = State::Zero();
    State drift = State::Zero();
    Control control = Control::Zero();
    Scalar accrued_cost = Scalar{0};
    std::size_t parent = kDetachedParent;
  };

  // An empty set: every stage, the initial one included, holds no particles.
  FbrrtParticleSet() noexcept : stages_(NumStages) {}

  // Fills stage 0 with `Capacity` initial particles.
  auto SetInitialStates(const StateBlock& states) noexcept -> void {
    Stage& initial = stages_.front();
    initial.states = states;
    initial.accrued_costs.setZero();
    initial.count = Capacity;
  }

  // Writes `particle` into slot `slot` of `stage`. Distinct slots may be written concurrently.
  //
  // Precondition: `0 < stage < NumStages` and `slot < Capacity`. The slot counts as live only once
  // SetCount() covers it.
  auto WriteParticle(std::size_t stage, std::size_t slot, const Particle& particle) noexcept -> void {
    Stage& target = stages_[stage];
    const auto column = static_cast<Eigen::Index>(slot);
    target.states.col(column) = particle.state;
    target.source_states.col(column) = particle.source_state;
    target.drifts.col(column) = particle.drift;
    target.controls.col(column) = particle.control;
    target.accrued_costs[column] = particle.accrued_cost;
    target.parents[slot] = particle.parent;
  }

  // Sets how many leading slots of `stage` are live.
  //
  // Fails if `stage` is not a stage or `count` exceeds the capacity.
  auto SetCount(std::size_t stage, std::size_t count) noexcept -> Result<> {
    RESULT_ASSERT(stage < NumStages, "FbrrtParticleSet::SetCount: no such stage.");
    RESULT_ASSERT(count <= Capacity, "FbrrtParticleSet::SetCount: the count exceeds the capacity.");
    stages_[stage].count = count;
    return SuccessResult();
  }

  // Records a heuristic value per slot of `stage`, for plotting and for thinning by it.
  auto SetHeuristics(std::size_t stage, const ValueBlock& heuristics) noexcept -> void {
    stages_[stage].heuristics = heuristics;
  }

  // Keeps the `keep_count` live particles of `stage` with the lowest `keys` and discards the rest,
  // compacting the survivors to the leading slots in their original order. The parent slots of the
  // next stage are remapped to match. Does nothing if the stage holds no more than `keep_count`.
  //
  // Fails if `stage` is the initial stage or not a stage.
  auto KeepLowest(std::size_t stage, const ValueBlock& keys, std::size_t keep_count) noexcept -> Result<> {
    RESULT_ASSERT(stage >= 1, "FbrrtParticleSet::KeepLowest: the initial stage is never thinned.");
    RESULT_ASSERT(stage < NumStages, "FbrrtParticleSet::KeepLowest: no such stage.");
    Stage& target = stages_[stage];
    if (target.count <= keep_count) {
      return SuccessResult();
    }

    // Rank the live slots by key, ties broken by slot so the result does not depend on the sort.
    std::vector<std::size_t> order(target.count);
    std::ranges::iota(order, std::size_t{0});
    std::ranges::nth_element(
        order, order.begin() + static_cast<std::ptrdiff_t>(keep_count), [&](std::size_t a, std::size_t b) {
          const Scalar key_a = keys[static_cast<Eigen::Index>(a)];
          const Scalar key_b = keys[static_cast<Eigen::Index>(b)];
          return key_a < key_b || (key_a == key_b && a < b);
        });
    order.resize(keep_count);
    std::ranges::sort(order);

    std::array<std::size_t, Capacity> remap{};
    remap.fill(kDetachedParent);
    for (std::size_t destination = 0; destination < keep_count; ++destination) {
      const std::size_t source = order[destination];
      remap[source] = destination;
      MoveSlot(target, source, destination);
    }
    target.count = keep_count;

    if (stage + 1 < NumStages) {
      Stage& children = stages_[stage + 1];
      for (std::size_t slot = 0; slot < children.count; ++slot) {
        std::size_t& parent = children.parents[slot];
        parent = parent == kDetachedParent ? kDetachedParent : remap[parent];
      }
    }
    return SuccessResult();
  }

  [[nodiscard]] auto count(std::size_t stage) const noexcept -> std::size_t { return stages_[stage].count; }

  // The per-slot blocks of `stage`. Only the first count(stage) columns are live.
  [[nodiscard]] auto states(std::size_t stage) const noexcept -> const StateBlock& { return stages_[stage].states; }

  [[nodiscard]] auto source_states(std::size_t stage) const noexcept -> const StateBlock& {
    return stages_[stage].source_states;
  }

  [[nodiscard]] auto drifts(std::size_t stage) const noexcept -> const StateBlock& { return stages_[stage].drifts; }

  [[nodiscard]] auto controls(std::size_t stage) const noexcept -> const ControlBlock& {
    return stages_[stage].controls;
  }

  [[nodiscard]] auto accrued_costs(std::size_t stage) const noexcept -> const ValueBlock& {
    return stages_[stage].accrued_costs;
  }

  [[nodiscard]] auto heuristics(std::size_t stage) const noexcept -> const ValueBlock& {
    return stages_[stage].heuristics;
  }

  // The parent slot of every live particle of `stage`, or kDetachedParent where it was removed.
  [[nodiscard]] auto parents(std::size_t stage) const noexcept -> std::span<const std::size_t> {
    return std::span<const std::size_t>(stages_[stage].parents).first(stages_[stage].count);
  }

 private:
  struct Stage {
    StateBlock states = StateBlock::Zero();
    StateBlock source_states = StateBlock::Zero();
    StateBlock drifts = StateBlock::Zero();
    ControlBlock controls = ControlBlock::Zero();
    ValueBlock accrued_costs = ValueBlock::Zero();
    ValueBlock heuristics = ValueBlock::Zero();
    std::array<std::size_t, Capacity> parents{};
    std::size_t count = 0;
  };

  // Copies slot `source` into slot `destination`; `destination <= source`, so a forward sweep over
  // ascending sources never overwrites a slot it has yet to read.
  static auto MoveSlot(Stage& stage, std::size_t source, std::size_t destination) noexcept -> void {
    if (source == destination) {
      return;
    }
    const auto from = static_cast<Eigen::Index>(source);
    const auto to = static_cast<Eigen::Index>(destination);
    stage.states.col(to) = stage.states.col(from);
    stage.source_states.col(to) = stage.source_states.col(from);
    stage.drifts.col(to) = stage.drifts.col(from);
    stage.controls.col(to) = stage.controls.col(from);
    stage.accrued_costs[to] = stage.accrued_costs[from];
    stage.heuristics[to] = stage.heuristics[from];
    stage.parents[destination] = stage.parents[source];
  }

  std::vector<Stage> stages_;
};

}  // namespace fbsde_traj_opt

#endif  // FBSDE_TRAJ_OPT_FBRRT_FBRRT_PARTICLE_SET_HPP_
