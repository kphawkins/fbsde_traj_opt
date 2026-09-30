// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#ifndef FBSDE_TRAJ_OPT_FBRRT_FORWARD_EXPANSION_HPP_
#define FBSDE_TRAJ_OPT_FBRRT_FORWARD_EXPANSION_HPP_

#include <cstddef>
#include <cstdint>

#include <Eigen/Core>

#include "fbsde_traj_opt/sampling/counter_based_normal_sampler.hpp"
#include "fbsde_traj_opt/utils/parallel_for.hpp"
#include "fbsde_traj_opt/utils/result.hpp"

namespace fbsde_traj_opt {

// The streams the forward pass's two random choices draw from, mixed into the caller's seed so that
// swapping one component for another never changes the other's draws.
inline constexpr std::uint64_t kFbrrtSelectorStream = 0x3C6EF372FE94F82BULL;
inline constexpr std::uint64_t kFbrrtControlStream = 0xA54FF53A5F1D36F1ULL;
inline constexpr std::uint64_t kFbrrtNoiseStream = 0x510E527FADE682D1ULL;

// The forward pass of the FBRRT method: refills every stage of an FbrrtParticleSet to capacity
// with new edges, stage by stage from the first.
//
// A new particle at stage `i + 1` is drawn as in Hawkins (2021), Algorithm 4.2. Its parent is
// chosen from the particles at stage `i` by `selector`, its control by `control_sampler`, and its
// drift is `K_i = f(i, x) + B(i, x) u`. By default it is placed at
//
//   x_{i+1} = x_i + K_i,
//
// with no noise term: the thesis added `Sigma_i w_i` with `w_i ~ N(0, I)`, and here `w_i = 0`.
// That noise does little for the search -- the exploration that matters comes from sampling the
// control -- and it is not needed for correctness either: the Taylor noiseless estimator of the
// backward pass reads only `x_i` and `K_i`, and charges for the noise analytically through its
// trace term. With `sample_noise`, the thesis's `x_{i+1} = x_i + K_i + Sigma_i w_i` is used
// instead, for comparison; the edges' targets are unchanged, and only where the children land --
// and so where the next stage's edges start -- differs. What the particle records is exactly what that estimator needs:
// the edge's source state and drift, the control, and the running cost accrued along the path, `L_{0:i}`, for the
// path-integral heuristic that erosion ranks particles by.
//
// Stage `i + 1` depends only on stage `i`, which is already complete by the time it is drawn, so
// the new particles of a stage are drawn in parallel. Particles already at a stage -- the
// survivors of the last erosion -- are left untouched; only the free slots are filled.
//
// Randomness: the selector, the control sampler, and the noise each draw from a
// CounterBasedNormalSampler of their own stream of `seed`, at coordinate (slot, stage + 1). The caller varies `seed`
// between iterations.
//
// Fails, leaving the stages before the failing one filled, if a stage has no particles to branch
// from.
template <typename ParticleSetT,
          typename ForwardModelT,
          typename RunningCostT,
          typename SelectorT,
          typename ControlSamplerT,
          typename PolicyT,
          typename Scalar>
auto ExpandParticles(ParticleSetT& particles,
                     const ForwardModelT& forward_model,
                     const RunningCostT& running_cost,
                     const SelectorT& selector,
                     const ControlSamplerT& control_sampler,
                     const PolicyT& policy,
                     Scalar rrt_probability,
                     Scalar exploit_probability,
                     std::uint64_t seed,
                     bool sample_noise = false) noexcept -> Result<> {
  using State = typename ParticleSetT::State;
  using Control = typename ParticleSetT::Control;
  using Particle = typename ParticleSetT::Particle;
  constexpr auto kCapacity = static_cast<std::size_t>(ParticleSetT::kCapacity);

  const CounterBasedNormalSampler selector_sampler(seed ^ kFbrrtSelectorStream);
  const CounterBasedNormalSampler control_sampler_stream(seed ^ kFbrrtControlStream);
  const CounterBasedNormalSampler noise_sampler(seed ^ kFbrrtNoiseStream);

  for (std::size_t stage = 0; stage + 1 < ParticleSetT::kNumStages; ++stage) {
    const std::size_t parents = particles.count(stage);
    RESULT_ASSERT(parents >= 1, "ExpandParticles: a stage has no particles to branch from.");
    const std::size_t existing = particles.count(stage + 1);

    ParallelFor(kCapacity - existing, [&](std::size_t index) noexcept {
      const std::size_t slot = existing + index;
      const std::size_t parent =
          selector(particles.states(stage), parents, rrt_probability, selector_sampler, slot, stage + 1);
      const auto column = static_cast<Eigen::Index>(parent);
      const State state = particles.states(stage).col(column);
      const Control control = control_sampler(policy, stage, state, exploit_probability, control_sampler_stream, slot);

      Particle particle;
      particle.source_state = state;
      particle.control = control;
      particle.drift =
          forward_model.state_drift()(stage, state) + (forward_model.control_drift_mat()(stage, state) * control);
      State noise = State::Zero();
      if (sample_noise) {
        noise_sampler.FillNoise(slot, stage + 1, noise);
      }
      particle.state = forward_model(stage, state, control, noise);
      particle.accrued_cost = particles.accrued_costs(stage)[column] + running_cost(stage, state, control);
      particle.parent = parent;
      particles.WriteParticle(stage + 1, slot, particle);
    });
    RESULT_RETURN_IF_ERROR(particles.SetCount(stage + 1, kCapacity));
  }
  return SuccessResult();
}

}  // namespace fbsde_traj_opt

#endif  // FBSDE_TRAJ_OPT_FBRRT_FORWARD_EXPANSION_HPP_
