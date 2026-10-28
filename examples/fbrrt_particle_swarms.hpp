// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#ifndef FBSDE_TRAJ_OPT_EXAMPLES_FBRRT_PARTICLE_SWARMS_HPP_
#define FBSDE_TRAJ_OPT_EXAMPLES_FBRRT_PARTICLE_SWARMS_HPP_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "fbsde_traj_opt/utils/result.hpp"
#include "fbsde_traj_opt/viz/plotly_figure.hpp"
#include "examples/fbrrt_comparison_experiment.hpp"

namespace fbsde_traj_opt::examples {

// Pictures of the FBRRT particle swarm on the two-dimensional benchmarks, the L1 double integrator
// and the L1 pendulum, run with the comparison's configuration (FbrrtComparisonOptions), so that
// what they show is the method the comparison measures.
//
// Every picture is of the particle set just before erosion: everything the expansion drew, before
// the heuristic thins it.
//
//   * The swarm over iterations: every stage's particles at once, in the state plane, colored by
//     time, one frame per iteration -- how the cloud the method samples changes as it learns --
//     with the current policy's noise-free rollout drawn through it.
//   * The swarm over time, at the first iteration and at the last: one frame per stage, the
//     particles at that stage colored by the heuristic erosion ranks them by,
//     `rho = L_{0:i-1} + V~_i(x)`, recomputed from the value functions erosion used -- the
//     thesis's picture of the forward sampling tree. Low `rho` survives.
//   * The policy's cost over the iterations, for reading the swarms against.
//
// The over-iterations picture shows a subsample of each stage, so that the page stays a few
// megabytes; the over-time pictures show every particle.

struct FbrrtParticleSwarmOptions {
  FbrrtComparisonOptions comparison{};

  // The seed of the one run each picture comes from.
  std::uint64_t seed = 20260928;

  // How many particles of each stage the over-iterations picture shows.
  std::size_t particles_per_stage_shown = 64;
};

// The pictures for one benchmark.
struct FbrrtParticleSwarmReport {
  std::string name{};
  std::string description{};
  std::vector<viz::PlotlyFigure> figures{};
};

// Runs FBRRT on each two-dimensional benchmark and draws its swarm.
//
// Fails if a model, the solver, or a figure is rejected, or if the solver fails mid-run.
[[nodiscard]] auto DrawFbrrtParticleSwarms(const FbrrtParticleSwarmOptions& options) noexcept
    -> Result<std::vector<FbrrtParticleSwarmReport>>;

}  // namespace fbsde_traj_opt::examples

#endif  // FBSDE_TRAJ_OPT_EXAMPLES_FBRRT_PARTICLE_SWARMS_HPP_
