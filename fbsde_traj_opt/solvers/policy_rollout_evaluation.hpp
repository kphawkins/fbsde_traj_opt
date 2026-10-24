// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#ifndef FBSDE_TRAJ_OPT_SOLVERS_POLICY_ROLLOUT_EVALUATION_HPP_
#define FBSDE_TRAJ_OPT_SOLVERS_POLICY_ROLLOUT_EVALUATION_HPP_

#include <cstddef>
#include <cstdint>
#include <vector>

#include <Eigen/Core>

#include "fbsde_traj_opt/sampling/counter_based_normal_sampler.hpp"
#include "fbsde_traj_opt/utils/parallel_for.hpp"
#include "fbsde_traj_opt/utils/phase_profiler.hpp"

namespace fbsde_traj_opt {

// Scoring a policy by simulation: the mean cost of `num_rollouts` closed-loop rollouts from the
// initial distribution, with no exploration.
//
// Every iterative solver needs this for the same reason -- a value function's own estimate of its
// cost is exactly what is in question -- and solvers that are to be compared must score their
// policies on the same noise, or a difference in cost could be a difference in draws. Hence one
// function, and one stream constant that every solver mixes into its seed for its evaluations,
// kept apart from the streams its forward passes draw from so that evaluation noise never changes
// and never leaks into sampling.
inline constexpr std::uint64_t kPolicyEvaluationStream = 0xD1B54A32D192ED03ULL;

// The mean cost of a set of rollouts, and its two parts.
template <typename Scalar = double>
struct PolicyRolloutCosts {
  Scalar total = Scalar{0};
  Scalar running = Scalar{0};
  Scalar terminal = Scalar{0};
};

// Rolls `policy` out `num_rollouts` times over `num_control_stages` stages and returns the mean
// costs. Rollout `r` starts at `initial_mean + initial_covariance_factor * w` with `w` drawn at
// sampler coordinate (r, 0), and draws its stage-`k` noise at (r, k + 1).
template <typename ForwardModelT,
          typename PolicyT,
          typename RunningCostT,
          typename TerminalCostT,
          typename Scalar,
          int N>
[[nodiscard]] auto EvaluatePolicyByRollouts(const ForwardModelT& forward_model,
                                            const PolicyT& policy,
                                            const RunningCostT& running_cost,
                                            const TerminalCostT& terminal_cost,
                                            const Eigen::Matrix<Scalar, N, 1>& initial_mean,
                                            const Eigen::Matrix<Scalar, N, N>& initial_covariance_factor,
                                            std::size_t num_rollouts,
                                            std::size_t num_control_stages,
                                            std::uint64_t seed) noexcept -> PolicyRolloutCosts<Scalar> {
  using State = Eigen::Matrix<Scalar, N, 1>;
  const CounterBasedNormalSampler sampler(seed);

  std::vector<Scalar> running(num_rollouts, Scalar{0});
  std::vector<Scalar> terminal(num_rollouts, Scalar{0});
  ParallelFor(num_rollouts, [&](std::size_t rollout) noexcept {
    const profiling::ScopedPhase rollout_phase(profiling::Phase::kRolloutSample);
    State noise;
    sampler.FillNoise(rollout, 0, noise);
    State state = initial_mean + (initial_covariance_factor * noise);
    for (std::size_t stage = 0; stage < num_control_stages; ++stage) {
      const auto control = [&] {
        const profiling::ScopedPhase policy_phase(profiling::Phase::kRolloutPolicy);
        return policy(stage, state);
      }();
      running[rollout] += running_cost(stage, state, control);
      sampler.FillNoise(rollout, stage + 1, noise);
      const profiling::ScopedPhase dynamics_phase(profiling::Phase::kRolloutDynamics);
      state = forward_model(stage, state, control, noise);
    }
    terminal[rollout] = terminal_cost(state);
  });

  PolicyRolloutCosts<Scalar> costs;
  for (std::size_t rollout = 0; rollout < num_rollouts; ++rollout) {
    costs.running += running[rollout];
    costs.terminal += terminal[rollout];
  }
  costs.running /= static_cast<Scalar>(num_rollouts);
  costs.terminal /= static_cast<Scalar>(num_rollouts);
  costs.total = costs.running + costs.terminal;
  return costs;
}

}  // namespace fbsde_traj_opt

#endif  // FBSDE_TRAJ_OPT_SOLVERS_POLICY_ROLLOUT_EVALUATION_HPP_
