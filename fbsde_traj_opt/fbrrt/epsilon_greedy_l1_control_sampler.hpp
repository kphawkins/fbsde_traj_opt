// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#ifndef FBSDE_TRAJ_OPT_FBRRT_EPSILON_GREEDY_L1_CONTROL_SAMPLER_HPP_
#define FBSDE_TRAJ_OPT_FBRRT_EPSILON_GREEDY_L1_CONTROL_SAMPLER_HPP_

#include <cstddef>
#include <utility>

#include <Eigen/Core>

#include "fbsde_traj_opt/sampling/counter_based_normal_sampler.hpp"
#include "fbsde_traj_opt/sampling/l1_exploratory_control.hpp"
#include "fbsde_traj_opt/utils/result.hpp"

namespace fbsde_traj_opt {

// Concept for the rule that chooses the control a new edge is drawn under: given the current
// policy, the parent's stage and state, the probability of exploiting the policy, and the new
// particle's sampler coordinate, return a control. A sampler owns every component of its
// coordinate, as an FbrrtExpansionSelector does.
template <typename T, typename PolicyT, typename State, typename Control, typename Scalar>
concept FbrrtControlSampler = requires(const T& control_sampler,
                                       const PolicyT& policy,
                                       std::size_t stage,
                                       const State& state,
                                       Scalar exploit_probability,
                                       const CounterBasedNormalSampler& sampler,
                                       std::size_t sample) {
  { control_sampler(policy, stage, state, exploit_probability, sampler, sample) } -> std::same_as<Control>;
};

// The drift choice of Hawkins (2021), Algorithm 4.2, lines 9-13, for an L1 problem: with
// probability `eps_opt` the policy's control, `K_i = F_i(x, mu_i(x))`, and otherwise a control drawn
// uniformly from `{lower, 0, upper}^M`, the set an L1-optimal policy's values lie in.
//
// Sampler components of the new particle's coordinate: 0 decides between the two, and 1..M pick
// the exploratory control's coordinates.
template <int N, int M, typename Scalar = double>
class EpsilonGreedyL1ControlSampler {
 public:
  using State = Eigen::Matrix<Scalar, N, 1>;
  using Control = Eigen::Matrix<Scalar, M, 1>;

  // Fails unless the bounds are finite and every one contains zero.
  static auto Make(const Control& lower_bounds, const Control& upper_bounds) noexcept
      -> Result<EpsilonGreedyL1ControlSampler> {
    RESULT_ASSERT(lower_bounds.allFinite() && upper_bounds.allFinite(),
                  "EpsilonGreedyL1ControlSampler::Make: the control bounds must be finite.");
    RESULT_ASSERT((lower_bounds.array() <= Scalar{0}).all() && (upper_bounds.array() >= Scalar{0}).all(),
                  "EpsilonGreedyL1ControlSampler::Make: every control's bounds must contain zero.");
    return SuccessResult(EpsilonGreedyL1ControlSampler(lower_bounds, upper_bounds));
  }

  template <typename PolicyT>
  auto operator()(const PolicyT& policy,
                  std::size_t stage,
                  const State& state,
                  Scalar exploit_probability,
                  const CounterBasedNormalSampler& sampler,
                  std::size_t sample) const noexcept -> Control {
    if (SampleUniform<Scalar>(sampler, sample, stage, kDecisionComponent) < exploit_probability) {
      return policy(stage, state);
    }
    return ExploratoryL1Control<M>(sampler, sample, stage, kControlComponent, lower_bounds_, upper_bounds_);
  }

  [[nodiscard]] auto lower_bounds() const noexcept -> const Control& { return lower_bounds_; }

  [[nodiscard]] auto upper_bounds() const noexcept -> const Control& { return upper_bounds_; }

 private:
  static constexpr std::size_t kDecisionComponent = 0;
  static constexpr std::size_t kControlComponent = 1;

  EpsilonGreedyL1ControlSampler(Control lower_bounds, Control upper_bounds) noexcept
      : lower_bounds_(std::move(lower_bounds)), upper_bounds_(std::move(upper_bounds)) {}

  Control lower_bounds_;
  Control upper_bounds_;
};

}  // namespace fbsde_traj_opt

#endif  // FBSDE_TRAJ_OPT_FBRRT_EPSILON_GREEDY_L1_CONTROL_SAMPLER_HPP_
