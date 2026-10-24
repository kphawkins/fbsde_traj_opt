// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#ifndef FBSDE_TRAJ_OPT_SAMPLING_L1_EXPLORATORY_CONTROL_HPP_
#define FBSDE_TRAJ_OPT_SAMPLING_L1_EXPLORATORY_CONTROL_HPP_

#include <cmath>
#include <cstddef>
#include <numbers>

#include <Eigen/Core>

#include "fbsde_traj_opt/sampling/counter_based_normal_sampler.hpp"

namespace fbsde_traj_opt {

// The exploratory controls of Hawkins (2021), Section 4.4.1, for a minimum-fuel (L1) problem with
// a box-constrained control: each coordinate drawn uniformly from `{lower, 0, upper}`. That set is
// the thesis's choice of `U_rand` because an L1-optimal policy only ever returns values in it, so
// exploring over it explores exactly the controls the policy could come to choose.
//
// Both helpers draw from a CounterBasedNormalSampler, so that one coordinate-addressed stream
// supplies every kind of randomness a sampling scheme needs and a run repeats exactly.

// A standard normal variate mapped to a uniform on (0, 1) through the normal CDF.
template <typename Scalar = double>
[[nodiscard]] inline auto UniformFromNormal(Scalar normal) noexcept -> Scalar {
  return Scalar{0.5} * std::erfc(-normal / std::numbers::sqrt2_v<Scalar>);
}

// A uniform on (0, 1) drawn at sampler coordinate (`sample`, `stage`, `component`).
template <typename Scalar = double>
[[nodiscard]] inline auto SampleUniform(const CounterBasedNormalSampler& sampler,
                                        std::size_t sample,
                                        std::size_t stage,
                                        std::size_t component) noexcept -> Scalar {
  return UniformFromNormal(static_cast<Scalar>(sampler.Sample(sample, stage, component)));
}

// One exploratory control, its coordinate `j` drawn from sampler component `first_component + j`.
//
// Precondition: every lower bound is at most zero and every upper bound at least zero.
template <int M, typename Scalar>
[[nodiscard]] auto ExploratoryL1Control(const CounterBasedNormalSampler& sampler,
                                        std::size_t sample,
                                        std::size_t stage,
                                        std::size_t first_component,
                                        const Eigen::Matrix<Scalar, M, 1>& lower_bounds,
                                        const Eigen::Matrix<Scalar, M, 1>& upper_bounds) noexcept
    -> Eigen::Matrix<Scalar, M, 1> {
  constexpr auto kControlDim = static_cast<std::size_t>(M);
  Eigen::Matrix<Scalar, M, 1> control;
  for (std::size_t coordinate = 0; coordinate < kControlDim; ++coordinate) {
    const auto index = static_cast<Eigen::Index>(coordinate);
    const auto pick = SampleUniform<Scalar>(sampler, sample, stage, first_component + coordinate);
    if (pick < Scalar{1} / Scalar{3}) {
      control[index] = lower_bounds[index];
    } else if (pick < Scalar{2} / Scalar{3}) {
      control[index] = Scalar{0};
    } else {
      control[index] = upper_bounds[index];
    }
  }
  return control;
}

}  // namespace fbsde_traj_opt

#endif  // FBSDE_TRAJ_OPT_SAMPLING_L1_EXPLORATORY_CONTROL_HPP_
