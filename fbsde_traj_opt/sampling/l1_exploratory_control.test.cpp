// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#include "fbsde_traj_opt/sampling/l1_exploratory_control.hpp"

#include <array>
#include <cstddef>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "fbsde_traj_opt/sampling/counter_based_normal_sampler.hpp"

namespace fbsde_traj_opt {
namespace {

TEST(UniformFromNormalTest, MapsTheNormalCdfOntoTheUnitInterval) {
  EXPECT_DOUBLE_EQ(UniformFromNormal(0.0), 0.5);
  EXPECT_NEAR(UniformFromNormal(1.0), 0.841344746, 1e-9);
  EXPECT_NEAR(UniformFromNormal(-1.0), 1.0 - 0.841344746, 1e-9);
}

TEST(ExploratoryL1ControlTest, DrawsEachCoordinateUniformlyFromTheBoundsAndZero) {
  const CounterBasedNormalSampler sampler(11);
  const Eigen::Vector2d lower(-1.0, -0.5);
  const Eigen::Vector2d upper(2.0, 0.5);

  constexpr std::size_t kDraws = 30000;
  std::array<std::array<std::size_t, 3>, 2> counts{};
  for (std::size_t draw = 0; draw < kDraws; ++draw) {
    const Eigen::Vector2d control = ExploratoryL1Control<2>(sampler, draw, 3, 5, lower, upper);
    for (std::size_t coordinate = 0; coordinate < 2; ++coordinate) {
      const auto index = static_cast<Eigen::Index>(coordinate);
      if (control[index] == lower[index]) {
        ++counts[coordinate][0];
      } else if (control[index] == 0.0) {
        ++counts[coordinate][1];
      } else {
        ASSERT_EQ(control[index], upper[index]);
        ++counts[coordinate][2];
      }
    }
  }
  for (const auto& coordinate_counts : counts) {
    for (const std::size_t count : coordinate_counts) {
      EXPECT_NEAR(static_cast<double>(count) / static_cast<double>(kDraws), 1.0 / 3.0, 0.015);
    }
  }
}

TEST(ExploratoryL1ControlTest, IsAPureFunctionOfItsSamplerCoordinate) {
  const CounterBasedNormalSampler sampler(11);
  const Eigen::Vector2d lower(-1.0, -1.0);
  const Eigen::Vector2d upper(1.0, 1.0);
  for (std::size_t draw = 0; draw < 64; ++draw) {
    EXPECT_EQ(ExploratoryL1Control<2>(sampler, draw, 2, 4, lower, upper),
              ExploratoryL1Control<2>(sampler, draw, 2, 4, lower, upper));
  }
}

}  // namespace
}  // namespace fbsde_traj_opt
