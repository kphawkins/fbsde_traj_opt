// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#include "fbsde_traj_opt/utils/parallel_for.hpp"

#include <atomic>
#include <cstddef>
#include <vector>

#include <gtest/gtest.h>

namespace fbsde_traj_opt {
namespace {

TEST(ParallelForTest, VisitsEveryIndexExactlyOnce) {
  for (const std::size_t count : {0UL, 1UL, 31UL, 32UL, 1000UL, 4097UL}) {
    std::vector<std::atomic<int>> visits(count);
    ParallelFor(count, [&visits](std::size_t index) { visits[index].fetch_add(1); });
    for (std::size_t index = 0; index < count; ++index) {
      EXPECT_EQ(visits[index].load(), 1) << "count " << count << " index " << index;
    }
  }
}

TEST(ParallelForTest, ResultsMatchASerialLoop) {
  constexpr std::size_t kCount = 5000;
  std::vector<double> parallel(kCount);
  ParallelFor(kCount, [&parallel](std::size_t index) { parallel[index] = static_cast<double>(index * index) * 0.5; });
  for (std::size_t index = 0; index < kCount; ++index) {
    EXPECT_DOUBLE_EQ(parallel[index], static_cast<double>(index * index) * 0.5);
  }
}

}  // namespace
}  // namespace fbsde_traj_opt
