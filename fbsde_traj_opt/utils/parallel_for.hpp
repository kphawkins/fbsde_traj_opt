// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#ifndef FBSDE_TRAJ_OPT_UTILS_PARALLEL_FOR_HPP_
#define FBSDE_TRAJ_OPT_UTILS_PARALLEL_FOR_HPP_

#include <algorithm>
#include <cstddef>
#include <thread>
#include <vector>

namespace fbsde_traj_opt {

// Calls `body(index)` for every index in [0, count), split into contiguous blocks across up to
// `std::thread::hardware_concurrency()` threads, and returns once every call has.
//
// For loops whose iterations are independent -- each writes only its own outputs and reads only
// shared state that nothing modifies during the loop -- which is what a batch of sampled
// trajectories, or of per-sample regression targets, is. The order of the calls is unspecified;
// the result must not depend on it. Contiguous blocks rather than interleaved indices so that
// each thread writes its own stretch of a column-per-sample matrix.
//
// A thread that cannot be started terminates the program, as an allocation failure does
// elsewhere in this `noexcept` codebase. Small loops run on the calling thread, since starting
// threads costs more than they would save.
template <typename Body>
auto ParallelFor(std::size_t count, const Body& body) noexcept -> void {
  constexpr std::size_t kMinimumPerThread = 32;
  const std::size_t hardware = std::max<std::size_t>(std::thread::hardware_concurrency(), 1);
  const std::size_t threads = std::min(hardware, std::max<std::size_t>(count / kMinimumPerThread, 1));
  if (threads == 1) {
    for (std::size_t index = 0; index < count; ++index) {
      body(index);
    }
    return;
  }

  const std::size_t block = (count + threads - 1) / threads;
  std::vector<std::jthread> workers;
  workers.reserve(threads - 1);
  for (std::size_t thread = 1; thread < threads; ++thread) {
    const std::size_t begin = std::min(thread * block, count);
    const std::size_t end = std::min(begin + block, count);
    workers.emplace_back([&body, begin, end] {
      for (std::size_t index = begin; index < end; ++index) {
        body(index);
      }
    });
  }
  for (std::size_t index = 0; index < std::min(block, count); ++index) {
    body(index);
  }
  // Every std::jthread joins as `workers` goes out of scope.
}

}  // namespace fbsde_traj_opt

#endif  // FBSDE_TRAJ_OPT_UTILS_PARALLEL_FOR_HPP_
