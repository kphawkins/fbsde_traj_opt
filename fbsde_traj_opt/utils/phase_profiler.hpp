// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#ifndef FBSDE_TRAJ_OPT_UTILS_PHASE_PROFILER_HPP_
#define FBSDE_TRAJ_OPT_UTILS_PHASE_PROFILER_HPP_

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <mutex>
#include <string_view>

namespace fbsde_traj_opt::profiling {

// Where the solvers spend their time: wall-clock seconds and call counts per named phase.
//
// Profiling is off until SetEnabled(true), and a disabled ScopedPhase costs one relaxed atomic load.
// Each thread accumulates into its own totals, which are merged into a process-wide table when the
// thread exits -- ParallelFor's workers exit before it returns -- or when that thread takes a
// Snapshot(). A phase timed on a worker thread therefore adds thread-seconds rather than wall time:
// the per-sample phases inside a parallel loop can sum to more than the loop's own wall time.
//
// The table is process-wide, so two solvers profiled concurrently in one process share it.
enum class Phase : std::size_t {
  // DtFbsdeIterativeSolver::Iterate(): the forward pass, each sample's step, and its policy query.
  kForwardPass,
  kForwardSample,
  kForwardPolicy,
  // The backward pass, the targets of one stage (and half), each sample's target, and within it the
  // chooser's argmin, the scorer's Q-value, and the slope target; then each fit and each averaging,
  // and a time-shared fit's closing joint solve (DtFbsdeIterativeOptions::time_sharing).
  kBackwardPass,
  kBackwardTargets,
  kBackwardTargetSample,
  kBackwardArgmin,
  kBackwardQValue,
  kBackwardSlopeTarget,
  kBackwardFit,
  kBackwardAverage,
  kBackwardTimeShare,
  // The line search: the whole of it, each blend, each screening evaluation, each full evaluation.
  kLineSearch,
  kLineSearchBlend,
  kLineSearchScreen,
  kLineSearchFull,
  // EvaluatePolicyByRollouts(): each rollout, and each step's policy query and dynamics.
  kRolloutSample,
  kRolloutPolicy,
  kRolloutDynamics,
  kCount,
};

inline constexpr std::size_t kPhaseCount = static_cast<std::size_t>(Phase::kCount);

inline constexpr std::array<std::string_view, kPhaseCount> kPhaseNames{
    "forward_pass",
    "forward.sample",
    "forward.policy",
    "backward_pass",
    "backward.targets",
    "backward.target_sample",
    "backward.argmin",
    "backward.qvalue",
    "backward.slope_target",
    "backward.fit",
    "backward.average",
    "backward.time_share",
    "line_search",
    "line_search.blend",
    "line_search.screen_eval",
    "line_search.full_eval",
    "rollout.sample",
    "rollout.policy",
    "rollout.dynamics",
};

[[nodiscard]] constexpr auto PhaseName(Phase phase) noexcept -> std::string_view {
  return kPhaseNames[static_cast<std::size_t>(phase)];
}

// Seconds and calls per phase.
struct PhaseTotals {
  std::array<double, kPhaseCount> seconds{};
  std::array<std::size_t, kPhaseCount> calls{};

  [[nodiscard]] auto Seconds(Phase phase) const noexcept -> double { return seconds[static_cast<std::size_t>(phase)]; }
  [[nodiscard]] auto Calls(Phase phase) const noexcept -> std::size_t { return calls[static_cast<std::size_t>(phase)]; }

  auto Add(const PhaseTotals& other) noexcept -> void {
    for (std::size_t index = 0; index < kPhaseCount; ++index) {
      seconds[index] += other.seconds[index];
      calls[index] += other.calls[index];
    }
  }
};

namespace internal {

inline auto EnabledFlag() noexcept -> std::atomic<bool>& {
  static std::atomic<bool> enabled{false};
  return enabled;
}

inline auto GlobalMutex() noexcept -> std::mutex& {
  static std::mutex mutex;
  return mutex;
}

inline auto GlobalTotals() noexcept -> PhaseTotals& {
  static PhaseTotals totals;
  return totals;
}

// One thread's totals, merged into the global table when the thread exits.
class ThreadTotals {
 public:
  ThreadTotals() noexcept = default;
  ThreadTotals(const ThreadTotals&) = delete;
  ThreadTotals(ThreadTotals&&) = delete;
  auto operator=(const ThreadTotals&) -> ThreadTotals& = delete;
  auto operator=(ThreadTotals&&) -> ThreadTotals& = delete;
  ~ThreadTotals() noexcept { Flush(); }

  auto Flush() noexcept -> void {
    const std::scoped_lock lock(GlobalMutex());
    GlobalTotals().Add(totals);
    totals = PhaseTotals{};
  }

  PhaseTotals totals;
};

inline auto Local() noexcept -> ThreadTotals& {
  thread_local ThreadTotals local;
  return local;
}

}  // namespace internal

// Turns the timers on or off for every thread.
inline auto SetEnabled(bool enabled) noexcept -> void {
  internal::EnabledFlag().store(enabled, std::memory_order_relaxed);
}

[[nodiscard]] inline auto Enabled() noexcept -> bool {
  return internal::EnabledFlag().load(std::memory_order_relaxed);
}

// The totals so far: every exited thread's, and the calling thread's. Threads still running keep
// theirs until they exit.
[[nodiscard]] inline auto Snapshot() noexcept -> PhaseTotals {
  internal::Local().Flush();
  const std::scoped_lock lock(internal::GlobalMutex());
  return internal::GlobalTotals();
}

// Clears the global table and the calling thread's totals.
inline auto Reset() noexcept -> void {
  internal::Local().totals = PhaseTotals{};
  const std::scoped_lock lock(internal::GlobalMutex());
  internal::GlobalTotals() = PhaseTotals{};
}

// Times its own lifetime as one call of `phase`, if profiling was enabled when it was constructed.
class ScopedPhase {
 public:
  explicit ScopedPhase(Phase phase) noexcept : phase_(phase), active_(Enabled()) {
    if (active_) {
      start_ = std::chrono::steady_clock::now();
    }
  }
  ScopedPhase(const ScopedPhase&) = delete;
  ScopedPhase(ScopedPhase&&) = delete;
  auto operator=(const ScopedPhase&) -> ScopedPhase& = delete;
  auto operator=(ScopedPhase&&) -> ScopedPhase& = delete;

  ~ScopedPhase() noexcept {
    if (!active_) {
      return;
    }
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
    PhaseTotals& totals = internal::Local().totals;
    totals.seconds[static_cast<std::size_t>(phase_)] += elapsed;
    ++totals.calls[static_cast<std::size_t>(phase_)];
  }

 private:
  Phase phase_;
  bool active_;
  std::chrono::steady_clock::time_point start_{};
};

}  // namespace fbsde_traj_opt::profiling

#endif  // FBSDE_TRAJ_OPT_UTILS_PHASE_PROFILER_HPP_
