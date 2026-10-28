// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#ifndef FBSDE_TRAJ_OPT_NEAREST_NEIGHBOR_DT_FBSDE_SOLVER_HPP_
#define FBSDE_TRAJ_OPT_NEAREST_NEIGHBOR_DT_FBSDE_SOLVER_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include "fbsde_traj_opt/costs/l1_control_sde_running_cost_term.hpp"
#include "fbsde_traj_opt/sampling/counter_based_normal_sampler.hpp"
#include "fbsde_traj_opt/sampling/l1_exploratory_control.hpp"
#include "fbsde_traj_opt/sampling/normal_distribution_concepts.hpp"
#include "fbsde_traj_opt/sde/sde_term_concepts.hpp"
#include "fbsde_traj_opt/utils/parallel_for.hpp"
#include "fbsde_traj_opt/utils/result.hpp"

namespace fbsde_traj_opt {

// A nonparametric DT-FBSDE iteration for a minimum-fuel (L1) problem: no value function model is
// fitted. Every sample's backward variable Y_s^i is its own parameter, and the value function the
// next iteration's policy reads at stage s is the nearest-neighbor interpolant of the last batch,
//
//   V_s(x) = Y_s^{i*},     i* = argmin_i || D (x - X_s^i) ||,
//
// with D the identity (Euclidean distance) unless a distance scale is given. The policy is the
// Q-value policy of the thesis's Section 4.4 on that table,
//
//   mu_s(x) = argmin_{u in {lower, 0, upper}^M} c || u ||_1 + V_{s+1}(x + f(s, x) + B(s, x) u),
//
// whose landing points are wherever the step goes, explored or not; the nearest sample answers for
// them. A nearest-neighbor table is piecewise constant, so it has no slope or curvature to put a
// minimizer between the vertices and zero, and the L1 problem's bang-off-bang controls are exactly
// that set anyway. With several controls the set is searched by coordinate sweeps.
//
// One iteration is a forward pass under the current policy and a backward pass that only computes
// Y: there is no regression, no blending, and no line search. Key to it is that the forward pass
// draws the same noise W_s^i -- the same initial states and the same increments -- every iteration
// (common random numbers), so the batch is a deterministic function of the policy, the change from
// one batch to the next is a change of policy and not of noise, and the batch's mean Y_0 is the
// policy's exact cost on that fixed noise. The price is that the policy is tuned to those noise
// draws; whether it is overfitted shows up when it is scored on fresh noise, which the solver
// itself never does.
//
// Y is estimated one of two ways (NearestNeighborYEstimate):
//
//   * kPathwise: Y_s^i = l(X_s^i, u_s^i) + Y_{s+1}^i, the cost-to-go the path actually incurred
//     under the fixed noise -- the BSDE's Y with the martingale term left in.
//   * kGreedyBackup: Y_s^i = min_u l(X_s^i, u) + V_{s+1}(X_s^i + F(X_s^i, u) + Sigma W_s^i), the
//     nearest-neighbor table of this same batch read where the path would have gone under each
//     vertex control with its own noise. The path's own control lands exactly on X_{s+1}^i, so
//     this is never above the pathwise estimate, and it lets an explored control's path be scored
//     as if the better control had been taken.
//
// Since the batch is a deterministic function of the policy, the iteration is a deterministic map
// on policies: once a batch takes every control the batch before it took, every later batch is the
// same, and the solver reports a fixed point. It keeps the policy whose batch cost least and hands
// that out as the current policy.
//
// The first iteration has no table to be greedy on and explores with probability
// `initial_exploration_probability` (one by default: every control drawn uniformly from the vertex
// set, as the thesis's first iteration does); later ones with `exploration_probability`, and only on
// the paths after the first `1 - exploring_path_fraction` of the batch. The exploration draws are
// fixed across iterations too. The batch cost reported is that of the paths that never explore when
// there are any, so it is the greedy policy's cost on the fixed noise.
//
// `N` and `M` are the state and control dimensions and `ForwardModelT` a ComposedForwardSdeModel
// of those dimensions.

// How the backward pass estimates each sample's Y; see the class comment.
enum class NearestNeighborYEstimate : std::uint8_t { kPathwise, kGreedyBackup };

template <int N, typename Scalar = double>
struct NearestNeighborDtFbsdeOptions {
  // The number of paths in the batch.
  std::size_t num_samples = 256;

  // The exploration probability of the first iteration, and of every later one.
  Scalar initial_exploration_probability = Scalar{1};
  Scalar exploration_probability = Scalar{0};

  // The fraction of paths that may explore after the first iteration; the rest always follow the
  // policy, and the batch cost is theirs. They are the last paths of the batch.
  Scalar exploring_path_fraction = Scalar{1};

  NearestNeighborYEstimate y_estimate = NearestNeighborYEstimate::kPathwise;

  // Read the terminal cost itself at the last stage rather than its nearest sample's value. The
  // terminal value function is given, not estimated, so there is nothing to look up.
  bool exact_terminal_value = true;

  // Average the Y of this many nearest samples rather than taking the nearest one alone.
  std::size_t num_neighbors = 1;

  // The per-coordinate scale `D` divides the state difference by; ones give Euclidean distance.
  Eigen::Matrix<Scalar, N, 1> distance_scale = Eigen::Matrix<Scalar, N, 1>::Ones();

  // An ablation: draw fresh noise every iteration instead of the same noise, so the batch is no
  // longer a deterministic function of the policy.
  bool fresh_noise_each_iteration = false;
};

// What one iteration did.
template <typename Scalar = double>
struct NearestNeighborDtFbsdeIterationReport {
  std::size_t iteration = 0;
  // The mean cost of this iteration's batch: of its non-exploring paths when it has any.
  Scalar batch_cost = Scalar{0};
  // The least batch cost of any greedy iteration so far: the cost of the policy CurrentPolicy() returns.
  Scalar best_batch_cost = Scalar{0};
  // The fraction of (path, stage) controls that differ from the previous batch's.
  Scalar changed_control_fraction = Scalar{1};
  // Whether this batch repeats the previous one, so every later iteration would too.
  bool fixed_point = false;
};

// The nearest-neighbor tables of one batch: every stage's sample states and Y.
template <int N, typename Scalar = double>
struct NearestNeighborValueTable {
  using StateColumns = Eigen::Matrix<Scalar, N, Eigen::Dynamic>;
  using ValueRow = Eigen::Matrix<Scalar, 1, Eigen::Dynamic>;

  std::vector<StateColumns> states{};
  std::vector<ValueRow> values{};

  // The nearest-neighbor value at `stage`, averaged over `num_neighbors` samples, with distances
  // measured after dividing by `inverse_scale`'s reciprocal, i.e. multiplying by `inverse_scale`.
  [[nodiscard]] auto Value(std::size_t stage,
                           const Eigen::Matrix<Scalar, N, 1>& state,
                           const Eigen::Matrix<Scalar, N, 1>& inverse_scale,
                           std::size_t num_neighbors) const noexcept -> Scalar {
    const StateColumns& columns = states[stage];
    const ValueRow& row = values[stage];
    const auto count = static_cast<std::size_t>(columns.cols());
    if (num_neighbors <= 1) {
      Scalar best = std::numeric_limits<Scalar>::infinity();
      std::size_t best_index = 0;
      for (std::size_t sample = 0; sample < count; ++sample) {
        const Scalar distance =
            (columns.col(static_cast<Eigen::Index>(sample)) - state).cwiseProduct(inverse_scale).squaredNorm();
        if (distance < best) {
          best = distance;
          best_index = sample;
        }
      }
      return row[static_cast<Eigen::Index>(best_index)];
    }
    const std::size_t neighbors = std::min(num_neighbors, count);
    std::vector<std::pair<Scalar, std::size_t>> distances(count);
    for (std::size_t sample = 0; sample < count; ++sample) {
      distances[sample] = {
          (columns.col(static_cast<Eigen::Index>(sample)) - state).cwiseProduct(inverse_scale).squaredNorm(), sample};
    }
    std::ranges::nth_element(distances, distances.begin() + static_cast<std::ptrdiff_t>(neighbors - 1));
    Scalar sum = Scalar{0};
    for (std::size_t index = 0; index < neighbors; ++index) {
      sum += row[static_cast<Eigen::Index>(distances[index].second)];
    }
    return sum / static_cast<Scalar>(neighbors);
  }
};

// The Q-value policy on a nearest-neighbor table; see NearestNeighborDtFbsdeSolver. Copies share the
// table. Without a table it applies zero control.
template <int N, int M, typename ForwardModelT, typename TerminalCostT, typename Scalar = double>
class NearestNeighborL1Policy {
 public:
  using State = Eigen::Matrix<Scalar, N, 1>;
  using Control = Eigen::Matrix<Scalar, M, 1>;
  using Table = NearestNeighborValueTable<N, Scalar>;

  static constexpr std::size_t kCoordinateSweeps = 2;
  static constexpr auto kControlDim = static_cast<std::size_t>(M);

  NearestNeighborL1Policy(ForwardModelT forward_model,
                          L1ControlSdeRunningCostTerm<N, M, Scalar> running_cost,
                          TerminalCostT terminal_cost,
                          Control lower_bounds,
                          Control upper_bounds,
                          State inverse_scale,
                          std::size_t num_neighbors,
                          bool exact_terminal_value,
                          std::shared_ptr<const Table> table) noexcept
      : forward_model_(std::move(forward_model)),
        running_cost_(std::move(running_cost)),
        terminal_cost_(std::move(terminal_cost)),
        lower_bounds_(std::move(lower_bounds)),
        upper_bounds_(std::move(upper_bounds)),
        inverse_scale_(std::move(inverse_scale)),
        num_neighbors_(num_neighbors),
        exact_terminal_value_(exact_terminal_value),
        table_(std::move(table)) {}

  // The value at `stage` of a state, read from the table or, at the terminal stage, optionally the
  // terminal cost.
  //
  // Precondition: the policy has a table.
  [[nodiscard]] auto Value(std::size_t stage, const State& state) const noexcept -> Scalar {
    if (exact_terminal_value_ && stage + 1 == table_->states.size()) {
      return terminal_cost_(state);
    }
    return table_->Value(stage, state, inverse_scale_, num_neighbors_);
  }

  // The Q-value of `control` at `state`, read at the noise-free landing point.
  [[nodiscard]] auto QValue(std::size_t stage, const State& state, const Control& control) const noexcept -> Scalar {
    const State landing = state + forward_model_.state_drift()(stage, state) +
                          (forward_model_.control_drift_mat()(stage, state) * control);
    return running_cost_(stage, state, control) + Value(stage + 1, landing);
  }

  auto operator()(std::size_t stage, const State& state) const noexcept -> Control {
    Control control = Control::Zero();
    if (table_ == nullptr) {
      return control;
    }
    const State uncontrolled = state + forward_model_.state_drift()(stage, state);
    const Eigen::Matrix<Scalar, N, M> control_matrix = forward_model_.control_drift_mat()(stage, state);
    const auto q_value = [&](const Control& candidate) noexcept {
      return running_cost_(stage, state, candidate) + Value(stage + 1, uncontrolled + (control_matrix * candidate));
    };
    Scalar best = q_value(control);
    const std::size_t sweeps = kControlDim == 1 ? 1 : kCoordinateSweeps;
    for (std::size_t sweep = 0; sweep < sweeps; ++sweep) {
      for (std::size_t coordinate = 0; coordinate < kControlDim; ++coordinate) {
        const auto index = static_cast<Eigen::Index>(coordinate);
        const Scalar kept = control[index];
        Scalar chosen = kept;
        for (const Scalar candidate : {Scalar{0}, lower_bounds_[index], upper_bounds_[index]}) {
          if (candidate == kept) {
            continue;
          }
          control[index] = candidate;
          const Scalar value = q_value(control);
          if (value < best) {
            best = value;
            chosen = candidate;
          }
        }
        control[index] = chosen;
      }
    }
    return control;
  }

  [[nodiscard]] auto HasTable() const noexcept -> bool { return table_ != nullptr; }

 private:
  ForwardModelT forward_model_;
  L1ControlSdeRunningCostTerm<N, M, Scalar> running_cost_;
  TerminalCostT terminal_cost_;
  Control lower_bounds_;
  Control upper_bounds_;
  State inverse_scale_;
  std::size_t num_neighbors_;
  bool exact_terminal_value_;
  std::shared_ptr<const Table> table_;
};

template <int N, int M, typename ForwardModelT, typename TerminalCostT, typename Scalar = double>
  requires SdeTerminalCostTerm<TerminalCostT, Eigen::Matrix<Scalar, N, 1>>
class NearestNeighborDtFbsdeSolver {
 public:
  using State = Eigen::Matrix<Scalar, N, 1>;
  using Control = Eigen::Matrix<Scalar, M, 1>;
  using StateMat = Eigen::Matrix<Scalar, N, N>;
  using RunningCost = L1ControlSdeRunningCostTerm<N, M, Scalar>;
  using Options = NearestNeighborDtFbsdeOptions<N, Scalar>;
  using IterationReport = NearestNeighborDtFbsdeIterationReport<Scalar>;
  using Table = NearestNeighborValueTable<N, Scalar>;
  using Policy = NearestNeighborL1Policy<N, M, ForwardModelT, TerminalCostT, Scalar>;

  // Mixed into the seed for the exploration draws, so they are independent of the noise.
  static constexpr std::uint64_t kExplorationStream = 0x9E3779B97F4A7C15ULL;
  static constexpr std::uint64_t kFreshNoiseStream = 0xBF58476D1CE4E5B9ULL;

  // Builds a solver over `num_stages` stages including the terminal one. `seed` fixes the batch's
  // noise and exploration draws.
  //
  // Fails if an option is out of range, the horizon has fewer than two stages, the initial
  // distribution's covariance is not positive definite, or the bounds do not contain zero.
  template <typename InitialDistributionT>
    requires NormalDistribution<InitialDistributionT, State>
  static auto Make(ForwardModelT forward_model,
                   RunningCost running_cost,
                   TerminalCostT terminal_cost,
                   const InitialDistributionT& initial_distribution,
                   const Control& lower_bounds,
                   const Control& upper_bounds,
                   std::size_t num_stages,
                   const Options& options,
                   std::uint64_t seed) noexcept -> Result<NearestNeighborDtFbsdeSolver> {
    RESULT_ASSERT(num_stages >= 2, "NearestNeighborDtFbsdeSolver::Make: a problem needs at least two stages.");
    RESULT_ASSERT(options.num_samples >= 1, "NearestNeighborDtFbsdeSolver::Make: a batch needs at least one path.");
    RESULT_ASSERT(options.num_neighbors >= 1, "NearestNeighborDtFbsdeSolver::Make: at least one neighbor is read.");
    RESULT_ASSERT(options.exploration_probability >= Scalar{0} && options.exploration_probability <= Scalar{1} &&
                      options.initial_exploration_probability >= Scalar{0} &&
                      options.initial_exploration_probability <= Scalar{1} &&
                      options.exploring_path_fraction >= Scalar{0} && options.exploring_path_fraction <= Scalar{1},
                  "NearestNeighborDtFbsdeSolver::Make: exploration probabilities and fractions lie in [0, 1].");
    RESULT_ASSERT(options.distance_scale.allFinite() && (options.distance_scale.array() > Scalar{0}).all(),
                  "NearestNeighborDtFbsdeSolver::Make: the distance scale must be finite and positive.");
    RESULT_ASSERT((lower_bounds.array() <= Scalar{0}).all() && (upper_bounds.array() >= Scalar{0}).all() &&
                      lower_bounds.allFinite() && upper_bounds.allFinite(),
                  "NearestNeighborDtFbsdeSolver::Make: every control's bounds must be finite and contain zero.");
    const StateMat covariance = initial_distribution.Covariance();
    const Eigen::LLT<StateMat> factorization(covariance);
    RESULT_ASSERT(factorization.info() == Eigen::Success,
                  "NearestNeighborDtFbsdeSolver::Make: the initial distribution's covariance is not positive definite.");
    return SuccessResult(NearestNeighborDtFbsdeSolver(std::move(forward_model),
                                                      std::move(running_cost),
                                                      std::move(terminal_cost),
                                                      State(initial_distribution.Mean()),
                                                      StateMat(factorization.matrixL()),
                                                      lower_bounds,
                                                      upper_bounds,
                                                      num_stages,
                                                      options,
                                                      seed));
  }

  // One forward pass under the latest table's policy, and one backward pass computing Y.
  auto Iterate() noexcept -> IterationReport {
    IterationReport report;
    report.iteration = iteration_;
    const bool first = latest_ == nullptr;
    const Scalar exploration = first ? options_.initial_exploration_probability : options_.exploration_probability;
    const Policy policy = MakePolicy(latest_);
    const std::uint64_t noise_seed =
        options_.fresh_noise_each_iteration ? seed_ ^ (kFreshNoiseStream * (iteration_ + 1)) : seed_;
    auto batch = std::make_shared<Table>();
    const Scalar cost = SampleBatch(policy, exploration, first, noise_seed, *batch);
    report.batch_cost = cost;

    // Compare with the previous batch's controls before they are replaced.
    std::size_t changed = 0;
    if (!previous_controls_.empty()) {
      for (std::size_t stage = 0; stage < controls_.size(); ++stage) {
        changed += static_cast<std::size_t>(
            ((controls_[stage] - previous_controls_[stage]).array() != Scalar{0}).colwise().any().count());
      }
      report.changed_control_fraction =
          static_cast<Scalar>(changed) / static_cast<Scalar>(options_.num_samples * controls_.size());
    }
    report.fixed_point = !first && !previous_controls_.empty() && changed == 0 && !options_.fresh_noise_each_iteration;
    previous_controls_ = controls_;

    // The batch's cost is the cost of the policy that drew it, so that policy is the one to keep.
    if (!first && cost < best_cost_) {
      best_cost_ = cost;
      best_ = latest_;
    }
    report.best_batch_cost = best_cost_;

    BackwardPass(noise_seed, *batch);
    latest_ = std::move(batch);
    ++iteration_;
    return report;
  }

  // The policy whose batch cost least so far; zero control before the first greedy iteration.
  [[nodiscard]] auto CurrentPolicy() const noexcept -> Policy { return MakePolicy(best_); }

  // The policy the next iteration will sample under.
  [[nodiscard]] auto LatestPolicy() const noexcept -> Policy { return MakePolicy(latest_); }

  [[nodiscard]] auto BestBatchCost() const noexcept -> Scalar { return best_cost_; }

  // The latest batch's table; null before the first iteration.
  [[nodiscard]] auto LatestTable() const noexcept -> const std::shared_ptr<const Table>& { return latest_; }

 private:
  NearestNeighborDtFbsdeSolver(ForwardModelT forward_model,
                               RunningCost running_cost,
                               TerminalCostT terminal_cost,
                               State initial_mean,
                               StateMat initial_factor,
                               Control lower_bounds,
                               Control upper_bounds,
                               std::size_t num_stages,
                               Options options,
                               std::uint64_t seed) noexcept
      : forward_model_(std::move(forward_model)),
        running_cost_(std::move(running_cost)),
        terminal_cost_(std::move(terminal_cost)),
        initial_mean_(std::move(initial_mean)),
        initial_factor_(std::move(initial_factor)),
        lower_bounds_(std::move(lower_bounds)),
        upper_bounds_(std::move(upper_bounds)),
        num_stages_(num_stages),
        options_(std::move(options)),
        inverse_scale_(options_.distance_scale.cwiseInverse()),
        seed_(seed) {}

  [[nodiscard]] auto MakePolicy(std::shared_ptr<const Table> table) const noexcept -> Policy {
    return Policy(forward_model_,
                  running_cost_,
                  terminal_cost_,
                  lower_bounds_,
                  upper_bounds_,
                  inverse_scale_,
                  options_.num_neighbors,
                  options_.exact_terminal_value,
                  std::move(table));
  }

  [[nodiscard]] auto ExploringPath(std::size_t sample, bool first) const noexcept -> bool {
    if (first) {
      return true;
    }
    const auto steady = static_cast<std::size_t>(
        std::llround((Scalar{1} - options_.exploring_path_fraction) * static_cast<Scalar>(options_.num_samples)));
    return sample >= steady;
  }

  // Samples the batch's states into `batch` and its controls into `controls_`, and returns the mean
  // cost over the paths that never explore, or over all of them if every path may explore.
  auto SampleBatch(const Policy& policy, Scalar exploration, bool first, std::uint64_t noise_seed, Table& batch) noexcept
      -> Scalar {
    const std::size_t samples = options_.num_samples;
    const auto columns = static_cast<Eigen::Index>(samples);
    batch.states.assign(num_stages_, typename Table::StateColumns(N, columns));
    batch.values.assign(num_stages_, typename Table::ValueRow(1, columns));
    controls_.assign(num_stages_ - 1, Eigen::Matrix<Scalar, M, Eigen::Dynamic>(M, columns));
    costs_.assign(samples, Scalar{0});
    const CounterBasedNormalSampler noise_sampler(noise_seed);
    const CounterBasedNormalSampler exploration_sampler(seed_ ^ kExplorationStream);
    ParallelFor(samples, [&](std::size_t sample) noexcept {
      const auto column = static_cast<Eigen::Index>(sample);
      const bool may_explore = ExploringPath(sample, first);
      State noise;
      noise_sampler.FillNoise(sample, 0, noise);
      State state = initial_mean_ + (initial_factor_ * noise);
      Scalar running = Scalar{0};
      for (std::size_t stage = 0; stage + 1 < num_stages_; ++stage) {
        batch.states[stage].col(column) = state;
        const bool explore =
            may_explore && exploration > Scalar{0} &&
            SampleUniform<Scalar>(exploration_sampler, sample, stage, 0) < exploration;
        const Control control =
            explore ? ExploratoryL1Control<M, Scalar>(exploration_sampler, sample, stage, 1, lower_bounds_, upper_bounds_)
                    : policy(stage, state);
        controls_[stage].col(column) = control;
        running += running_cost_(stage, state, control);
        noise_sampler.FillNoise(sample, stage + 1, noise);
        state = forward_model_(stage, state, control, noise);
      }
      batch.states[num_stages_ - 1].col(column) = state;
      costs_[sample] = running + terminal_cost_(state);
    });
    Scalar total = Scalar{0};
    std::size_t counted = 0;
    for (std::size_t sample = 0; sample < samples; ++sample) {
      if (first || !ExploringPath(sample, false) || options_.exploring_path_fraction == Scalar{1}) {
        total += costs_[sample];
        ++counted;
      }
    }
    return total / static_cast<Scalar>(counted);
  }

  // Fills `batch.values` with every sample's Y, from the terminal stage back.
  auto BackwardPass(std::uint64_t noise_seed, Table& batch) const noexcept -> void {
    const std::size_t samples = options_.num_samples;
    const std::size_t terminal = num_stages_ - 1;
    for (std::size_t sample = 0; sample < samples; ++sample) {
      const auto column = static_cast<Eigen::Index>(sample);
      batch.values[terminal][column] = terminal_cost_(State(batch.states[terminal].col(column)));
    }
    const CounterBasedNormalSampler noise_sampler(noise_seed);
    // The greedy backup reads this batch's own later stages through a policy over it.
    const Policy backup(forward_model_,
                        running_cost_,
                        terminal_cost_,
                        lower_bounds_,
                        upper_bounds_,
                        inverse_scale_,
                        options_.num_neighbors,
                        options_.exact_terminal_value,
                        std::shared_ptr<const Table>(std::shared_ptr<const Table>{}, &batch));
    for (std::size_t stage = terminal; stage-- > 0;) {
      ParallelFor(samples, [&](std::size_t sample) noexcept {
        const auto column = static_cast<Eigen::Index>(sample);
        const State state = batch.states[stage].col(column);
        const Control taken = controls_[stage].col(column);
        const Scalar pathwise = running_cost_(stage, state, taken) + batch.values[stage + 1][column];
        if (options_.y_estimate == NearestNeighborYEstimate::kPathwise) {
          batch.values[stage][column] = pathwise;
          return;
        }
        State noise;
        noise_sampler.FillNoise(sample, stage + 1, noise);
        Scalar best = pathwise;
        for (std::size_t coordinate = 0; coordinate < static_cast<std::size_t>(M); ++coordinate) {
          const auto index = static_cast<Eigen::Index>(coordinate);
          for (const Scalar candidate : {Scalar{0}, lower_bounds_[index], upper_bounds_[index]}) {
            Control control = taken;
            control[index] = candidate;
            if (control == taken) {
              continue;
            }
            const State landing = forward_model_(stage, state, control, noise);
            best = std::min(best, running_cost_(stage, state, control) + backup.Value(stage + 1, landing));
          }
        }
        batch.values[stage][column] = best;
      });
    }
  }

  ForwardModelT forward_model_;
  RunningCost running_cost_;
  TerminalCostT terminal_cost_;
  State initial_mean_;
  StateMat initial_factor_;
  Control lower_bounds_;
  Control upper_bounds_;
  std::size_t num_stages_;
  Options options_;
  State inverse_scale_;
  std::uint64_t seed_;

  std::size_t iteration_ = 0;
  std::shared_ptr<const Table> latest_{};
  std::shared_ptr<const Table> best_{};
  Scalar best_cost_ = std::numeric_limits<Scalar>::infinity();
  std::vector<Eigen::Matrix<Scalar, M, Eigen::Dynamic>> controls_{};
  std::vector<Eigen::Matrix<Scalar, M, Eigen::Dynamic>> previous_controls_{};
  std::vector<Scalar> costs_{};
};

}  // namespace fbsde_traj_opt

#endif  // FBSDE_TRAJ_OPT_NEAREST_NEIGHBOR_DT_FBSDE_SOLVER_HPP_
