// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#ifndef FBSDE_TRAJ_OPT_EXAMPLES_L1_GRID_DP_REFERENCE_HPP_
#define FBSDE_TRAJ_OPT_EXAMPLES_L1_GRID_DP_REFERENCE_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "fbsde_traj_opt/utils/parallel_for.hpp"
#include "fbsde_traj_opt/utils/result.hpp"

namespace fbsde_traj_opt::examples {

// A near-exact optimal policy for a two-dimensional L1 benchmark (l1_benchmark_models.hpp), by
// dynamic programming on a grid, to tell whether an approximate method has found the optimum or
// only a policy it cannot improve on.
//
// The value function at every stage is held at the nodes of a regular grid and read between them
// by bilinear interpolation. The backward recursion is the stochastic Bellman equation of the
// benchmark's own discrete model,
//
//   V_k(x) = min_u [ l_k(x, u) + E_w V_{k+1}(x + f_k(x) + B_k(x) u + S w) ],   V_K = terminal cost,
//
// with the minimum over an evenly spaced grid of controls in [-1, 1] -- one that contains -1, 0,
// and 1, the controls an L1 problem favors -- and the expectation over the standard normal `w` by
// tensor Gauss-Hermite quadrature. The model is control-affine and its diffusion constant, so
// `f`, `B`, and `S` are read off the forward model itself, exactly, from one call per control and
// per noise direction; nothing about the dynamics is restated here.
//
// The policy is greedy on the grid's value function, evaluated at the state it is asked about
// rather than interpolated from the nodes, so it can be rolled out on the same noise as any other
// policy. A state off the grid reads the value at the nearest point of its boundary.
//
// Discretization error -- the interpolation's smoothing, compounded over the stages, and the
// control grid's spacing -- makes the policy slightly suboptimal, never better than optimal: its
// rolled-out cost is an upper bound on the optimum up to sampling error, and refining the grid
// shows how far above it is.
template <typename BenchmarkT>
class L1GridDpReference {
 public:
  using State = typename BenchmarkT::State;
  using Control = Eigen::Matrix<double, 1, 1>;
  static_assert(State::RowsAtCompileTime == 2, "The grid reference is for two-dimensional benchmarks.");

  struct Options {
    State lower = State::Zero();
    State upper = State::Zero();
    std::size_t cells_per_axis = 400;
    // Odd, so that the grid contains zero.
    std::size_t num_controls = 41;
    // Gauss-Hermite points per noise dimension: 3 or 5.
    std::size_t quadrature_order = 3;
  };

  // Runs the backward recursion over `num_stages` stages (the terminal one included).
  //
  // Fails if the box is empty, the grid has fewer than two cells per axis, the control count is
  // not odd and at least three, or the quadrature order is not 3 or 5.
  [[nodiscard]] static auto Make(const BenchmarkT& benchmark, std::size_t num_stages, const Options& options) noexcept
      -> Result<L1GridDpReference> {
    RESULT_ASSERT((options.upper.array() > options.lower.array()).all(), "L1GridDpReference::Make: the box is empty.");
    RESULT_ASSERT(options.cells_per_axis >= 2, "L1GridDpReference::Make: the grid needs at least two cells per axis.");
    RESULT_ASSERT(options.num_controls >= 3 && options.num_controls % 2 == 1,
                  "L1GridDpReference::Make: the control grid needs an odd count of at least three.");
    RESULT_ASSERT(options.quadrature_order == 3 || options.quadrature_order == 5,
                  "L1GridDpReference::Make: the quadrature order is 3 or 5.");
    RESULT_ASSERT(num_stages >= 2, "L1GridDpReference::Make: a problem needs an initial and a terminal stage.");
    L1GridDpReference reference(benchmark, num_stages, options);
    reference.Solve();
    return SuccessResult(std::move(reference));
  }

  // The grid's value function at `stage`, interpolated at `state`.
  [[nodiscard]] auto Value(std::size_t stage, const State& state) const noexcept -> double {
    return Interpolate(values_[stage], state);
  }

  // The greedy control at `stage` and `state`.
  [[nodiscard]] auto operator()(std::size_t stage, const State& state) const noexcept -> Control {
    const State drifted = benchmark_.forward_model(stage, state, Control::Zero(), State::Zero());
    const State control_column = benchmark_.forward_model(stage, state, Control::Ones(), State::Zero()) - drifted;
    return Control(BestControl(stage, state, drifted, control_column).first);
  }

 private:
  L1GridDpReference(const BenchmarkT& benchmark, std::size_t num_stages, const Options& options) noexcept
      : benchmark_(benchmark),
        options_(options),
        num_stages_(num_stages),
        nodes_per_axis_(options.cells_per_axis + 1),
        spacing_((options.upper - options.lower) / static_cast<double>(options.cells_per_axis)),
        values_(num_stages, std::vector<double>(nodes_per_axis_ * nodes_per_axis_, 0.0)) {
    for (std::size_t index = 0; index < options.num_controls; ++index) {
      controls_.push_back(-1.0 + (2.0 * static_cast<double>(index) / static_cast<double>(options.num_controls - 1)));
    }
    // Probabilists' Gauss-Hermite rules: nodes and weights for E[g(w)], w ~ N(0, 1).
    std::vector<std::pair<double, double>> rule;
    if (options.quadrature_order == 3) {
      rule = {{-std::sqrt(3.0), 1.0 / 6.0}, {0.0, 2.0 / 3.0}, {std::sqrt(3.0), 1.0 / 6.0}};
    } else {
      const double outer = std::sqrt(5.0 + std::sqrt(10.0));
      const double inner = std::sqrt(5.0 - std::sqrt(10.0));
      const double outer_weight = (7.0 - (2.0 * std::sqrt(10.0))) / 60.0;
      const double inner_weight = (7.0 + (2.0 * std::sqrt(10.0))) / 60.0;
      rule = {{-outer, outer_weight},
              {-inner, inner_weight},
              {0.0, 8.0 / 15.0},
              {inner, inner_weight},
              {outer, outer_weight}};
    }
    // The diffusion is constant, so its columns can be read once, at any state and stage.
    const State origin = benchmark_.forward_model(0, State::Zero(), Control::Zero(), State::Zero());
    const State first = benchmark_.forward_model(0, State::Zero(), Control::Zero(), State::UnitX()) - origin;
    const State second = benchmark_.forward_model(0, State::Zero(), Control::Zero(), State::UnitY()) - origin;
    for (const auto& [node_x, weight_x] : rule) {
      for (const auto& [node_y, weight_y] : rule) {
        offsets_.push_back((node_x * first) + (node_y * second));
        weights_.push_back(weight_x * weight_y);
      }
    }
  }

  [[nodiscard]] auto NodeState(std::size_t index) const noexcept -> State {
    const std::size_t column = index % nodes_per_axis_;
    const std::size_t row = index / nodes_per_axis_;
    return {options_.lower[0] + (static_cast<double>(column) * spacing_[0]),
            options_.lower[1] + (static_cast<double>(row) * spacing_[1])};
  }

  [[nodiscard]] auto Interpolate(const std::vector<double>& table, const State& state) const noexcept -> double {
    const double last = static_cast<double>(options_.cells_per_axis);
    const double u = std::clamp((state[0] - options_.lower[0]) / spacing_[0], 0.0, last);
    const double v = std::clamp((state[1] - options_.lower[1]) / spacing_[1], 0.0, last);
    const auto column = std::min(static_cast<std::size_t>(u), options_.cells_per_axis - 1);
    const auto row = std::min(static_cast<std::size_t>(v), options_.cells_per_axis - 1);
    const double a = u - static_cast<double>(column);
    const double b = v - static_cast<double>(row);
    const std::size_t base = (row * nodes_per_axis_) + column;
    return ((1.0 - b) * (((1.0 - a) * table[base]) + (a * table[base + 1]))) +
           (b * (((1.0 - a) * table[base + nodes_per_axis_]) + (a * table[base + nodes_per_axis_ + 1])));
  }

  // The best control on the control grid at `stage` and `state`, and its Q-value, given the
  // state's noise-free step under zero control and its control column.
  [[nodiscard]] auto BestControl(std::size_t stage,
                                 const State& state,
                                 const State& drifted,
                                 const State& control_column) const noexcept -> std::pair<double, double> {
    const std::vector<double>& next = values_[stage + 1];
    double best_control = 0.0;
    double best_value = std::numeric_limits<double>::infinity();
    for (const double control : controls_) {
      const State mean = drifted + (control * control_column);
      double expected = 0.0;
      for (std::size_t point = 0; point < offsets_.size(); ++point) {
        expected += weights_[point] * Interpolate(next, State(mean + offsets_[point]));
      }
      const double value = benchmark_.running_cost(stage, state, Control(control)) + expected;
      if (value < best_value) {
        best_value = value;
        best_control = control;
      }
    }
    return {best_control, best_value};
  }

  auto Solve() noexcept -> void {
    const std::size_t count = nodes_per_axis_ * nodes_per_axis_;
    std::vector<double>& terminal = values_.back();
    for (std::size_t index = 0; index < count; ++index) {
      terminal[index] = benchmark_.terminal_cost(NodeState(index));
    }
    for (std::size_t stage = num_stages_ - 1; stage-- > 0;) {
      std::vector<double>& table = values_[stage];
      ParallelFor(count, [&](std::size_t index) noexcept {
        const State state = NodeState(index);
        const State drifted = benchmark_.forward_model(stage, state, Control::Zero(), State::Zero());
        const State control_column = benchmark_.forward_model(stage, state, Control::Ones(), State::Zero()) - drifted;
        table[index] = BestControl(stage, state, drifted, control_column).second;
      });
    }
  }

  BenchmarkT benchmark_;
  Options options_;
  std::size_t num_stages_;
  std::size_t nodes_per_axis_;
  State spacing_;
  std::vector<std::vector<double>> values_;
  std::vector<double> controls_;
  std::vector<State> offsets_;
  std::vector<double> weights_;
};

}  // namespace fbsde_traj_opt::examples

#endif  // FBSDE_TRAJ_OPT_EXAMPLES_L1_GRID_DP_REFERENCE_HPP_
