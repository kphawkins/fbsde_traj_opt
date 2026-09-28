// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#ifndef FBSDE_TRAJ_OPT_EXAMPLES_DOUBLE_INVERTED_PENDULUM_EXPERIMENT_HPP_
#define FBSDE_TRAJ_OPT_EXAMPLES_DOUBLE_INVERTED_PENDULUM_EXPERIMENT_HPP_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "fbsde_traj_opt/dt_fbsde_iterative_solver.hpp"
#include "fbsde_traj_opt/utils/result.hpp"
#include "fbsde_traj_opt/viz/plotly_figure.hpp"

namespace fbsde_traj_opt::examples {

// The L1 double inverted pendulum swing-up of Hawkins (2021), Section 5.5.3, solved by the
// DT-FBSDE iterative method of Section 4.6 rather than by the thesis's FBRRT: parallel sampled
// trajectories, a value function fitted at every stage by a backward pass, the policy improved
// through the Taylor Q-function of Section 4.4, and the trajectories resampled under the
// improved policy.
//
// The problem is the thesis's (5.19): starting from hanging at rest, minimize
//
//   E[ integral of c0 |u| dt + sum_j c_j X_T[j]^2 ],     |u| <= 1,
//
// the L1 -- minimum-fuel -- running cost and a quadratic terminal cost at upright, under the
// thesis's diffusion diag(0.03, 0.03, 0.18, 0.18). The thesis does not give its physical
// parameters, so the pendulum here is two uniform 1 kg, 0.5 m rods with a 5 N m motor, chosen so
// that a swing-up within the 2.5 s horizon is possible but needs most of the torque available
// (see the model header for the dynamics).
//
// The report shows three things, each answering a question a reader should ask of the result:
//
//   * Did it invert? An animation of the two links over the horizon, the link angles of a
//     hundred-odd rollouts, and the cost-to-go.
//   * How did it get there? The expected cost over the iterations, measured and as the value
//     function predicts it, and each iteration's policy: its noise-free rollout and the policy
//     around that rollout, on sliders over the iterations.
//   * Is the simulation right? The discrete model against a fourth-order Runge-Kutta reference
//     on a fine grid, energy conservation of the unforced frictionless pendulum, and the
//     discretization error's order of convergence.
//   * Was the L1 cost actually minimized? The control signal, how often it sits at each of
//     -1, 0 and +1 -- an L1-optimal control is bang-off-bang -- the fuel it spends against the
//     fuel a saturated control would, and the policy itself as a map over the state.

// The problem and method settings the experiment runs with, exposed so that a caller -- a test,
// say -- can run a shorter version.
struct DoubleInvertedPendulumExperimentOptions {
  std::uint64_t seed = 20260926;
  std::size_t iterations = 150;
};

// One experiment's output.
struct DoubleInvertedPendulumReport {
  std::string name{};
  std::string description{};

  // Several lines naming what the run showed, for the terminal.
  std::string summary{};

  std::vector<viz::PlotlyFigure> figures{};

  // One entry per iteration of the method.
  std::vector<DtFbsdeIterationReport<double>> iterations{};

  // One entry per iteration: the value function's own estimate of the expected cost from the
  // start, V~_0(x0), after that iteration -- to set beside the mean cost the iteration measured.
  std::vector<double> predicted_costs{};

  // The final policy, on a fresh set of rollouts: its mean cost and the two parts of it.
  double final_mean_cost = 0.0;
  double final_mean_fuel = 0.0;
  double final_mean_terminal_cost = 0.0;

  // The fraction of those rollouts that end with both links within 20 degrees of upright.
  double fraction_inverted = 0.0;

  // The mean terminal state of those rollouts.
  Eigen::Vector4d terminal_mean = Eigen::Vector4d::Zero();

  // The largest one-step difference between the discrete model and a fine Runge-Kutta reference,
  // stepping from each state the representative rollout visited under the control it applied: in
  // either link angle, and in either rate.
  double max_one_step_angle_error = 0.0;
  double max_one_step_rate_error = 0.0;

  // The largest difference in either link angle between the discrete model and the reference,
  // replaying the same controls open loop over the whole horizon -- which on a chaotic system also
  // measures how fast small differences grow.
  double max_reference_angle_error = 0.0;

  // The largest relative change in total energy of the unforced frictionless pendulum under the
  // discrete model over the horizon.
  double max_relative_energy_error = 0.0;
};

// Called after every iteration of the method, so a caller can report progress.
using DoubleInvertedPendulumProgress = std::function<void(const DtFbsdeIterationReport<double>&)>;

// Runs the method and builds the report.
//
// Fails if any model, the solver, a rollout, or a figure is rejected.
[[nodiscard]] auto RunDoubleInvertedPendulumExperiment(const DoubleInvertedPendulumExperimentOptions& options,
                                                       const DoubleInvertedPendulumProgress& progress) noexcept
    -> Result<DoubleInvertedPendulumReport>;

}  // namespace fbsde_traj_opt::examples

#endif  // FBSDE_TRAJ_OPT_EXAMPLES_DOUBLE_INVERTED_PENDULUM_EXPERIMENT_HPP_
