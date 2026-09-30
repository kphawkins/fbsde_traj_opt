// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#ifndef FBSDE_TRAJ_OPT_FBRRT_TAYLOR_BACKWARD_PASS_HPP_
#define FBSDE_TRAJ_OPT_FBRRT_TAYLOR_BACKWARD_PASS_HPP_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "fbsde_traj_opt/solvers/taylor_noiseless_backward_step_estimator.hpp"
#include "fbsde_traj_opt/utils/parallel_for.hpp"
#include "fbsde_traj_opt/utils/result.hpp"
#include "fbsde_traj_opt/value_function/value_function_sgd_fitter.hpp"

namespace fbsde_traj_opt {

// The knobs of the SGD backward pass.
template <typename Scalar = double>
struct TaylorBackwardPassOptions {
  // The fitter's options -- learning rate, trust region, and the rest; see ValueFunctionSgdFitter.
  ValueFunctionSgdOptions<Scalar> sgd{};

  // Passes over a stage's samples per backward pass. Few, on purpose: each backward pass is meant
  // to move a value function part of the way toward its targets, not all of it.
  std::size_t epochs_per_stage = 2;

  // A floor under every target. The true value function of a problem whose costs are all
  // non-negative is itself non-negative, so a target below zero is extrapolation error by
  // construction; flooring it stops a value function from learning a basin that does not exist and
  // the policy from steering into it. Negative infinity disables the floor.
  Scalar target_floor = -std::numeric_limits<Scalar>::infinity();

  // `lambda` of the optimal trajectory distribution weighting, in units of the heuristic's spread over
  // the batch; zero -- the default -- fits every edge equally, leaving the heuristic to erosion. With
  // a positive value each edge into stage `i + 1` is weighted by `exp(-rho / lambda)` on
  //
  //   rho = L_{0:i} + V~_{i+1}(x_{i+1}),
  //
  // the optimal trajectory distribution heuristic of Hawkins (2021), Chapter 4 -- the running cost the edge's path
  // accrued plus the estimated cost from where it landed -- as Algorithm 4.3 weights its regression. The heuristic is
  // shifted by its minimum and divided by its standard deviation over the batch, so `lambda`
  // transfers between problems, and the weights are then scaled to average one, so that weighting
  // changes which edges the fit attends to without also shrinking its effective learning rate.
  Scalar entropy_temperature = Scalar{0};

  // Whether each stage's Adam moments are forgotten at the start of every backward pass. The
  // targets change between passes -- they are computed from a next-stage function that has itself
  // moved -- so moments carried across describe a problem that no longer exists; keeping them is
  // the variant to try when the targets move slowly.
  bool reset_optimizer_each_pass = true;
};

// What one backward pass did.
template <typename Scalar = double>
struct TaylorBackwardPassReport {
  // Per stage 0..K-1, the root-mean-square estimator error at the start of that stage's fit -- the
  // residual `V~_i(x) - Y^_i` the pass then descends, over its first epoch.
  std::vector<Scalar> stage_rms_residuals{};

  // The root mean square of the above over the stages.
  Scalar rms_residual = Scalar{0};

  // With the entropy weighting, the effective sample fraction `(sum w)^2 / (N sum w^2)` of the
  // weights, averaged over the stages: one when every edge counts equally, and near `1 / N` when
  // one edge dominates. One without the weighting.
  Scalar effective_sample_fraction = Scalar{1};
};

// The backward pass of the FBRRT method, with the least-squares Monte Carlo regression of Hawkins
// (2021), Algorithm 4.3, replaced by a few steps of stochastic gradient descent on the estimator's
// error.
//
// For each stage `i` from the last down, every edge into stage `i + 1` -- a source state `x_i`
// with the drift `K_i` it was drawn under -- yields a Taylor noiseless target
//
//   Y^_i = L^mu_i + V~_{i+1}(x_i + K_i) + g' dF + 1/2 dF' H dF + 1/2 tr(Sigma' H Sigma),
//
// (TaylorNoiselessBackwardStepEstimator, which explains the terms) for the target policy `mu`.
// The stage's value function is then moved *toward* those targets by `epochs_per_stage` passes of
// damped minibatch Adam (ValueFunctionSgdFitter), starting from where the previous iteration left
// it. By default the regression is unweighted: the path-integral heuristic acts on which particles
// survive erosion, not on how much each one counts here. `entropy_temperature` weights it as well.
//
// Why not solve the regression outright, as LSMC does. An exact fit makes each value function the
// best answer to the *current* targets, which are computed from a next-stage function and a policy
// that are themselves still wrong. The value functions then converge far faster than the policy
// they are meant to evaluate, and lock in whatever the early, poorly-sampled iterations believed.
// A few damped steps per iteration keep the value functions and the policy moving together.
//
// The target policy reads the same table that is being updated, stage by stage from the end, so
// the targets at stage `i` see the already-updated `V~_{i+1}`, as in the thesis's recursion. The
// terminal entry of the table is never touched.
//
// `ValueFunctionT` is the per-stage representation, `Capacity` the number of edges into each stage,
// `MinibatchSize` the fitter's minibatch, and `TargetBatchSize` the width of the blocks the targets
// are computed in, in parallel.
template <typename ValueFunctionT,
          typename ForwardModelT,
          typename RunningCostT,
          int Capacity,
          int MinibatchSize = 64,
          int TargetBatchSize = 128>
class TaylorBackwardPass {
  static_assert(Capacity % TargetBatchSize == 0, "Targets are computed in whole blocks of TargetBatchSize.");

 public:
  using State = typename ValueFunctionT::State;
  using Scalar = typename State::Scalar;
  using Options = TaylorBackwardPassOptions<Scalar>;
  using Report = TaylorBackwardPassReport<Scalar>;
  using Fitter = ValueFunctionSgdFitter<ValueFunctionT, MinibatchSize>;

  static constexpr int kStateDim = State::RowsAtCompileTime;

  // Builds a backward pass for a table of `num_stages` value functions, one fitter per non-terminal
  // stage, each anchored at that stage's initial representation.
  //
  // Fails if there are fewer than two stages, if `epochs_per_stage` is zero, or if the fitter
  // rejects the SGD options.
  static auto Make(ForwardModelT forward_model,
                   RunningCostT running_cost,
                   const Options& options,
                   const std::vector<ValueFunctionT>& initial_table) noexcept -> Result<TaylorBackwardPass> {
    RESULT_ASSERT(initial_table.size() >= 2, "TaylorBackwardPass::Make: the table needs at least two stages.");
    RESULT_ASSERT(options.epochs_per_stage >= 1, "TaylorBackwardPass::Make: a pass needs at least one epoch.");
    RESULT_ASSERT(std::isfinite(options.entropy_temperature) && options.entropy_temperature >= Scalar{0},
                  "TaylorBackwardPass::Make: the entropy temperature must be finite and non-negative.");
    std::vector<Fitter> fitters;
    fitters.reserve(initial_table.size() - 1);
    for (std::size_t stage = 0; stage + 1 < initial_table.size(); ++stage) {
      RESULT_ASSIGN_OR_RETURN(Fitter fitter, Fitter::Make(options.sgd, initial_table[stage].Parameters()));
      fitters.push_back(std::move(fitter));
    }
    return SuccessResult(
        TaylorBackwardPass(std::move(forward_model), std::move(running_cost), options, std::move(fitters)));
  }

  // Runs one backward pass over `table`, whose policy is `policy`, from the edges of `particles`.
  //
  // `policy` must read `table` live (TaylorQFunctionL1Policy does), so that the targets at each
  // stage see the next stage's update. `seed` shuffles the minibatches.
  //
  // Fails if a stage past the first is not full, or if a fit diverges -- which leaves the stages
  // already updated as they are.
  template <typename ParticleSetT, typename PolicyT>
  auto operator()(std::vector<ValueFunctionT>& table,
                  const ParticleSetT& particles,
                  const PolicyT& policy,
                  std::uint64_t seed) noexcept -> Result<Report> {
    RESULT_ASSERT(table.size() == fitters_.size() + 1,
                  "TaylorBackwardPass: the table does not have the stage count the pass was built for.");
    using Estimator = TaylorNoiselessBackwardStepEstimator<kStateDim,
                                                           ParticleSetT::Control::RowsAtCompileTime,
                                                           ForwardModelT,
                                                           PolicyT,
                                                           RunningCostT,
                                                           Scalar>;
    const Estimator estimator(forward_model_, policy, running_cost_);

    Report report;
    report.stage_rms_residuals.assign(fitters_.size(), Scalar{0});
    std::vector<typename Fitter::StepReport> history;
    auto fraction_sum = Scalar{0};
    for (std::size_t stage = fitters_.size(); stage-- > 0;) {
      RESULT_ASSERT(particles.count(stage + 1) == static_cast<std::size_t>(Capacity),
                    "TaylorBackwardPass: every stage past the first must be full; expand before the backward pass.");
      const ValueFunctionStateBatch<State, Capacity>& states = particles.source_states(stage + 1);
      const ValueFunctionStateBatch<State, Capacity>& drifts = particles.drifts(stage + 1);

      const bool weighted = options_.entropy_temperature > Scalar{0};
      ValueFunctionValueBatch<State, Capacity> targets;
      ValueFunctionValueBatch<State, Capacity> heuristics;
      ParallelFor(kTargetBlocks, [&](std::size_t block) noexcept {
        const auto first = static_cast<Eigen::Index>(block * static_cast<std::size_t>(TargetBatchSize));
        ValueFunctionValueBatch<State, TargetBatchSize> block_targets;
        estimator.template EstimateTargets<TargetBatchSize>(stage,
                                                            states.template middleCols<TargetBatchSize>(first),
                                                            drifts.template middleCols<TargetBatchSize>(first),
                                                            table[stage + 1],
                                                            block_targets);
        targets.template middleCols<TargetBatchSize>(first) = block_targets.cwiseMax(options_.target_floor);
        if (weighted) {
          ValueFunctionValueBatch<State, TargetBatchSize> landing_values;
          table[stage + 1].template Values<TargetBatchSize>(
              particles.states(stage + 1).template middleCols<TargetBatchSize>(first), landing_values);
          heuristics.template middleCols<TargetBatchSize>(first) =
              particles.accrued_costs(stage + 1).template middleCols<TargetBatchSize>(first) + landing_values;
        }
      });

      ValueFunctionValueBatch<State, Capacity> weights = ValueFunctionValueBatch<State, Capacity>::Ones();
      if (weighted) {
        const Scalar mean = heuristics.mean();
        const Scalar spread =
            std::sqrt((heuristics.array() - mean).square().mean()) + std::numeric_limits<Scalar>::epsilon();
        weights = (-(heuristics.array() - heuristics.minCoeff()) / (spread * options_.entropy_temperature)).exp();
        weights /= weights.mean();
        fraction_sum += (weights.sum() * weights.sum()) / (static_cast<Scalar>(Capacity) * weights.squaredNorm());
      }

      Fitter& fitter = fitters_[stage];
      RESULT_RETURN_IF_ERROR(fitter.SetAnchor(table[stage].Parameters()));
      if (options_.reset_optimizer_each_pass) {
        fitter.ResetOptimizer();
      }
      history.clear();
      RESULT_RETURN_IF_ERROR(fitter.template FitEpochs<Capacity>(table[stage],
                                                                 states,
                                                                 targets,
                                                                 weights,
                                                                 options_.epochs_per_stage,
                                                                 seed + (0x9E3779B97F4A7C15ULL * stage),
                                                                 history));
      if (options_.sgd.parameter_averaging_rate > Scalar{0}) {
        RESULT_RETURN_IF_ERROR(table[stage].SetParameters(fitter.AveragedParameters()));
      }

      const std::size_t first_epoch_steps = std::max<std::size_t>(history.size() / options_.epochs_per_stage, 1);
      auto squared = Scalar{0};
      for (std::size_t step = 0; step < first_epoch_steps; ++step) {
        squared += history[step].mean_squared_residual;
      }
      report.stage_rms_residuals[stage] = std::sqrt(squared / static_cast<Scalar>(first_epoch_steps));
    }

    if (options_.entropy_temperature > Scalar{0}) {
      report.effective_sample_fraction = fraction_sum / static_cast<Scalar>(fitters_.size());
    }
    auto total = Scalar{0};
    for (const Scalar residual : report.stage_rms_residuals) {
      total += residual * residual;
    }
    report.rms_residual = std::sqrt(total / static_cast<Scalar>(report.stage_rms_residuals.size()));
    return SuccessResult(std::move(report));
  }

  [[nodiscard]] auto options() const noexcept -> const Options& { return options_; }

 private:
  static constexpr auto kTargetBlocks = static_cast<std::size_t>(Capacity / TargetBatchSize);

  TaylorBackwardPass(ForwardModelT forward_model,
                     RunningCostT running_cost,
                     const Options& options,
                     std::vector<Fitter> fitters) noexcept
      : forward_model_(std::move(forward_model)),
        running_cost_(std::move(running_cost)),
        options_(options),
        fitters_(std::move(fitters)) {}

  ForwardModelT forward_model_;
  RunningCostT running_cost_;
  Options options_;
  std::vector<Fitter> fitters_;
};

}  // namespace fbsde_traj_opt

#endif  // FBSDE_TRAJ_OPT_FBRRT_TAYLOR_BACKWARD_PASS_HPP_
