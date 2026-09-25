// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#ifndef FBSDE_TRAJ_OPT_VALUE_FUNCTION_SGD_FITTER_HPP_
#define FBSDE_TRAJ_OPT_VALUE_FUNCTION_SGD_FITTER_HPP_

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "fbsde_traj_opt/adam_optimizer.hpp"
#include "fbsde_traj_opt/utils/result.hpp"
#include "fbsde_traj_opt/value_function_approx_concepts.hpp"

namespace fbsde_traj_opt {

// Stochastic gradient fitting of a ParameterizedValueFunctionApprox to a set of target values,
// damped so that the representation improves without lurching.
//
// The problem this solves is the regression half of a DT-FBSDE backward step. An estimator turns
// forward samples into targets `y_s` for the stage-i value function -- for the Taylor Noiseless
// estimator those targets are deterministic given the samples, so this is ordinary nonlinear
// least squares -- and the representation `V~_i` has to be moved toward them. The loss is
//
//   (1 / 2B) sum_s rho( V~(x_s) - y_s ) + (lambda / 2) || theta - anchor ||^2,
//
// with `rho` either the square or its Huber variant, and the parameter gradient of the first term
// is `(1/B) G r`, one matrix-vector product against the `P x B` matrix of per-sample parameter
// gradients the model hands back.
//
// Why damping, and why in function space. An FBSDE backward pass is a fixed-point iteration: the
// targets at stage i are computed from the representation at stage i+1, so an over-eager fit at
// one stage is amplified by every stage after it, and the usual failure is not a bad fit but a
// divergent one. Three mechanisms are offered against that, and they act in different places:
//
//   * A proximal term, pulling the parameters back toward an anchor -- the representation this
//     fitting session started from. It bounds how far the whole session can travel, but it is
//     stated in parameter norm, whose scale is a property of the model rather than the problem.
//
//   * A trust region on the change in the function's *values* over the batch. After Adam proposes
//     an increment, the first-order change it would produce at each sample is `G' dtheta`, and
//     the increment is shortened until the root-mean-square of that is within a bound. This is
//     the mechanism to reach for first: a bound of 0.1 means "no step may move the value function
//     by more than about 0.1 on average over the samples it was fitted on", which is a statement
//     about the problem and transfers unchanged between models.
//
//   * Polyak averaging, keeping a slowly-moving shadow copy of the parameters. The fitted
//     representation is then the average rather than the last iterate, which removes the
//     minibatch-to-minibatch jitter that the next stage would otherwise have to estimate through.
//
// All three are off by default; each is enabled by giving its option a positive value.
//
// `ValueFunctionT` is the model being fitted and `BatchSize` the minibatch size. Both the model's
// evaluation and the optimizer state are fixed-size, so a fitting step allocates nothing; the
// per-step history FitEpochs() appends to is the one exception, and it is owned by the caller.

// The knobs of one fitting session. Every damping mechanism is off at its default, so a fitter
// built from a default-constructed set of options is plain minibatch Adam.
template <typename Scalar = double>
struct ValueFunctionSgdOptions {
  // Adam's hyperparameters; see AdamOptimizer for what each one does.
  Scalar learning_rate = Scalar{1e-2};
  Scalar first_moment_decay = Scalar{0.9};
  Scalar second_moment_decay = Scalar{0.999};
  Scalar adam_epsilon = Scalar{1e-8};

  // The `lambda` of the proximal term. Zero disables it.
  Scalar proximal_weight = Scalar{0};

  // An upper bound on the Euclidean norm of the loss gradient, applied before the optimizer sees
  // it. Zero disables it. This is a guard against a single pathological minibatch, not a tuning
  // knob: Adam already normalizes per coordinate, so clipping rarely binds on a healthy problem.
  Scalar max_gradient_norm = Scalar{0};

  // An upper bound on the root-mean-square first-order change a single step may make to the
  // model's values over the minibatch. Zero disables it. This is the damping mechanism to prefer.
  Scalar max_rms_value_change = Scalar{0};

  // The Huber threshold: residuals larger than this contribute a gradient of constant magnitude
  // rather than one growing with the residual. Zero selects the plain squared loss. Worth setting
  // when the targets come from a sampling scheme that occasionally produces an outlier -- a state
  // far out in the tail, where the next-stage representation is extrapolating.
  Scalar huber_threshold = Scalar{0};

  // The rate at which the Polyak shadow parameters follow the live ones, in (0, 1]. Zero disables
  // the averaging, and one makes the shadow track the live parameters exactly.
  Scalar parameter_averaging_rate = Scalar{0};
};

// What one minibatch step did. Reported rather than logged, so a caller can plot the history or
// assert on it.
template <typename Scalar = double>
struct ValueFunctionSgdStepReport {
  // The plain mean squared residual over the minibatch, before the step and before any Huber or
  // sample weighting -- the fit quality, not the loss that was descended.
  Scalar mean_squared_residual = Scalar{0};

  // The square root of the above, in the units of the value function itself.
  Scalar root_mean_squared_residual = Scalar{0};

  // The Euclidean norm of the loss gradient, after clipping.
  Scalar gradient_norm = Scalar{0};

  // The root-mean-square change in the model's values the optimizer's unshortened increment would
  // have made, to first order. Compare against `max_rms_value_change` to see how hard the trust
  // region was working.
  Scalar predicted_rms_value_change = Scalar{0};

  // The factor the trust region applied to the increment: 1 when it did not bind.
  Scalar step_scale = Scalar{1};

  // The root-mean-square change the step actually made to the model's values over the minibatch,
  // measured by re-evaluating after applying it. Close to `predicted_rms_value_change *
  // step_scale` while the model is locally linear in its parameters, and visibly above it when it
  // is not -- which is the signal that the step was too long for the linearization it was chosen
  // by.
  Scalar rms_value_change = Scalar{0};
};

template <typename ValueFunctionT, int BatchSize>
  requires requires { typename ValueFunctionT::State; } &&
           BatchedParameterizedValueFunctionApprox<ValueFunctionT, typename ValueFunctionT::State, BatchSize>
class ValueFunctionSgdFitter {
 public:
  using State = typename ValueFunctionT::State;
  using Scalar = typename State::Scalar;
  using ParameterVector = typename ValueFunctionT::ParameterVector;

  static constexpr int kNumParameters = ValueFunctionT::kNumParameters;
  static constexpr auto kBatchSize = static_cast<std::size_t>(BatchSize);

  using StepReport = ValueFunctionSgdStepReport<Scalar>;

  using StateBatch = ValueFunctionStateBatch<State, BatchSize>;
  using ValueBatch = ValueFunctionValueBatch<State, BatchSize>;
  using ParameterGradientBatch = ValueFunctionParameterGradientBatch<State, kNumParameters, BatchSize>;

  // Builds a fitter.
  //
  // `anchor` is the parameter vector the proximal term pulls back toward and the Polyak shadow
  // starts from -- normally the parameters the model already holds, which is to say the
  // representation this session is refining rather than replacing.
  //
  // Fails if any option is out of range; the Adam hyperparameters are checked by AdamOptimizer
  // and its message is passed through.
  static auto Make(const ValueFunctionSgdOptions<Scalar>& options, const ParameterVector& anchor) noexcept
      -> Result<ValueFunctionSgdFitter> {
    RESULT_ASSERT(std::isfinite(options.proximal_weight) && options.proximal_weight >= Scalar{0},
                  "ValueFunctionSgdFitter::Make: the proximal weight must be finite and non-negative.");
    RESULT_ASSERT(std::isfinite(options.max_gradient_norm) && options.max_gradient_norm >= Scalar{0},
                  "ValueFunctionSgdFitter::Make: the maximum gradient norm must be finite and non-negative.");
    RESULT_ASSERT(std::isfinite(options.max_rms_value_change) && options.max_rms_value_change >= Scalar{0},
                  "ValueFunctionSgdFitter::Make: the maximum RMS value change must be finite and non-negative.");
    RESULT_ASSERT(std::isfinite(options.huber_threshold) && options.huber_threshold >= Scalar{0},
                  "ValueFunctionSgdFitter::Make: the Huber threshold must be finite and non-negative.");
    RESULT_ASSERT(options.parameter_averaging_rate >= Scalar{0} && options.parameter_averaging_rate <= Scalar{1},
                  "ValueFunctionSgdFitter::Make: the parameter averaging rate must lie in [0, 1].");
    RESULT_ASSERT(anchor.allFinite(), "ValueFunctionSgdFitter::Make: the anchor parameters must be finite.");

    const Result<Optimizer> optimizer = Optimizer::Make(
        options.learning_rate, options.first_moment_decay, options.second_moment_decay, options.adam_epsilon);
    if (!optimizer.has_value()) {
      return ErrorResult(optimizer.error().message, optimizer.error().location);
    }

    return SuccessResult(ValueFunctionSgdFitter(options, anchor, *optimizer));
  }

  // Takes one damped Adam step on `value_function` toward `targets`, weighting each sample's
  // residual by the corresponding entry of `sample_weights`.
  //
  // Sample weights exist for the off-policy case: when the forward samples were drawn under a
  // drift other than the policy's, the likelihood ratio between the two measures is what makes
  // the weighted mean an estimate of the on-policy one, and it belongs here rather than folded
  // into the targets.
  //
  // Fails if the step would leave the model with non-finite parameters, which is what a diverging
  // fit looks like from here; the model is then left holding the parameters it had.
  auto FitMinibatch(ValueFunctionT& value_function,
                    const StateBatch& states,
                    const ValueBatch& targets,
                    const ValueBatch& sample_weights,
                    StepReport& report_out) noexcept -> Result<> {
    const ParameterVector parameters = value_function.Parameters();

    ValueBatch values;
    ParameterGradientBatch parameter_gradients;
    value_function.template ValuesAndParameterGradients<BatchSize>(states, values, parameter_gradients);

    const ValueBatch residuals = values - targets;
    report_out.mean_squared_residual = residuals.squaredNorm() / static_cast<Scalar>(BatchSize);
    report_out.root_mean_squared_residual = std::sqrt(report_out.mean_squared_residual);

    // The derivative of the loss with respect to each sample's value. For the squared loss that
    // is the residual itself; the Huber variant scales any residual past the threshold down to a
    // constant magnitude, so one wild sample cannot dominate the batch.
    ValueBatch loss_derivatives = residuals.cwiseProduct(sample_weights);
    if (options_.huber_threshold > Scalar{0}) {
      loss_derivatives = loss_derivatives.cwiseProduct(
          (options_.huber_threshold * residuals.cwiseAbs().cwiseMax(options_.huber_threshold).cwiseInverse()));
    }

    ParameterVector gradient = (parameter_gradients * loss_derivatives.transpose()) / static_cast<Scalar>(BatchSize);
    if (options_.proximal_weight > Scalar{0}) {
      gradient += options_.proximal_weight * (parameters - anchor_);
    }

    const Scalar gradient_norm = gradient.norm();
    if (options_.max_gradient_norm > Scalar{0} && gradient_norm > options_.max_gradient_norm) {
      gradient *= options_.max_gradient_norm / gradient_norm;
    }
    report_out.gradient_norm = gradient.norm();

    ParameterVector increment = optimizer_.Step(gradient);

    // The first-order change this increment would make to the model's value at each sample. `G`
    // holds one parameter gradient per column, so `G' dtheta` is that change, one entry per
    // sample, at the cost of a single matrix-vector product.
    const Eigen::Matrix<Scalar, BatchSize, 1> predicted_changes = parameter_gradients.transpose() * increment;
    report_out.predicted_rms_value_change = std::sqrt(predicted_changes.squaredNorm() / static_cast<Scalar>(BatchSize));
    report_out.step_scale = Scalar{1};
    if (options_.max_rms_value_change > Scalar{0} &&
        report_out.predicted_rms_value_change > options_.max_rms_value_change) {
      report_out.step_scale = options_.max_rms_value_change / report_out.predicted_rms_value_change;
      increment *= report_out.step_scale;
    }

    const ParameterVector updated = parameters + increment;
    RESULT_ASSERT(updated.allFinite(),
                  "ValueFunctionSgdFitter::FitMinibatch: the step produced non-finite parameters, which means the fit "
                  "is diverging rather than converging.");
    if (const Result<> assigned = value_function.SetParameters(updated); !assigned.has_value()) {
      return ErrorResult(assigned.error().message, assigned.error().location);
    }

    if (options_.parameter_averaging_rate > Scalar{0}) {
      averaged_parameters_ += options_.parameter_averaging_rate * (updated - averaged_parameters_);
    }

    // Measured rather than predicted: the gap between the two is the only visible sign that the
    // step outran the linearization it was sized by.
    ValueBatch updated_values;
    value_function.template Values<BatchSize>(states, updated_values);
    report_out.rms_value_change = (updated_values - values).norm() / std::sqrt(static_cast<Scalar>(BatchSize));

    return SuccessResult();
  }

  // As above, with every sample weighted equally.
  auto FitMinibatch(ValueFunctionT& value_function,
                    const StateBatch& states,
                    const ValueBatch& targets,
                    StepReport& report_out) noexcept -> Result<> {
    return FitMinibatch(value_function, states, targets, ValueBatch::Ones(), report_out);
  }

  // Runs `num_epochs` passes of minibatch fitting over a whole set of `NumSamples` samples,
  // appending one report per step to `history_out`.
  //
  // Each epoch draws a fresh permutation of the samples and walks it in blocks of `BatchSize`,
  // wrapping around the end so that every minibatch is full even when `NumSamples` is not a
  // multiple of `BatchSize`. The wrap means the last block of an epoch revisits a few samples the
  // epoch already used; padding with a partial batch instead would either bias the mean by the
  // padding or require a second, differently sized instantiation of every batch operation.
  //
  // `history_out` is appended to rather than cleared, so a caller sweeping a backward pass can
  // accumulate one continuous history across stages. It is the one thing here that allocates.
  template <int NumSamples>
  auto FitEpochs(ValueFunctionT& value_function,
                 const ValueFunctionStateBatch<State, NumSamples>& states,
                 const ValueFunctionValueBatch<State, NumSamples>& targets,
                 const ValueFunctionValueBatch<State, NumSamples>& sample_weights,
                 std::size_t num_epochs,
                 std::uint64_t seed,
                 std::vector<StepReport>& history_out) noexcept -> Result<> {
    static_assert(NumSamples >= 1, "Fitting needs at least one sample.");
    constexpr auto kSampleCount = static_cast<std::size_t>(NumSamples);
    constexpr std::size_t kBatchesPerEpoch = (kSampleCount + kBatchSize - 1) / kBatchSize;

    std::array<std::size_t, kSampleCount> order{};
    std::iota(order.begin(), order.end(), std::size_t{0});

    for (std::size_t epoch = 0; epoch < num_epochs; ++epoch) {
      Shuffle(order, seed + epoch);

      for (std::size_t batch = 0; batch < kBatchesPerEpoch; ++batch) {
        StateBatch batch_states;
        ValueBatch batch_targets;
        ValueBatch batch_weights;
        for (std::size_t slot = 0; slot < kBatchSize; ++slot) {
          const std::size_t index = order[((batch * kBatchSize) + slot) % kSampleCount];
          const auto column = static_cast<Eigen::Index>(slot);
          const auto source = static_cast<Eigen::Index>(index);
          batch_states.col(column) = states.col(source);
          batch_targets[column] = targets[source];
          batch_weights[column] = sample_weights[source];
        }

        StepReport report;
        if (const Result<> stepped = FitMinibatch(value_function, batch_states, batch_targets, batch_weights, report);
            !stepped.has_value()) {
          return ErrorResult(stepped.error().message, stepped.error().location);
        }
        history_out.push_back(report);
      }
    }

    return SuccessResult();
  }

  // As above, with every sample weighted equally.
  template <int NumSamples>
  auto FitEpochs(ValueFunctionT& value_function,
                 const ValueFunctionStateBatch<State, NumSamples>& states,
                 const ValueFunctionValueBatch<State, NumSamples>& targets,
                 std::size_t num_epochs,
                 std::uint64_t seed,
                 std::vector<StepReport>& history_out) noexcept -> Result<> {
    return FitEpochs<NumSamples>(value_function,
                                 states,
                                 targets,
                                 ValueFunctionValueBatch<State, NumSamples>::Ones(),
                                 num_epochs,
                                 seed,
                                 history_out);
  }

  // The parameters the proximal term pulls toward.
  [[nodiscard]] auto Anchor() const noexcept -> const ParameterVector& { return anchor_; }

  // Moves the anchor and, with it, restarts the Polyak shadow. Called between the stages of a
  // backward pass, where the representation being refined changes and an anchor left behind would
  // pull the new stage toward the old one's answer.
  //
  // Fails if `anchor` is not finite.
  auto SetAnchor(const ParameterVector& anchor) noexcept -> Result<> {
    RESULT_ASSERT(anchor.allFinite(), "ValueFunctionSgdFitter::SetAnchor: the anchor parameters must be finite.");
    anchor_ = anchor;
    averaged_parameters_ = anchor;
    return SuccessResult();
  }

  // The Polyak shadow parameters. Equal to the anchor until the first step, and thereafter a
  // slowly-moving average of the live ones -- meaningful only when `parameter_averaging_rate` is
  // positive, and equal to the anchor forever when it is not.
  [[nodiscard]] auto AveragedParameters() const noexcept -> const ParameterVector& { return averaged_parameters_; }

  // Forgets the optimizer's moments, so the next step behaves like a first step. Worth doing
  // whenever the objective changes -- a new stage of a backward pass computes its targets against
  // a different next-stage representation, and moments carried across that boundary describe
  // gradients of a problem that no longer exists.
  auto ResetOptimizer() noexcept -> void { optimizer_.Reset(); }

  [[nodiscard]] auto Options() const noexcept -> const ValueFunctionSgdOptions<Scalar>& { return options_; }

 private:
  using Optimizer = AdamOptimizer<kNumParameters, Scalar>;

  ValueFunctionSgdFitter(const ValueFunctionSgdOptions<Scalar>& options,
                         const ParameterVector& anchor,
                         Optimizer optimizer) noexcept
      : options_(options), anchor_(anchor), averaged_parameters_(anchor), optimizer_(std::move(optimizer)) {}

  // A Fisher-Yates shuffle over a SplitMix64 stream. Deterministic in `seed`, so a fitting run
  // repeats exactly; not a statistical generator, and not asked to be one.
  template <std::size_t Count>
  static auto Shuffle(std::array<std::size_t, Count>& order, std::uint64_t seed) noexcept -> void {
    std::uint64_t state = seed + 0x9E3779B97F4A7C15ULL;
    for (std::size_t position = Count; position-- > 1;) {
      state += 0x9E3779B97F4A7C15ULL;
      std::uint64_t mixed = state;
      mixed = (mixed ^ (mixed >> 30U)) * 0xBF58476D1CE4E5B9ULL;
      mixed = (mixed ^ (mixed >> 27U)) * 0x94D049BB133111EBULL;
      mixed ^= mixed >> 31U;
      const auto other = static_cast<std::size_t>(mixed % static_cast<std::uint64_t>(position + 1));
      std::swap(order[position], order[other]);
    }
  }

  ValueFunctionSgdOptions<Scalar> options_;
  ParameterVector anchor_;
  ParameterVector averaged_parameters_;
  Optimizer optimizer_;
};

}  // namespace fbsde_traj_opt

#endif  // FBSDE_TRAJ_OPT_VALUE_FUNCTION_SGD_FITTER_HPP_
