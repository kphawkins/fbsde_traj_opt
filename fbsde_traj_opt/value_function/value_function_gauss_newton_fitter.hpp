// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#ifndef FBSDE_TRAJ_OPT_VALUE_FUNCTION_GAUSS_NEWTON_FITTER_HPP_
#define FBSDE_TRAJ_OPT_VALUE_FUNCTION_GAUSS_NEWTON_FITTER_HPP_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include "fbsde_traj_opt/utils/result.hpp"
#include "fbsde_traj_opt/value_function/value_function_approx_concepts.hpp"

namespace fbsde_traj_opt {

// Damped Gauss-Newton (Levenberg-Marquardt) fitting of a ParameterizedValueFunctionApprox to fixed
// target values, with a trust region on how far each step moves the function.
//
// The problem is the same weighted nonlinear least squares ValueFunctionSgdFitter descends,
//
//   minimize  1/2 sum_s w_s ( V~(x_s; theta) - y_s )^2,
//
// and for the Taylor noiseless estimator the targets `y_s` are fixed labels, so the objective is
// deterministic and can be minimized by a method that uses its structure. With `r` the residuals
// and `J` the Jacobian of the values in the parameters -- the per-sample parameter gradients the
// model already computes -- linearizing the model and solving the resulting linear least squares
// with a damping term gives the step
//
//   ( J' W J + lambda D ) delta = -J' W r,       D = diag( J' W J ),
//
// Marquardt's form: `lambda = 0` is the full Gauss-Newton step, which for a model linear in its
// parameters lands on the least-squares fit in one step, and a large `lambda` shrinks it toward a
// short gradient step. Scaling the damping by the diagonal measures each parameter's step in how
// much that parameter moves the values, so parameters in very different units -- a soft-min
// component's constant against the entries of its metric -- are stepped on their own scales with
// no per-parameter tuning.
//
// How far a step may go. `J delta` is the first-order change the step makes to the value at every
// sample, so its weighted root mean square is the natural size of a step, in the value function's
// own units. With `max_rms_value_change` finite, `lambda` is raised until that size is within the
// bound: the step is then the one that reduces the error most among those that move the function
// by no more than the bound -- a trust region in function space. Infinity lets every step go as far
// as Gauss-Newton wants, which, iterated, is an ordinary nonlinear least-squares solve.
//
// Nonlinear models. The linearization is only locally right, so each step is checked: the actual
// reduction of the objective is compared with the reduction the linear model predicted. A poor
// ratio rejects the step and raises the damping; a good one keeps it and relaxes the damping for
// the next. A model with a direction the samples cannot see -- a soft-min component that no sample
// weights -- makes `J' W J` singular; the damping, with a floor under `D`, keeps the solve posed.
//
// `ValueFunctionT` is the model and `BlockSize` the width of the batches the normal equations are
// accumulated over.

// The knobs of one fit.
template <typename Scalar = double>
struct ValueFunctionGaussNewtonOptions {
  // Accepted steps per fit.
  std::size_t max_steps = 2;

  // The bound on each step's weighted root-mean-square first-order value change. Infinity disables
  // it.
  Scalar max_rms_value_change = std::numeric_limits<Scalar>::infinity();

  // The damping each fit starts from, relative to the diagonal. Zero starts from the full
  // Gauss-Newton step.
  Scalar initial_damping = Scalar{1e-4};

  // How the damping moves: raised by `damping_increase` after a rejected or poorly predicted step,
  // lowered by `damping_decrease` after a well predicted one.
  Scalar damping_increase = Scalar{10};
  Scalar damping_decrease = Scalar{0.3};

  // How many times one step may be rejected before the fit stops where it is.
  std::size_t max_rejections = 10;
};

// What one fit did.
template <typename Scalar = double>
struct ValueFunctionGaussNewtonReport {
  // The weighted root-mean-square residual before and after the fit.
  Scalar initial_rms_residual = Scalar{0};
  Scalar final_rms_residual = Scalar{0};

  std::size_t steps_accepted = 0;
  std::size_t steps_rejected = 0;

  // The largest predicted weighted root-mean-square value change of any accepted step.
  Scalar max_step_rms_value_change = Scalar{0};
};

template <typename ValueFunctionT, int BlockSize = 128>
  requires requires { typename ValueFunctionT::State; } &&
           BatchedParameterizedValueFunctionApprox<ValueFunctionT, typename ValueFunctionT::State, BlockSize>
class ValueFunctionGaussNewtonFitter {
 public:
  using State = typename ValueFunctionT::State;
  using Scalar = typename State::Scalar;
  using ParameterVector = typename ValueFunctionT::ParameterVector;
  using Options = ValueFunctionGaussNewtonOptions<Scalar>;
  using Report = ValueFunctionGaussNewtonReport<Scalar>;

  static constexpr int kNumParameters = ValueFunctionT::kNumParameters;

  // Fails unless the damping is non-negative, it rises by more than one and falls by less, and at
  // least one step is allowed.
  static auto Make(const Options& options) noexcept -> Result<ValueFunctionGaussNewtonFitter> {
    RESULT_ASSERT(options.max_steps >= 1, "ValueFunctionGaussNewtonFitter::Make: a fit needs at least one step.");
    RESULT_ASSERT(std::isfinite(options.initial_damping) && options.initial_damping >= Scalar{0},
                  "ValueFunctionGaussNewtonFitter::Make: the initial damping must be finite and non-negative.");
    RESULT_ASSERT(options.damping_increase > Scalar{1} && std::isfinite(options.damping_increase),
                  "ValueFunctionGaussNewtonFitter::Make: the damping increase must be finite and above one.");
    RESULT_ASSERT(options.damping_decrease > Scalar{0} && options.damping_decrease < Scalar{1},
                  "ValueFunctionGaussNewtonFitter::Make: the damping decrease must lie in (0, 1).");
    RESULT_ASSERT(options.max_rms_value_change > Scalar{0},
                  "ValueFunctionGaussNewtonFitter::Make: the trust region must be positive.");
    return SuccessResult(ValueFunctionGaussNewtonFitter(options));
  }

  // Fits `value_function` to `targets` at `states`, weighting sample `s` by `weights[s]`.
  //
  // Fails, leaving the model with the last accepted parameters, if the weights are not finite and
  // non-negative with a positive sum, or a target is not finite.
  template <int NumSamples>
  auto Fit(ValueFunctionT& value_function,
           const ValueFunctionStateBatch<State, NumSamples>& states,
           const ValueFunctionValueBatch<State, NumSamples>& targets,
           const ValueFunctionValueBatch<State, NumSamples>& weights,
           Report& report_out) const noexcept -> Result<> {
    static_assert(NumSamples % BlockSize == 0, "The samples are processed in whole blocks of BlockSize.");
    RESULT_ASSERT(
        weights.allFinite() && (weights.array() >= Scalar{0}).all() && weights.sum() > Scalar{0},
        "ValueFunctionGaussNewtonFitter::Fit: the weights must be finite and non-negative, and not all zero.");
    RESULT_ASSERT(targets.allFinite(), "ValueFunctionGaussNewtonFitter::Fit: every target must be finite.");

    const Scalar weight_sum = weights.sum();
    report_out = Report{};
    Scalar damping = options_.initial_damping;

    NormalEquations equations = Accumulate<NumSamples>(value_function, states, targets, weights);
    report_out.initial_rms_residual = std::sqrt(Scalar{2} * equations.loss / weight_sum);

    while (report_out.steps_accepted < options_.max_steps) {
      const ParameterVector scale = DampingScale(equations.normal);
      bool accepted = false;
      for (std::size_t attempt = 0; attempt <= options_.max_rejections; ++attempt) {
        const ParameterVector step = TrustRegionStep(equations, scale, weight_sum, damping);
        const Scalar predicted = -(equations.gradient.dot(step) + (Scalar{0.5} * step.dot(equations.normal * step)));
        if (!(predicted > Scalar{0}) || !step.allFinite()) {
          break;  // Nothing left to gain from this linearization.
        }

        const ParameterVector previous = value_function.Parameters();
        RESULT_RETURN_IF_ERROR(value_function.SetParameters(previous + step));
        const Scalar loss = Loss<NumSamples>(value_function, states, targets, weights);
        const Scalar ratio = (equations.loss - loss) / predicted;
        if (std::isfinite(loss) && ratio > kAcceptRatio) {
          damping *= ratio > kGoodRatio ? options_.damping_decrease : Scalar{1};
          report_out.max_step_rms_value_change =
              std::max(report_out.max_step_rms_value_change, RmsValueChange(equations.normal, step, weight_sum));
          accepted = true;
          break;
        }
        RESULT_RETURN_IF_ERROR(value_function.SetParameters(previous));
        ++report_out.steps_rejected;
        damping = std::max(damping, kMinimumRaisedDamping) * options_.damping_increase;
      }
      if (!accepted) {
        break;
      }
      ++report_out.steps_accepted;
      equations = Accumulate<NumSamples>(value_function, states, targets, weights);
    }
    report_out.final_rms_residual = std::sqrt(Scalar{2} * equations.loss / weight_sum);
    return SuccessResult();
  }

  [[nodiscard]] auto options() const noexcept -> const Options& { return options_; }

 private:
  using ParameterMat = Eigen::Matrix<Scalar, kNumParameters, kNumParameters>;
  using Block = ValueFunctionStateBatch<State, BlockSize>;
  using BlockValues = ValueFunctionValueBatch<State, BlockSize>;
  using BlockGradients = ValueFunctionParameterGradientBatch<State, kNumParameters, BlockSize>;

  // A step is kept if it achieves at least this fraction of its predicted reduction, and relaxes
  // the damping if it achieves this much more.
  static constexpr Scalar kAcceptRatio = Scalar{1e-4};
  static constexpr Scalar kGoodRatio = Scalar{0.75};

  // Raising a damping of zero must still get somewhere.
  static constexpr Scalar kMinimumRaisedDamping = Scalar{1e-6};

  // The floor under each diagonal scale, relative to the largest, for directions the samples do
  // not see.
  static constexpr Scalar kRelativeScaleFloor = Scalar{1e-10};

  // The weighted normal equations at the current parameters: `J' W J`, `J' W r`, and the objective.
  struct NormalEquations {
    ParameterMat normal = ParameterMat::Zero();
    ParameterVector gradient = ParameterVector::Zero();
    Scalar loss = Scalar{0};
  };

  explicit ValueFunctionGaussNewtonFitter(const Options& options) noexcept : options_(options) {}

  template <int NumSamples>
  static auto Accumulate(const ValueFunctionT& value_function,
                         const ValueFunctionStateBatch<State, NumSamples>& states,
                         const ValueFunctionValueBatch<State, NumSamples>& targets,
                         const ValueFunctionValueBatch<State, NumSamples>& weights) noexcept -> NormalEquations {
    NormalEquations equations;
    for (Eigen::Index first = 0; first < NumSamples; first += BlockSize) {
      const Block block = states.template middleCols<BlockSize>(first);
      BlockValues values;
      BlockGradients gradients;
      value_function.template ValuesAndParameterGradients<BlockSize>(block, values, gradients);
      const BlockValues block_weights = weights.template middleCols<BlockSize>(first);
      const BlockValues residuals = values - targets.template middleCols<BlockSize>(first);
      const BlockGradients weighted = gradients * block_weights.asDiagonal();
      equations.normal.noalias() += weighted * gradients.transpose();
      equations.gradient.noalias() += weighted * residuals.transpose();
      equations.loss += Scalar{0.5} * block_weights.cwiseProduct(residuals.cwiseAbs2()).sum();
    }
    return equations;
  }

  template <int NumSamples>
  static auto Loss(const ValueFunctionT& value_function,
                   const ValueFunctionStateBatch<State, NumSamples>& states,
                   const ValueFunctionValueBatch<State, NumSamples>& targets,
                   const ValueFunctionValueBatch<State, NumSamples>& weights) noexcept -> Scalar {
    auto loss = Scalar{0};
    for (Eigen::Index first = 0; first < NumSamples; first += BlockSize) {
      BlockValues values;
      value_function.template Values<BlockSize>(Block(states.template middleCols<BlockSize>(first)), values);
      const BlockValues residuals = values - targets.template middleCols<BlockSize>(first);
      loss += Scalar{0.5} * weights.template middleCols<BlockSize>(first).cwiseProduct(residuals.cwiseAbs2()).sum();
    }
    return loss;
  }

  // Marquardt's scale: the diagonal of `J' W J`, floored so that an unseen direction is still damped.
  static auto DampingScale(const ParameterMat& normal) noexcept -> ParameterVector {
    const ParameterVector diagonal = normal.diagonal();
    const Scalar floor = std::max(diagonal.maxCoeff() * kRelativeScaleFloor, std::numeric_limits<Scalar>::min());
    return diagonal.cwiseMax(floor);
  }

  static auto SolveStep(const NormalEquations& equations, const ParameterVector& scale, Scalar damping) noexcept
      -> ParameterVector {
    ParameterMat system = equations.normal;
    system.diagonal() += damping * scale;
    // A zero damping on a singular system fails the factorization; the least damping that does not
    // is used instead.
    for (Scalar extra = kMinimumRaisedDamping; extra < Scalar{1e12}; extra *= Scalar{10}) {
      const Eigen::LLT<ParameterMat> factorization(system);
      if (factorization.info() == Eigen::Success) {
        return factorization.solve(-equations.gradient);
      }
      system.diagonal() += extra * scale;
    }
    return ParameterVector::Zero();
  }

  // The weighted root-mean-square first-order value change of `step`: sqrt(step' J' W J step / sum w).
  static auto RmsValueChange(const ParameterMat& normal, const ParameterVector& step, Scalar weight_sum) noexcept
      -> Scalar {
    return std::sqrt(std::max(step.dot(normal * step), Scalar{0}) / weight_sum);
  }

  // The step at `damping`, or, if that moves the values by more than the trust region allows, the
  // step at the damping that puts it on the boundary, found by bisection in the logarithm of the
  // damping. `damping` is raised in place to match.
  auto TrustRegionStep(const NormalEquations& equations,
                       const ParameterVector& scale,
                       Scalar weight_sum,
                       Scalar& damping) const noexcept -> ParameterVector {
    ParameterVector step = SolveStep(equations, scale, damping);
    const Scalar bound = options_.max_rms_value_change;
    if (!std::isfinite(bound) || RmsValueChange(equations.normal, step, weight_sum) <= bound) {
      return step;
    }
    Scalar low = damping;
    Scalar high = std::max(damping, kMinimumRaisedDamping);
    do {
      low = high;
      high *= options_.damping_increase;
      step = SolveStep(equations, scale, high);
    } while (RmsValueChange(equations.normal, step, weight_sum) > bound && high < Scalar{1e12});
    for (std::size_t bisection = 0; bisection < kBisections; ++bisection) {
      const Scalar middle = std::sqrt(std::max(low, kMinimumRaisedDamping) * high);
      const ParameterVector candidate = SolveStep(equations, scale, middle);
      if (RmsValueChange(equations.normal, candidate, weight_sum) > bound) {
        low = middle;
      } else {
        high = middle;
        step = candidate;
      }
    }
    damping = high;
    return step;
  }

  static constexpr std::size_t kBisections = 20;

  Options options_;
};

}  // namespace fbsde_traj_opt

#endif  // FBSDE_TRAJ_OPT_VALUE_FUNCTION_GAUSS_NEWTON_FITTER_HPP_
