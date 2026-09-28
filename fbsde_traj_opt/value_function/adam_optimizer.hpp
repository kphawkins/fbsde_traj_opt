// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#ifndef FBSDE_TRAJ_OPT_ADAM_OPTIMIZER_HPP_
#define FBSDE_TRAJ_OPT_ADAM_OPTIMIZER_HPP_

#include <cmath>
#include <cstddef>

#include <Eigen/Core>

#include "fbsde_traj_opt/utils/result.hpp"

namespace fbsde_traj_opt {

// The Adam update rule over a fixed-size parameter vector.
//
// Adam keeps two exponential moving averages of the gradients it is shown -- the first moment,
// which is the direction, and the second, which is the per-coordinate scale -- and takes a step
// whose size in each coordinate is the first divided by the square root of the second. The effect
// is that every coordinate moves at roughly the learning rate regardless of how large that
// coordinate's gradients happen to be, which is what makes it usable on a parameter vector whose
// entries mean different things at different scales. That is exactly the situation here: the
// parameters of a value function model mix the entries of a metric, the coordinates of a center,
// and a bare constant, and a plain gradient step on all of them at one rate would either crawl in
// the constant or diverge in the metric.
//
// Step() returns the increment rather than applying it. A caller that only wants to descend adds
// it; ValueFunctionSgdFitter instead shortens it first, so that the change it produces in the
// function's values stays within a bound. Handing back the increment is what lets the trust
// region live outside this class.
//
// `NumParameters` is the compile-time parameter count and `Scalar` defaults to `double`. The
// optimizer holds two vectors of that size and a step counter, and allocates nothing.
template <int NumParameters, typename Scalar = double>
class AdamOptimizer {
  static_assert(NumParameters >= 1, "An optimizer needs at least one parameter to optimize.");

 public:
  using ParameterVector = Eigen::Matrix<Scalar, NumParameters, 1>;

  // Builds an optimizer with the given hyperparameters.
  //
  // The defaults are the ones from the original paper, and are a reasonable starting point: the
  // first moment averages over roughly the last ten gradients and the second over roughly the
  // last thousand. `epsilon` keeps the division finite for a coordinate whose gradients have all
  // been zero, and also sets the largest step such a coordinate can take once a gradient does
  // arrive.
  //
  // Fails if `learning_rate` or `epsilon` is not finite and strictly positive, or if either decay
  // lies outside [0, 1). A decay of exactly 1 would freeze the corresponding moment at zero
  // forever, which is a silently useless optimizer rather than a slow one.
  static auto Make(Scalar learning_rate,
                   Scalar first_moment_decay = Scalar{0.9},
                   Scalar second_moment_decay = Scalar{0.999},
                   Scalar epsilon = Scalar{1e-8}) noexcept -> Result<AdamOptimizer> {
    RESULT_ASSERT(std::isfinite(learning_rate) && learning_rate > Scalar{0},
                  "AdamOptimizer::Make: the learning rate must be finite and positive.");
    RESULT_ASSERT(std::isfinite(epsilon) && epsilon > Scalar{0},
                  "AdamOptimizer::Make: epsilon must be finite and positive.");
    RESULT_ASSERT(first_moment_decay >= Scalar{0} && first_moment_decay < Scalar{1},
                  "AdamOptimizer::Make: the first moment decay must lie in [0, 1).");
    RESULT_ASSERT(second_moment_decay >= Scalar{0} && second_moment_decay < Scalar{1},
                  "AdamOptimizer::Make: the second moment decay must lie in [0, 1).");

    return SuccessResult(AdamOptimizer(learning_rate, first_moment_decay, second_moment_decay, epsilon));
  }

  // Advances the moments by `gradient` and returns the increment to add to the parameters.
  //
  // The increment already carries the descent sign, so a caller descends with
  // `parameters += optimizer.Step(gradient)`.
  //
  // Both moments start at zero and are therefore biased toward it for the first several steps;
  // the returned increment divides that bias out, which is why the very first step has magnitude
  // close to the learning rate rather than the vanishing size the raw moments would give.
  auto Step(const ParameterVector& gradient) noexcept -> ParameterVector {
    ++step_count_;

    first_moment_ = (first_moment_decay_ * first_moment_) + ((Scalar{1} - first_moment_decay_) * gradient);
    second_moment_ =
        (second_moment_decay_ * second_moment_) + ((Scalar{1} - second_moment_decay_) * gradient.cwiseAbs2());

    const auto steps = static_cast<Scalar>(step_count_);
    const Scalar first_bias = Scalar{1} - std::pow(first_moment_decay_, steps);
    const Scalar second_bias = Scalar{1} - std::pow(second_moment_decay_, steps);

    const ParameterVector corrected_first = first_moment_ / first_bias;
    const ParameterVector corrected_second = second_moment_ / second_bias;

    return (-learning_rate_ * corrected_first.array() / (corrected_second.array().sqrt() + epsilon_)).matrix();
  }

  // Forgets both moments and the step count, so the next Step() behaves like the first.
  //
  // Worth doing whenever the objective changes out from under the optimizer -- moving to the next
  // stage of a backward pass, for instance, where the targets are recomputed against a different
  // next-stage value function. Moments carried across that boundary describe the gradients of a
  // problem that no longer exists.
  auto Reset() noexcept -> void {
    first_moment_.setZero();
    second_moment_.setZero();
    step_count_ = 0;
  }

  // The number of steps taken since construction or the last Reset().
  [[nodiscard]] auto StepCount() const noexcept -> std::size_t { return step_count_; }

  [[nodiscard]] auto LearningRate() const noexcept -> Scalar { return learning_rate_; }

 private:
  AdamOptimizer(Scalar learning_rate, Scalar first_moment_decay, Scalar second_moment_decay, Scalar epsilon) noexcept
      : learning_rate_(learning_rate),
        first_moment_decay_(first_moment_decay),
        second_moment_decay_(second_moment_decay),
        epsilon_(epsilon) {}

  Scalar learning_rate_;
  Scalar first_moment_decay_;
  Scalar second_moment_decay_;
  Scalar epsilon_;

  ParameterVector first_moment_ = ParameterVector::Zero();
  ParameterVector second_moment_ = ParameterVector::Zero();
  std::size_t step_count_ = 0;
};

}  // namespace fbsde_traj_opt

#endif  // FBSDE_TRAJ_OPT_ADAM_OPTIMIZER_HPP_
