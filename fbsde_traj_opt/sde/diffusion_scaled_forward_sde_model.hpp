// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#ifndef FBSDE_TRAJ_OPT_SDE_DIFFUSION_SCALED_FORWARD_SDE_MODEL_HPP_
#define FBSDE_TRAJ_OPT_SDE_DIFFUSION_SCALED_FORWARD_SDE_MODEL_HPP_

#include <cstddef>
#include <utility>

#include <Eigen/Core>

namespace fbsde_traj_opt {

// A forward SDE model with its diffusion multiplied by a scale `s`: the step
//
//   x_{k+1} = x_k + f(k, x_k) + B(k, x_k) u_k + s Sigma(k, x_k) z_k,
//
// of the wrapped ComposedForwardSdeModel, with the same drift terms and a diffusion term that
// reports `s Sigma`. Everything that reads the model -- a rollout, or a policy's noise-curvature
// correction -- therefore sees one consistent problem, the original with `s` times its noise. A
// solver that walks `s` down to one solves a sequence of smoother problems that ends at the
// original; at `s = 1` the model is the wrapped one exactly.
template <typename ForwardModelT>
class DiffusionScaledForwardSdeModel {
 public:
  using State = typename ForwardModelT::State;
  using Control = typename ForwardModelT::Control;
  using Scalar = typename State::Scalar;
  using StateMat = Eigen::Matrix<Scalar, State::RowsAtCompileTime, State::RowsAtCompileTime>;

  // The wrapped diffusion term, scaled. It refers to the model it came from, so it is meant to be
  // called where it is obtained rather than kept.
  class ScaledDiffusionTerm {
   public:
    ScaledDiffusionTerm(const typename ForwardModelT::DiffusionTerm& term, Scalar scale) noexcept
        : term_(&term), scale_(scale) {}

    auto operator()(std::size_t stage, const State& state) const noexcept -> StateMat {
      return scale_ * StateMat((*term_)(stage, state));
    }

   private:
    const typename ForwardModelT::DiffusionTerm* term_;
    Scalar scale_;
  };

  explicit DiffusionScaledForwardSdeModel(ForwardModelT model, Scalar scale = Scalar{1}) noexcept
      : model_(std::move(model)), scale_(scale) {}

  auto operator()(std::size_t stage, const State& state, const Control& control, const State& noise) const noexcept
      -> State {
    return model_(stage, state, control, State(scale_ * noise));
  }

  [[nodiscard]] auto state_drift() const noexcept -> const auto& { return model_.state_drift(); }

  [[nodiscard]] auto control_drift_mat() const noexcept -> const auto& { return model_.control_drift_mat(); }

  [[nodiscard]] auto diffusion() const noexcept -> ScaledDiffusionTerm {
    return ScaledDiffusionTerm(model_.diffusion(), scale_);
  }

  [[nodiscard]] auto scale() const noexcept -> Scalar { return scale_; }

  // Precondition: `scale` is finite and positive.
  auto set_scale(Scalar scale) noexcept -> void { scale_ = scale; }

  [[nodiscard]] auto unscaled() const noexcept -> const ForwardModelT& { return model_; }

 private:
  ForwardModelT model_;
  Scalar scale_;
};

}  // namespace fbsde_traj_opt

#endif  // FBSDE_TRAJ_OPT_SDE_DIFFUSION_SCALED_FORWARD_SDE_MODEL_HPP_
