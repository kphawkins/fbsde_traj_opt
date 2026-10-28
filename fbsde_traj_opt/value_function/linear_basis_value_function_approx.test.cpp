// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#include "fbsde_traj_opt/value_function/linear_basis_value_function_approx.hpp"

#include <cmath>
#include <cstddef>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "fbsde_traj_opt/value_function/polynomial_value_function_approx.hpp"
#include "fbsde_traj_opt/value_function/shallow_tanh_value_function_approx.hpp"
#include "fbsde_traj_opt/value_function/value_function_approx_concepts.hpp"
#include "fbsde_traj_opt/value_function/value_function_slope_targets.hpp"

namespace fbsde_traj_opt {
namespace {

using State = Eigen::Vector2d;

template <typename BasisT>
using Model = LinearBasisValueFunctionApprox<2, BasisT>;

static_assert(BatchedParameterizedValueFunctionApprox<Model<LegendreBasis<2, 4>>, State, 8>);
static_assert(BatchedParameterizedValueFunctionApprox<Model<GaussianRbfBasis<2, 5>>, State, 8>);
static_assert(BatchedParameterizedValueFunctionApprox<Model<CubicBSplineBasis<2, 4>>, State, 8>);

template <typename BasisT>
auto MakeModel(BasisT basis, std::size_t seed, RidgeAnchor anchor = RidgeAnchor::kZero) -> Model<BasisT> {
  using Parameters = typename Model<BasisT>::ParameterVector;
  std::srand(static_cast<unsigned>(seed));
  const Parameters parameters = Parameters::Random();
  auto model = Model<BasisT>::Make(parameters, State(0.5, -0.25), State(2.0, 3.0), basis, anchor);
  EXPECT_TRUE(model.has_value());
  return *model;
}

// The gradient and Hessian agree with central differences of the value and the gradient.
template <typename BasisT>
auto ExpectDerivativesMatchDifferences(const Model<BasisT>& model) -> void {
  for (const State state : {State(0.3, 0.7), State(-1.2, 2.1), State(2.4, -2.9)}) {
    constexpr double kStep = 1e-5;
    const State gradient = model.Gradient(state);
    const Eigen::Matrix2d hessian = model.Hessian(state);
    for (int axis = 0; axis < 2; ++axis) {
      const State offset = State::Unit(axis) * kStep;
      const double slope = (model(state + offset) - model(state - offset)) / (2 * kStep);
      EXPECT_NEAR(gradient[axis], slope, 1e-5 * std::max(1.0, std::abs(slope)));
      const State column = (model.Gradient(state + offset) - model.Gradient(state - offset)) / (2 * kStep);
      for (int row = 0; row < 2; ++row) {
        EXPECT_NEAR(hessian(row, axis), column[row], 1e-4 * std::max(1.0, std::abs(column[row])));
      }
    }
    EXPECT_NEAR(hessian(0, 1), hessian(1, 0), 1e-12);
  }
}

TEST(LinearBasisValueFunctionApproxTest, DerivativesMatchDifferencesForEveryBasis) {
  ExpectDerivativesMatchDifferences(MakeModel(MonomialBasis<2, 6>{}, 1));
  ExpectDerivativesMatchDifferences(MakeModel(LegendreBasis<2, 6>{}, 2));
  ExpectDerivativesMatchDifferences(MakeModel(GaussianRbfBasis<2, 5>{.width_factor = 0.8}, 3));
  ExpectDerivativesMatchDifferences(MakeModel(CubicBSplineBasis<2, 4>{}, 4));
}

// The monomial instance spans exactly what PolynomialValueFunctionApprox does, so the two fits
// agree wherever the samples determine the coefficients.
TEST(LinearBasisValueFunctionApproxTest, MonomialFitMatchesThePolynomialModel) {
  constexpr int kSamples = 64;
  Eigen::Matrix<double, 2, kSamples> states = Eigen::Matrix<double, 2, kSamples>::Random() * 2.0;
  Eigen::Matrix<double, 1, kSamples> targets;
  for (int column = 0; column < kSamples; ++column) {
    targets[column] = std::cos(states(0, column)) + (states(1, column) * states(1, column));
  }
  const Eigen::Matrix<double, 1, kSamples> weights = Eigen::Matrix<double, 1, kSamples>::Ones();
  auto polynomial = PolynomialValueFunctionApprox<2, 4>::Make(
      PolynomialValueFunctionApprox<2, 4>::ParameterVector::Zero(), State(0.5, -0.25), State(2.0, 3.0));
  ASSERT_TRUE(polynomial.has_value());
  Model<MonomialBasis<2, 4>> linear = MakeModel(MonomialBasis<2, 4>{}, 5);
  ASSERT_TRUE(polynomial->FitWeightedLeastSquares<kSamples>(states, targets, weights, 0.0).has_value());
  ASSERT_TRUE(linear.FitWeightedLeastSquares<kSamples>(states, targets, weights, 0.0).has_value());
  for (const State state : {State(0.1, 0.2), State(-1.5, 1.0), State(1.9, -1.7)}) {
    EXPECT_NEAR(linear(state), (*polynomial)(state), 1e-8);
  }
}

// Cubic splines reproduce a quadratic inside their domain, so a quadratic is fitted exactly.
TEST(LinearBasisValueFunctionApproxTest, SplinesAndRbfsFitAQuadraticExactly) {
  constexpr int kSamples = 400;
  Eigen::Matrix<double, 2, kSamples> states = Eigen::Matrix<double, 2, kSamples>::Random() * 2.0;
  states.row(0).array() += 0.5;
  Eigen::Matrix<double, 1, kSamples> targets;
  const auto quadratic = [](const State& x) { return (3.0 * x[0] * x[0]) - (x[0] * x[1]) + (0.5 * x[1]) + 1.0; };
  for (int column = 0; column < kSamples; ++column) {
    targets[column] = quadratic(states.col(column));
  }
  const Eigen::Matrix<double, 1, kSamples> weights = Eigen::Matrix<double, 1, kSamples>::Ones();
  Model<CubicBSplineBasis<2, 4>> spline = MakeModel(CubicBSplineBasis<2, 4>{}, 6);
  Model<GaussianRbfBasis<2, 4>> rbf = MakeModel(GaussianRbfBasis<2, 4>{}, 7);
  ASSERT_TRUE(spline.FitWeightedLeastSquares<kSamples>(states, targets, weights, 1e-12).has_value());
  ASSERT_TRUE(rbf.FitWeightedLeastSquares<kSamples>(states, targets, weights, 1e-12).has_value());
  for (const State state : {State(0.1, 0.2), State(-1.0, 1.0), State(2.0, -1.5)}) {
    EXPECT_NEAR(spline(state), quadratic(state), 1e-6);
    EXPECT_NEAR(rbf(state), quadratic(state), 1e-6);
  }
}

// Anchored at the current coefficients, a coefficient the samples do not touch keeps its value; at
// zero it is shrunk to zero.
TEST(LinearBasisValueFunctionApproxTest, ACurrentAnchorKeepsUnexcitedCoefficients) {
  constexpr int kSamples = 4;
  Eigen::Matrix<double, 2, kSamples> states;
  states << -1.0, -0.9, -1.1, -1.0, -2.0, -2.1, -1.9, -2.0;  // all near one corner of the box
  const Eigen::Matrix<double, 1, kSamples> targets = Eigen::Matrix<double, 1, kSamples>::Constant(1.0);
  const Eigen::Matrix<double, 1, kSamples> weights = Eigen::Matrix<double, 1, kSamples>::Ones();
  // The last spline sits in the opposite corner, where no sample's row reaches it.
  constexpr int kFar = CubicBSplineBasis<2, 4>::kNumSplines - 1;
  for (const RidgeAnchor anchor : {RidgeAnchor::kZero, RidgeAnchor::kCurrent}) {
    Model<CubicBSplineBasis<2, 4>> spline = MakeModel(CubicBSplineBasis<2, 4>{}, 8, anchor);
    const double before = spline.Parameters()[kFar];
    ASSERT_NE(before, 0.0);
    ASSERT_TRUE(spline.FitWeightedLeastSquares<kSamples>(states, targets, weights, 1e-3).has_value());
    EXPECT_NEAR(spline.Parameters()[kFar], anchor == RidgeAnchor::kCurrent ? before : 0.0, 1e-12);
  }
}

// The roughness penalty leaves affine functions alone and charges curvature: with it, two samples of
// a line are fitted by that line even in a basis rich enough to bend through them any other way.
TEST(LinearBasisValueFunctionApproxTest, RoughnessPenaltyPrefersTheFlattestFit) {
  constexpr int kSamples = 2;
  Eigen::Matrix<double, 2, kSamples> states;
  states << -1.0, 2.0, 0.5, -1.0;
  Eigen::Matrix<double, 1, kSamples> targets;
  const auto line = [](const State& x) { return (2.0 * x[0]) - x[1] + 0.5; };
  targets << line(states.col(0)), line(states.col(1));
  const Eigen::Matrix<double, 1, kSamples> weights = Eigen::Matrix<double, 1, kSamples>::Ones();
  auto model = Model<LegendreBasis<2, 8>>::Make(Model<LegendreBasis<2, 8>>::ParameterVector::Zero(),
                                                 State(0.5, -0.25),
                                                 State(2.0, 3.0),
                                                 LegendreBasis<2, 8>{},
                                                 RidgeAnchor::kZero,
                                                 1.0);
  ASSERT_TRUE(model.has_value());
  ASSERT_TRUE(model->FitWeightedLeastSquares<kSamples>(states, targets, weights, 1e-9).has_value());
  for (const State state : {State(0.0, 0.0), State(1.5, 1.5)}) {
    EXPECT_NEAR(model->Hessian(state).norm(), 0.0, 1e-3);
  }
}

// A slope row is the model's own directional derivative: FeatureSlopes(x, d)' theta = d' grad V(x),
// through the normalization's chain rule, for every basis.
template <typename BasisT>
auto ExpectSlopeRowsMatchTheGradient(const Model<BasisT>& model) -> void {
  for (const State state : {State(0.3, 0.7), State(-1.2, 2.1)}) {
    for (const State direction : {State(1.0, 0.0), State(0.25, -0.8)}) {
      const double from_rows = model.FeatureSlopes(state, direction).dot(model.Parameters());
      EXPECT_NEAR(from_rows, direction.dot(model.Gradient(state)), 1e-9 * (1.0 + std::abs(from_rows)));
    }
  }
}

TEST(LinearBasisValueFunctionApproxTest, SlopeRowsAreTheDirectionalDerivative) {
  ExpectSlopeRowsMatchTheGradient(MakeModel(MonomialBasis<2, 4>{}, 21));
  ExpectSlopeRowsMatchTheGradient(MakeModel(LegendreBasis<2, 6>{}, 22));
  ExpectSlopeRowsMatchTheGradient(MakeModel(GaussianRbfBasis<2, 5>{}, 23));
  ExpectSlopeRowsMatchTheGradient(MakeModel(CubicBSplineBasis<2, 4>{}, 24));
}

// A slope-matched fit of a function in the span recovers it, values and slopes, from fewer samples
// than coefficients -- which only the slope rows can make determined.
TEST(LinearBasisValueFunctionApproxTest, SlopeMatchedFitRecoversAFunctionInTheSpan) {
  using Poly = Model<MonomialBasis<2, 2>>;
  Poly truth = MakeModel(MonomialBasis<2, 2>{}, 25);
  Poly fitted = MakeModel(MonomialBasis<2, 2>{}, 26);
  constexpr int kSamples = 4;  // 4 values + 8 slopes for 6 coefficients
  Eigen::Matrix<double, 2, kSamples> states;
  states << -1.0, 0.5, 2.0, 0.1, 0.3, -1.5, 1.0, 2.5;
  Eigen::Matrix<double, 1, kSamples> targets;
  ValueFunctionSlopeTargets<State, kSamples, 2> slopes;
  for (int column = 0; column < kSamples; ++column) {
    const State x = states.col(column);
    targets[column] = truth(x);
    for (int axis = 0; axis < 2; ++axis) {
      slopes.directions[static_cast<std::size_t>(axis)].col(column) = State::Unit(axis);
      slopes.slopes[static_cast<std::size_t>(axis)][column] = truth.Gradient(x)[axis];
    }
  }
  ASSERT_TRUE((fitted.FitWeightedLeastSquares<kSamples, 2>(
                   states, targets, Eigen::Matrix<double, 1, kSamples>::Ones(), slopes, 1.0, 0.0))
                  .has_value());
  for (const State x : {State(0.0, 0.0), State(1.7, -0.9)}) {
    EXPECT_NEAR(fitted(x), truth(x), 1e-8);
  }
}

// Two batches accumulated into one set of normal equations fit exactly what one fit of their union
// does; decaying the first batch's rows before adding the second weights it down accordingly.
TEST(LinearBasisValueFunctionApproxTest, AccumulatedNormalEquationsFitTheUnionOfTheirBatches) {
  using Poly = Model<MonomialBasis<2, 4>>;
  constexpr int kHalf = 32;
  Eigen::Matrix<double, 2, 2 * kHalf> states = Eigen::Matrix<double, 2, 2 * kHalf>::Random() * 2.0;
  Eigen::Matrix<double, 1, 2 * kHalf> targets;
  for (int column = 0; column < 2 * kHalf; ++column) {
    targets[column] = std::sin(states(0, column)) + (states(1, column) * states(0, column));
  }
  const Eigen::Matrix<double, 1, 2 * kHalf> ones = Eigen::Matrix<double, 1, 2 * kHalf>::Ones();
  Poly whole = MakeModel(MonomialBasis<2, 4>{}, 9);
  Poly pieces = whole;
  ASSERT_TRUE(whole.FitWeightedLeastSquares<2 * kHalf>(states, targets, ones, 1e-6).has_value());
  Poly::NormalEquations equations;
  for (int part = 0; part < 2; ++part) {
    const Eigen::Matrix<double, 2, kHalf> part_states = states.middleCols<kHalf>(part * kHalf);
    const Eigen::Matrix<double, 1, kHalf> part_targets = targets.middleCols<kHalf>(part * kHalf);
    ASSERT_TRUE(pieces
                    .AccumulateNormalEquations<kHalf>(
                        part_states, part_targets, Eigen::Matrix<double, 1, kHalf>::Ones(), equations)
                    .has_value());
  }
  EXPECT_DOUBLE_EQ(equations.samples, 2.0 * kHalf);
  ASSERT_TRUE(pieces.FitNormalEquations(equations, 1e-6).has_value());
  EXPECT_LT((whole.Parameters() - pieces.Parameters()).norm(), 1e-9 * (1.0 + whole.Parameters().norm()));

  // Decaying the first half by 0.5 is the same as weighting it 0.5 in one fit.
  Eigen::Matrix<double, 1, 2 * kHalf> weights = ones;
  weights.leftCols<kHalf>().setConstant(0.5);
  Poly weighted = MakeModel(MonomialBasis<2, 4>{}, 9);
  ASSERT_TRUE(weighted.FitWeightedLeastSquares<2 * kHalf>(states, targets, weights, 0.0).has_value());
  Poly::NormalEquations decayed;
  ASSERT_TRUE(pieces
                  .AccumulateNormalEquations<kHalf>(states.leftCols<kHalf>(),
                                                    targets.leftCols<kHalf>(),
                                                    Eigen::Matrix<double, 1, kHalf>::Ones(),
                                                    decayed)
                  .has_value());
  decayed.Decay(0.5);
  ASSERT_TRUE(pieces
                  .AccumulateNormalEquations<kHalf>(states.rightCols<kHalf>(),
                                                    targets.rightCols<kHalf>(),
                                                    Eigen::Matrix<double, 1, kHalf>::Ones(),
                                                    decayed)
                  .has_value());
  ASSERT_TRUE(pieces.FitNormalEquations(decayed, 0.0).has_value());
  EXPECT_LT((weighted.Parameters() - pieces.Parameters()).norm(), 1e-8 * (1.0 + weighted.Parameters().norm()));
}

TEST(ShallowTanhValueFunctionApproxTest, DerivativesAndParameterGradientMatchDifferences) {
  using Net = ShallowTanhValueFunctionApprox<2, 8>;
  static_assert(BatchedParameterizedValueFunctionApprox<Net, State, 8>);
  const Net::ParameterVector parameters =
      Net::RandomParameters(Eigen::Matrix<double, Net::kNumQuadratic, 1>::Random(), 0.7, 11);
  auto net = Net::Make(parameters, State(0.5, -0.25), State(2.0, 3.0));
  ASSERT_TRUE(net.has_value());
  for (const State state : {State(0.3, 0.7), State(-1.2, 2.1), State(2.4, -2.9)}) {
    constexpr double kStep = 1e-5;
    const State gradient = net->Gradient(state);
    const Eigen::Matrix2d hessian = net->Hessian(state);
    for (int axis = 0; axis < 2; ++axis) {
      const State offset = State::Unit(axis) * kStep;
      EXPECT_NEAR(gradient[axis], ((*net)(state + offset) - (*net)(state - offset)) / (2 * kStep), 1e-6);
      const State column = (net->Gradient(state + offset) - net->Gradient(state - offset)) / (2 * kStep);
      EXPECT_NEAR(hessian(0, axis), column[0], 1e-5);
      EXPECT_NEAR(hessian(1, axis), column[1], 1e-5);
    }
    const Net::ParameterVector parameter_gradient = net->ParameterGradient(state);
    for (int index = 0; index < Net::kNumParameters; ++index) {
      Net plus = *net;
      Net minus = *net;
      ASSERT_TRUE(plus.SetParameters(parameters + Net::ParameterVector::Unit(index) * kStep).has_value());
      ASSERT_TRUE(minus.SetParameters(parameters - Net::ParameterVector::Unit(index) * kStep).has_value());
      EXPECT_NEAR(parameter_gradient[index], (plus(state) - minus(state)) / (2 * kStep), 1e-6);
    }
  }
}

}  // namespace
}  // namespace fbsde_traj_opt
