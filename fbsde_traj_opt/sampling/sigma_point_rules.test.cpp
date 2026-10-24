// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#include "fbsde_traj_opt/sampling/sigma_point_rules.hpp"

#include <cmath>
#include <cstddef>

#include <Eigen/Core>
#include <gtest/gtest.h>

namespace fbsde_traj_opt {
namespace {

// The weighted moment E[prod_k xi_k^{e_k}] of a rule.
template <int N>
auto Moment(const SigmaPointSet<N>& set, const std::array<int, N>& exponents) -> double {
  double total = 0.0;
  for (std::size_t index = 0; index < set.points.size(); ++index) {
    double term = set.weights[index];
    for (int axis = 0; axis < N; ++axis) {
      term *= std::pow(set.points[index][axis], exponents[static_cast<std::size_t>(axis)]);
    }
    total += term;
  }
  return total;
}

// The standard normal's moment: the product of (e - 1)!! over even exponents, zero if any is odd.
template <int N>
auto GaussianMoment(const std::array<int, N>& exponents) -> double {
  double moment = 1.0;
  for (const int e : exponents) {
    if (e % 2 == 1) {
      return 0.0;
    }
    for (int k = e - 1; k > 0; k -= 2) {
      moment *= k;
    }
  }
  return moment;
}

// Every moment of total degree up to `degree` matches the Gaussian's.
template <int N>
auto ExpectExactToDegree(SigmaPointRule rule, int degree, std::size_t expected_count) -> void {
  const SigmaPointSet<N> set = MakeSigmaPoints<N>(rule);
  EXPECT_EQ(set.points.size(), expected_count);
  std::array<int, N> exponents{};
  const auto check = [&](auto&& self, int axis, int remaining) -> void {
    if (axis == N) {
      EXPECT_NEAR(Moment<N>(set, exponents), GaussianMoment<N>(exponents), 1e-9);
      return;
    }
    for (int power = 0; power <= remaining; ++power) {
      exponents[static_cast<std::size_t>(axis)] = power;
      self(self, axis + 1, remaining - power);
    }
  };
  check(check, 0, degree);
}

TEST(SigmaPointRulesTest, EachRuleIsExactToItsDegree) {
  ExpectExactToDegree<2>(SigmaPointRule::kUnscented, 3, 5);
  ExpectExactToDegree<4>(SigmaPointRule::kUnscented, 3, 9);
  ExpectExactToDegree<2>(SigmaPointRule::kCubature5, 5, 9);
  ExpectExactToDegree<4>(SigmaPointRule::kCubature5, 5, 33);
  ExpectExactToDegree<2>(SigmaPointRule::kConjugate, 5, 9);
  ExpectExactToDegree<4>(SigmaPointRule::kConjugate, 5, 25);
  ExpectExactToDegree<2>(SigmaPointRule::kGaussHermite3, 5, 9);
  ExpectExactToDegree<2>(SigmaPointRule::kGaussHermite5, 9, 25);
}

TEST(SigmaPointRulesTest, ConjugateSetMatchesTheSixthMarginalMoment) {
  const SigmaPointSet<3> set = MakeSigmaPoints<3>(SigmaPointRule::kConjugate);
  EXPECT_NEAR(Moment<3>(set, {6, 0, 0}), 15.0, 1e-9);
  for (const double weight : set.weights) {
    EXPECT_GE(weight, 0.0);
  }
}

}  // namespace
}  // namespace fbsde_traj_opt
