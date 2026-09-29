// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#include "fbsde_traj_opt/fbrrt/brute_force_nearest_neighbor.hpp"

#include <cstddef>
#include <limits>

#include <Eigen/Core>
#include <gtest/gtest.h>

namespace fbsde_traj_opt {
namespace {

using Points = Eigen::Matrix<double, 3, 64>;
using Search = BruteForceNearestNeighbor<3>;

static_assert(NearestNeighborSearch<Search, Points, Eigen::Vector3d>);

// The reference answer: a plain loop over the live columns.
auto NaiveNearest(const Points& points, std::size_t count, const Eigen::Vector3d& query, const Eigen::Vector3d& weights)
    -> std::size_t {
  std::size_t best = 0;
  double best_distance = std::numeric_limits<double>::infinity();
  for (std::size_t slot = 0; slot < count; ++slot) {
    const double distance =
        (weights.asDiagonal() * (points.col(static_cast<Eigen::Index>(slot)) - query)).squaredNorm();
    if (distance < best_distance) {
      best_distance = distance;
      best = slot;
    }
  }
  return best;
}

TEST(BruteForceNearestNeighborTest, MatchesANaiveLoopOverTheLiveColumns) {
  const Eigen::Vector3d weights(1.0, 4.0, 0.25);
  const Result<Search> search = Search::Make(weights);
  ASSERT_TRUE(search.has_value());

  for (std::size_t trial = 0; trial < 200; ++trial) {
    const Points points = Points::Random();
    const Eigen::Vector3d query = Eigen::Vector3d::Random();
    const std::size_t count = 1 + (trial % 64);
    EXPECT_EQ((*search)(points, count, query), NaiveNearest(points, count, query, weights));
  }
}

TEST(BruteForceNearestNeighborTest, IgnoresColumnsPastTheLiveCount) {
  const Result<Search> search = Search::Make(Eigen::Vector3d::Ones());
  ASSERT_TRUE(search.has_value());
  Points points = Points::Constant(10.0);
  points.col(40) = Eigen::Vector3d::Zero();
  EXPECT_EQ((*search)(points, 64, Eigen::Vector3d::Zero()), 40U);
  EXPECT_EQ((*search)(points, 40, Eigen::Vector3d::Zero()), 0U);
}

TEST(BruteForceNearestNeighborTest, TheWeightsDecideWhichAxisMatters) {
  Points points = Points::Zero();
  points.col(0) = Eigen::Vector3d(1.0, 0.0, 0.0);
  points.col(1) = Eigen::Vector3d(0.0, 2.0, 0.0);
  // Unweighted, column 0 is nearer the origin; shrinking the second axis makes column 1 nearer.
  EXPECT_EQ((*Search::Make(Eigen::Vector3d::Ones()))(points, 2, Eigen::Vector3d::Zero()), 0U);
  EXPECT_EQ((*Search::Make(Eigen::Vector3d(1.0, 0.1, 1.0)))(points, 2, Eigen::Vector3d::Zero()), 1U);
}

TEST(BruteForceNearestNeighborTest, MakeRejectsNonPositiveOrNonFiniteWeights) {
  EXPECT_FALSE(Search::Make(Eigen::Vector3d(1.0, 0.0, 1.0)).has_value());
  EXPECT_FALSE(Search::Make(Eigen::Vector3d(1.0, -1.0, 1.0)).has_value());
  EXPECT_FALSE(Search::Make(Eigen::Vector3d(1.0, std::numeric_limits<double>::infinity(), 1.0)).has_value());
}

}  // namespace
}  // namespace fbsde_traj_opt
