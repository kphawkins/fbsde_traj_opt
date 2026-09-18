// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#include "fbsde_traj_opt/counter_based_normal_sampler.hpp"

#include <cmath>
#include <cstddef>
#include <set>

#include <Eigen/Core>
#include <gtest/gtest.h>

namespace fbsde_traj_opt {
namespace {

TEST(CounterBasedNormalSamplerTest, SameCoordinateAlwaysGivesSameVariate) {
  const CounterBasedNormalSampler sampler(12345);

  EXPECT_DOUBLE_EQ(sampler.Sample(3, 7, 1), sampler.Sample(3, 7, 1));
}

TEST(CounterBasedNormalSamplerTest, TwoSamplersWithTheSameSeedAgreeEverywhere) {
  const CounterBasedNormalSampler first(2026);
  const CounterBasedNormalSampler second(2026);

  for (std::size_t trajectory = 0; trajectory < 8; ++trajectory) {
    for (std::size_t stage = 0; stage < 8; ++stage) {
      for (std::size_t component = 0; component < 4; ++component) {
        EXPECT_DOUBLE_EQ(first.Sample(trajectory, stage, component), second.Sample(trajectory, stage, component));
      }
    }
  }
}

// The property that makes batches comparable across models: a variate depends only on its own
// coordinate, never on how many variates were drawn before it or in what order.
TEST(CounterBasedNormalSamplerTest, VariateIsIndependentOfDrawOrder) {
  const CounterBasedNormalSampler sampler(99);

  const double drawn_directly = sampler.Sample(5, 3, 2);

  double drawn_after_a_sweep = 0.0;
  for (std::size_t trajectory = 0; trajectory < 6; ++trajectory) {
    for (std::size_t stage = 0; stage < 4; ++stage) {
      for (std::size_t component = 0; component < 3; ++component) {
        const double variate = sampler.Sample(trajectory, stage, component);
        if (trajectory == 5 && stage == 3 && component == 2) {
          drawn_after_a_sweep = variate;
        }
      }
    }
  }

  EXPECT_DOUBLE_EQ(drawn_directly, drawn_after_a_sweep);
}

TEST(CounterBasedNormalSamplerTest, DifferentSeedsGiveDifferentVariates) {
  const CounterBasedNormalSampler first(1);
  const CounterBasedNormalSampler second(2);

  EXPECT_NE(first.Sample(0, 0, 0), second.Sample(0, 0, 0));
}

TEST(CounterBasedNormalSamplerTest, DistinctCoordinatesGiveDistinctVariates) {
  const CounterBasedNormalSampler sampler(7);

  std::set<double> variates;
  for (std::size_t trajectory = 0; trajectory < 16; ++trajectory) {
    for (std::size_t stage = 0; stage < 16; ++stage) {
      for (std::size_t component = 0; component < 4; ++component) {
        variates.insert(sampler.Sample(trajectory, stage, component));
      }
    }
  }

  EXPECT_EQ(variates.size(), 16U * 16U * 4U);
}

TEST(CounterBasedNormalSamplerTest, VariatesHaveUnitNormalMomentsInAggregate) {
  const CounterBasedNormalSampler sampler(4242);

  constexpr std::size_t kCount = 200000;
  double sum = 0.0;
  double sum_of_squares = 0.0;
  for (std::size_t index = 0; index < kCount; ++index) {
    const double variate = sampler.Sample(index, index / 97, index % 5);
    sum += variate;
    sum_of_squares += variate * variate;
  }

  const double mean = sum / static_cast<double>(kCount);
  const double variance = (sum_of_squares / static_cast<double>(kCount)) - (mean * mean);

  // The standard error of the mean of kCount unit normals is 1/sqrt(kCount) ~= 0.0022, so a 0.02
  // band is roughly nine standard errors: tight enough to catch a real bias, loose enough never
  // to flake.
  EXPECT_NEAR(mean, 0.0, 0.02);
  EXPECT_NEAR(variance, 1.0, 0.02);
}

// Adjacent coordinates must not be correlated; a weak hash would show up here as a nonzero
// correlation between a trajectory's noise and its neighbour's.
TEST(CounterBasedNormalSamplerTest, AdjacentTrajectoriesAreUncorrelated) {
  const CounterBasedNormalSampler sampler(31337);

  constexpr std::size_t kCount = 100000;
  double sum_of_products = 0.0;
  for (std::size_t index = 0; index < kCount; ++index) {
    sum_of_products += sampler.Sample(index, 0, 0) * sampler.Sample(index + 1, 0, 0);
  }

  EXPECT_NEAR(sum_of_products / static_cast<double>(kCount), 0.0, 0.02);
}

TEST(CounterBasedNormalSamplerTest, FillNoiseMatchesPerComponentSamples) {
  const CounterBasedNormalSampler sampler(555);

  Eigen::Vector3d noise = Eigen::Vector3d::Zero();
  sampler.FillNoise(4, 2, noise);

  EXPECT_DOUBLE_EQ(noise[0], sampler.Sample(4, 2, 0));
  EXPECT_DOUBLE_EQ(noise[1], sampler.Sample(4, 2, 1));
  EXPECT_DOUBLE_EQ(noise[2], sampler.Sample(4, 2, 2));
}

TEST(CounterBasedNormalSamplerTest, FillNoiseNarrowsToTheVectorScalarType) {
  const CounterBasedNormalSampler sampler(555);

  Eigen::Vector3f noise = Eigen::Vector3f::Zero();
  sampler.FillNoise(4, 2, noise);

  EXPECT_FLOAT_EQ(noise[0], static_cast<float>(sampler.Sample(4, 2, 0)));
}

TEST(CounterBasedNormalSamplerTest, EveryVariateIsFinite) {
  const CounterBasedNormalSampler sampler(0);

  for (std::size_t index = 0; index < 50000; ++index) {
    EXPECT_TRUE(std::isfinite(sampler.Sample(index, 0, 0)));
  }
}

TEST(CounterBasedNormalSamplerTest, SeedAccessorReturnsTheConstructorSeed) {
  const CounterBasedNormalSampler sampler(8675309);

  EXPECT_EQ(sampler.Seed(), 8675309U);
}

}  // namespace
}  // namespace fbsde_traj_opt
