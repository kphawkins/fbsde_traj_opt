// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#ifndef FBSDE_TRAJ_OPT_FBRRT_BRUTE_FORCE_NEAREST_NEIGHBOR_HPP_
#define FBSDE_TRAJ_OPT_FBRRT_BRUTE_FORCE_NEAREST_NEIGHBOR_HPP_

#include <concepts>
#include <cstddef>
#include <utility>

#include <Eigen/Core>

#include "fbsde_traj_opt/utils/result.hpp"

namespace fbsde_traj_opt {

// Concept for a nearest-neighbor search over the live columns of a stage's particle block: given
// the block, how many leading columns are live, and a query state, return the slot of the live
// particle closest to the query.
template <typename T, typename StateBlock, typename State>
concept NearestNeighborSearch =
    requires(const T& search, const StateBlock& points, std::size_t count, const State& query) {
      { search(points, count, query) } -> std::same_as<std::size_t>;
    };

// Nearest neighbor by exhaustive search, under the diagonally weighted distance
// `|| diag(inverse_scale) (x - query) ||`.
//
// A k-d tree would answer in logarithmic rather than linear time, but FBRRT keeps the width of
// every stage at a fixed `M` from one iteration to the next -- erosion removes as many particles
// as expansion adds -- so the linear cost is a constant per draw, paid as one vectorized pass over
// a contiguous block. The weighting puts dimensions with different units on a common footing;
// the natural choice is the inverse half-width of the region the RRT draws its targets from.
template <int N, typename Scalar = double>
class BruteForceNearestNeighbor {
 public:
  using State = Eigen::Matrix<Scalar, N, 1>;

  // Fails unless every weight is finite and positive.
  static auto Make(const State& inverse_scale) noexcept -> Result<BruteForceNearestNeighbor> {
    RESULT_ASSERT(inverse_scale.allFinite() && (inverse_scale.array() > Scalar{0}).all(),
                  "BruteForceNearestNeighbor::Make: every metric weight must be finite and positive.");
    return SuccessResult(BruteForceNearestNeighbor(inverse_scale));
  }

  // Returns the slot among the first `count` columns of `points` nearest to `query`; ties go to the
  // lowest slot.
  //
  // Precondition: `1 <= count <= points.cols()`.
  template <int Columns>
  auto operator()(const Eigen::Matrix<Scalar, N, Columns>& points, std::size_t count, const State& query) const noexcept
      -> std::size_t {
    const auto live = static_cast<Eigen::Index>(count);
    Eigen::Index nearest = 0;
    ((points.leftCols(live).colwise() - query).array().colwise() * inverse_scale_.array())
        .square()
        .colwise()
        .sum()
        .minCoeff(&nearest);
    return static_cast<std::size_t>(nearest);
  }

  [[nodiscard]] auto inverse_scale() const noexcept -> const State& { return inverse_scale_; }

 private:
  explicit BruteForceNearestNeighbor(State inverse_scale) noexcept : inverse_scale_(std::move(inverse_scale)) {}

  State inverse_scale_;
};

}  // namespace fbsde_traj_opt

#endif  // FBSDE_TRAJ_OPT_FBRRT_BRUTE_FORCE_NEAREST_NEIGHBOR_HPP_
