// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#ifndef FBSDE_TRAJ_OPT_SAMPLING_SIGMA_POINT_RULES_HPP_
#define FBSDE_TRAJ_OPT_SAMPLING_SIGMA_POINT_RULES_HPP_

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <Eigen/Core>

namespace fbsde_traj_opt {

// Deterministic point sets for the standard normal distribution in N dimensions: points `xi_j` and
// weights `w_j`, summing to one, with
//
//   E[g(xi)] ~ sum_j w_j g(xi_j),    xi ~ N(0, I),
//
// exact for every polynomial `g` up to the rule's degree. A Gaussian N(m, L L') is represented by
// the points `m + L xi_j` with the same weights. These are the point sets of the sigma-point
// (unscented, cubature, quadrature) filters, used here to stand in for a sampled batch.
//
//   kUnscented   Julier and Uhlmann's 2N+1 points: the center and +-sqrt(N + kappa) along each
//                axis, kappa = max(3 - N, 0). Degree 3, and the fourth marginal moment as well
//                when N <= 3.
//   kCubature5   The fifth-degree spherical-radial rule of the high-degree cubature Kalman filter
//                (Jia, Xin and Cheng, Automatica 2013): the center, +-sqrt(N + 2) along each axis,
//                and +-sqrt((N + 2) / 2) along every pair of axes, 2N^2 + 1 points. Degree 5;
//                weights stay non-negative for N <= 4.
//   kConjugate   A symmetric set in the spirit of the conjugate unscented transform (Adurthi,
//                Singla and Singh, ASME JDSMC 2018): the center, +-a along each axis, and every
//                corner b(+-1, ..., +-1), 2N + 2^N + 1 points. Its weights and radii are derived
//                here, not taken from the paper: they match every moment through degree five and
//                the sixth marginal moment E[xi_k^6] = 15, which fixes b^2 = 6 - sqrt(21).
//   kGaussHermite3, kGaussHermite5
//                Tensor products of the 3- and 5-point Gauss-Hermite rules: 3^N and 5^N points,
//                exact to degree 5 and 9 in each coordinate.
enum class SigmaPointRule : std::uint8_t { kNone, kUnscented, kCubature5, kConjugate, kGaussHermite3, kGaussHermite5 };

template <int N, typename Scalar = double>
struct SigmaPointSet {
  using Point = Eigen::Matrix<Scalar, N, 1>;
  std::vector<Point> points;
  std::vector<Scalar> weights;
};

namespace sigma_point_detail {

template <int N, typename Scalar>
auto TensorGaussHermite(const std::vector<Scalar>& nodes, const std::vector<Scalar>& weights)
    -> SigmaPointSet<N, Scalar> {
  SigmaPointSet<N, Scalar> set;
  std::size_t total = 1;
  for (int axis = 0; axis < N; ++axis) {
    total *= nodes.size();
  }
  for (std::size_t index = 0; index < total; ++index) {
    typename SigmaPointSet<N, Scalar>::Point point;
    Scalar weight = Scalar{1};
    std::size_t rest = index;
    for (int axis = 0; axis < N; ++axis) {
      point[axis] = nodes[rest % nodes.size()];
      weight *= weights[rest % nodes.size()];
      rest /= nodes.size();
    }
    set.points.push_back(point);
    set.weights.push_back(weight);
  }
  return set;
}

}  // namespace sigma_point_detail

// The rule's points and weights; empty for kNone.
template <int N, typename Scalar = double>
auto MakeSigmaPoints(SigmaPointRule rule) -> SigmaPointSet<N, Scalar> {
  using Point = typename SigmaPointSet<N, Scalar>::Point;
  SigmaPointSet<N, Scalar> set;
  const auto n = static_cast<Scalar>(N);
  switch (rule) {
    case SigmaPointRule::kNone:
      break;
    case SigmaPointRule::kUnscented: {
      const Scalar kappa = std::max(Scalar{3} - n, Scalar{0});
      const Scalar radius = std::sqrt(n + kappa);
      set.points.push_back(Point::Zero());
      set.weights.push_back(kappa / (n + kappa));
      for (int axis = 0; axis < N; ++axis) {
        for (const Scalar sign : {Scalar{1}, Scalar{-1}}) {
          set.points.push_back(sign * radius * Point::Unit(axis));
          set.weights.push_back(Scalar{1} / (Scalar{2} * (n + kappa)));
        }
      }
      break;
    }
    case SigmaPointRule::kCubature5: {
      const Scalar radius = std::sqrt(n + Scalar{2});
      set.points.push_back(Point::Zero());
      set.weights.push_back(Scalar{2} / (n + Scalar{2}));
      for (int axis = 0; axis < N; ++axis) {
        for (const Scalar sign : {Scalar{1}, Scalar{-1}}) {
          set.points.push_back(sign * radius * Point::Unit(axis));
          set.weights.push_back((Scalar{4} - n) / (Scalar{2} * (n + Scalar{2}) * (n + Scalar{2})));
        }
      }
      const Scalar pair_radius = radius / std::sqrt(Scalar{2});
      for (int first = 0; first < N; ++first) {
        for (int second = first + 1; second < N; ++second) {
          for (const Scalar first_sign : {Scalar{1}, Scalar{-1}}) {
            for (const Scalar second_sign : {Scalar{1}, Scalar{-1}}) {
              set.points.push_back(pair_radius * ((first_sign * Point::Unit(first)) + (second_sign * Point::Unit(second))));
              set.weights.push_back(Scalar{1} / ((n + Scalar{2}) * (n + Scalar{2})));
            }
          }
        }
      }
      break;
    }
    case SigmaPointRule::kConjugate: {
      // Corners b(+-1, ...) weighted w2, axes +-a weighted w1, center w0. With s = b^2:
      //   corners alone carry E[xi_i^2 xi_j^2] = 1:  2^N s^2 w2 = 1;
      //   E[xi_i^2] = 1 and E[xi_i^4] = 3 then fix a^2 = 2s / (s - 1), w1 = (s - 1)^2 / (4 s^2);
      //   E[xi_i^6] = 15 gives s^2 - 12 s + 15 = 0, of which the root below 3 keeps a real.
      const Scalar s = Scalar{6} - std::sqrt(Scalar{21});
      const Scalar corner = std::sqrt(s);
      const Scalar axis_radius = std::sqrt(Scalar{2} * s / (s - Scalar{1}));
      const Scalar corner_weight = Scalar{1} / (std::pow(Scalar{2}, n) * s * s);
      const Scalar axis_weight = (s - Scalar{1}) * (s - Scalar{1}) / (Scalar{4} * s * s);
      set.points.push_back(Point::Zero());
      set.weights.push_back(Scalar{1} - (Scalar{2} * n * axis_weight) - (Scalar{1} / (s * s)));
      for (int axis = 0; axis < N; ++axis) {
        for (const Scalar sign : {Scalar{1}, Scalar{-1}}) {
          set.points.push_back(sign * axis_radius * Point::Unit(axis));
          set.weights.push_back(axis_weight);
        }
      }
      for (int mask = 0; mask < (1 << N); ++mask) {
        Point point;
        for (int axis = 0; axis < N; ++axis) {
          point[axis] = ((mask >> axis) & 1) != 0 ? corner : -corner;
        }
        set.points.push_back(point);
        set.weights.push_back(corner_weight);
      }
      break;
    }
    case SigmaPointRule::kGaussHermite3:
      set = sigma_point_detail::TensorGaussHermite<N, Scalar>(
          {-std::sqrt(Scalar{3}), Scalar{0}, std::sqrt(Scalar{3})},
          {Scalar{1} / Scalar{6}, Scalar{2} / Scalar{3}, Scalar{1} / Scalar{6}});
      break;
    case SigmaPointRule::kGaussHermite5: {
      const Scalar outer = std::sqrt(Scalar{5} + std::sqrt(Scalar{10}));
      const Scalar inner = std::sqrt(Scalar{5} - std::sqrt(Scalar{10}));
      const Scalar outer_weight = (Scalar{7} - (Scalar{2} * std::sqrt(Scalar{10}))) / Scalar{60};
      const Scalar inner_weight = (Scalar{7} + (Scalar{2} * std::sqrt(Scalar{10}))) / Scalar{60};
      set = sigma_point_detail::TensorGaussHermite<N, Scalar>(
          {-outer, -inner, Scalar{0}, inner, outer},
          {outer_weight, inner_weight, Scalar{8} / Scalar{15}, inner_weight, outer_weight});
      break;
    }
  }
  return set;
}

}  // namespace fbsde_traj_opt

#endif  // FBSDE_TRAJ_OPT_SAMPLING_SIGMA_POINT_RULES_HPP_
