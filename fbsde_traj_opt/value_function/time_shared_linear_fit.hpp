// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#ifndef FBSDE_TRAJ_OPT_TIME_SHARED_LINEAR_FIT_HPP_
#define FBSDE_TRAJ_OPT_TIME_SHARED_LINEAR_FIT_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/QR>
#include <Eigen/SVD>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>

#include "fbsde_traj_opt/utils/result.hpp"

namespace fbsde_traj_opt {

// A value function table whose stages share their parameters through time.
//
// DT-FBSDE fits one linear-in-parameters model per stage, `V~_k(x) = phi(x)' theta_k`, so a table of
// K stages over a basis of P functions has K P free coefficients, each stage's estimated from that
// stage's few samples alone. This ties the stages together as
//
//   theta_k = U C b(k),       U: P x r,   C: r x R,   b(k): R,
//
// a time basis `b` of R functions of the stage index (TimeBasis) and, optionally, a shared basis of
// r directions in coefficient space. Three cases matter:
//
//   U = I, b = e_k            the per-stage table: K P coefficients, what DT-FBSDE does now
//   U = I, b piecewise linear the coefficients interpolated between R knots in time: R P
//   U of rank r, b = e_k      a basis of r value functions shared by every stage, each stage's own
//                             mix of them: P r + K r
//   U of rank r, b knots      both: P r + R r
//
// The time basis keeps the ability to differ between regimes -- it is local, so two knots far apart
// are independent -- but forces the coefficients to move continuously between knots. The low-rank
// form keeps every stage free, and instead forces all of them into one r-dimensional family of
// functions, which every stage's samples help to find.
//
// Everything works from each stage's normal equations, `G_k = sum w phi phi'` and `h_k = sum w phi y`
// (slope rows included; see LinearBasisValueFunctionApprox::NormalEquations), so the per-stage
// ridge is unchanged: `ridge S_k || theta_k - anchor_k ||^2`, folded into `G_k` and `h_k`. With the
// time basis and no rank the fit is linear least squares in C; with a rank it is bilinear, and is
// solved by alternating least squares, U being kept orthonormal so that the ridge on theta_k is a
// ridge on the stage's mixing weights.
//
// The fit can be re-solved locally (SolveAroundStage): only the knots whose support contains one
// stage, everything else held. That is block Gauss-Seidel on the joint problem, and is what lets a
// backward pass keep its stage-by-stage recursion -- each stage's targets read the stage after it,
// which has just been refitted -- while the parameters are shared.
enum class TimeBasisKind : std::uint8_t { kPerStage, kPiecewiseConstant, kPiecewiseLinear };

// R functions of the stage index 0..K-1, each stage touching at most two of them.
template <typename Scalar = double>
class TimeBasis {
 public:
  struct Term {
    std::size_t knot = 0;
    Scalar weight = Scalar{0};
  };

  // kPerStage ignores `num_knots` (it is K). kPiecewiseConstant splits the stages into `num_knots`
  // blocks of nearly equal length. kPiecewiseLinear places `num_knots` knots evenly from stage 0 to
  // stage K-1 and interpolates linearly between the two around each stage.
  //
  // Fails if there are no stages, or, for the other two kinds, if `num_knots` is below one
  // (constant) or two (linear) or above K.
  static auto Make(TimeBasisKind kind, std::size_t num_stages, std::size_t num_knots) noexcept -> Result<TimeBasis> {
    RESULT_ASSERT(num_stages >= 1, "TimeBasis::Make: there must be at least one stage.");
    if (kind == TimeBasisKind::kPerStage) {
      num_knots = num_stages;
    }
    RESULT_ASSERT(num_knots >= (kind == TimeBasisKind::kPiecewiseLinear ? 2U : 1U) && num_knots <= num_stages,
                  "TimeBasis::Make: the knot count must be between 1 (2 for piecewise linear) and the stage count.");
    TimeBasis basis(kind, num_stages, num_knots);
    for (std::size_t stage = 0; stage < num_stages; ++stage) {
      std::array<Term, 2>& terms = basis.terms_[stage];
      if (kind == TimeBasisKind::kPerStage) {
        terms[0] = {.knot = stage, .weight = Scalar{1}};
        basis.counts_[stage] = 1;
      } else if (kind == TimeBasisKind::kPiecewiseConstant) {
        terms[0] = {.knot = (stage * num_knots) / num_stages, .weight = Scalar{1}};
        basis.counts_[stage] = 1;
      } else {
        const Scalar position = num_stages == 1 ? Scalar{0}
                                                : static_cast<Scalar>(stage) * static_cast<Scalar>(num_knots - 1) /
                                                      static_cast<Scalar>(num_stages - 1);
        const std::size_t left = std::min(static_cast<std::size_t>(std::floor(position)), num_knots - 2);
        const Scalar fraction = position - static_cast<Scalar>(left);
        terms[0] = {.knot = left, .weight = Scalar{1} - fraction};
        terms[1] = {.knot = left + 1, .weight = fraction};
        basis.counts_[stage] = 2;
      }
    }
    return SuccessResult(std::move(basis));
  }

  // The nonzero terms of b(stage).
  [[nodiscard]] auto Terms(std::size_t stage) const noexcept -> std::span<const Term> {
    return {terms_[stage].data(), counts_[stage]};
  }

  [[nodiscard]] auto NumKnots() const noexcept -> std::size_t { return num_knots_; }

  [[nodiscard]] auto NumStages() const noexcept -> std::size_t { return num_stages_; }

  [[nodiscard]] auto Kind() const noexcept -> TimeBasisKind { return kind_; }

 private:
  TimeBasis(TimeBasisKind kind, std::size_t num_stages, std::size_t num_knots) noexcept
      : kind_(kind), num_stages_(num_stages), num_knots_(num_knots), terms_(num_stages), counts_(num_stages, 0) {}

  TimeBasisKind kind_;
  std::size_t num_stages_;
  std::size_t num_knots_;
  std::vector<std::array<Term, 2>> terms_;
  std::vector<std::size_t> counts_;
};

struct TimeSharedFitOptions {
  TimeBasisKind kind = TimeBasisKind::kPerStage;
  // Ignored for kPerStage.
  std::size_t num_knots = 8;
  // The shared basis's rank r; zero is no shared basis (U = I).
  std::size_t rank = 0;
  // Alternating least-squares sweeps of a full Fit() with a rank.
  std::size_t sweeps = 6;
  // Weight each stage by one over its targets' variance; see TimeSharedLinearFit::SetStage. Read by
  // the callers that compute the weights.
  bool balance_stages = false;
};

// The fit itself, for a linear-in-parameters value function type `ValueFunctionT` whose
// NormalEquations it reads (LinearBasisValueFunctionApprox). Holds one stage's regularized normal
// equations per stage, the shared basis U, and the coefficients C.
template <typename ValueFunctionT>
class TimeSharedLinearFit {
 public:
  using Scalar = typename ValueFunctionT::ParameterVector::Scalar;
  using ParameterVector = typename ValueFunctionT::ParameterVector;
  using NormalEquations = typename ValueFunctionT::NormalEquations;
  using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

  static constexpr auto kNumFeatures = static_cast<std::size_t>(ValueFunctionT::kNumParameters);

  // Fails if the time basis cannot be built, or if the rank exceeds the feature count.
  static auto Make(const TimeSharedFitOptions& options,
                   std::size_t num_stages) noexcept -> Result<TimeSharedLinearFit> {
    RESULT_ASSIGN_OR_RETURN(TimeBasis<Scalar> basis,
                            TimeBasis<Scalar>::Make(options.kind, num_stages, options.num_knots));
    RESULT_ASSERT(options.rank <= kNumFeatures, "TimeSharedLinearFit::Make: the rank cannot exceed the feature count.");
    return SuccessResult(TimeSharedLinearFit(options, std::move(basis)));
  }

  // Free parameters of the shared table: R P without a rank, P r + R r with one.
  [[nodiscard]] auto NumParameters() const noexcept -> std::size_t {
    const std::size_t knots = basis_.NumKnots();
    return options_.rank == 0 ? knots * kNumFeatures : (kNumFeatures * options_.rank) + (knots * options_.rank);
  }

  [[nodiscard]] auto Basis() const noexcept -> const TimeBasis<Scalar>& { return basis_; }

  [[nodiscard]] auto Rank() const noexcept -> std::size_t { return options_.rank; }

  [[nodiscard]] auto HasSharedBasis() const noexcept -> bool { return options_.rank > 0 && initialized_; }

  // Replaces stage `stage`'s data with `equations` under the per-sample ridge `ridge`, anchored at
  // `anchor` (zero when null): G_k + ridge S_k I and h_k + ridge S_k anchor, all times
  // `stage_weight`. The weight leaves a stage's own least-squares solution alone and sets how much
  // the stage counts against the others wherever they share parameters: the targets of a swing-up's
  // early stages run to hundreds and its last stages' to a few units, and unweighted, a shared basis
  // is spent on the former. One over the stage's target variance gives every stage an equal say.
  auto SetStage(std::size_t stage,
                const NormalEquations& equations,
                Scalar ridge,
                const ParameterVector* anchor = nullptr,
                Scalar stage_weight = Scalar{1}) noexcept -> Result<> {
    RESULT_ASSERT(stage < basis_.NumStages(), "TimeSharedLinearFit::SetStage: the stage is out of range.");
    RESULT_ASSERT(std::isfinite(ridge) && ridge >= Scalar{0},
                  "TimeSharedLinearFit::SetStage: the ridge weight must be finite and non-negative.");
    RESULT_ASSERT(std::isfinite(stage_weight) && stage_weight > Scalar{0},
                  "TimeSharedLinearFit::SetStage: the stage weight must be finite and positive.");
    const Scalar weight = ridge * equations.samples;
    Matrix matrix = equations.matrix.template selfadjointView<Eigen::Lower>();
    matrix.diagonal().array() += weight;
    Vector vector = equations.vector;
    if (anchor != nullptr) {
      vector += weight * (*anchor);
    }
    matrices_[stage] = stage_weight * matrix;
    vectors_[stage] = stage_weight * vector;
    has_data_[stage] = true;
    fresh_[stage] = true;
    Project(stage);
    return SuccessResult();
  }

  // A full fit on the stored data: for a rank, the basis from the per-stage solutions' leading
  // singular vectors (first time only), then `sweeps` alternations of coefficients and basis; and a
  // final coefficient solve. Every stage must have data.
  auto Fit() noexcept -> Result<> {
    RESULT_ASSERT(std::ranges::all_of(has_data_, [](bool has) { return has; }),
                  "TimeSharedLinearFit::Fit: every stage needs data.");
    if (options_.rank > 0) {
      if (!initialized_) {
        RESULT_RETURN_IF_ERROR(InitializeBasis());
      }
      for (std::size_t sweep = 0; sweep < options_.sweeps; ++sweep) {
        RESULT_RETURN_IF_ERROR(SolveCoefficients());
        RESULT_RETURN_IF_ERROR(SolveSharedBasis());
      }
    }
    return SolveCoefficients();
  }

  // Marks every stage's data stale: a pass about to replace them all begins.
  auto BeginPass() noexcept -> void { std::ranges::fill(fresh_, false); }

  // Re-solves the coefficients of the knots stage `stage` touches, holding the basis and every
  // other knot. Stages without data contribute nothing, and stages whose data predate the last
  // BeginPass() contribute them scaled by `stale_weight`: one counts them fully, zero drops them.
  auto SolveAroundStage(std::size_t stage, Scalar stale_weight = Scalar{1}) noexcept -> Result<> {
    std::vector<std::size_t> knots;
    for (const auto& term : basis_.Terms(stage)) {
      if (term.weight != Scalar{0}) {
        knots.push_back(term.knot);
      }
    }
    return SolveKnots(knots, stale_weight);
  }

  // Solves every knot's coefficients jointly, the basis held.
  auto SolveCoefficients() noexcept -> Result<> {
    std::vector<std::size_t> knots(basis_.NumKnots());
    for (std::size_t knot = 0; knot < knots.size(); ++knot) {
      knots[knot] = knot;
    }
    return SolveKnots(knots);
  }

  // Solves the shared basis U with the coefficients held, then re-orthonormalizes it, carrying the
  // triangular factor into C so that every stage's parameters are unchanged by the rotation.
  auto SolveSharedBasis() noexcept -> Result<> {
    const std::size_t rank = options_.rank;
    if (rank == 0) {
      return SuccessResult();
    }
    const auto p = static_cast<Eigen::Index>(kNumFeatures);
    const auto r = static_cast<Eigen::Index>(rank);
    Matrix normal = Matrix::Zero(p * r, p * r);
    Vector rhs = Vector::Zero(p * r);
    for (std::size_t stage = 0; stage < basis_.NumStages(); ++stage) {
      if (!has_data_[stage]) {
        continue;
      }
      const Vector mix = Mix(stage);
      for (Eigen::Index i = 0; i < r; ++i) {
        rhs.segment(i * p, p) += mix[i] * vectors_[stage];
        for (Eigen::Index j = 0; j <= i; ++j) {
          normal.block(i * p, j * p, p, p) += (mix[i] * mix[j]) * matrices_[stage];
        }
      }
    }
    // A direction no stage uses is undetermined; a whisper of ridge keeps it where it was.
    const Scalar floor = Scalar{1e-10} * (normal.diagonal().cwiseAbs().maxCoeff() + Scalar{1});
    normal.diagonal().array() += floor;
    for (Eigen::Index i = 0; i < r; ++i) {
      rhs.segment(i * p, p) += floor * shared_.col(i);
    }
    const Eigen::LLT<Matrix> factorization(normal.template selfadjointView<Eigen::Lower>());
    RESULT_ASSERT(factorization.info() == Eigen::Success,
                  "TimeSharedLinearFit::SolveSharedBasis: the normal equations are not positive definite.");
    const Vector solution = factorization.solve(rhs);
    RESULT_ASSERT(solution.allFinite(), "TimeSharedLinearFit::SolveSharedBasis: the fit is not finite.");
    const Matrix shared = Eigen::Map<const Matrix>(solution.data(), p, r);
    const Eigen::HouseholderQR<Matrix> qr(shared);
    Matrix q = qr.householderQ() * Matrix::Identity(p, r);
    Matrix upper = qr.matrixQR().topRows(r).template triangularView<Eigen::Upper>();
    shared_ = std::move(q);
    coefficients_ = upper * coefficients_;
    for (std::size_t stage = 0; stage < basis_.NumStages(); ++stage) {
      if (has_data_[stage]) {
        Project(stage);
      }
    }
    return SuccessResult();
  }

  // theta_k = U C b(k).
  [[nodiscard]] auto StageParameters(std::size_t stage) const noexcept -> ParameterVector {
    const Vector mix = Mix(stage);
    if (options_.rank == 0) {
      return ParameterVector(mix);
    }
    return ParameterVector(shared_ * mix);
  }

  // Writes theta_k into table[k] for every stage the basis covers.
  auto WriteTable(std::span<ValueFunctionT> table) const noexcept -> Result<> {
    RESULT_ASSERT(table.size() >= basis_.NumStages(), "TimeSharedLinearFit::WriteTable: the table is too short.");
    for (std::size_t stage = 0; stage < basis_.NumStages(); ++stage) {
      RESULT_RETURN_IF_ERROR(table[stage].SetParameters(StageParameters(stage)));
    }
    return SuccessResult();
  }

  // Writes theta for the stages whose parameters a local solve around `stage` changed: every stage
  // touching a knot that `stage` touches.
  auto WriteAroundStage(std::size_t stage, std::span<ValueFunctionT> table) const noexcept -> Result<> {
    for (const std::size_t other : StagesAround(stage)) {
      RESULT_RETURN_IF_ERROR(table[other].SetParameters(StageParameters(other)));
    }
    return SuccessResult();
  }

  // Sets the coefficients so that the table reproduces `table` as closely as the parameterization
  // allows in plain coefficient space (no data): the starting point of a fit.
  auto Initialize(std::span<const ValueFunctionT> table) noexcept -> Result<> {
    const std::size_t stages = basis_.NumStages();
    RESULT_ASSERT(table.size() >= stages, "TimeSharedLinearFit::Initialize: the table is too short.");
    Matrix thetas(static_cast<Eigen::Index>(kNumFeatures), static_cast<Eigen::Index>(stages));
    for (std::size_t stage = 0; stage < stages; ++stage) {
      thetas.col(static_cast<Eigen::Index>(stage)) = table[stage].Parameters();
    }
    if (options_.rank > 0) {
      SetSharedBasisFromColumns(thetas);
    }
    // Least squares of C against the thetas, in the basis's coordinates.
    const Matrix coordinates = options_.rank > 0 ? Matrix(shared_.transpose() * thetas) : thetas;
    Matrix gram =
        Matrix::Zero(static_cast<Eigen::Index>(basis_.NumKnots()), static_cast<Eigen::Index>(basis_.NumKnots()));
    Matrix cross = Matrix::Zero(coordinates.rows(), static_cast<Eigen::Index>(basis_.NumKnots()));
    for (std::size_t stage = 0; stage < stages; ++stage) {
      for (const auto& a : basis_.Terms(stage)) {
        cross.col(static_cast<Eigen::Index>(a.knot)) += a.weight * coordinates.col(static_cast<Eigen::Index>(stage));
        for (const auto& b : basis_.Terms(stage)) {
          gram(static_cast<Eigen::Index>(a.knot), static_cast<Eigen::Index>(b.knot)) += a.weight * b.weight;
        }
      }
    }
    gram.diagonal().array() += Scalar{1e-12};
    coefficients_ = Eigen::LDLT<Matrix>(gram).solve(cross.transpose()).transpose();
    return SuccessResult();
  }

 private:
  TimeSharedLinearFit(const TimeSharedFitOptions& options, TimeBasis<Scalar> basis) noexcept
      : options_(options),
        basis_(std::move(basis)),
        matrices_(basis_.NumStages()),
        vectors_(basis_.NumStages()),
        projected_matrices_(basis_.NumStages()),
        projected_vectors_(basis_.NumStages()),
        has_data_(basis_.NumStages(), false),
        fresh_(basis_.NumStages(), false),
        coefficients_(Matrix::Zero(static_cast<Eigen::Index>(Width()), static_cast<Eigen::Index>(basis_.NumKnots()))),
        shared_(Matrix::Identity(static_cast<Eigen::Index>(kNumFeatures), static_cast<Eigen::Index>(Width()))) {}

  // The coefficient width per knot: r with a shared basis, P without.
  [[nodiscard]] auto Width() const noexcept -> std::size_t { return options_.rank > 0 ? options_.rank : kNumFeatures; }

  // C b(k): the stage's coordinates in the shared basis (or its parameters, without one).
  [[nodiscard]] auto Mix(std::size_t stage) const noexcept -> Vector {
    Vector mix = Vector::Zero(coefficients_.rows());
    for (const auto& term : basis_.Terms(stage)) {
      mix += term.weight * coefficients_.col(static_cast<Eigen::Index>(term.knot));
    }
    return mix;
  }

  // The stage's equations in the shared basis's coordinates: U' G U and U' h.
  auto Project(std::size_t stage) noexcept -> void {
    if (options_.rank == 0) {
      return;
    }
    projected_matrices_[stage] = shared_.transpose() * matrices_[stage] * shared_;
    projected_vectors_[stage] = shared_.transpose() * vectors_[stage];
  }

  [[nodiscard]] auto ProjectedMatrix(std::size_t stage) const noexcept -> const Matrix& {
    return options_.rank > 0 ? projected_matrices_[stage] : matrices_[stage];
  }

  [[nodiscard]] auto ProjectedVector(std::size_t stage) const noexcept -> const Vector& {
    return options_.rank > 0 ? projected_vectors_[stage] : vectors_[stage];
  }

  // Every stage touching one of the knots `stage` touches.
  [[nodiscard]] auto StagesAround(std::size_t stage) const noexcept -> std::vector<std::size_t> {
    if (basis_.Kind() == TimeBasisKind::kPerStage) {
      return {stage};
    }
    std::vector<std::size_t> stages;
    for (std::size_t other = 0; other < basis_.NumStages(); ++other) {
      bool shares = false;
      for (const auto& a : basis_.Terms(other)) {
        for (const auto& b : basis_.Terms(stage)) {
          shares = shares || (a.knot == b.knot && a.weight != Scalar{0} && b.weight != Scalar{0});
        }
      }
      if (shares) {
        stages.push_back(other);
      }
    }
    return stages;
  }

  // Block least squares over the coefficients of `knots`, everything else held:
  //   sum_k [b_t(k) b_t'(k) Gp_k] C_t' = sum_k b_t(k) (hp_k - Gp_k * held part of C b(k)).
  // Each stage couples only the (at most two) knots it touches, so the system is block-banded; it
  // is assembled and factored sparse, which keeps a joint solve over many knots cheap.
  auto SolveKnots(const std::vector<std::size_t>& knots, Scalar stale_weight = Scalar{1}) noexcept -> Result<> {
    const Eigen::Index width = coefficients_.rows();
    const auto count = static_cast<Eigen::Index>(knots.size());
    std::vector<Eigen::Index> slot(basis_.NumKnots(), -1);
    for (Eigen::Index index = 0; index < count; ++index) {
      slot[knots[static_cast<std::size_t>(index)]] = index;
    }
    std::vector<Eigen::Triplet<Scalar>> entries;
    Vector rhs = Vector::Zero(width * count);
    Vector diagonal = Vector::Zero(width * count);
    for (std::size_t stage = 0; stage < basis_.NumStages(); ++stage) {
      const Scalar stage_scale = fresh_[stage] ? Scalar{1} : stale_weight;
      if (!has_data_[stage] || stage_scale == Scalar{0}) {
        continue;
      }
      const std::span<const typename TimeBasis<Scalar>::Term> terms = basis_.Terms(stage);
      Vector held = Vector::Zero(width);
      bool touches = false;
      for (const auto& term : terms) {
        if (slot[term.knot] < 0) {
          held += term.weight * coefficients_.col(static_cast<Eigen::Index>(term.knot));
        } else {
          touches = touches || term.weight != Scalar{0};
        }
      }
      if (!touches) {
        continue;
      }
      const Matrix matrix = stage_scale * ProjectedMatrix(stage);
      const Vector residual = (stage_scale * ProjectedVector(stage)) - (matrix * held);
      for (const auto& a : terms) {
        const Eigen::Index i = slot[a.knot];
        if (i < 0 || a.weight == Scalar{0}) {
          continue;
        }
        rhs.segment(i * width, width) += a.weight * residual;
        diagonal.segment(i * width, width) += (a.weight * a.weight) * matrix.diagonal();
        for (const auto& b : terms) {
          const Eigen::Index j = slot[b.knot];
          if (j < 0 || b.weight == Scalar{0}) {
            continue;
          }
          const Scalar weight = a.weight * b.weight;
          for (Eigen::Index column = 0; column < width; ++column) {
            for (Eigen::Index row = 0; row < width; ++row) {
              entries.emplace_back((i * width) + row, (j * width) + column, weight * matrix(row, column));
            }
          }
        }
      }
    }
    if (entries.empty()) {
      return SuccessResult();
    }
    // A whisper of ridge toward the current coefficients, relative to each row's own scale (stages can
    // be weighted very differently), so that a direction no data excites keeps its coefficient.
    const Scalar smallest = Scalar{1e-14} * (diagonal.cwiseAbs().maxCoeff() + Scalar{1});
    const Vector floor = (Scalar{1e-12} * diagonal.cwiseAbs()).array() + smallest;
    for (Eigen::Index index = 0; index < width * count; ++index) {
      entries.emplace_back(index, index, floor[index]);
    }
    for (Eigen::Index index = 0; index < count; ++index) {
      rhs.segment(index * width, width) +=
          floor.segment(index * width, width)
              .cwiseProduct(coefficients_.col(static_cast<Eigen::Index>(knots[static_cast<std::size_t>(index)])));
    }
    Eigen::SparseMatrix<Scalar> normal(width * count, width * count);
    normal.setFromTriplets(entries.begin(), entries.end());
    const Eigen::SimplicialLDLT<Eigen::SparseMatrix<Scalar>> factorization(normal);
    RESULT_ASSERT(factorization.info() == Eigen::Success,
                  "TimeSharedLinearFit::SolveKnots: the normal equations could not be factored.");
    const Vector solution = factorization.solve(rhs);
    RESULT_ASSERT(solution.allFinite(), "TimeSharedLinearFit::SolveKnots: the fit is not finite.");
    for (Eigen::Index index = 0; index < count; ++index) {
      coefficients_.col(static_cast<Eigen::Index>(knots[static_cast<std::size_t>(index)])) =
          solution.segment(index * width, width);
    }
    return SuccessResult();
  }

  // The shared basis from the per-stage ridge solutions: their r leading left singular vectors.
  auto InitializeBasis() noexcept -> Result<> {
    const std::size_t stages = basis_.NumStages();
    Matrix thetas(static_cast<Eigen::Index>(kNumFeatures), static_cast<Eigen::Index>(stages));
    for (std::size_t stage = 0; stage < stages; ++stage) {
      const Eigen::LDLT<Matrix> factorization(matrices_[stage]);
      RESULT_ASSERT(factorization.info() == Eigen::Success,
                    "TimeSharedLinearFit::InitializeBasis: a stage's normal equations could not be factored.");
      thetas.col(static_cast<Eigen::Index>(stage)) = factorization.solve(vectors_[stage]);
    }
    RESULT_ASSERT(thetas.allFinite(), "TimeSharedLinearFit::InitializeBasis: a stage's solution is not finite.");
    SetSharedBasisFromColumns(thetas);
    return SuccessResult();
  }

  auto SetSharedBasisFromColumns(const Matrix& thetas) noexcept -> void {
    const Eigen::JacobiSVD<Matrix, Eigen::ComputeThinU> svd(thetas);
    const auto r = static_cast<Eigen::Index>(options_.rank);
    const Eigen::Index available = std::min<Eigen::Index>(r, svd.matrixU().cols());
    Matrix shared = Matrix::Zero(static_cast<Eigen::Index>(kNumFeatures), r);
    shared.leftCols(available) = svd.matrixU().leftCols(available);
    if (available < r) {
      // Fewer stages than the rank: complete the basis with an orthonormal complement.
      const Eigen::HouseholderQR<Matrix> qr(
          (Matrix(shared.rows(), r) << shared.leftCols(available), Matrix::Identity(shared.rows(), r - available))
              .finished());
      shared = qr.householderQ() * Matrix::Identity(shared.rows(), r);
    }
    shared_ = std::move(shared);
    initialized_ = true;
    for (std::size_t stage = 0; stage < basis_.NumStages(); ++stage) {
      if (has_data_[stage]) {
        Project(stage);
      }
    }
  }

  TimeSharedFitOptions options_;
  TimeBasis<Scalar> basis_;
  std::vector<Matrix> matrices_;
  std::vector<Vector> vectors_;
  std::vector<Matrix> projected_matrices_;
  std::vector<Vector> projected_vectors_;
  std::vector<bool> has_data_;
  std::vector<bool> fresh_;
  // C: one column per knot.
  Matrix coefficients_;
  // U: orthonormal columns; the identity without a rank.
  Matrix shared_;
  bool initialized_ = false;
};

}  // namespace fbsde_traj_opt

#endif  // FBSDE_TRAJ_OPT_TIME_SHARED_LINEAR_FIT_HPP_
