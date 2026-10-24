// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

// Which value-function model should DT-FBSDE use? Two experiments on the L1 benchmarks
// (l1_benchmark_models.hpp), over a menu of models named on the command line:
//
//   oracle   On the two 2-D benchmarks, fit each model at every stage to the grid dynamic-programming
//            value function (l1_grid_dp_reference.hpp) -- values and slopes along the control
//            column, on states drawn the way DT-FBSDE draws them -- and score the Taylor Q-function
//            policy it induces on the held-out rollouts. The ceiling a model puts on DT-FBSDE when
//            the data are perfect, and what one policy evaluation costs online.
//   policy   On the 2-D benchmarks, score policies built from value functions: the Taylor Q-function
//            policy and an exact-expectation minimization, on the grid's own value function (with
//            finite-difference derivatives), on each model's oracle fit, and on the tables DT-FBSDE
//            itself fits in a budget of charged time -- which value function, and which policy rule,
//            the cost comes from.
//   dagger   On the 2-D benchmarks, the oracle refitted DAgger-style: fit to the grid on the optimal
//            policy's states, then repeatedly add the states the fitted policy itself visits, labeled
//            by the grid, and refit. Keys: rounds (default 6), probes (1 adds DT-FBSDE-style probes).
//   dt       Run DT-FBSDE with each model for a budget of charged solver time, over seeds, scoring
//            the policy on the held-out rollouts at geometrically spaced checkpoints: what the
//            model does inside the method, which is what matters.
//
//   bazel run -c opt //examples:value_model_study -- --phase oracle --models "poly6;leg8;rbf7"
//   bazel run -c opt //examples:value_model_study -- --phase dt --only pendulum --seeds 3 \
//       --models "poly6;rbf7:anchor=current,ridge=1e-3"
//
// A model is written `name[:key=value,...]`. Names: poly2..poly10 and leg4..leg10 (monomials and
// Legendre polynomials of that total degree), rbf5/rbf7/rbf9 (Gaussian RBFs on that many centers
// per axis, plus a quadratic), spline3/spline5/spline7 (tensor cubic B-splines with that many knot
// intervals per axis, plus a quadratic), softmin1..softmin8 (soft minimum of that many quadratics),
// tanh16/tanh32 (one tanh layer of that width, plus a quadratic); and for the double inverted
// pendulum the 4-D menu poly2/poly4/poly6, leg4/leg6, rbf4, spline2, softmin2/softmin4/softmin8,
// tanh32. Keys: ridge, anchor (zero|current), width (RBF width per spacing), extent (spline domain
// half-width in normalized units), sharpness (soft-min beta), lr and epochs (SGD), slope (DT-FBSDE's
// slope weight; default the batch sweep's), scale (tanh output-weight scale), rough (the linear
// models' curvature penalty; see LinearBasisValueFunctionApprox), temperature (the oracle fit's
// sample-weight temperature), memory and reset (DT-FBSDE's carried normal equations: the forgetting
// factor, and 1 to clear them on every accepted update; linear models only), replay (fit the last
// that many batches together, targets recomputed; linear models only), temp (DT-FBSDE's entropy
// temperature; 0 weights every sample equally), target (greedy, the default
// double estimation; current: the current policy's targets from one table; or taylor: the thesis's
// off-policy drifted Taylor noiseless estimator of the current policy's value).

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <format>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include "fbsde_traj_opt/costs/quadratic_regulator_sde_terminal_cost_term.hpp"
#include "fbsde_traj_opt/policies/taylor_q_function_l1_policy.hpp"
#include "fbsde_traj_opt/sampling/counter_based_normal_sampler.hpp"
#include "fbsde_traj_opt/solvers/dt_fbsde_iterative_solver.hpp"
#include "fbsde_traj_opt/solvers/policy_rollout_evaluation.hpp"
#include "fbsde_traj_opt/utils/result.hpp"
#include "fbsde_traj_opt/value_function/linear_basis_value_function_approx.hpp"
#include "fbsde_traj_opt/value_function/shallow_tanh_value_function_approx.hpp"
#include "fbsde_traj_opt/value_function/soft_min_quadratic_value_function_approx.hpp"
#include "fbsde_traj_opt/value_function/value_function_sgd_fitter.hpp"
#include "fbsde_traj_opt/value_function/value_function_slope_targets.hpp"
#include "examples/fbrrt_comparison_experiment.hpp"
#include "examples/fbrrt_comparison_setup.hpp"
#include "examples/l1_benchmark_models.hpp"
#include "examples/l1_grid_dp_reference.hpp"

namespace fbsde_traj_opt::examples {
namespace {

using Clock = std::chrono::steady_clock;
using Control = Eigen::Matrix<double, 1, 1>;

// The same held-out noise as the method comparison (method_comparison_main.cpp), so that costs
// carry over between the two.
constexpr std::uint64_t kHeldOutSeed = 0x5851F42D4C957F2DULL;
constexpr std::uint64_t kOracleDataSeed = 0x2545F4914F6CDD1DULL;
constexpr std::uint64_t kOracleTestSeed = 0x9E3779B97F4A7C15ULL;
constexpr std::uint64_t kNetworkSeed = 0xC2B2AE3D27D4EB4FULL;

// ---------------------------------------------------------------------------------------------
// Model specifications
// ---------------------------------------------------------------------------------------------

struct ModelSpec {
  std::string text;
  std::string name;
  std::map<std::string, double> numbers;
  std::map<std::string, std::string> words;

  [[nodiscard]] auto Number(const std::string& key, double fallback) const -> double {
    const auto found = numbers.find(key);
    return found == numbers.end() ? fallback : found->second;
  }
  [[nodiscard]] auto Word(const std::string& key, const std::string& fallback) const -> std::string {
    const auto found = words.find(key);
    return found == words.end() ? fallback : found->second;
  }
};

auto ParseSpec(std::string_view text) -> Result<ModelSpec> {
  ModelSpec spec;
  spec.text = std::string(text);
  const std::size_t colon = text.find(':');
  spec.name = std::string(text.substr(0, colon));
  if (colon == std::string_view::npos) {
    return SuccessResult(std::move(spec));
  }
  std::string_view rest = text.substr(colon + 1);
  while (!rest.empty()) {
    const std::size_t comma = std::min(rest.find(','), rest.size());
    const std::string_view pair = rest.substr(0, comma);
    const std::size_t equals = pair.find('=');
    RESULT_ASSERT(equals != std::string_view::npos, "a model option is key=value.");
    const std::string key(pair.substr(0, equals));
    const std::string value(pair.substr(equals + 1));
    char* end = nullptr;
    const double number = std::strtod(value.c_str(), &end);
    if (end != nullptr && *end == '\0' && !value.empty()) {
      spec.numbers[key] = number;
    } else {
      spec.words[key] = value;
    }
    rest = comma < rest.size() ? rest.substr(comma + 1) : std::string_view();
  }
  return SuccessResult(std::move(spec));
}

auto SplitList(std::string_view text, char separator) -> std::vector<std::string> {
  std::vector<std::string> items;
  while (!text.empty()) {
    const std::size_t end = std::min(text.find(separator), text.size());
    if (end > 0) {
      items.emplace_back(text.substr(0, end));
    }
    text = end < text.size() ? text.substr(end + 1) : std::string_view();
  }
  return items;
}

// ---------------------------------------------------------------------------------------------
// Model traits: how each family is built, fitted to the terminal cost, and fitted to data
// ---------------------------------------------------------------------------------------------

template <typename T>
struct IsLinearBasis : std::false_type {};
template <int N, typename B, typename S>
struct IsLinearBasis<LinearBasisValueFunctionApprox<N, B, S>> : std::true_type {};

template <typename T>
struct IsTanh : std::false_type {};
template <int N, int W, typename S>
struct IsTanh<ShallowTanhValueFunctionApprox<N, W, S>> : std::true_type {};

template <typename BasisT>
auto MakeBasis(const ModelSpec& spec) -> BasisT {
  BasisT basis{};
  if constexpr (requires { basis.width_factor; }) {
    basis.width_factor = spec.Number("width", 1.0);
  }
  if constexpr (requires { basis.extent; }) {
    basis.extent = spec.Number("extent", 1.25);
  }
  return basis;
}

// A dense grid of states over the normalized box [-1.5, 1.5]^N, for fitting the terminal cost.
template <int N>
auto BoxGrid(const Eigen::Matrix<double, N, 1>& center, const Eigen::Matrix<double, N, 1>& scale)
    -> std::vector<Eigen::Matrix<double, N, 1>> {
  const int per_axis = N <= 2 ? 31 : 9;
  int total = 1;
  for (int axis = 0; axis < N; ++axis) {
    total *= per_axis;
  }
  std::vector<Eigen::Matrix<double, N, 1>> states;
  for (int index = 0; index < total; ++index) {
    Eigen::Matrix<double, N, 1> z;
    int rest = index;
    for (int axis = 0; axis < N; ++axis) {
      z[axis] = -1.5 + (3.0 * static_cast<double>(rest % per_axis) / static_cast<double>(per_axis - 1));
      rest /= per_axis;
    }
    states.push_back(center + scale.cwiseProduct(z));
  }
  return states;
}

// Least squares of a linear-in-parameters model's features against `values` at `states`.
template <typename ParameterVector, typename FeatureFn, typename State>
auto LeastSquares(const std::vector<State>& states, const std::vector<double>& values, const FeatureFn& features)
    -> ParameterVector {
  const auto count = static_cast<Eigen::Index>(ParameterVector::RowsAtCompileTime);
  Eigen::MatrixXd normal = Eigen::MatrixXd::Zero(count, count);
  Eigen::VectorXd rhs = Eigen::VectorXd::Zero(count);
  for (std::size_t index = 0; index < states.size(); ++index) {
    const Eigen::VectorXd phi = features(states[index]);
    normal.selfadjointView<Eigen::Lower>().rankUpdate(phi);
    rhs += values[index] * phi;
  }
  normal.diagonal().array() += 1e-10 * static_cast<double>(states.size());
  return ParameterVector(Eigen::LDLT<Eigen::MatrixXd>(normal.selfadjointView<Eigen::Lower>()).solve(rhs));
}

// The terminal value function and the starting guess elsewhere, for any model of the menu.
template <typename VF, typename BenchmarkT>
auto MakeTable(const BenchmarkT& benchmark, const ModelSpec& spec) -> Result<StartingTable<VF>> {
  constexpr int N = BenchmarkT::State::RowsAtCompileTime;
  using State = Eigen::Matrix<double, N, 1>;
  if constexpr (IsLinearBasis<VF>::value) {
    using Basis = typename VF::Basis;
    const RidgeAnchor anchor = spec.Word("anchor", "zero") == "current" ? RidgeAnchor::kCurrent : RidgeAnchor::kZero;
    RESULT_ASSIGN_OR_RETURN(VF model,
                            VF::Make(VF::ParameterVector::Zero(),
                                     benchmark.normalization_center,
                                     benchmark.normalization_scale,
                                     MakeBasis<Basis>(spec),
                                     anchor,
                                     spec.Number("rough", 0.0)));
    const std::vector<State> states = BoxGrid<N>(benchmark.normalization_center, benchmark.normalization_scale);
    std::vector<double> values;
    for (const State& state : states) {
      values.push_back(benchmark.terminal_cost(state));
    }
    RESULT_RETURN_IF_ERROR(model.SetParameters(LeastSquares<typename VF::ParameterVector>(
        states, values, [&](const State& x) { return Eigen::VectorXd(model.Features(x)); })));
    return SuccessResult(StartingTable<VF>{.terminal = model, .initial = model});
  } else if constexpr (IsTanh<VF>::value) {
    using Quadratic = Eigen::Matrix<double, VF::kNumQuadratic, 1>;
    const std::vector<State> states = BoxGrid<N>(benchmark.normalization_center, benchmark.normalization_scale);
    // The terminal table is the quadratic alone; the starting table adds random units and refits
    // the quadratic so the sum still matches the terminal cost on the box.
    typename VF::ParameterVector parameters =
        VF::RandomParameters(Quadratic::Zero(), spec.Number("scale", 0.1), kNetworkSeed);
    RESULT_ASSIGN_OR_RETURN(VF network,
                            VF::Make(parameters, benchmark.normalization_center, benchmark.normalization_scale));
    std::vector<double> residual;
    std::vector<double> terminal;
    for (const State& state : states) {
      terminal.push_back(benchmark.terminal_cost(state));
      residual.push_back(terminal.back() - network(state));
    }
    const auto features = [&](const State& x) { return Eigen::VectorXd(network.QuadraticFeatures(network.Normalize(x))); };
    parameters.template tail<VF::kNumQuadratic>() = LeastSquares<Quadratic>(states, residual, features);
    RESULT_RETURN_IF_ERROR(network.SetParameters(parameters));
    typename VF::ParameterVector terminal_parameters = VF::ParameterVector::Zero();
    terminal_parameters.template tail<VF::kNumQuadratic>() = LeastSquares<Quadratic>(states, terminal, features);
    RESULT_ASSIGN_OR_RETURN(
        VF terminal_model,
        VF::Make(terminal_parameters, benchmark.normalization_center, benchmark.normalization_scale));
    return SuccessResult(StartingTable<VF>{.terminal = terminal_model, .initial = network});
  } else {
    FbrrtComparisonOptions options;
    options.soft_min_sharpness = spec.Number("sharpness", 1.0);
    if constexpr (VF::kNumComponents == 1) {
      // One component cannot be spread along the way to the start; it is the terminal bowl.
      RESULT_ASSIGN_OR_RETURN(
          const typename VF::ParameterVector terminal,
          VF::QuadraticParameters(benchmark.terminal_cost_mat, State::Zero(), 0.0, options.soft_min_sharpness, 1e-3));
      RESULT_ASSIGN_OR_RETURN(VF model, VF::Make(terminal, options.soft_min_sharpness, 1e-3));
      return SuccessResult(StartingTable<VF>{.terminal = model, .initial = model});
    } else {
      return MakeStartingTable(benchmark, options, static_cast<const VF*>(nullptr));
    }
  }
}

// Calls `body(type_tag, spec)` with the model type a spec names, for N-dimensional states.
template <int N, typename BodyT>
auto WithModel(const ModelSpec& spec, const BodyT& body) -> Result<> {
  const std::string& name = spec.name;
  const auto run = [&]<typename VF>() -> Result<> { return body(static_cast<VF*>(nullptr)); };
  if constexpr (N == 2) {
    if (name == "poly2") return run.template operator()<LinearBasisValueFunctionApprox<2, MonomialBasis<2, 2>>>();
    if (name == "poly4") return run.template operator()<LinearBasisValueFunctionApprox<2, MonomialBasis<2, 4>>>();
    if (name == "poly6") return run.template operator()<LinearBasisValueFunctionApprox<2, MonomialBasis<2, 6>>>();
    if (name == "poly8") return run.template operator()<LinearBasisValueFunctionApprox<2, MonomialBasis<2, 8>>>();
    if (name == "poly10") return run.template operator()<LinearBasisValueFunctionApprox<2, MonomialBasis<2, 10>>>();
    if (name == "leg4") return run.template operator()<LinearBasisValueFunctionApprox<2, LegendreBasis<2, 4>>>();
    if (name == "leg6") return run.template operator()<LinearBasisValueFunctionApprox<2, LegendreBasis<2, 6>>>();
    if (name == "leg8") return run.template operator()<LinearBasisValueFunctionApprox<2, LegendreBasis<2, 8>>>();
    if (name == "leg10") return run.template operator()<LinearBasisValueFunctionApprox<2, LegendreBasis<2, 10>>>();
    if (name == "rbf5") return run.template operator()<LinearBasisValueFunctionApprox<2, GaussianRbfBasis<2, 5>>>();
    if (name == "rbf7") return run.template operator()<LinearBasisValueFunctionApprox<2, GaussianRbfBasis<2, 7>>>();
    if (name == "rbf9") return run.template operator()<LinearBasisValueFunctionApprox<2, GaussianRbfBasis<2, 9>>>();
    if (name == "spline3") return run.template operator()<LinearBasisValueFunctionApprox<2, CubicBSplineBasis<2, 3>>>();
    if (name == "spline5") return run.template operator()<LinearBasisValueFunctionApprox<2, CubicBSplineBasis<2, 5>>>();
    if (name == "spline7") return run.template operator()<LinearBasisValueFunctionApprox<2, CubicBSplineBasis<2, 7>>>();
    if (name == "softmin1") return run.template operator()<SoftMinQuadraticValueFunctionApprox<2, 1>>();
    if (name == "softmin2") return run.template operator()<SoftMinQuadraticValueFunctionApprox<2, 2>>();
    if (name == "softmin4") return run.template operator()<SoftMinQuadraticValueFunctionApprox<2, 4>>();
    if (name == "softmin8") return run.template operator()<SoftMinQuadraticValueFunctionApprox<2, 8>>();
    if (name == "tanh16") return run.template operator()<ShallowTanhValueFunctionApprox<2, 16>>();
    if (name == "tanh32") return run.template operator()<ShallowTanhValueFunctionApprox<2, 32>>();
  } else {
    if (name == "poly2") return run.template operator()<LinearBasisValueFunctionApprox<4, MonomialBasis<4, 2>>>();
    if (name == "poly4") return run.template operator()<LinearBasisValueFunctionApprox<4, MonomialBasis<4, 4>>>();
    if (name == "poly6") return run.template operator()<LinearBasisValueFunctionApprox<4, MonomialBasis<4, 6>>>();
    if (name == "leg4") return run.template operator()<LinearBasisValueFunctionApprox<4, LegendreBasis<4, 4>>>();
    if (name == "leg6") return run.template operator()<LinearBasisValueFunctionApprox<4, LegendreBasis<4, 6>>>();
    if (name == "rbf4") return run.template operator()<LinearBasisValueFunctionApprox<4, GaussianRbfBasis<4, 4>>>();
    if (name == "spline2") return run.template operator()<LinearBasisValueFunctionApprox<4, CubicBSplineBasis<4, 2>>>();
    if (name == "softmin2") return run.template operator()<SoftMinQuadraticValueFunctionApprox<4, 2>>();
    if (name == "softmin4") return run.template operator()<SoftMinQuadraticValueFunctionApprox<4, 4>>();
    if (name == "softmin8") return run.template operator()<SoftMinQuadraticValueFunctionApprox<4, 8>>();
    if (name == "tanh32") return run.template operator()<ShallowTanhValueFunctionApprox<4, 32>>();
  }
  return ErrorResult("unknown model name; run with --help for the menu.");
}

// ---------------------------------------------------------------------------------------------
// Shared evaluation
// ---------------------------------------------------------------------------------------------

template <typename BenchmarkT>
auto InitialFactor(const BenchmarkT& benchmark) {
  using StateMat = Eigen::Matrix<double, BenchmarkT::State::RowsAtCompileTime, BenchmarkT::State::RowsAtCompileTime>;
  return StateMat(Eigen::LLT<StateMat>(benchmark.initial_distribution.Covariance()).matrixL());
}

template <typename BenchmarkT, typename PolicyT>
auto HeldOutCost(const BenchmarkT& benchmark, const PolicyT& policy, std::size_t num_stages, std::size_t rollouts)
    -> double {
  return EvaluatePolicyByRollouts(benchmark.forward_model,
                                  policy,
                                  benchmark.running_cost,
                                  benchmark.terminal_cost,
                                  typename BenchmarkT::State(benchmark.initial_distribution.Mean()),
                                  InitialFactor(benchmark),
                                  rollouts,
                                  num_stages - 1,
                                  kHeldOutSeed)
      .total;
}

// Microseconds per policy call, from serial rollouts: what the policy costs online.
template <typename BenchmarkT, typename PolicyT>
auto PolicyMicroseconds(const BenchmarkT& benchmark, const PolicyT& policy, std::size_t num_stages) -> double {
  using State = typename BenchmarkT::State;
  constexpr std::size_t kRollouts = 16;
  const CounterBasedNormalSampler sampler(kHeldOutSeed);
  double seconds = 0.0;
  for (std::size_t rollout = 0; rollout < kRollouts; ++rollout) {
    State noise;
    sampler.FillNoise(rollout, 0, noise);
    State state = benchmark.initial_distribution.Mean() + (InitialFactor(benchmark) * noise);
    for (std::size_t stage = 0; stage + 1 < num_stages; ++stage) {
      const Clock::time_point start = Clock::now();
      const auto control = policy(stage, state);
      seconds += std::chrono::duration<double>(Clock::now() - start).count();
      sampler.FillNoise(rollout, stage + 1, noise);
      state = benchmark.forward_model(stage, state, control, noise);
    }
  }
  return 1e6 * seconds / static_cast<double>(kRollouts * (num_stages - 1));
}

// ---------------------------------------------------------------------------------------------
// The oracle experiment
// ---------------------------------------------------------------------------------------------

constexpr int kOracleSamples = 512;

// Per stage: states drawn by rolling the grid policy out, half at the problem's noise and half at
// six times it (DT-FBSDE's mixed exploration), with the grid's values and slopes along the control
// column there; and a separate test set at the problem's own noise.
template <typename BenchmarkT>
struct OracleData {
  using State = typename BenchmarkT::State;
  std::vector<ValueFunctionStateBatch<State, kOracleSamples>> states;
  std::vector<ValueFunctionValueBatch<State, kOracleSamples>> values;
  std::vector<ValueFunctionSlopeTargets<State, kOracleSamples, 1>> slopes;
  // The running cost each training sample accrued on its way to the stage.
  std::vector<ValueFunctionValueBatch<State, kOracleSamples>> accrued;
  std::vector<ValueFunctionStateBatch<State, kOracleSamples>> test_states;
  std::vector<ValueFunctionValueBatch<State, kOracleSamples>> test_values;
  std::vector<ValueFunctionSlopeTargets<State, kOracleSamples, 1>> test_slopes;
};

template <typename BenchmarkT>
auto MakeOracleData(const BenchmarkT& benchmark,
                    const L1GridDpReference<BenchmarkT>& grid,
                    std::size_t num_stages,
                    const Eigen::Vector2d& lower,
                    const Eigen::Vector2d& upper,
                    double cell) -> OracleData<BenchmarkT> {
  using State = Eigen::Vector2d;
  OracleData<BenchmarkT> data;
  data.states.resize(num_stages);
  data.values.resize(num_stages);
  data.slopes.resize(num_stages);
  data.accrued.resize(num_stages);
  data.test_states.resize(num_stages);
  data.test_values.resize(num_stages);
  data.test_slopes.resize(num_stages);
  std::vector<ValueFunctionValueBatch<State, kOracleSamples>> unused_accrued(num_stages);
  const auto fill = [&](std::uint64_t seed, bool mixed, auto& states, auto& values, auto& slopes, auto& accrued) {
    const CounterBasedNormalSampler sampler(seed);
    for (int sample = 0; sample < kOracleSamples; ++sample) {
      const auto s = static_cast<std::size_t>(sample);
      const double inflation = (mixed && sample % 2 == 1) ? 6.0 : 1.0;
      State noise;
      sampler.FillNoise(s, 0, noise);
      State state = benchmark.initial_distribution.Mean() + (InitialFactor(benchmark) * noise);
      double running = 0.0;
      for (std::size_t stage = 0; stage < num_stages; ++stage) {
        accrued[stage][sample] = running;
        const State clipped = state.cwiseMax(lower).cwiseMin(upper);
        states[stage].col(sample) = clipped;
        values[stage][sample] = grid.Value(stage, clipped);
        const State drifted = benchmark.forward_model(std::min(stage, num_stages - 2), clipped, Control::Zero(), State::Zero());
        const State column =
            benchmark.forward_model(std::min(stage, num_stages - 2), clipped, Control::Ones(), State::Zero()) - drifted;
        State gradient;
        for (int axis = 0; axis < 2; ++axis) {
          const State offset = State::Unit(axis) * cell;
          gradient[axis] = (grid.Value(stage, clipped + offset) - grid.Value(stage, clipped - offset)) / (2 * cell);
        }
        slopes[stage].directions[0].col(sample) = column;
        slopes[stage].slopes[0][sample] = column.dot(gradient);
        if (stage + 1 < num_stages) {
          sampler.FillNoise(s, stage + 1, noise);
          const Control control = grid(stage, state);
          running += benchmark.running_cost(stage, state, control);
          state = benchmark.forward_model(stage, state, control, inflation * noise);
        }
      }
    }
  };
  fill(kOracleDataSeed, true, data.states, data.values, data.slopes, data.accrued);
  fill(kOracleTestSeed, false, data.test_states, data.test_values, data.test_slopes, unused_accrued);
  return data;
}

// DT-FBSDE's local-entropy weights (DtFbsdeIterativeSolver::SampleWeights) for one stage of the
// oracle's training data: on the accrued cost plus the grid's value, at temperature `temperature`
// in units of the stage's spread; zero gives equal weights.
template <typename BenchmarkT>
auto OracleWeights(const OracleData<BenchmarkT>& data, std::size_t stage, double temperature)
    -> ValueFunctionValueBatch<typename BenchmarkT::State, kOracleSamples> {
  using Values = ValueFunctionValueBatch<typename BenchmarkT::State, kOracleSamples>;
  if (temperature <= 0.0) {
    return Values::Ones();
  }
  const Values heuristic = data.accrued[stage] + data.values[stage];
  const double mean = heuristic.mean();
  const double spread = std::sqrt((heuristic.array() - mean).square().mean()) + 1e-12;
  return (-(heuristic.array() - heuristic.minCoeff()) / (spread * temperature)).exp();
}

struct OracleResult {
  int num_parameters = 0;
  double policy_cost = 0.0;
  // The median over stages of the weights' effective sample size, (sum w)^2 / sum w^2, out of
  // kOracleSamples: how many of the training samples a stage's fit effectively rests on.
  double effective_samples = 0.0;
  // The share of held-out states on the optimal policy's trajectories where the fitted table's
  // policy picks a control more than 0.5 from the grid's: error where it decides the control.
  double disagreement = 0.0;
  // Value error at the states the fitted table's *own* policy visits (128 held-out rollouts), against
  // the grid: RMSE, and the share of visited states where the fit is off by more than 1.
  double on_policy_rmse = 0.0;
  double on_policy_large_error = 0.0;
  double value_rmse = 0.0;
  double slope_rmse = 0.0;
  double fit_seconds = 0.0;
  double policy_microseconds = 0.0;
};

// Fits a table to the grid's values and slopes, stage by stage; see RunOracle().
template <typename VF, typename BenchmarkT>
auto FitOracleTable(const BenchmarkT& benchmark,
                    const OracleData<BenchmarkT>& data,
                    const ModelSpec& spec,
                    std::size_t num_stages) -> Result<std::vector<VF>> {
  using State = typename BenchmarkT::State;
  RESULT_ASSIGN_OR_RETURN(const StartingTable<VF> start, MakeTable<VF>(benchmark, spec));
  std::vector<VF> table(num_stages, start.initial);
  table.back() = start.terminal;
  const double ridge = spec.Number("ridge", 1e-6);
  const double slope_weight = spec.Number("slope", 1.0);
  const Clock::time_point fit_start = Clock::now();
  for (std::size_t stage = 0; stage + 1 < num_stages; ++stage) {
    const ValueFunctionValueBatch<State, kOracleSamples> weights =
        OracleWeights(data, stage, spec.Number("temperature", 2.0));
    if constexpr (requires {
                    table[stage].template FitWeightedLeastSquares<kOracleSamples, 1>(
                        data.states[stage], data.values[stage], weights, data.slopes[stage], 1.0, 1.0);
                  }) {
      RESULT_RETURN_IF_ERROR(table[stage].template FitWeightedLeastSquares<kOracleSamples, 1>(
          data.states[stage], data.values[stage], weights, data.slopes[stage], slope_weight, ridge));
    } else {
      ValueFunctionSgdOptions<double> sgd;
      sgd.learning_rate = spec.Number("lr", 0.05);
      using Fitter = ValueFunctionSgdFitter<VF, 64>;
      RESULT_ASSIGN_OR_RETURN(Fitter fitter, Fitter::Make(sgd, table[stage].Parameters()));
      std::vector<typename Fitter::StepReport> history;
      RESULT_RETURN_IF_ERROR(fitter.template FitEpochs<kOracleSamples, 1>(table[stage],
                                                                          data.states[stage],
                                                                          data.values[stage],
                                                                          weights,
                                                                          data.slopes[stage],
                                                                          slope_weight,
                                                                          static_cast<std::size_t>(
                                                                              spec.Number("epochs", 400)),
                                                                          7 + stage,
                                                                          history));
    }
  }
  (void)fit_start;
  return SuccessResult(std::move(table));
}

template <typename VF, typename BenchmarkT>
auto RunOracle(const BenchmarkT& benchmark,
               const L1GridDpReference<BenchmarkT>& grid,
               const OracleData<BenchmarkT>& data,
               const ModelSpec& spec,
               std::size_t num_stages,
               std::size_t held_out) -> Result<OracleResult> {
  using State = typename BenchmarkT::State;
  const Clock::time_point fit_start = Clock::now();
  RESULT_ASSIGN_OR_RETURN(std::vector<VF> table, (FitOracleTable<VF>(benchmark, data, spec, num_stages)));
  OracleResult result;
  result.num_parameters = VF::kNumParameters;
  result.fit_seconds = std::chrono::duration<double>(Clock::now() - fit_start).count();
  double value_sum = 0.0;
  double slope_sum = 0.0;
  double count = 0.0;
  for (std::size_t stage = 0; stage + 1 < num_stages; ++stage) {
    for (int sample = 0; sample < kOracleSamples; ++sample) {
      const State state = data.test_states[stage].col(sample);
      const double error = table[stage](state) - data.test_values[stage][sample];
      const State column = data.test_slopes[stage].directions[0].col(sample);
      const double slope_error = column.dot(table[stage].Gradient(state)) - data.test_slopes[stage].slopes[0][sample];
      value_sum += error * error;
      slope_sum += slope_error * slope_error;
      count += 1.0;
    }
  }
  result.value_rmse = std::sqrt(value_sum / count);
  result.slope_rmse = std::sqrt(slope_sum / count);
  using Policy = TaylorQFunctionL1Policy<2, 1, typename BenchmarkT::ForwardModel, VF>;
  RESULT_ASSIGN_OR_RETURN(
      const Policy policy,
      Policy::Make(benchmark.forward_model, benchmark.running_cost.L1Weight(), Control(-1.0), Control(1.0), table));
  result.policy_cost = HeldOutCost(benchmark, policy, num_stages, held_out);
  std::vector<double> effective(num_stages - 1);
  std::size_t disagreements = 0;
  std::size_t decisions = 0;
  for (std::size_t stage = 0; stage + 1 < num_stages; ++stage) {
    const auto weights = OracleWeights(data, stage, spec.Number("temperature", 2.0));
    effective[stage] = weights.sum() * weights.sum() / weights.squaredNorm();
    for (int sample = 0; sample < kOracleSamples; ++sample) {
      const State state = data.test_states[stage].col(sample);
      disagreements += std::abs(policy(stage, state)[0] - grid(stage, state)[0]) > 0.5 ? 1U : 0U;
      ++decisions;
    }
  }
  {
    const CounterBasedNormalSampler sampler(kHeldOutSeed);
    double squared = 0.0;
    std::size_t large = 0;
    std::size_t visited = 0;
    for (std::size_t rollout = 0; rollout < 128; ++rollout) {
      State noise;
      sampler.FillNoise(rollout, 0, noise);
      State state = benchmark.initial_distribution.Mean() + (InitialFactor(benchmark) * noise);
      for (std::size_t stage = 0; stage + 1 < num_stages; ++stage) {
        const double error = table[stage](state) - grid.Value(stage, state);
        squared += error * error;
        large += std::abs(error) > 1.0 ? 1U : 0U;
        ++visited;
        sampler.FillNoise(rollout, stage + 1, noise);
        state = benchmark.forward_model(stage, state, policy(stage, state), noise);
      }
    }
    result.on_policy_rmse = std::sqrt(squared / static_cast<double>(visited));
    result.on_policy_large_error = static_cast<double>(large) / static_cast<double>(visited);
  }
  std::ranges::nth_element(effective, effective.begin() + static_cast<std::ptrdiff_t>(effective.size() / 2));
  result.effective_samples = effective[effective.size() / 2];
  result.disagreement = static_cast<double>(disagreements) / static_cast<double>(decisions);
  result.policy_microseconds = PolicyMicroseconds(benchmark, policy, num_stages);
  return SuccessResult(result);
}

// ---------------------------------------------------------------------------------------------
// The policy experiment: which policy a value function produces
// ---------------------------------------------------------------------------------------------

// The grid's value function at one stage, as a ValueFunctionApprox: bilinear values, and a gradient
// and Hessian by central differences of step `step` per axis. The interpolant is piecewise
// bilinear, so its own curvature is zero inside every cell; differences across several cells
// measure the curvature at that scale instead.
template <typename BenchmarkT>
struct GridValueFunction {
  using State = typename BenchmarkT::State;
  using StateMat = Eigen::Matrix<double, 2, 2>;
  const L1GridDpReference<BenchmarkT>* grid = nullptr;
  std::size_t stage = 0;
  State step = State::Ones();
  bool curvature = true;

  auto operator()(const State& x) const -> double { return grid->Value(stage, x); }
  [[nodiscard]] auto Gradient(const State& x) const -> State {
    State gradient;
    for (int axis = 0; axis < 2; ++axis) {
      const State offset = State::Unit(axis) * step[axis];
      gradient[axis] = ((*this)(x + offset) - (*this)(x - offset)) / (2.0 * step[axis]);
    }
    return gradient;
  }
  [[nodiscard]] auto Hessian(const State& x) const -> StateMat {
    StateMat hessian = StateMat::Zero();
    if (!curvature) {
      return hessian;
    }
    const double center = (*this)(x);
    for (int a = 0; a < 2; ++a) {
      const State da = State::Unit(a) * step[a];
      hessian(a, a) = ((*this)(x + da) - (2.0 * center) + (*this)(x - da)) / (step[a] * step[a]);
    }
    const State d0 = State::Unit(0) * step[0];
    const State d1 = State::Unit(1) * step[1];
    hessian(0, 1) = ((*this)(x + d0 + d1) - (*this)(x + d0 - d1) - (*this)(x - d0 + d1) + (*this)(x - d0 - d1)) /
                    (4.0 * step[0] * step[1]);
    hessian(1, 0) = hessian(0, 1);
    return hessian;
  }
};

// The policy that minimizes the *exact* expected next value over a set of controls: c|u| plus a
// 3x3 Gauss-Hermite expectation of V_{k+1}(x + f + B u + Sigma w), as the grid's own policy does,
// but over any table and any control set. It is the Bellman minimization without the Taylor
// expansion, so comparing it with the Taylor Q-function policy on the same table isolates the
// Q-function.
template <typename BenchmarkT, typename TableT>
class ExpectationPolicy {
 public:
  using State = typename BenchmarkT::State;
  using Control = Eigen::Matrix<double, 1, 1>;

  ExpectationPolicy(const BenchmarkT& benchmark, const TableT& table, std::vector<double> controls)
      : benchmark_(&benchmark), table_(&table), controls_(std::move(controls)) {}

  auto operator()(std::size_t stage, const State& x) const -> Control {
    static constexpr std::array<std::pair<double, double>, 3> kRule{
        {{-1.7320508075688772, 1.0 / 6.0}, {0.0, 2.0 / 3.0}, {1.7320508075688772, 1.0 / 6.0}}};
    double best = std::numeric_limits<double>::infinity();
    double best_control = 0.0;
    const auto& next = (*table_)[stage + 1];
    for (const double u : controls_) {
      double value = benchmark_->running_cost(stage, x, Control(u));
      for (const auto& [first, first_weight] : kRule) {
        for (const auto& [second, second_weight] : kRule) {
          value += first_weight * second_weight *
                   next(benchmark_->forward_model(stage, x, Control(u), State(first, second)));
        }
      }
      if (value < best) {
        best = value;
        best_control = u;
      }
    }
    return Control(best_control);
  }

 private:
  const BenchmarkT* benchmark_;
  const TableT* table_;
  std::vector<double> controls_;
};

auto ControlGrid(std::size_t count) -> std::vector<double> {
  std::vector<double> controls;
  for (std::size_t index = 0; index < count; ++index) {
    controls.push_back(-1.0 + (2.0 * static_cast<double>(index) / static_cast<double>(count - 1)));
  }
  return controls;
}

// ---------------------------------------------------------------------------------------------
// The DT-FBSDE experiment
// ---------------------------------------------------------------------------------------------

struct StudyOptions {
  std::string phase = "oracle";
  std::uint64_t seed = 20260928;
  std::size_t num_seeds = 3;
  double time_budget = 30.0;
  double double_pendulum_time_budget = 180.0;
  std::size_t held_out = 1024;
  double first_checkpoint = 1e-3;
  double checkpoint_growth = 1.15;
  bool include_integrator = true;
  bool include_pendulum = true;
  bool include_double_pendulum = false;
  std::vector<std::string> models{};
  std::string csv_path = "value_model_study.csv";
  // Per run, for the DT-FBSDE phase: iterations, accepted updates, failures, time, final cost.
  std::string runs_path = "value_model_study_runs.csv";
  std::ofstream* runs_csv = nullptr;
  std::size_t dp_cells = 400;
};

template <typename SolverT, typename BenchmarkT, typename VF>
auto RunDtFbsde(const BenchmarkT& benchmark,
                const StartingTable<VF>& table,
                const DtFbsdeIterativeOptions<double>& solver_options,
                const StudyOptions& options,
                std::size_t num_stages,
                double budget,
                std::size_t seed_index,
                std::ofstream& csv,
                const std::string& problem,
                const std::string& label) -> Result<std::pair<double, double>> {
  std::size_t accepted = 0;
  std::size_t iterations = 0;
  double girsanov_sum = 0.0;
  double correction_sum = 0.0;
  double target_sum = 0.0;
  double signed_sum = 0.0;
  double curvature_sum = 0.0;
  double off_policy_sum = 0.0;
  double off_policy_correction_sum = 0.0;
  double residual_sum = 0.0;
  const std::uint64_t seed = options.seed + (1000U * static_cast<std::uint64_t>(seed_index));
  auto solver = std::make_unique<Result<SolverT>>(SolverT::Make(benchmark.forward_model,
                                                                benchmark.running_cost,
                                                                benchmark.terminal_cost,
                                                                benchmark.initial_distribution,
                                                                Control(-1.0),
                                                                Control(1.0),
                                                                table.initial,
                                                                table.terminal,
                                                                solver_options,
                                                                seed));
  if (!solver->has_value()) {
    return ErrorResult(solver->error().message, solver->error().location);
  }
  SolverT& dt = **solver;
  double charged = 0.0;
  double next = options.first_checkpoint;
  double cost = HeldOutCost(benchmark, dt.CurrentPolicy(), num_stages, options.held_out);
  std::size_t failures = 0;
  const auto record = [&](std::size_t iteration) {
    cost = HeldOutCost(benchmark, dt.CurrentPolicy(), num_stages, options.held_out);
    csv << std::format("{},\"{}\",{},{},{:.6f},{:.6f},{:.6f}\n",
                       problem,
                       label,
                       seed_index,
                       iteration,
                       charged,
                       cost,
                       dt.CurrentCost());
  };
  record(0);
  for (std::size_t iteration = 1;; ++iteration) {
    const Clock::time_point start = Clock::now();
    const auto report = dt.Iterate();
    charged += std::chrono::duration<double>(Clock::now() - start).count();
    if (!report.has_value()) {
      // A model that makes the method fail is a result, not a crash: record it and stop the run.
      ++failures;
      std::cout << std::format("{:<40} seed {}: DT-FBSDE failed at iteration {}: {}\n",
                               label,
                               seed_index,
                               iteration,
                               report.error().message);
      csv << std::format("{},\"{}\",{},{},{:.6f},nan,nan\n", problem, label, seed_index, iteration, charged);
      break;
    }
    iterations = iteration;
    accepted += report->accepted_step > 0.0 ? 1U : 0U;
    girsanov_sum += report->mean_girsanov_drift;
    correction_sum += report->mean_first_order_correction;
    target_sum += report->mean_target_magnitude;
    signed_sum += report->mean_signed_first_order_correction;
    curvature_sum += report->mean_second_order_correction;
    off_policy_sum += report->off_policy_fraction;
    off_policy_correction_sum += report->mean_off_policy_correction;
    residual_sum += report->mean_taylor_residual;
    const bool done = charged >= budget;
    if (done || charged >= next) {
      record(iteration);
      while (next <= charged) {
        next *= options.checkpoint_growth;
      }
    }
    if (done) {
      break;
    }
  }
  csv.flush();
  if (options.runs_csv != nullptr) {
    const double per_iteration = iterations > 0 ? 1.0 / static_cast<double>(iterations) : 0.0;
    *options.runs_csv << std::format("{},\"{}\",{},{},{},{},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f}\n",
                                     problem,
                                     label,
                                     seed_index,
                                     iterations,
                                     accepted,
                                     failures,
                                     charged,
                                     failures > 0 ? std::nan("") : cost,
                                     girsanov_sum * per_iteration,
                                     correction_sum * per_iteration,
                                     target_sum * per_iteration,
                                     signed_sum * per_iteration,
                                     curvature_sum * per_iteration,
                                     off_policy_sum * per_iteration,
                                     off_policy_correction_sum * per_iteration,
                                     residual_sum * per_iteration);
    options.runs_csv->flush();
  }
  return SuccessResult(std::make_pair(failures > 0 ? std::nan("") : cost, charged));
}

// The knobs every DT-FBSDE run takes from its spec: targets, exploration, and the Taylor noiseless
// estimator's options.
auto ApplyTargetKnobs(const ModelSpec& spec, DtFbsdeIterativeOptions<double>& solver) -> void {
  solver.entropy_temperature = spec.Number("temp", solver.entropy_temperature);
  if (spec.Word("target", "greedy") == "current") {
    solver.backward_target = DtFbsdeBackwardTarget::kCurrentPolicy;
  } else if (spec.Word("target", "greedy") == "taylor") {
    solver.backward_target = DtFbsdeBackwardTarget::kTaylorNoiseless;
  }
  solver.exploration_probability = spec.Number("explore", solver.exploration_probability);
  solver.initial_exploration_probability = spec.Number("explore0", solver.initial_exploration_probability);
  solver.exploration_decay = spec.Number("decay", solver.exploration_decay);
  solver.exploration_hold_stages = static_cast<std::size_t>(spec.Number("hold", 1.0));
  solver.girsanov_weight = spec.Number("gweight", 0.0);
  solver.taylor_correction_order = static_cast<std::size_t>(spec.Number("order", 2.0));
  solver.exploring_path_fraction = spec.Number("explorers", 1.0);
  solver.measure_taylor_residual = spec.Number("residual", 0.0) != 0.0;
  if (spec.Word("tpolicy", "current") == "greedy") {
    solver.taylor_target_policy = DtFbsdeTaylorTargetPolicy::kGreedy;
  }
  if (spec.Number("inflate", 0.0) == 1.0) {
    solver.base_diffusion_inflation = 1.0;
    solver.max_diffusion_inflation = 1.0;
    solver.uninflated_path_fraction = 0.0;
  }
}

// The batch sweep's best 2-D configuration (dt_fbsde_batch_sweep_main.cpp), with the spec's slope
// weight and SGD settings.
auto TwoDimensionalDtOptions(const ModelSpec& spec, double default_slope) -> DtFbsdeIterativeOptions<double> {
  FbrrtComparisonOptions comparison;
  DtFbsdeIterativeOptions<double> solver = DtFbsdeOptions(comparison);
  solver.stop_uphill_line_search = true;
  solver.screening_rollouts = 32;
  solver.base_diffusion_inflation = 6.0;
  solver.max_diffusion_inflation = 6.0;
  solver.uninflated_path_fraction = 0.5;
  solver.slope_weight = spec.Number("slope", default_slope);
  solver.ridge = spec.Number("ridge", 1e-6);
  solver.sgd.learning_rate = spec.Number("lr", 0.05);
  solver.sgd_epochs = static_cast<std::size_t>(spec.Number("epochs", 50));
  solver.normal_equation_memory = spec.Number("memory", 0.0);
  solver.reset_normal_equations_on_acceptance = spec.Number("reset", 0.0) != 0.0;
  solver.replay_batches = static_cast<std::size_t>(spec.Number("replay", 0.0));
  ApplyTargetKnobs(spec, solver);
  return solver;
}

// ---------------------------------------------------------------------------------------------
// DAgger: the oracle, refitted on the states its own policy visits
// ---------------------------------------------------------------------------------------------

using State2 = Eigen::Vector2d;

// One stage's worth of oracle training data, labeled by the grid: states, values, slopes along the
// control column, and the running cost each state's path accrued.
template <typename BenchmarkT>
struct DaggerBlock {
  using State = typename BenchmarkT::State;
  ValueFunctionStateBatch<State, kOracleSamples> states;
  ValueFunctionValueBatch<State, kOracleSamples> values;
  ValueFunctionSlopeTargets<State, kOracleSamples, 1> slopes;
  ValueFunctionValueBatch<State, kOracleSamples> accrued;
};

// Rolls `policy` out kOracleSamples times -- every second rollout at six times the noise, as the
// oracle's data and DT-FBSDE's mixed exploration do -- and labels every visited state with the
// grid's value and slope. With `probes`, each stage also gets a block of one-step branches from the
// stage before under u in {-1, 0, 1} (cycled over the samples), noise-free, as DT-FBSDE's probes are.
// Returns, per stage, one block of path states and, with probes, one of probe states.
template <typename BenchmarkT, typename PolicyT>
auto SampleDaggerBlocks(const BenchmarkT& benchmark,
                        const L1GridDpReference<BenchmarkT>& grid,
                        const PolicyT& policy,
                        std::size_t num_stages,
                        const Eigen::Vector2d& lower,
                        const Eigen::Vector2d& upper,
                        double cell,
                        std::uint64_t seed,
                        bool probes) -> std::vector<std::vector<DaggerBlock<BenchmarkT>>> {
  using State = Eigen::Vector2d;
  const auto label = [&](std::size_t stage, const State& state, DaggerBlock<BenchmarkT>& block, int column, double accrued) {
    const State clipped = state.cwiseMax(lower).cwiseMin(upper);
    block.states.col(column) = clipped;
    block.values[column] = grid.Value(stage, clipped);
    block.accrued[column] = accrued;
    const std::size_t model_stage = std::min(stage, num_stages - 2);
    const State drifted = benchmark.forward_model(model_stage, clipped, Control::Zero(), State::Zero());
    const State direction = benchmark.forward_model(model_stage, clipped, Control::Ones(), State::Zero()) - drifted;
    State gradient;
    for (int axis = 0; axis < 2; ++axis) {
      const State offset = State::Unit(axis) * cell;
      gradient[axis] = (grid.Value(stage, clipped + offset) - grid.Value(stage, clipped - offset)) / (2 * cell);
    }
    block.slopes.directions[0].col(column) = direction;
    block.slopes.slopes[0][column] = direction.dot(gradient);
  };
  std::vector<std::vector<DaggerBlock<BenchmarkT>>> blocks(num_stages,
                                                           std::vector<DaggerBlock<BenchmarkT>>(probes ? 2 : 1));
  const CounterBasedNormalSampler sampler(seed);
  for (int sample = 0; sample < kOracleSamples; ++sample) {
    const auto s = static_cast<std::size_t>(sample);
    const double inflation = sample % 2 == 1 ? 6.0 : 1.0;
    State noise;
    sampler.FillNoise(s, 0, noise);
    State state = benchmark.initial_distribution.Mean() + (InitialFactor(benchmark) * noise);
    double running = 0.0;
    for (std::size_t stage = 0; stage < num_stages; ++stage) {
      label(stage, state, blocks[stage][0], sample, running);
      if (stage + 1 == num_stages) {
        break;
      }
      if (probes) {
        const double branch = static_cast<double>(sample % 3) - 1.0;
        const State probe = benchmark.forward_model(stage, state, Control(branch), State::Zero());
        label(stage + 1, probe, blocks[stage + 1][1], sample, running + benchmark.running_cost(stage, state, Control(branch)));
      }
      sampler.FillNoise(s, stage + 1, noise);
      const Control control = policy(stage, state);
      running += benchmark.running_cost(stage, state, control);
      state = benchmark.forward_model(stage, state, control, inflation * noise);
    }
  }
  if (probes) {
    // Stage 0 has no stage before it to branch from: its probe block repeats its paths.
    blocks[0][1] = blocks[0][0];
  }
  return blocks;
}

// Errors of a set of value functions at the states its own policy visits, against the grid, over
// 128 held-out rollouts: the value RMSE, the share of states off by more than 1, and the RMSE of the
// slope along the stage's control column -- the difference the policy actually compares, which a
// constant offset in the value leaves untouched. The grid's slope is a central difference over 4
// cells, the scale at which its bilinear kinks average out.
struct OnPolicyErrors {
  double value_rmse = 0.0;
  double large_fraction = 0.0;
  double slope_rmse = 0.0;
};

template <typename BenchmarkT, typename TableT, typename PolicyT>
auto OnPolicyError(const BenchmarkT& benchmark,
                   const L1GridDpReference<BenchmarkT>& grid,
                   const TableT& table,
                   const PolicyT& policy,
                   std::size_t num_stages,
                   const Eigen::Vector2d& spacing) -> OnPolicyErrors {
  using State = typename BenchmarkT::State;
  const CounterBasedNormalSampler sampler(kHeldOutSeed);
  double squared = 0.0;
  double slope_squared = 0.0;
  std::size_t large = 0;
  std::size_t visited = 0;
  for (std::size_t rollout = 0; rollout < 128; ++rollout) {
    State noise;
    sampler.FillNoise(rollout, 0, noise);
    State state = benchmark.initial_distribution.Mean() + (InitialFactor(benchmark) * noise);
    for (std::size_t stage = 0; stage + 1 < num_stages; ++stage) {
      const double error = table[stage](state) - grid.Value(stage, state);
      squared += error * error;
      large += std::abs(error) > 1.0 ? 1U : 0U;
      const State drifted = benchmark.forward_model(stage, state, Control::Zero(), State::Zero());
      const State direction = benchmark.forward_model(stage, state, Control::Ones(), State::Zero()) - drifted;
      State grid_gradient;
      for (int axis = 0; axis < 2; ++axis) {
        const State offset = State::Unit(axis) * (4.0 * spacing[axis]);
        grid_gradient[axis] =
            (grid.Value(stage, state + offset) - grid.Value(stage, state - offset)) / (8.0 * spacing[axis]);
      }
      const double slope_error = direction.dot(table[stage].Gradient(state) - grid_gradient);
      slope_squared += slope_error * slope_error;
      ++visited;
      sampler.FillNoise(rollout, stage + 1, noise);
      state = benchmark.forward_model(stage, state, policy(stage, state), noise);
    }
  }
  const auto count = static_cast<double>(visited);
  return {.value_rmse = std::sqrt(squared / count),
          .large_fraction = static_cast<double>(large) / count,
          .slope_rmse = std::sqrt(slope_squared / count)};
}

// DAgger on the oracle: fit the set of value functions to the grid on the optimal policy's states,
// then for `rounds` rounds roll out the fitted policy, label the states it visits with the grid,
// add them to the data, and refit on all of it. Each block keeps its own entropy weights, as each
// DT-FBSDE batch does. Prints, per round, the policy's gap and the fit's error on its own rollouts.
template <typename VF, typename BenchmarkT>
auto RunDagger(const BenchmarkT& benchmark,
               const L1GridDpReference<BenchmarkT>& grid,
               const ModelSpec& spec,
               std::size_t num_stages,
               const Eigen::Vector2d& lower,
               const Eigen::Vector2d& upper,
               double cell,
               double optimum,
               const StudyOptions& options,
               std::ofstream& csv) -> Result<> {
  if constexpr (!requires { typename VF::NormalEquations; }) {
    return ErrorResult("the DAgger phase needs a linear model.");
  } else {
    using Policy = TaylorQFunctionL1Policy<2, 1, typename BenchmarkT::ForwardModel, VF>;
    const bool probes = spec.Number("probes", 0.0) != 0.0;
    const auto rounds = static_cast<std::size_t>(spec.Number("rounds", 6.0));
    const double temperature = spec.Number("temperature", 2.0);
    const double slope_weight = spec.Number("slope", 1.0);
    const double ridge = spec.Number("ridge", 1e-6);
    RESULT_ASSIGN_OR_RETURN(const StartingTable<VF> start, MakeTable<VF>(benchmark, spec));
    std::vector<std::vector<DaggerBlock<BenchmarkT>>> data =
        SampleDaggerBlocks(benchmark, grid, grid, num_stages, lower, upper, cell, kOracleDataSeed, probes);
    std::vector<VF> table(num_stages, start.initial);
    table.back() = start.terminal;
    for (std::size_t round = 0;; ++round) {
      for (std::size_t stage = 0; stage + 1 < num_stages; ++stage) {
        typename VF::NormalEquations equations;
        for (const DaggerBlock<BenchmarkT>& block : data[stage]) {
          ValueFunctionValueBatch<State2, kOracleSamples> weights = ValueFunctionValueBatch<State2, kOracleSamples>::Ones();
          if (temperature > 0.0) {
            const ValueFunctionValueBatch<State2, kOracleSamples> heuristic = block.accrued + block.values;
            const double mean = heuristic.mean();
            const double spread = std::sqrt((heuristic.array() - mean).square().mean()) + 1e-12;
            weights = (-(heuristic.array() - heuristic.minCoeff()) / (spread * temperature)).exp();
          }
          RESULT_RETURN_IF_ERROR((table[stage].template AccumulateNormalEquations<kOracleSamples, 1>(
              block.states, block.values, weights, block.slopes, slope_weight, equations)));
        }
        RESULT_RETURN_IF_ERROR(table[stage].FitNormalEquations(equations, ridge));
      }
      RESULT_ASSIGN_OR_RETURN(const Policy policy,
                              Policy::Make(benchmark.forward_model,
                                           benchmark.running_cost.L1Weight(),
                                           Control(-1.0),
                                           Control(1.0),
                                           std::span<const VF>(table)));
      const double cost = HeldOutCost(benchmark, policy, num_stages, options.held_out);
      const OnPolicyErrors errors = OnPolicyError(benchmark, grid, table, policy, num_stages, (upper - lower) / 400.0);
      const double rmse = errors.value_rmse;
      const double large = errors.large_fraction;
      std::cout << std::format("{:<36} round {}: {:>5} samples per stage, gap {:+9.2f}%, error on own rollouts {:7.3f} "
                               "({:5.1f}% off by >1), slope error {:7.4f}\n",
                               spec.text,
                               round,
                               data[1].size() * static_cast<std::size_t>(kOracleSamples),
                               100.0 * (cost / optimum - 1.0),
                               rmse,
                               100.0 * large,
                               errors.slope_rmse);
      std::cout.flush();
      csv << std::format("{},\"{}\",{},{},{:.6f},{:.6f},{:.6f},{:.5f}\n",
                         benchmark.name,
                         spec.text,
                         round,
                         data[1].size() * static_cast<std::size_t>(kOracleSamples),
                         cost,
                         optimum,
                         rmse,
                         large);
      csv.flush();
      if (round == rounds) {
        break;
      }
      const auto added = SampleDaggerBlocks(
          benchmark, grid, policy, num_stages, lower, upper, cell, kOracleDataSeed + (7919U * (round + 1)), probes);
      for (std::size_t stage = 0; stage < num_stages; ++stage) {
        data[stage].insert(data[stage].end(), added[stage].begin(), added[stage].end());
      }
    }
    return SuccessResult();
  }
}

// ---------------------------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------------------------

template <std::size_t NumStages, typename BenchmarkT>
auto StudyTwoDimensional(const BenchmarkT& benchmark,
                         const Eigen::Vector2d& lower,
                         const Eigen::Vector2d& upper,
                         double default_slope,
                         const StudyOptions& options,
                         std::ofstream& csv) -> Result<> {
  const std::string& problem = benchmark.name;
  typename L1GridDpReference<BenchmarkT>::Options grid_options;
  grid_options.lower = lower;
  grid_options.upper = upper;
  grid_options.cells_per_axis = options.dp_cells;
  RESULT_ASSIGN_OR_RETURN(const L1GridDpReference<BenchmarkT> grid,
                          L1GridDpReference<BenchmarkT>::Make(benchmark, NumStages, grid_options));
  const double optimum = HeldOutCost(benchmark, grid, NumStages, options.held_out);
  std::cout << std::format("\n=== {}: grid optimum {:.4f} ===\n", problem, optimum);
  const double cell = ((upper - lower) / static_cast<double>(options.dp_cells)).minCoeff();

  std::unique_ptr<OracleData<BenchmarkT>> data;
  if (options.phase == "oracle" || options.phase == "policy") {
    data = std::make_unique<OracleData<BenchmarkT>>(MakeOracleData(benchmark, grid, NumStages, lower, upper, cell));
  }
  // One row of the policy experiment: a policy built from a value function, scored on the held-out
  // rollouts against the grid's own policy.
  const auto report = [&](const std::string& value_function, const std::string& policy_name, double cost) {
    std::cout << std::format("{:<44} {:<34} cost {:8.4f}  gap {:+8.2f}%\n",
                             value_function,
                             policy_name,
                             cost,
                             100.0 * (cost / optimum - 1.0));
    csv << std::format("{},\"{}\",\"{}\",{:.6f},{:.6f}\n", problem, value_function, policy_name, cost, optimum);
    csv.flush();
  };
  if (options.phase == "policy") {
    using GridVF = GridValueFunction<BenchmarkT>;
    const Eigen::Vector2d spacing = (upper - lower) / static_cast<double>(options.dp_cells);
    const auto grid_table = [&](double cells, bool curvature) {
      std::vector<GridVF> table;
      for (std::size_t stage = 0; stage < NumStages; ++stage) {
        table.push_back(GridVF{.grid = &grid, .stage = stage, .step = cells * spacing, .curvature = curvature});
      }
      return table;
    };
    report("grid V", "grid's own policy (reference)", optimum);
    const std::vector<GridVF> plain = grid_table(2.0, true);
    report("grid V", "exact expectation, 41 controls",
           HeldOutCost(benchmark, ExpectationPolicy<BenchmarkT, std::vector<GridVF>>(benchmark, plain, ControlGrid(41)),
                       NumStages, options.held_out));
    report("grid V", "exact expectation, controls {-1,0,1}",
           HeldOutCost(benchmark, ExpectationPolicy<BenchmarkT, std::vector<GridVF>>(benchmark, plain, ControlGrid(3)),
                       NumStages, options.held_out));
    using GridPolicy = TaylorQFunctionL1Policy<2, 1, typename BenchmarkT::ForwardModel, GridVF>;
    for (const double cells : {1.0, 2.0, 4.0, 8.0}) {
      for (const bool curvature : {true, false}) {
        if (!curvature && cells != 2.0) {
          continue;
        }
        const std::vector<GridVF> table = grid_table(cells, curvature);
        RESULT_ASSIGN_OR_RETURN(const GridPolicy policy,
                                GridPolicy::Make(benchmark.forward_model,
                                                 benchmark.running_cost.L1Weight(),
                                                 Control(-1.0),
                                                 Control(1.0),
                                                 std::span<const GridVF>(table)));
        report(std::format("grid V, differences over {:g} cells", cells),
               curvature ? "Taylor Q-function" : "Taylor Q-function, no curvature term",
               HeldOutCost(benchmark, policy, NumStages, options.held_out));
      }
    }
  }
  for (const std::string& text : options.models) {
    RESULT_ASSIGN_OR_RETURN(const ModelSpec spec, ParseSpec(text));
    RESULT_RETURN_IF_ERROR(WithModel<2>(spec, [&]<typename VF>(VF* /*tag*/) -> Result<> {
      if (options.phase == "dagger") {
        return RunDagger<VF>(benchmark, grid, spec, NumStages, lower, upper, cell, optimum, options, csv);
      }
      if (options.phase == "policy") {
        using Policy = TaylorQFunctionL1Policy<2, 1, typename BenchmarkT::ForwardModel, VF>;
        const auto score_table = [&](const std::string& name, const std::vector<VF>& table) -> Result<> {
          RESULT_ASSIGN_OR_RETURN(const Policy policy,
                                  Policy::Make(benchmark.forward_model,
                                               benchmark.running_cost.L1Weight(),
                                               Control(-1.0),
                                               Control(1.0),
                                               std::span<const VF>(table)));
          report(name, "Taylor Q-function", HeldOutCost(benchmark, policy, NumStages, options.held_out));
          const OnPolicyErrors errors =
              OnPolicyError(benchmark, grid, table, policy, NumStages, (upper - lower) / static_cast<double>(options.dp_cells));
          std::cout << std::format(
              "{:<44} {:<34} value error on own rollouts {:7.3f} ({:5.1f}% off by >1), slope error {:7.4f}\n",
              name,
              "",
              errors.value_rmse,
              100.0 * errors.large_fraction,
              errors.slope_rmse);
          report(name, "exact expectation, 41 controls",
                 HeldOutCost(benchmark, ExpectationPolicy<BenchmarkT, std::vector<VF>>(benchmark, table, ControlGrid(41)),
                             NumStages, options.held_out));
          report(name, "exact expectation, controls {-1,0,1}",
                 HeldOutCost(benchmark, ExpectationPolicy<BenchmarkT, std::vector<VF>>(benchmark, table, ControlGrid(3)),
                             NumStages, options.held_out));
          return SuccessResult();
        };
        RESULT_ASSIGN_OR_RETURN(const std::vector<VF> oracle_table,
                                (FitOracleTable<VF>(benchmark, *data, spec, NumStages)));
        RESULT_RETURN_IF_ERROR(score_table(std::format("{} fitted to grid V (oracle)", spec.text), oracle_table));
        RESULT_ASSIGN_OR_RETURN(const StartingTable<VF> start, MakeTable<VF>(benchmark, spec));
        using Solver = DtFbsdeIterativeSolver<2,
                                              1,
                                              16,
                                              NumStages,
                                              256,
                                              typename BenchmarkT::ForwardModel,
                                              QuadraticRegulatorSdeTerminalCostTerm<2>,
                                              VF>;
        const DtFbsdeIterativeOptions<double> solver_options = TwoDimensionalDtOptions(spec, default_slope);
        for (std::size_t index = 0; index < options.num_seeds; ++index) {
          const std::uint64_t seed = options.seed + (1000U * static_cast<std::uint64_t>(index));
          auto solver = std::make_unique<Result<Solver>>(Solver::Make(benchmark.forward_model,
                                                                      benchmark.running_cost,
                                                                      benchmark.terminal_cost,
                                                                      benchmark.initial_distribution,
                                                                      Control(-1.0),
                                                                      Control(1.0),
                                                                      start.initial,
                                                                      start.terminal,
                                                                      solver_options,
                                                                      seed));
          if (!solver->has_value()) {
            return ErrorResult(solver->error().message, solver->error().location);
          }
          double charged = 0.0;
          while (charged < options.time_budget) {
            const Clock::time_point begin = Clock::now();
            RESULT_RETURN_IF_ERROR((*solver)->Iterate());
            charged += std::chrono::duration<double>(Clock::now() - begin).count();
          }
          const std::span<const VF> fitted = (*solver)->ValueFunctions();
          const std::vector<VF> table(fitted.begin(), fitted.end());
          RESULT_RETURN_IF_ERROR(
              score_table(std::format("{} from DT-FBSDE, seed {}, {:.0f} s", spec.text, index, charged), table));
        }
        return SuccessResult();
      }
      if (options.phase == "oracle") {
        RESULT_ASSIGN_OR_RETURN(const OracleResult result,
                                (RunOracle<VF>(benchmark, grid, *data, spec, NumStages, options.held_out)));
        std::cout << std::format(
            "{:<32} gap {:+9.2f}%  rmse on optimal tube {:7.3f}  on own rollouts {:8.3f} ({:5.1f}% off by >1)  "
            "disagree {:5.1f}%\n",
            spec.text,
            100.0 * (result.policy_cost / optimum - 1.0),
            result.value_rmse,
            result.on_policy_rmse,
            100.0 * result.on_policy_large_error,
            100.0 * result.disagreement);
        csv << std::format("{},\"{}\",{},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.4f},{:.3f},{:.5f}\n",
                           problem,
                           spec.text,
                           result.num_parameters,
                           result.policy_cost,
                           optimum,
                           result.value_rmse,
                           result.slope_rmse,
                           result.fit_seconds,
                           result.policy_microseconds,
                           result.effective_samples,
                           result.disagreement);
        csv.flush();
        return SuccessResult();
      }
      RESULT_ASSIGN_OR_RETURN(const StartingTable<VF> table, MakeTable<VF>(benchmark, spec));
      using Solver = DtFbsdeIterativeSolver<2,
                                            1,
                                            16,
                                            NumStages,
                                            256,
                                            typename BenchmarkT::ForwardModel,
                                            QuadraticRegulatorSdeTerminalCostTerm<2>,
                                            VF>;
      const DtFbsdeIterativeOptions<double> solver_options = TwoDimensionalDtOptions(spec, default_slope);
      for (std::size_t index = 0; index < options.num_seeds; ++index) {
        RESULT_ASSIGN_OR_RETURN(const auto outcome,
                                (RunDtFbsde<Solver>(benchmark,
                                                    table,
                                                    solver_options,
                                                    options,
                                                    NumStages,
                                                    options.time_budget,
                                                    index,
                                                    csv,
                                                    problem,
                                                    spec.text)));
        std::cout << std::format("{:<40} seed {}: final gap {:+8.2f}% after {:.1f} s\n",
                                 spec.text,
                                 index,
                                 100.0 * (outcome.first / optimum - 1.0),
                                 outcome.second);
        std::cout.flush();
      }
      return SuccessResult();
    }));
  }
  return SuccessResult();
}

auto StudyDoublePendulum(const StudyOptions& options, std::ofstream& csv) -> Result<> {
  constexpr std::size_t kStages = kL1DoubleInvertedPendulumNumStages;
  RESULT_ASSIGN_OR_RETURN(const L1DoubleInvertedPendulum benchmark, MakeL1DoubleInvertedPendulum());
  std::cout << "\n=== L1 double inverted pendulum ===\n";
  for (const std::string& text : options.models) {
    RESULT_ASSIGN_OR_RETURN(const ModelSpec spec, ParseSpec(text));
    RESULT_RETURN_IF_ERROR(WithModel<4>(spec, [&]<typename VF>(VF* /*tag*/) -> Result<> {
      RESULT_ASSIGN_OR_RETURN(const StartingTable<VF> table, MakeTable<VF>(benchmark, spec));
      // The FBRRT comparison's configuration, which is the one DT-FBSDE has been run with here.
      FbrrtComparisonOptions comparison;
      DtFbsdeIterativeOptions<double> solver_options = DtFbsdeOptions(comparison);
      solver_options.slope_weight = spec.Number("slope", 0.0);
      solver_options.ridge = spec.Number("ridge", 1e-6);
      solver_options.sgd.learning_rate = spec.Number("lr", 0.05);
      solver_options.sgd_epochs = static_cast<std::size_t>(spec.Number("epochs", 50));
      solver_options.normal_equation_memory = spec.Number("memory", 0.0);
      solver_options.reset_normal_equations_on_acceptance = spec.Number("reset", 0.0) != 0.0;
      if (const double inflation = spec.Number("inflation", 1.0); inflation > 1.0) {
        solver_options.base_diffusion_inflation = inflation;
        solver_options.max_diffusion_inflation = inflation;
        solver_options.uninflated_path_fraction = 0.5;
      }
      if (spec.Number("uphill", 0.0) != 0.0) {
        solver_options.stop_uphill_line_search = true;
        solver_options.screening_rollouts = 32;
      }
      ApplyTargetKnobs(spec, solver_options);
      // The batch is a compile-time size: the comparison's 4096, or the 128 the sigma-point study
      // found faster here.
      const auto run = [&]<std::size_t Batch>() -> Result<> {
        using Solver = DtFbsdeIterativeSolver<4,
                                              1,
                                              Batch,
                                              kStages,
                                              256,
                                              L1DoubleInvertedPendulum::ForwardModel,
                                              QuadraticRegulatorSdeTerminalCostTerm<4>,
                                              VF>;
        for (std::size_t index = 0; index < options.num_seeds; ++index) {
          RESULT_ASSIGN_OR_RETURN(const auto outcome,
                                  (RunDtFbsde<Solver>(benchmark,
                                                      table,
                                                      solver_options,
                                                      options,
                                                      kStages,
                                                      options.double_pendulum_time_budget,
                                                      index,
                                                      csv,
                                                      benchmark.name,
                                                      spec.text)));
          std::cout << std::format(
              "{:<40} seed {}: final cost {:9.4f} after {:.1f} s\n", spec.text, index, outcome.first, outcome.second);
          std::cout.flush();
        }
        return SuccessResult();
      };
      const double batch = spec.Number("batch", 4096.0);
      RESULT_ASSERT(batch == 128.0 || batch == 4096.0, "the double pendulum's batch is 128 or 4096.");
      return batch == 128.0 ? run.template operator()<128>() : run.template operator()<4096>();
    }));
  }
  return SuccessResult();
}

auto Run(StudyOptions options) -> Result<> {
  std::ofstream runs(options.runs_path, std::ios::app);
  if (options.phase == "dt") {
    RESULT_ASSERT(runs.good(), "could not open the runs CSV file.");
    if (runs.tellp() == 0) {
      runs << "problem,model,seed,iterations,accepted,failures,seconds,final_cost,mean_girsanov_drift,"
              "mean_first_order_correction,mean_target_magnitude,mean_signed_first_order_correction,"
              "mean_second_order_correction,off_policy_fraction,mean_off_policy_correction,mean_taylor_residual\n";
    }
    options.runs_csv = &runs;
  }
  std::ofstream csv(options.csv_path);
  RESULT_ASSERT(csv.good(), "could not open the CSV file.");
  csv << (options.phase == "oracle"
              ? "problem,model,num_parameters,policy_cost,grid_optimum,value_rmse,slope_rmse,fit_seconds,policy_us,"
                "effective_samples,disagreement\n"
          : options.phase == "policy" ? "problem,value_function,policy,held_out_cost,grid_optimum\n"
          : options.phase == "dagger" ? "problem,model,round,samples_per_stage,held_out_cost,grid_optimum,on_policy_rmse,"
                                        "on_policy_large_error\n"
                                      : "problem,model,seed,iteration,seconds,held_out_cost,line_search_cost\n");
  if (options.include_integrator) {
    RESULT_ASSIGN_OR_RETURN(const L1DoubleIntegrator integrator, MakeL1DoubleIntegrator());
    RESULT_RETURN_IF_ERROR((StudyTwoDimensional<kL1DoubleIntegratorNumStages>(
        integrator, {-2.5, -3.5}, {4.5, 2.5}, 10.0, options, csv)));
  }
  if (options.include_pendulum) {
    RESULT_ASSIGN_OR_RETURN(const L1Pendulum pendulum, MakeL1Pendulum());
    RESULT_RETURN_IF_ERROR(
        (StudyTwoDimensional<kL1PendulumNumStages>(pendulum, {-2.0, -9.0}, {5.0, 9.0}, 1.0, options, csv)));
  }
  if (options.include_double_pendulum && options.phase == "dt") {
    RESULT_RETURN_IF_ERROR(StudyDoublePendulum(options, csv));
  }
  return SuccessResult();
}

auto ParseOptions(int argc, char** argv, StudyOptions& options) -> Result<> {
  const std::vector<std::string_view> arguments(argv + 1, argv + argc);
  for (std::size_t index = 0; index < arguments.size(); ++index) {
    const std::string_view argument = arguments[index];
    RESULT_ASSERT(index + 1 < arguments.size(), "every option needs a value.");
    const std::string value(arguments[++index]);
    if (argument == "--phase") {
      RESULT_ASSERT(value == "oracle" || value == "dt" || value == "policy" || value == "dagger",
                    "--phase is oracle, dt, policy, or dagger.");
      options.phase = value;
    } else if (argument == "--models") {
      options.models = SplitList(value, ';');
    } else if (argument == "--seed") {
      options.seed = std::strtoull(value.c_str(), nullptr, 10);
    } else if (argument == "--seeds") {
      options.num_seeds = std::strtoull(value.c_str(), nullptr, 10);
    } else if (argument == "--time-budget") {
      options.time_budget = std::strtod(value.c_str(), nullptr);
    } else if (argument == "--dip-time-budget") {
      options.double_pendulum_time_budget = std::strtod(value.c_str(), nullptr);
    } else if (argument == "--held-out") {
      options.held_out = std::strtoull(value.c_str(), nullptr, 10);
    } else if (argument == "--only") {
      const std::vector<std::string> problems = SplitList(value, ',');
      const auto has = [&](const char* name) { return std::ranges::find(problems, name) != problems.end(); };
      options.include_integrator = has("integrator");
      options.include_pendulum = has("pendulum");
      options.include_double_pendulum = has("double-pendulum");
    } else if (argument == "--csv") {
      options.csv_path = value;
    } else if (argument == "--runs-csv") {
      options.runs_path = value;
    } else {
      return ErrorResult("unrecognized argument; see the comment at the top of value_model_study_main.cpp.");
    }
  }
  RESULT_ASSERT(!options.models.empty(), "name at least one model with --models.");
  return SuccessResult();
}

}  // namespace
}  // namespace fbsde_traj_opt::examples

auto main(int argc, char** argv) -> int {
  fbsde_traj_opt::examples::StudyOptions options;
  fbsde_traj_opt::Result<> result = fbsde_traj_opt::examples::ParseOptions(argc, argv, options);
  if (result.has_value()) {
    result = fbsde_traj_opt::examples::Run(options);
  }
  if (!result.has_value()) {
    RESULT_REPORT_RESULT(result);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
