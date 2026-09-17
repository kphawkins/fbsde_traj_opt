// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#ifndef FBSDE_TRAJ_OPT_UTILS_RESULT_HPP_
#define FBSDE_TRAJ_OPT_UTILS_RESULT_HPP_

#include <expected>
#include <iosfwd>
#include <source_location>
#include <string_view>
#include <type_traits>
#include <utility>

namespace fbsde_traj_opt {

// The error payload carried by every failed Result<T>.
//
// `message` is expected to point at a string literal (or other storage that outlives the
// ExpectedError) -- the type stores a view rather than an owned std::string so that constructing
// an error never allocates. `location` defaults to the call site of whichever helper created the
// error (ErrorResult(), RESULT_ASSERT, ...), which is what makes error reports point at the
// actual failure instead of wherever the Result was eventually inspected.
struct ExpectedError {
  std::string_view message;
  std::source_location location = std::source_location::current();
};

// Pretty-prints `error` as "<message> [<file>:<line>, in <function>]".
//
// Defining this operator is also what lets GoogleTest's UniversalPrinter find it through ADL, so
// EXPECT_EQ/ASSERT_EQ on a failed Result<T> print something readable instead of a byte dump.
std::ostream& operator<<(std::ostream& os, const ExpectedError& error);

// std::expected specialized to this project's error type. `T` defaults to void, for operations
// that either succeed with no value to report or fail with an ExpectedError -- see the Divide()
// out-parameter overload below.
template <typename T = void>
using Result = std::expected<T, ExpectedError>;

// ---------------------------------------------------------------------------------------------
// Helpers
//
// Together these let most functions be written without ever naming std::expected, std::unexpected
// or ExpectedError directly:
//
//   auto Divide(int a, int b) noexcept -> Result<int> {
//     RESULT_ASSERT(b != 0, "Cannot divide by zero.");
//     return SuccessResult(a / b);
//   }
//
//   auto Divide(int a, int b, int& divide_out) noexcept -> Result<> {
//     if (b == 0) {
//       return ErrorResult("Cannot divide by zero.");
//     }
//     divide_out = a / b;
//     return SuccessResult();
//   }
// ---------------------------------------------------------------------------------------------

// Builds a successful Result<void>.
inline auto SuccessResult() noexcept -> Result<> {
  return {};
}

// Builds a successful Result<std::decay_t<T>> holding `value`.
template <typename T>
auto SuccessResult(T&& value) noexcept(std::is_nothrow_constructible_v<std::decay_t<T>, T&&>)
    -> Result<std::decay_t<T>> {
  return Result<std::decay_t<T>>(std::forward<T>(value));
}

// Builds an error, capturing the call site by default.
//
// The return type converts to any fbsde_traj_opt::Result<T>, so `return ErrorResult(...);` works
// regardless of which T the enclosing function's Result<T> uses -- exactly like returning
// std::unexpected(...) does for a bare std::expected.
inline auto ErrorResult(std::string_view message,
                        std::source_location location = std::source_location::current()) noexcept
    -> std::unexpected<ExpectedError> {
  return std::unexpected<ExpectedError>(ExpectedError{.message = message, .location = location});
}

// Pretty-prints `error` to stderr. Returns false, so it composes with early-return macros.
bool ReportError(const ExpectedError& error);

// Pretty-prints `result`'s error to stderr if it failed; does nothing on success. Returns
// `result.has_value()`.
template <typename T>
auto ReportResult(const Result<T>& result) noexcept -> bool {
  return result.has_value() || ReportError(result.error());
}

}  // namespace fbsde_traj_opt

// ---------------------------------------------------------------------------------------------
// Macros
// ---------------------------------------------------------------------------------------------

// Generates an error: returns ErrorResult(message) from the enclosing function unless `condition`
// holds. The enclosing function must return some fbsde_traj_opt::Result<T>.
#define RESULT_ASSERT(condition, message)            \
  do {                                               \
    if (!(condition)) {                              \
      return ::fbsde_traj_opt::ErrorResult(message); \
    }                                                \
  } while (false)

// Reports (pretty-prints to stderr) a standalone ExpectedError.
#define RESULT_REPORT_ERROR(error) (::fbsde_traj_opt::ReportError(error))

// Reports a Result<T>: pretty-prints its error to stderr if it failed, does nothing on success.
#define RESULT_REPORT_RESULT(result) (::fbsde_traj_opt::ReportResult(result))

#endif  // FBSDE_TRAJ_OPT_UTILS_RESULT_HPP_
