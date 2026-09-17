// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#include "fbsde_traj_opt/utils/result.hpp"

#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>

#include <gtest/gtest.h>

namespace fbsde_traj_opt {
namespace {

// Mirrors the two usage patterns documented in result.hpp: a value-returning Result<T> built with
// RESULT_ASSERT + SuccessResult(), and an out-parameter Result<> built with an explicit early
// return + SuccessResult().
auto Divide(int a, int b) noexcept -> Result<int> {
  RESULT_ASSERT(b != 0, "Cannot divide by zero.");
  return SuccessResult(a / b);
}

auto Divide(int a, int b, int& divide_out) noexcept -> Result<> {
  if (b == 0) {
    return ErrorResult("Cannot divide by zero.");
  }
  divide_out = a / b;
  return SuccessResult();
}

TEST(DivideTest, SucceedsWithAValueResult) {
  const Result<int> result = Divide(6, 3);

  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, 2);
}

TEST(DivideTest, FailsWithAValueResult) {
  const Result<int> result = Divide(6, 0);

  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error().message, "Cannot divide by zero.");
}

TEST(DivideTest, SucceedsWithAnOutParameterResult) {
  int quotient = 0;

  const Result<> result = Divide(6, 3, quotient);

  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(quotient, 2);
}

TEST(DivideTest, FailsWithAnOutParameterResultAndLeavesItUntouched) {
  int quotient = -1;

  const Result<> result = Divide(6, 0, quotient);

  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error().message, "Cannot divide by zero.");
  EXPECT_EQ(quotient, -1);
}

TEST(SuccessResultTest, WithNoArgumentsBuildsAnEngagedResultOfVoid) {
  const Result<> result = SuccessResult();

  EXPECT_TRUE(result.has_value());
}

TEST(SuccessResultTest, WithAValueBuildsAnEngagedResultHoldingIt) {
  const Result<int> result = SuccessResult(42);

  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, 42);
}

TEST(SuccessResultTest, DecaysTheValueTypeSoALvalueIsCopiedRatherThanReferenced) {
  const std::string original = "hello";

  const auto result = SuccessResult(original);

  static_assert(std::is_same_v<decltype(result), const Result<std::string>>);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, "hello");
}

TEST(ErrorResultTest, ConvertsToAResultCarryingTheMessage) {
  const Result<int> result = ErrorResult("Something went wrong.");

  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error().message, "Something went wrong.");
}

TEST(ErrorResultTest, CapturesTheCallSiteByDefault) {
  const Result<> result = ErrorResult("Something went wrong.");
  constexpr int kExpectedLine = __LINE__ - 1;

  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error().location.line(), kExpectedLine);
  EXPECT_NE(std::string_view(result.error().location.function_name()).find("ErrorResultTest_CapturesTheCallSite"),
            std::string_view::npos);
}

TEST(ExpectedErrorPrintTest, IncludesTheMessageFileLineAndFunction) {
  const ExpectedError error{"Something went wrong."};
  const int expected_line = error.location.line();

  std::ostringstream stream;
  stream << error;
  const std::string printed = stream.str();

  EXPECT_NE(printed.find("Something went wrong."), std::string::npos);
  EXPECT_NE(printed.find("result.test.cpp"), std::string::npos);
  EXPECT_NE(printed.find(std::to_string(expected_line)), std::string::npos);
  EXPECT_NE(printed.find("IncludesTheMessageFileLineAndFunction"), std::string::npos);
}

TEST(ReportErrorTest, PrintsTheErrorToStderrAndReturnsFalse) {
  const ExpectedError error{"Something went wrong."};

  testing::internal::CaptureStderr();
  const bool reported_success = ReportError(error);
  const std::string printed = testing::internal::GetCapturedStderr();

  EXPECT_FALSE(reported_success);
  EXPECT_NE(printed.find("Something went wrong."), std::string::npos);
}

TEST(ReportResultTest, PrintsNothingAndReturnsTrueOnSuccess) {
  const Result<int> result = SuccessResult(1);

  testing::internal::CaptureStderr();
  const bool reported_success = ReportResult(result);
  const std::string printed = testing::internal::GetCapturedStderr();

  EXPECT_TRUE(reported_success);
  EXPECT_TRUE(printed.empty());
}

TEST(ReportResultTest, PrintsTheErrorAndReturnsFalseOnFailure) {
  const Result<int> result = ErrorResult("Cannot divide by zero.");

  testing::internal::CaptureStderr();
  const bool reported_success = ReportResult(result);
  const std::string printed = testing::internal::GetCapturedStderr();

  EXPECT_FALSE(reported_success);
  EXPECT_NE(printed.find("Cannot divide by zero."), std::string::npos);
}

TEST(ResultReportErrorMacroTest, BehavesLikeReportError) {
  const ExpectedError error{"Macro-reported error."};

  testing::internal::CaptureStderr();
  const bool reported_success = RESULT_REPORT_ERROR(error);
  const std::string printed = testing::internal::GetCapturedStderr();

  EXPECT_FALSE(reported_success);
  EXPECT_NE(printed.find("Macro-reported error."), std::string::npos);
}

TEST(ResultReportResultMacroTest, BehavesLikeReportResult) {
  const Result<int> result = ErrorResult("Macro-reported result error.");

  testing::internal::CaptureStderr();
  const bool reported_success = RESULT_REPORT_RESULT(result);
  const std::string printed = testing::internal::GetCapturedStderr();

  EXPECT_FALSE(reported_success);
  EXPECT_NE(printed.find("Macro-reported result error."), std::string::npos);
}

}  // namespace
}  // namespace fbsde_traj_opt
