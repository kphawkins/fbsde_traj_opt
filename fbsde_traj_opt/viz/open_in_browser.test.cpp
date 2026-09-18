// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#include "fbsde_traj_opt/viz/open_in_browser.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace fbsde_traj_opt::viz {
namespace {

// Returns the program name of each launcher in `commands`, in order.
auto LauncherNames(const std::vector<BrowserLaunchCommand>& commands) -> std::vector<std::string> {
  std::vector<std::string> names;
  names.reserve(commands.size());
  for (const BrowserLaunchCommand& command : commands) {
    names.push_back(command.argv.front());
  }
  return names;
}

TEST(FileUrlTest, PlainPathBecomesAFileUrl) {
  EXPECT_EQ(FileUrl("/home/user/report.html"), "file:///home/user/report.html");
}

TEST(FileUrlTest, SeparatorsAreLeftLiteral) {
  EXPECT_EQ(FileUrl("/a/b/c"), "file:///a/b/c");
}

TEST(FileUrlTest, AwkwardCharactersArePercentEncoded) {
  EXPECT_EQ(FileUrl("/tmp/my report #1.html"), "file:///tmp/my%20report%20%231.html");
}

TEST(FileUrlTest, UnreservedCharactersAreNotEncoded) {
  EXPECT_EQ(FileUrl("/tmp/a-b_c.d~e"), "file:///tmp/a-b_c.d~e");
}

TEST(WindowsPathForWslPathTest, MountedWindowsVolumeMapsBackToItsDriveLetter) {
  const auto windows_path = WindowsPathForWslPath("/mnt/c/Users/me/report.html", "Ubuntu");
  ASSERT_TRUE(windows_path.has_value()) << windows_path.error();

  EXPECT_EQ(*windows_path, R"(C:\Users\me\report.html)");
}

TEST(WindowsPathForWslPathTest, DriveLetterIsUppercased) {
  const auto windows_path = WindowsPathForWslPath("/mnt/d/data/report.html", "Ubuntu");
  ASSERT_TRUE(windows_path.has_value()) << windows_path.error();

  EXPECT_EQ(*windows_path, R"(D:\data\report.html)");
}

TEST(WindowsPathForWslPathTest, LinuxPathMapsToTheDistributionShare) {
  const auto windows_path = WindowsPathForWslPath("/home/me/out/report.html", "Ubuntu-24.04");
  ASSERT_TRUE(windows_path.has_value()) << windows_path.error();

  EXPECT_EQ(*windows_path, R"(\\wsl.localhost\Ubuntu-24.04\home\me\out\report.html)");
}

TEST(WindowsPathForWslPathTest, LinuxPathWithoutADistributionNameFails) {
  EXPECT_FALSE(WindowsPathForWslPath("/home/me/report.html", "").has_value());
}

// A path that merely begins with the letters of the mount prefix is not a mounted volume.
TEST(WindowsPathForWslPathTest, MountLookalikePathsGoToTheShare) {
  const auto windows_path = WindowsPathForWslPath("/mnt/data/report.html", "Ubuntu");
  ASSERT_TRUE(windows_path.has_value()) << windows_path.error();

  EXPECT_EQ(*windows_path, R"(\\wsl.localhost\Ubuntu\mnt\data\report.html)");
}

TEST(BrowserLaunchCommandsTest, ExplicitBrowserIsTriedFirst) {
  const BrowserEnvironment environment{.is_wsl = false, .browser_command = "my-browser", .wsl_distro_name = ""};

  const std::vector<BrowserLaunchCommand> commands = BrowserLaunchCommands("/tmp/report.html", environment);

  ASSERT_FALSE(commands.empty());
  EXPECT_EQ(commands.front().argv, (std::vector<std::string>{"my-browser", "file:///tmp/report.html"}));
}

TEST(BrowserLaunchCommandsTest, PlainLinuxTriesXdgOpen) {
  const BrowserEnvironment environment{.is_wsl = false, .browser_command = "", .wsl_distro_name = ""};

  const std::vector<std::string> names = LauncherNames(BrowserLaunchCommands("/tmp/report.html", environment));

  EXPECT_EQ(names, (std::vector<std::string>{"xdg-open", "open"}));
}

TEST(BrowserLaunchCommandsTest, WslSkipsXdgOpenAndReachesForTheWindowsSide) {
  const BrowserEnvironment environment{.is_wsl = true, .browser_command = "", .wsl_distro_name = "Ubuntu"};

  const std::vector<BrowserLaunchCommand> commands = BrowserLaunchCommands("/home/me/report.html", environment);
  const std::vector<std::string> names = LauncherNames(commands);

  EXPECT_EQ(names, (std::vector<std::string>{"wslview", "explorer.exe", "open"}));
  // explorer.exe takes the Windows spelling of the path, not the URL.
  EXPECT_EQ(commands.at(1).argv.at(1), R"(\\wsl.localhost\Ubuntu\home\me\report.html)");
}

// explorer.exe reports a nonzero exit status even when it has successfully opened the file, so it
// is the one launcher whose exit status is not believed.
TEST(BrowserLaunchCommandsTest, OnlyExplorerIgnoresItsExitStatus) {
  const BrowserEnvironment environment{.is_wsl = true, .browser_command = "my-browser", .wsl_distro_name = "Ubuntu"};

  for (const BrowserLaunchCommand& command : BrowserLaunchCommands("/home/me/report.html", environment)) {
    EXPECT_EQ(command.ignore_exit_status, command.argv.front() == "explorer.exe") << command.argv.front();
  }
}

TEST(BrowserLaunchCommandsTest, WslWithoutADistributionNameOmitsExplorer) {
  const BrowserEnvironment environment{.is_wsl = true, .browser_command = "", .wsl_distro_name = ""};

  const std::vector<std::string> names = LauncherNames(BrowserLaunchCommands("/home/me/report.html", environment));

  EXPECT_EQ(names, (std::vector<std::string>{"wslview", "open"}));
}

TEST(BrowserLaunchCommandsTest, WslUnderAMountedVolumeNeedsNoDistributionName) {
  const BrowserEnvironment environment{.is_wsl = true, .browser_command = "", .wsl_distro_name = ""};

  const std::vector<BrowserLaunchCommand> commands = BrowserLaunchCommands("/mnt/c/tmp/report.html", environment);

  ASSERT_EQ(commands.size(), 3U);
  EXPECT_EQ(commands.at(1).argv.at(1), R"(C:\tmp\report.html)");
}

TEST(OpenInBrowserTest, MissingFileFails) {
  const std::filesystem::path path = std::filesystem::temp_directory_path() / "fbsde_no_such_report.html";
  std::filesystem::remove(path);

  EXPECT_FALSE(OpenInBrowser(path).has_value());
}

// Not a launch: BROWSER is pointed at `true`, which exits zero without doing anything, so the
// success path is exercised without a browser window appearing.
TEST(OpenInBrowserTest, SucceedsWhenTheFirstLauncherSucceeds) {
  const std::filesystem::path path = std::filesystem::temp_directory_path() / "fbsde_open_in_browser_test.html";
  {
    const std::ofstream file(path);
  }

  const std::string previous_browser = DetectBrowserEnvironment().browser_command;
  ASSERT_EQ(setenv("BROWSER", "true", 1), 0);

  EXPECT_TRUE(OpenInBrowser(path).has_value());

  if (previous_browser.empty()) {
    ASSERT_EQ(unsetenv("BROWSER"), 0);
  } else {
    ASSERT_EQ(setenv("BROWSER", previous_browser.c_str(), 1), 0);
  }
  std::filesystem::remove(path);
}

}  // namespace
}  // namespace fbsde_traj_opt::viz
