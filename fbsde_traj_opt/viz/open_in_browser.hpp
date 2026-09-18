// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#ifndef FBSDE_TRAJ_OPT_VIZ_OPEN_IN_BROWSER_HPP_
#define FBSDE_TRAJ_OPT_VIZ_OPEN_IN_BROWSER_HPP_

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include "fbsde_traj_opt/utils/result.hpp"

namespace fbsde_traj_opt::viz {

// Handing a generated report to the user's browser.
//
// This is the last step of the workflow the visualization exists for: an experiment binary writes
// a PlotlyReport and the plots appear on screen, with no second command to run. There is no
// portable way to do that, so what follows is a list of launchers tried in order until one works.
//
// The launcher logic is separated from the launching -- BrowserLaunchCommands() decides what to
// run, OpenInBrowser() runs it -- because the decision is the part with the platform-specific
// reasoning in it, and it is the part worth testing without spawning a browser.

// The pieces of the environment that decide how a file gets opened.
struct BrowserEnvironment {
  // True when running under the Windows Subsystem for Linux, where none of the usual Linux
  // launchers work: there is no desktop environment for xdg-open to hand off to, and the browser
  // that should open the file lives on the Windows side.
  bool is_wsl = false;

  // The value of $BROWSER, empty if unset. An explicit choice by the user, so it is tried first.
  std::string browser_command{};

  // The value of $WSL_DISTRO_NAME, empty if unset. Needed to name this filesystem from the
  // Windows side; see WindowsPathForWslPath().
  std::string wsl_distro_name{};
};

// One launcher attempt.
struct BrowserLaunchCommand {
  // The program and its arguments, as passed to execvp().
  std::vector<std::string> argv{};

  // Whether a nonzero exit status still counts as success. Set for explorer.exe, which reports
  // failure even when it has successfully handed the file to the default browser -- a long-
  // standing quirk. Only a failure to exec the program at all is then treated as a failure.
  bool ignore_exit_status = false;
};

// Reads the environment: $BROWSER, $WSL_DISTRO_NAME, and whether this is WSL.
[[nodiscard]] auto DetectBrowserEnvironment() noexcept -> BrowserEnvironment;

// Returns the `file://` URL for `absolute_path`, percent-encoding everything outside the
// unreserved set so that a path with spaces or other awkward characters survives.
[[nodiscard]] auto FileUrl(const std::filesystem::path& absolute_path) noexcept -> std::string;

// Returns the Windows path naming `absolute_path` from the Windows side of a WSL install.
//
// A path under `/mnt/<drive>` is already on a Windows volume and maps to `<DRIVE>:\...`. Anything
// else lives in the Linux filesystem, which Windows reaches through the UNC share
// `\\wsl.localhost\<distro>\...` -- hence the need for the distribution name.
//
// Fails if `absolute_path` is outside `/mnt` and `wsl_distro_name` is empty, since there is then
// no way to name the share.
[[nodiscard]] auto WindowsPathForWslPath(const std::filesystem::path& absolute_path,
                                         std::string_view wsl_distro_name) noexcept -> Result<std::string>;

// Returns the launchers to try for `absolute_path` under `environment`, in the order to try them.
//
// The order is: $BROWSER if the user set one, then xdg-open as the Linux standard, then wslview
// (from wslu) and explorer.exe for WSL, then open for macOS. Launchers for other platforms are
// harmless in the list: one that is not installed fails to exec and the next is tried.
[[nodiscard]] auto BrowserLaunchCommands(const std::filesystem::path& absolute_path,
                                         const BrowserEnvironment& environment) noexcept
    -> std::vector<BrowserLaunchCommand>;

// Opens `path` in the user's browser, trying each launcher in turn until one succeeds.
//
// Fails if `path` does not exist, or if every launcher fails -- in which case the report is still
// written and the caller can print its path for the user to open by hand.
[[nodiscard]] auto OpenInBrowser(const std::filesystem::path& path) noexcept -> Result<>;

}  // namespace fbsde_traj_opt::viz

#endif  // FBSDE_TRAJ_OPT_VIZ_OPEN_IN_BROWSER_HPP_
