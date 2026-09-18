// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#include "fbsde_traj_opt/viz/open_in_browser.hpp"

#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include "fbsde_traj_opt/utils/result.hpp"

namespace fbsde_traj_opt::viz {
namespace {

// Returns `name` from the environment, or an empty string if it is unset.
auto EnvironmentVariable(const char* name) noexcept -> std::string {
  const char* const value = std::getenv(name);
  return value == nullptr ? std::string() : std::string(value);
}

// Returns `text` lowercased, for case-insensitive matching.
auto Lowercased(std::string_view text) noexcept -> std::string {
  std::string lowered(text);
  std::ranges::transform(lowered, lowered.begin(), [](unsigned char character) noexcept {
    return static_cast<char>(std::tolower(character));
  });
  return lowered;
}

// True for the characters RFC 3986 calls unreserved, plus `/`, which separates path segments and
// must stay literal.
auto IsSafeInUrlPath(char character) noexcept -> bool {
  const auto unsigned_character = static_cast<unsigned char>(character);
  return (std::isalnum(unsigned_character) != 0) || character == '-' || character == '.' || character == '_' ||
         character == '~' || character == '/';
}

// Runs one launcher, waiting for it to finish. Returns true if it opened the file.
auto RunLaunchCommand(const BrowserLaunchCommand& command) noexcept -> bool {
  if (command.argv.empty()) {
    return false;
  }

  // execvp() wants a null-terminated array of mutable C strings. The strings themselves are only
  // read, but the signature predates const-correctness.
  std::vector<char*> argv;
  argv.reserve(command.argv.size() + 1);
  for (const std::string& argument : command.argv) {
    argv.push_back(const_cast<char*>(argument.c_str()));  // NOLINT(cppcoreguidelines-pro-type-const-cast)
  }
  argv.push_back(nullptr);

  // The exit status a failed exec reports, distinct from anything a launcher would return itself.
  // It is what lets a launcher that is simply not installed be told apart from one that ran and
  // complained -- the distinction explorer.exe's ignore_exit_status depends on.
  constexpr int kExecFailedStatus = 127;

  const pid_t child = fork();
  if (child < 0) {
    return false;
  }
  if (child == 0) {
    // Detach from the terminal: a launcher that decides to print is not this program's output,
    // and a browser that inherits stdout will interleave with it for as long as it runs.
    const int null_descriptor = open("/dev/null", O_WRONLY);
    if (null_descriptor >= 0) {
      dup2(null_descriptor, STDOUT_FILENO);
      dup2(null_descriptor, STDERR_FILENO);
      if (null_descriptor > STDERR_FILENO) {
        close(null_descriptor);
      }
    }
    execvp(argv[0], argv.data());
    _exit(kExecFailedStatus);
  }

  int status = 0;
  if (waitpid(child, &status, 0) < 0) {
    return false;
  }
  if (!WIFEXITED(status)) {
    return false;
  }

  const int exit_status = WEXITSTATUS(status);
  if (exit_status == kExecFailedStatus) {
    return false;
  }
  return command.ignore_exit_status || exit_status == 0;
}

}  // namespace

auto DetectBrowserEnvironment() noexcept -> BrowserEnvironment {
  BrowserEnvironment environment;
  environment.browser_command = EnvironmentVariable("BROWSER");
  environment.wsl_distro_name = EnvironmentVariable("WSL_DISTRO_NAME");

  // WSL advertises itself in the kernel release string, which is the one marker present in every
  // WSL version and independent of how the distribution was installed.
  std::ifstream release("/proc/sys/kernel/osrelease");
  std::string release_text;
  if (release.is_open()) {
    std::getline(release, release_text);
  }
  environment.is_wsl = Lowercased(release_text).find("microsoft") != std::string::npos;

  return environment;
}

auto FileUrl(const std::filesystem::path& absolute_path) noexcept -> std::string {
  const std::string path = absolute_path.string();

  std::string url = "file://";
  url.reserve(url.size() + path.size());
  for (const char character : path) {
    if (IsSafeInUrlPath(character)) {
      url += character;
    } else {
      constexpr std::array<char, 16> kHexDigits{
          '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F'};
      const auto value = static_cast<unsigned char>(character);
      url += '%';
      url += kHexDigits.at(value >> 4U);
      url += kHexDigits.at(value & 0x0FU);
    }
  }
  return url;
}

auto WindowsPathForWslPath(const std::filesystem::path& absolute_path, std::string_view wsl_distro_name) noexcept
    -> Result<std::string> {
  const std::string path = absolute_path.string();

  // `/mnt/c/users/...` is a Windows volume mounted into the Linux filesystem; it maps straight
  // back to `C:\users\...`.
  constexpr std::string_view kMountPrefix = "/mnt/";
  if (path.size() > kMountPrefix.size() + 1 && path.starts_with(kMountPrefix) &&
      (std::isalpha(static_cast<unsigned char>(path[kMountPrefix.size()])) != 0) &&
      path[kMountPrefix.size() + 1] == '/') {
    std::string windows_path;
    windows_path += static_cast<char>(std::toupper(static_cast<unsigned char>(path[kMountPrefix.size()])));
    windows_path += ':';
    windows_path += path.substr(kMountPrefix.size() + 1);
    std::ranges::replace(windows_path, '/', '\\');
    return SuccessResult(windows_path);
  }

  RESULT_ASSERT(!wsl_distro_name.empty(),
                "WindowsPathForWslPath: WSL_DISTRO_NAME is unset, so the Linux filesystem cannot be named from "
                "Windows. Write the report under /mnt, or set BROWSER.");

  std::string windows_path = R"(\\wsl.localhost\)";
  windows_path += wsl_distro_name;
  windows_path += path;
  std::ranges::replace(windows_path, '/', '\\');
  return SuccessResult(windows_path);
}

auto BrowserLaunchCommands(const std::filesystem::path& absolute_path, const BrowserEnvironment& environment) noexcept
    -> std::vector<BrowserLaunchCommand> {
  const std::string url = FileUrl(absolute_path);
  const std::string path = absolute_path.string();

  std::vector<BrowserLaunchCommand> commands;

  if (!environment.browser_command.empty()) {
    commands.push_back({.argv = {environment.browser_command, url}, .ignore_exit_status = false});
  }

  if (environment.is_wsl) {
    // wslview understands a Linux path and hands it to the Windows default handler. explorer.exe
    // needs the path spelled the Windows way, and only gets added when that spelling exists.
    commands.push_back({.argv = {"wslview", path}, .ignore_exit_status = false});

    const Result<std::string> windows_path = WindowsPathForWslPath(absolute_path, environment.wsl_distro_name);
    if (windows_path.has_value()) {
      commands.push_back({.argv = {"explorer.exe", *windows_path}, .ignore_exit_status = true});
    }
  } else {
    // xdg-open is the Linux standard, but on WSL it finds no desktop environment to hand off to
    // and fails slowly, so it is skipped there rather than tried first.
    commands.push_back({.argv = {"xdg-open", url}, .ignore_exit_status = false});
  }

  commands.push_back({.argv = {"open", url}, .ignore_exit_status = false});

  return commands;
}

auto OpenInBrowser(const std::filesystem::path& path) noexcept -> Result<> {
  std::error_code error;
  const std::filesystem::path absolute_path = std::filesystem::absolute(path, error);
  RESULT_ASSERT(!error, "OpenInBrowser: the path could not be resolved to an absolute path.");
  RESULT_ASSERT(std::filesystem::exists(absolute_path), "OpenInBrowser: there is no file at the given path.");

  for (const BrowserLaunchCommand& command : BrowserLaunchCommands(absolute_path, DetectBrowserEnvironment())) {
    if (RunLaunchCommand(command)) {
      return SuccessResult();
    }
  }

  return ErrorResult(
      "OpenInBrowser: no browser launcher succeeded. Set BROWSER to a command that opens a URL, or open the file "
      "by hand.");
}

}  // namespace fbsde_traj_opt::viz
