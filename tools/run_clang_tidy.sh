#!/usr/bin/env bash
#
# Runs clang-tidy over the files passed by pre-commit.
#
# clang-tidy needs the same compiler flags Bazel would use. There are two ways to supply them,
# and this script prefers the first:
#
#   1. A compile_commands.json in the repository root. Generate one with a tool such as
#      hedronvision/bazel-compile-commands-extractor. This is the correct answer once the build
#      grows generated headers, more dependencies, or select()-conditional copts.
#
#   2. A hand-maintained flag set, below. Viable only because the project's compile flags are
#      currently four items: the standard, the include root, Eigen, and GoogleTest. When this
#      stops being true, switch to (1); nothing else in this script has to change.
#
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "${repo_root}"

if [[ $# -eq 0 ]]; then
  exit 0
fi

if [[ -f compile_commands.json ]]; then
  exec clang-tidy --quiet -p "${repo_root}" "$@"
fi

# --- Fallback: synthesize the flag set. ---

if ! command -v bazel >/dev/null 2>&1; then
  echo "run_clang_tidy.sh: bazel not found on PATH, and no compile_commands.json present." >&2
  echo "Install Bazelisk, or generate a compilation database." >&2
  exit 1
fi

output_base="$(bazel info output_base)"
external="${output_base}/external"

# Canonical repository names under bzlmod are versioned and the separator has changed across
# Bazel releases ("eigen~" on Bazel 7, "eigen+" on Bazel 8+), so glob rather than hardcode.
find_external() {
  local pattern="$1"
  local match
  match="$(find "${external}" -maxdepth 1 -name "${pattern}" -print -quit 2>/dev/null || true)"
  if [[ -z "${match}" ]]; then
    echo "run_clang_tidy.sh: could not locate external repository '${pattern}'." >&2
    echo "Run 'bazel build //...' once to fetch dependencies, then retry." >&2
    exit 1
  fi
  printf '%s' "${match}"
}

eigen_include="$(find_external 'eigen*')"
googletest_include="$(find_external 'googletest*')/googletest/include"

exec clang-tidy --quiet "$@" -- \
  -std=c++23 \
  -I"${repo_root}" \
  -isystem "${eigen_include}" \
  -isystem "${googletest_include}"
