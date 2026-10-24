#!/usr/bin/env bash
#
# Provisions a Claude Code cloud environment to build and test this repository.
#
# Paste this file's contents into the environment's setup script (claude.ai/code > environment
# settings). Setup scripts run as root on Ubuntu 24.04 before the session starts, and the result
# is snapshotted, so this runs once per environment rather than once per session.
#
# It installs the same Clang major release CI uses (the image's default Clang is too old for the
# C++23 this project uses) and bazelisk as `bazel`, then downloads the Bazel version pinned in
# .bazelversion so the first build doesn't have to.
#
# The environment's network access must reach apt.llvm.org and *.bazel.build. Neither is on the
# default Trusted list: use Custom with those two domains plus the default list, or Full.

set -euo pipefail

readonly LLVM_VERSION=21
readonly BAZEL_VERSION=9.2.0

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y lsb-release wget gnupg software-properties-common

wget -qO /tmp/llvm.sh https://apt.llvm.org/llvm.sh
chmod +x /tmp/llvm.sh
/tmp/llvm.sh "${LLVM_VERSION}"
apt-get install -y "clang-tidy-${LLVM_VERSION}" "clang-format-${LLVM_VERSION}"
for tool in clang clang++ clang-tidy clang-format; do
  ln -sf "/usr/lib/llvm-${LLVM_VERSION}/bin/${tool}" "/usr/local/bin/${tool}"
done

npm install -g @bazel/bazelisk
ln -sf "$(command -v bazelisk)" /usr/local/bin/bazel
USE_BAZEL_VERSION="${BAZEL_VERSION}" bazel version
