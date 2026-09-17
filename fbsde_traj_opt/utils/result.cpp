// Copyright 2026 Kelsey P. Hawkins.
// SPDX-License-Identifier: MIT

#include "fbsde_traj_opt/utils/result.hpp"

#include <iostream>
#include <ostream>

namespace fbsde_traj_opt {

std::ostream& operator<<(std::ostream& os, const ExpectedError& error) {
  return os << error.message << " [" << error.location.file_name() << ":" << error.location.line() << ", in "
            << error.location.function_name() << "]";
}

bool ReportError(const ExpectedError& error) {
  std::cerr << error << '\n';
  return false;
}

}  // namespace fbsde_traj_opt
