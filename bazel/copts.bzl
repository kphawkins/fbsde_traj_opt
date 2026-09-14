"""Shared compiler options for first-party fbsde_traj_opt targets.

These live in per-target `copts` rather than in a global `--copt` in .bazelrc on purpose. A global
`-Werror` would also apply to the compiled sources of external modules -- abseil and re2 arrive
transitively through googletest -- and those are not warning-clean under this flag set.

Warnings arising from inside Eigen's headers are handled separately, by
`--features=external_include_paths` in .bazelrc, which makes external repositories' include paths
system include paths again.
"""

FBSDE_WARNING_COPTS = [
    "-Wall",
    "-Wextra",
    "-Wpedantic",
    "-Wshadow",
    "-Wnon-virtual-dtor",
    "-Woverloaded-virtual",
    "-Wold-style-cast",
    "-Wdouble-promotion",
    "-Wformat=2",
    "-Werror",
]

FBSDE_COPTS = FBSDE_WARNING_COPTS

# GoogleTest's macros expand into code that is attributed to our source location, which
# occasionally trips the stricter cast diagnostics. Tests therefore get a slightly relaxed set.
FBSDE_TEST_COPTS = [copt for copt in FBSDE_WARNING_COPTS if copt != "-Wold-style-cast"]
