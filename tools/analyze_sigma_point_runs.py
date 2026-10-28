#!/usr/bin/env python3
"""Per-configuration summary of sigma_point_study's run file: cost, value estimate, iterations, where the time went.

    python3 tools/analyze_sigma_point_runs.py RUNS_CSV [RUNS_CSV ...] [--json OUT.json]

Gaps are against the grid optimum's held-out cost on the 2-D problems; the value estimate V0(x0) is
compared with the grid's own V0(x0), which is what an estimate of the optimal expected cost should
match. The double inverted pendulum's reference is passed with --reference.
"""

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

GRID = {
    "L1 double integrator": {"optimum": 4.1367, "v0": 4.2353},
    "L1 inverted pendulum": {"optimum": 2.2977, "v0": 2.3134},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="+", type=Path)
    parser.add_argument("--reference", action="append", default=[])
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    for item in args.reference:
        name, value = item.split("=")
        GRID[name] = {"optimum": float(value), "v0": math.nan}
    groups = defaultdict(list)
    for path in args.csv:
        with open(path) as f:
            for row in csv.DictReader(f):
                groups[(row["problem"], row["config"])].append(row)
    summary = defaultdict(dict)
    for (problem, config), rows in groups.items():
        ref = GRID.get(problem, {"optimum": math.nan, "v0": math.nan})
        finals = [float(r["final_cost"]) for r in rows]
        gaps = [c / ref["optimum"] - 1 for c in finals]
        v0 = [float(r["v0_estimate"]) for r in rows]
        seconds = [float(r["seconds"]) for r in rows]
        iterations = [int(r["iterations"]) for r in rows]
        accepted = [int(r["accepted"]) for r in rows]
        share = lambda key: statistics.median(float(r[key]) / float(r["seconds"]) for r in rows)
        summary[problem][config] = {
            "batch": int(rows[0]["batch"]),
            "runs": len(rows),
            "failures": sum(int(r["failed"]) for r in rows),
            "gap_median": statistics.median(gaps),
            "gap_spread": max(gaps) - min(gaps),
            "gap_worst": max(gaps),
            "gaps": gaps,
            "v0_median": statistics.median(v0),
            "v0_error_median": statistics.median(abs(v / ref["v0"] - 1) for v in v0) if not math.isnan(ref["v0"]) else None,
            "ms_per_iteration": 1e3 * statistics.median(s / max(i, 1) for s, i in zip(seconds, iterations)),
            "accepted_fraction": statistics.median(a / max(i, 1) for a, i in zip(accepted, iterations)),
            "forward_share": share("forward_seconds"),
            "backward_share": share("backward_seconds"),
            "line_search_share": share("line_search_seconds"),
        }
    for problem, configs in summary.items():
        print(f"\n=== {problem}")
        print(f"{'config':<30} {'batch':>5} {'gap med':>8} {'spread':>7} {'worst':>8} {'V0 med':>8} {'V0 err':>7}"
              f" {'ms/it':>7} {'acc%':>5} {'fwd%':>5} {'bwd%':>5} {'ls%':>5}")
        for config, s in sorted(configs.items(), key=lambda kv: kv[1]["gap_median"]):
            v0e = "   n/a" if s["v0_error_median"] is None else f"{100 * s['v0_error_median']:6.0f}%"
            print(f"{config:<30} {s['batch']:>5} {100 * s['gap_median']:+7.1f}% {100 * s['gap_spread']:6.1f}% "
                  f"{100 * s['gap_worst']:+7.1f}% {s['v0_median']:8.3f} {v0e} {s['ms_per_iteration']:7.2f}"
                  f" {100 * s['accepted_fraction']:5.1f} {100 * s['forward_share']:5.0f} {100 * s['backward_share']:5.0f}"
                  f" {100 * s['line_search_share']:5.0f}")
    if args.json:
        args.json.write_text(json.dumps(summary))


if __name__ == "__main__":
    main()
