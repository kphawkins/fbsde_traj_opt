#!/usr/bin/env python3
"""Summarizes the line search study: python3 analyze.py [results_dir].

For every problem and configuration: the median held-out cost at a few charged times, the final costs, how many
runs reached the threshold (2% above the grid optimum in 2-D, 4.5 on the double inverted pendulum), and the
acceptance rate and time per iteration from results/iterations_summary.csv. Runs named `<config>B` are the second
round's 8 seeds (--seed 7); they are pooled with the first round's 4 seeds of the same configuration.
"""

import collections
import csv
import os
import statistics
import sys

PROBLEMS = {
    "integrator": [2, 5, 10, 20, 40, 60],
    "pendulum": [2, 5, 10, 20, 40, 60],
    "double-pendulum": [10, 20, 40, 60, 90, 120],
}
CONFIGS = ["base", "half", "quarter", "adapt", "pool4", "pool16", "anchor03", "anchor3"]


def load_runs(results, config, problem):
    runs = []
    for name in (config, config + "B"):
        path = os.path.join(results, f"{name}.{problem}.ckpt.csv")
        if not os.path.exists(path):
            continue
        by_seed = collections.defaultdict(list)
        for row in csv.DictReader(open(path)):
            by_seed[row["seed"]].append((float(row["seconds"]), float(row["held_out_cost"])))
        runs.extend(by_seed.values())
    return runs


def cost_at(run, seconds):
    cost = None
    for time, value in run:
        if time <= seconds:
            cost = value
    return cost


def main():
    results = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "results")
    summary = collections.defaultdict(list)
    for row in csv.DictReader(open(os.path.join(results, "iterations_summary.csv"))):
        summary[(row["config"].removesuffix("B"), row["problem"])].append(row)
    for problem, times in PROBLEMS.items():
        reference = os.path.join(results, f"{problem}.ref.csv")
        optimum = float(next(csv.DictReader(open(reference)))["grid_optimum"]) if os.path.exists(reference) else None
        threshold = 1.02 * optimum if optimum else 4.5
        print(f"\n== {problem}" + (f" (grid optimum {optimum:.4f})" if optimum else ""))
        header = " ".join(f"{t:>7}s" for t in times)
        print(f"{'config':9} {'runs':>4} {header}   {'final range':>17} {'<= thr':>6} {'acc %':>6} {'ms/it':>7}")
        for config in CONFIGS:
            runs = load_runs(results, config, problem)
            if not runs:
                continue
            medians = " ".join(f"{statistics.median([cost_at(r, t) for r in runs]):8.3f}" for t in times)
            finals = [r[-1][1] for r in runs]
            reached = sum(f <= threshold for f in finals)
            rows = summary[(config, problem)]
            iterations = sum(int(r["iterations"]) for r in rows)
            accepted = sum(int(r["accepted"]) for r in rows)
            seconds = sum(float(r["forward_s"]) + float(r["backward_s"]) + float(r["line_search_s"]) for r in rows)
            print(f"{config:9} {len(runs):4} {medians}   [{min(finals):7.3f},{max(finals):7.3f}] "
                  f"{reached:3}/{len(runs):<2} {100 * accepted / iterations:6.1f} {1000 * seconds / iterations:7.1f}")


if __name__ == "__main__":
    main()
