#!/usr/bin/env python3
"""Compares DT-FBSDE's old and new method_comparison defaults on the held-out rollouts.

    python3 comparison_runs/default_taylor_poly4/analyze.py comparison_runs/default_taylor_poly4 [--json OUT]

The directory holds old.csv and new.csv (method_comparison --csv output) and old_reference.csv. Old is
double-greedy targets with a degree-6 polynomial in 2-D and the soft-min model on the DIP; new is the
Taylor noiseless target with a degree-4 polynomial everywhere. For every problem and method it reports the
held-out cost each run reached by a set of charged times, as the median over seeds, and the relative
gap to the grid DP optimum on the 2-D benchmarks.
"""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

TIMES = (1.0, 5.0, 15.0, 30.0, None)  # None: the end of the run.


def load(path: Path):
    runs = defaultdict(lambda: {"seconds": [], "cost": [], "iteration": []})
    with open(path) as f:
        for row in csv.DictReader(f):
            run = runs[(row["problem"], row["method"], int(row["seed"]))]
            run["seconds"].append(float(row["seconds"]))
            run["cost"].append(float(row["held_out_cost"]))
            run["iteration"].append(int(row["iteration"]))
    return runs


def cost_at(run, time):
    """The held-out cost of the last checkpoint at or before `time` charged seconds."""
    if time is None:
        return run["cost"][-1]
    cost = run["cost"][0]
    for seconds, value in zip(run["seconds"], run["cost"]):
        if seconds > time:
            break
        cost = value
    return cost


def references(path: Path):
    optima = {}
    if path.exists():
        with open(path) as f:
            for row in csv.DictReader(f):
                optima[row["problem"]] = float(row["grid_optimum"])
    return optima


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    configs = {name: load(args.run_dir / f"{name}.csv") for name in ("old", "new")}
    optima = references(args.run_dir / "old_reference.csv")

    summary = []
    keys = sorted({(problem, method) for runs in configs.values() for problem, method, _ in runs})
    for problem, method in keys:
        print(f"\n{problem} / {method}" + (f"  (grid optimum {optima[problem]:.4f})" if problem in optima else ""))
        print(f"  {'config':<6}" + "".join(f"{('t=' + str(t) + 's') if t else 'final':>12}" for t in TIMES)
              + f"{'final min':>11}{'final max':>11}{'iters':>8}")
        for name, runs in configs.items():
            seeds = [run for (p, m, _), run in sorted(runs.items()) if p == problem and m == method]
            if not seeds:
                continue
            medians = [statistics.median(cost_at(run, t) for run in seeds) for t in TIMES]
            finals = [run["cost"][-1] for run in seeds]
            iterations = statistics.median(run["iteration"][-1] for run in seeds)
            print(f"  {name:<6}" + "".join(f"{m:>12.4f}" for m in medians)
                  + f"{min(finals):>11.4f}{max(finals):>11.4f}{iterations:>8.0f}")
            entry = {
                "problem": problem,
                "method": method,
                "config": name,
                "seeds": len(seeds),
                "median_cost_at": {("final" if t is None else str(t)): m for t, m in zip(TIMES, medians)},
                "final_costs": finals,
                "median_iterations": iterations,
            }
            if problem in optima:
                entry["median_final_gap"] = statistics.median(f / optima[problem] - 1.0 for f in finals)
                print(f"  {'':<6}median final gap to grid optimum: {100 * entry['median_final_gap']:.2f}%")
            summary.append(entry)
    if args.json:
        args.json.write_text(json.dumps({"grid_optima": optima, "runs": summary}, indent=2))


if __name__ == "__main__":
    main()
