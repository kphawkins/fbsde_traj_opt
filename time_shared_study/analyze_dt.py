#!/usr/bin/env python3
"""Summarizes the in-solver arm of the time-shared study (dt_queue.sh).

For every problem and model: the held-out gap to the grid optimum at a few charged-time
checkpoints (median over seeds, and the worst seed), iterations, and the share of accepted updates.
The optimum is read from the run's log; a problem without one (the double inverted pendulum) reports
the held-out cost itself in the gap columns. Writes dt_summary.json and prints a table. Each name
reads dt_<name>.csv, dt_<name>_runs.csv and dt_<name>.log.

    python3 analyze_dt.py integrator pendulum gs_integrator gs_pendulum dip
"""

import csv
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

CHECKPOINTS = (5.0, 10.0, 30.0, 180.0)
HERE = Path(__file__).resolve().parent


def optimum_of(log: Path) -> float | None:
    match = re.search(r"grid optimum ([0-9.]+)", log.read_text())
    return float(match.group(1)) if match else None


def cost_at(trace: list[tuple[float, float]], seconds: float) -> float:
    """The held-out cost of the last checkpoint at or before `seconds`."""
    cost = trace[0][1]
    for time, value in trace:
        if time <= seconds + 1e-9:
            cost = value
    return cost


def main(problems: list[str]) -> None:
    summary = []
    for problem in problems:
        optimum = optimum_of(HERE / f"dt_{problem}.log")
        traces = defaultdict(lambda: defaultdict(list))
        with open(HERE / f"dt_{problem}.csv", newline="") as handle:
            for row in csv.DictReader(handle):
                traces[row["model"]][row["seed"]].append((float(row["seconds"]), float(row["held_out_cost"])))
        runs = defaultdict(list)
        with open(HERE / f"dt_{problem}_runs.csv", newline="") as handle:
            for row in csv.DictReader(handle):
                runs[row["model"]].append(row)
        for model, seeds in traces.items():
            entry = {"problem": problem, "model": model, "seeds": len(seeds), "optimum": optimum}
            for checkpoint in CHECKPOINTS:
                costs = [cost_at(trace, checkpoint) for trace in seeds.values()]
                gaps = [100.0 * (cost / optimum - 1.0) for cost in costs] if optimum else costs
                entry[f"median_gap_{checkpoint:g}s"] = statistics.median(gaps)
                entry[f"worst_gap_{checkpoint:g}s"] = max(gaps)
            iterations = [int(row["iterations"]) for row in runs[model]]
            accepted = [int(row["accepted"]) for row in runs[model]]
            entry["median_iterations"] = statistics.median(iterations) if iterations else 0
            entry["accepted_share"] = sum(accepted) / max(1, sum(iterations))
            summary.append(entry)
    (HERE / "dt_summary.json").write_text(json.dumps(summary, indent=1))
    header = "".join(f" {f'gap@{c:g}s':>10}" for c in CHECKPOINTS)
    print(f"{'problem':<14} {'model':<62}{header} {'worst@end':>10} {'iters':>6} {'acc':>5}")
    for entry in summary:
        gaps = "".join(f" {entry[f'median_gap_{c:g}s']:>+9.2f}%" for c in CHECKPOINTS)
        print(f"{entry['problem']:<14} {entry['model']:<62}{gaps} {entry['worst_gap_180s']:>+9.2f}% "
              f"{entry['median_iterations']:>6} {entry['accepted_share']:>5.2f}")


if __name__ == "__main__":
    main(sys.argv[1:])
