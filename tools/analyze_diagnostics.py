#!/usr/bin/env python3
"""Summarizes a diagnostic run of value_model_study: per setting, the final held-out gap, the line
search's own final cost, how much the line search's rollouts flatter the policy, and how often
updates were accepted.

    python3 tools/analyze_diagnostics.py CHECKPOINTS_CSV RUNS_CSV

Flattery is held-out cost / line-search cost - 1, both from each run's last checkpoint. Even an
honest method shows some, because the line search keeps only candidates that look better on its own
256 rollouts; a value that grows with a setting means the method is fitting those rollouts' noise.
"""

import csv
import math
import statistics
import sys
from collections import defaultdict

OPTIMA = {"L1 double integrator": 4.1367, "L1 inverted pendulum": 2.2977}


def main():
    checkpoints, runs_path = sys.argv[1], sys.argv[2]
    last = {}
    with open(checkpoints) as f:
        for row in csv.DictReader(f):
            key = (row["problem"], row["model"], int(row["seed"]))
            if row["held_out_cost"] == "nan":
                last[key] = None
            elif last.get(key, 0) is not None:
                last[key] = (float(row["held_out_cost"]), float(row["line_search_cost"]))
    accepted = defaultdict(list)
    with open(runs_path) as f:
        for row in csv.DictReader(f):
            accepted[(row["problem"], row["model"])].append(int(row["accepted"]) / max(int(row["iterations"]), 1))
    groups = defaultdict(list)
    for (problem, model, seed), value in last.items():
        groups[(problem, model)].append(value)
    order = []
    with open(checkpoints) as f:
        for row in csv.DictReader(f):
            if (row["problem"], row["model"]) not in order:
                order.append((row["problem"], row["model"]))
    print(f"{'setting':<42} {'runs':>4} {'broke':>5} {'gap med':>8} {'worst':>8} {'flattery':>9} {'accepted':>9}")
    for problem, model in order:
        values = groups[(problem, model)]
        good = [v for v in values if v is not None]
        gaps = [h / OPTIMA[problem] - 1 for h, _ in good]
        flattery = [h / l - 1 for h, l in good]
        acc = accepted.get((problem, model), [])
        print(f"{model:<42} {len(values):>4} {len(values) - len(good):>5}"
              f" {100 * statistics.median(gaps):+7.1f}% {100 * max(gaps):+7.1f}%"
              f" {100 * statistics.median(flattery):+8.1f}% {100 * statistics.median(acc) if acc else math.nan:8.1f}%")


if __name__ == "__main__":
    main()
