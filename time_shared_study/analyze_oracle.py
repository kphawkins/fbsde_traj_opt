#!/usr/bin/env python3
"""Summarizes the oracle sweep of the time-shared study (oracle_queue.sh).

For every problem, sample count, ridge and shape: the table's parameter count and the policy's
gap to the grid optimum over the disjoint training subsets (median, and the worst), plus the value
RMSE on the optimal tube. Writes oracle_summary.json next to the CSVs and prints a table.

    python3 analyze_oracle.py oracle_integrator.csv oracle_pendulum.csv
"""

import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def parse(model: str) -> dict:
    name, _, rest = model.partition(":")
    keys = {}
    for pair in rest.split(","):
        if "=" in pair:
            key, value = pair.split("=", 1)
            keys[key] = value
    shape = []
    if "rank" in keys:
        shape.append(f"rank {keys['rank']}")
    kind = keys.get("time", "stage")
    if kind != "stage":
        shape.append(f"{kind} x{keys.get('knots', '8')}")
    return {
        "basis": name,
        "samples": int(keys.get("samples", "512")),
        "ridge": keys.get("ridge", "1e-6"),
        "noise": float(keys.get("noise", "0")),
        "shape": " + ".join(shape) if shape else "per stage",
    }


def main(paths: list[str]) -> None:
    groups = defaultdict(list)
    for path in paths:
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                spec = parse(row["model"])
                key = (row["problem"], spec["samples"], spec["noise"], spec["ridge"], spec["shape"])
                gap = 100.0 * (float(row["policy_cost"]) / float(row["grid_optimum"]) - 1.0)
                groups[key].append(
                    {
                        "gap": gap,
                        "parameters": int(row["table_parameters"]),
                        "value_rmse": float(row["value_rmse"]),
                        "on_policy_rmse": float(row["on_policy_rmse"]),
                    }
                )
    summary = []
    for (problem, samples, noise, ridge, shape), runs in sorted(groups.items()):
        gaps = [run["gap"] for run in runs]
        summary.append(
            {
                "problem": problem,
                "samples": samples,
                "noise": noise,
                "ridge": ridge,
                "shape": shape,
                "parameters": runs[0]["parameters"],
                "subsets": len(runs),
                "median_gap": statistics.median(gaps),
                "worst_gap": max(gaps),
                "median_value_rmse": statistics.median(run["value_rmse"] for run in runs),
                "median_on_policy_rmse": statistics.median(run["on_policy_rmse"] for run in runs),
            }
        )
    out = Path(paths[0]).with_name("oracle_summary.json")
    out.write_text(json.dumps(summary, indent=1))
    print(f"{'problem':<26} {'S':>4} {'noise':>5} {'ridge':>6} {'shape':<24} {'params':>6} {'n':>2} "
          f"{'med gap':>9} {'worst':>9} {'rmse':>7}")
    for row in summary:
        print(f"{row['problem']:<26} {row['samples']:>4} {row['noise']:>5g} {row['ridge']:>6} {row['shape']:<24} "
              f"{row['parameters']:>6} {row['subsets']:>2} {row['median_gap']:>+8.2f}% {row['worst_gap']:>+8.2f}% "
              f"{row['median_value_rmse']:>7.3f}")


if __name__ == "__main__":
    main(sys.argv[1:])
