#!/usr/bin/env python3
"""Summarizes value_model_study DT-FBSDE runs: per problem and model, how fast, how close, how reliably.

    python3 tools/analyze_value_model_study.py DT_CSV [DT_CSV ...] [--oracle ORACLE_CSV] [--json OUT.json]

The reference on the 2-D problems is the grid optimum on the same held-out rollouts. Pass
--reference "L1 double inverted pendulum=2.4928" for the 4-D problem, which has none.
A run whose solver failed is marked by a `nan` cost on its last row and counts as not converged.
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

GRID_OPTIMA = {"L1 double integrator": 4.1367, "L1 inverted pendulum": 2.2977}
TOLERANCES = (0.05, 0.10, 0.25)
CURVE_POINTS = 50


def load_runs(paths):
    runs = defaultdict(lambda: {"seconds": [], "cost": [], "iteration": [], "failed": False})
    for path in paths:
        with open(path) as f:
            for row in csv.DictReader(f):
                key = (row["problem"], row["model"], int(row["seed"]))
                if row["held_out_cost"] == "nan":
                    runs[key]["failed"] = True
                    continue
                runs[key]["seconds"].append(float(row["seconds"]))
                runs[key]["cost"].append(float(row["held_out_cost"]))
                runs[key]["iteration"].append(int(row["iteration"]))
    grouped = defaultdict(lambda: defaultdict(list))
    for (problem, model, seed), run in runs.items():
        run["seed"] = seed
        grouped[problem][model].append(run)
    return grouped


def quantile(values, q):
    values = sorted(values)
    if not values:
        return math.nan
    position = q * (len(values) - 1)
    low, high = math.floor(position), math.ceil(position)
    if math.isinf(values[low]) or math.isinf(values[high]):
        return values[high] if q >= 0.5 else values[low]
    return values[low] + (values[high] - values[low]) * (position - low)


def cost_at(run, t):
    cost = run["cost"][0]
    for seconds, value in zip(run["seconds"], run["cost"]):
        if seconds > t:
            break
        cost = value
    return cost


def summarize(grouped, references):
    summary = {}
    for problem, models in grouped.items():
        reference = references.get(problem) or GRID_OPTIMA.get(problem)
        budget = max(run["seconds"][-1] for runs in models.values() for run in runs if run["seconds"])
        times = [1e-3 * (budget / 1e-3) ** (i / (CURVE_POINTS - 1)) for i in range(CURVE_POINTS)]
        entry = {"reference": reference, "budget": budget, "times": times, "models": {}}
        for model, runs in models.items():
            finals = [math.inf if run["failed"] else run["cost"][-1] / reference - 1 for run in runs]
            reach = {}
            for tolerance in TOLERANCES:
                hits = []
                for run in runs:
                    hit = math.inf
                    for seconds, value in zip(run["seconds"], run["cost"]):
                        if value <= (1 + tolerance) * reference:
                            hit = seconds
                            break
                    hits.append(hit)
                reach[f"{tolerance:.2f}"] = {
                    "at_end": sum(1 for g in finals if g <= tolerance),
                    "median_seconds": quantile(hits, 0.5),
                }
            curve = []
            for t in times:
                gaps = [cost_at(run, t) / reference - 1 for run in runs if run["cost"]]
                curve.append([quantile(gaps, 0.25), quantile(gaps, 0.5), quantile(gaps, 0.75)])
            per_iteration = [run["seconds"][-1] / max(run["iteration"][-1], 1) for run in runs if run["iteration"]]
            entry["models"][model] = {
                "runs": len(runs),
                "failures": sum(1 for run in runs if run["failed"]),
                "final_gaps": finals,
                "final_gap_median": quantile(finals, 0.5),
                "final_gap_worst": max(finals),
                "seconds_per_iteration": quantile(per_iteration, 0.5),
                "reach": reach,
                "curve": curve,
            }
        summary[problem] = entry
    return summary


def pct(value):
    if value is None or math.isnan(value):
        return "    n/a"
    if math.isinf(value):
        return "   FAIL"
    return f"{100 * value:+7.1f}%"


def secs(value):
    return "never" if math.isinf(value) or math.isnan(value) else f"{value:.3g}s"


def finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="+", type=Path)
    parser.add_argument("--reference", action="append", default=[])
    parser.add_argument("--json", type=Path)
    parser.add_argument("--sort", default="final", choices=["final", "name"])
    args = parser.parse_args()
    references = dict((item.split("=")[0], float(item.split("=")[1])) for item in args.reference)
    summary = summarize(load_runs(args.csv), references)
    for problem, entry in summary.items():
        print(f"\n=== {problem}: reference {entry['reference']:.4f}, budget {entry['budget']:.0f} s")
        print(f"{'model':<42} {'runs':>4} {'fail':>4} {'final med':>9} {'worst':>9} {'ms/it':>8}"
              f" {'<=25% end, t':>16} {'<=10% end, t':>16} {'<=5% end, t':>16}")
        items = entry["models"].items()
        items = sorted(items, key=lambda kv: (kv[1]["final_gap_median"]) if args.sort == "final" else kv[0])
        for model, m in items:
            cells = [f"{m['reach'][t]['at_end']}/{m['runs']}, {secs(m['reach'][t]['median_seconds'])}"
                     for t in ("0.25", "0.10", "0.05")]
            print(f"{model:<42} {m['runs']:>4} {m['failures']:>4} {pct(m['final_gap_median'])} {pct(m['final_gap_worst'])}"
                  f" {1e3 * m['seconds_per_iteration']:8.2f} {cells[0]:>16} {cells[1]:>16} {cells[2]:>16}")
    if args.json:
        args.json.write_text(json.dumps(finite(summary)))


if __name__ == "__main__":
    main()
