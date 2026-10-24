#!/usr/bin/env python3
"""Summarizes a method_comparison run: speed, accuracy, and reliability of each method on each benchmark.

    python3 tools/analyze_method_comparison.py RUN_DIR [--json OUT.json]

RUN_DIR holds method_comparison.csv, method_comparison_mpc.csv, and method_comparison_reference.csv.

The reference on a 2-D benchmark is the grid DP optimum on the held-out rollouts. The double inverted
pendulum has none, so its reference is the lowest held-out cost of any policy any offline run produced
at any checkpoint: an upper bound on the optimum, up to the held-out sampling error.
"""

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

TOLERANCES = (0.05, 0.10, 0.25)
# Charged times at which the median gap curve is sampled, log-spaced from 1 ms to the budget.
CURVE_POINTS = 60


def load(run_dir: Path):
    traces = defaultdict(lambda: defaultdict(list))  # problem -> method -> [run]
    runs = defaultdict(lambda: {"seconds": [], "cost": [], "iteration": []})
    with open(run_dir / "method_comparison.csv") as f:
        for row in csv.DictReader(f):
            key = (row["problem"], row["method"], int(row["seed"]))
            runs[key]["seconds"].append(float(row["seconds"]))
            runs[key]["cost"].append(float(row["held_out_cost"]))
            runs[key]["iteration"].append(int(row["iteration"]))
    for (problem, method, seed), run in runs.items():
        run["seed"] = seed
        traces[problem][method].append(run)
    mpc = defaultdict(lambda: defaultdict(list))
    mpc_path = run_dir / "method_comparison_mpc.csv"
    if mpc_path.exists():
        with open(mpc_path) as f:
            for row in csv.DictReader(f):
                mpc[row["problem"]][row["method"]].append(
                    {"seed": int(row["seed"]), "cost": float(row["cost"]), "ms_per_step": 1e3 * float(row["seconds_per_step"])}
                )
    references = {}
    ref_path = run_dir / "method_comparison_reference.csv"
    if ref_path.exists():
        with open(ref_path) as f:
            for row in csv.DictReader(f):
                references[row["problem"]] = {
                    "optimum": float(row["grid_optimum"]),
                    "optimum_mpc_subset": float(row["grid_optimum_mpc_subset"]),
                    "kind": "grid DP optimum",
                }
    return traces, mpc, references


def cost_at(run, t):
    """The held-out cost of the policy a run had at charged time t (its last checkpoint at or before t)."""
    cost = run["cost"][0]
    for seconds, value in zip(run["seconds"], run["cost"]):
        if seconds > t:
            break
        cost = value
    return cost


def first_time_within(run, threshold):
    for seconds, value in zip(run["seconds"], run["cost"]):
        if value <= threshold:
            return seconds
    return math.inf


def quantile(values, q):
    values = sorted(values)
    if not values:
        return math.nan
    position = q * (len(values) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if any(math.isinf(values[i]) for i in (low, high)):
        return values[high] if q >= 0.5 else values[low]
    return values[low] + (values[high] - values[low]) * (position - low)


def summarize(traces, mpc, references):
    summary = {}
    for problem, methods in traces.items():
        if problem not in references:
            # Every checkpoint's policy is a real policy, and its held-out cost an honest estimate of
            # its expected cost, so the lowest is an upper bound on the optimum up to sampling error.
            seen = [cost for runs in methods.values() for run in runs for cost in run["cost"]]
            references[problem] = {"optimum": min(seen), "kind": "lowest held-out cost of any policy found"}
        reference = references[problem]["optimum"]
        budget = max(run["seconds"][-1] for runs in methods.values() for run in runs)
        times = [1e-3 * (budget / 1e-3) ** (i / (CURVE_POINTS - 1)) for i in range(CURVE_POINTS)]
        entry = {"reference": references[problem], "budget": budget, "times": times, "methods": {}}
        for method, runs in sorted(methods.items()):
            gaps_final = [run["cost"][-1] / reference - 1 for run in runs]
            gaps_best = [min(run["cost"]) / reference - 1 for run in runs]
            reach = {}
            for tolerance in TOLERANCES:
                hits = [first_time_within(run, (1 + tolerance) * reference) for run in runs]
                finite = [h for h in hits if math.isfinite(h)]
                # "Within tolerance at the end": the final policy, not any checkpoint along the way.
                at_end = sum(1 for g in gaps_final if g <= tolerance)
                reach[f"{tolerance:.2f}"] = {
                    "ever": len(finite),
                    "at_end": at_end,
                    "median_seconds": quantile(hits, 0.5),
                    "fastest_seconds": min(finite) if finite else math.inf,
                }
            curve = []
            for t in times:
                gaps = [cost_at(run, t) / reference - 1 for run in runs]
                curve.append([quantile(gaps, 0.25), quantile(gaps, 0.5), quantile(gaps, 0.75)])
            entry["methods"][method] = {
                "runs": len(runs),
                "final_gap_median": quantile(gaps_final, 0.5),
                "final_gap_q25": quantile(gaps_final, 0.25),
                "final_gap_q75": quantile(gaps_final, 0.75),
                "final_gap_worst": max(gaps_final),
                "final_gap_best": min(gaps_final),
                "best_seen_gap_median": quantile(gaps_best, 0.5),
                "final_gaps": gaps_final,
                "median_seconds_used": quantile([run["seconds"][-1] for run in runs], 0.5),
                "median_iterations": quantile([run["iteration"][-1] for run in runs], 0.5),
                "reach": reach,
                "curve": curve,
            }
        entry["mpc"] = {}
        for method, results in mpc.get(problem, {}).items():
            subset = references[problem].get("optimum_mpc_subset", reference)
            gaps = [r["cost"] / subset - 1 for r in results]
            entry["mpc"][method] = {
                "runs": len(results),
                "gap_median": quantile(gaps, 0.5),
                "gap_worst": max(gaps),
                "gaps": gaps,
                "ms_per_step_median": quantile([r["ms_per_step"] for r in results], 0.5),
                "reference_note": "grid optimum on the same rollouts"
                if "optimum_mpc_subset" in references[problem]
                else "best policy found offline (different rollouts: 128 vs 1024)",
            }
        summary[problem] = entry
    return summary


def pct(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "   n/a"
    if math.isinf(value):
        return "   inf"
    return f"{100 * value:+6.1f}%"


def secs(value):
    return "never" if math.isinf(value) else f"{value:.3g}s"


def print_summary(summary):
    for problem, entry in summary.items():
        ref = entry["reference"]
        print(f"\n=== {problem}: reference {ref['optimum']:.4f} ({ref['kind']}), budget {entry['budget']:.0f} s")
        print(f"{'method':<18} {'runs':>4} {'final gap (median, IQR, worst)':>36} {'best-seen':>9}"
              f" {'<=5% ever/end, median t':>26} {'<=10% ever/end, t':>22} {'<=25% ever/end, t':>22}")
        for method, m in entry["methods"].items():
            cells = []
            for tol in ("0.05", "0.10", "0.25"):
                r = m["reach"][tol]
                cells.append(f"{r['ever']}/{r['at_end']} of {m['runs']}, {secs(r['median_seconds'])}")
            print(f"{method:<18} {m['runs']:>4} {pct(m['final_gap_median'])} [{pct(m['final_gap_q25'])},"
                  f"{pct(m['final_gap_q75'])}] {pct(m['final_gap_worst'])} {pct(m['best_seen_gap_median'])}"
                  f" {cells[0]:>26} {cells[1]:>22} {cells[2]:>22}")
        for method, m in entry["mpc"].items():
            print(f"{method:<18} {m['runs']:>4} gap median {pct(m['gap_median'])}, worst {pct(m['gap_worst'])},"
                  f" {m['ms_per_step_median']:.3f} ms/step ({m['reference_note']})")


def finite(value):
    """JSON has no infinity or NaN: both become null."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    traces, mpc, references = load(args.run_dir)
    summary = summarize(traces, mpc, references)
    print_summary(summary)
    if args.json:
        args.json.write_text(json.dumps(finite(summary)))


if __name__ == "__main__":
    main()
