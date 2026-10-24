#!/usr/bin/env python3
"""Summarizes the random-drift Taylor noiseless study: per problem and setting, the final gap to the
reference and how much the off-policy (K) term of the estimator was used.

    python3 tools/analyze_drift_study.py RUNS_CSV [RUNS_CSV ...] [--json OUT.json]

Each RUNS_CSV is a value_model_study --runs-csv file. Columns the older driver did not write are
treated as missing, not zero. Pass --reference "L1 double inverted pendulum=2.4928" for a problem
without a grid optimum.
"""

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict

REFERENCES = {"L1 double integrator": 4.1367, "L1 inverted pendulum": 2.2977}
DIAGNOSTICS = (
    "mean_girsanov_drift",
    "mean_first_order_correction",
    "mean_target_magnitude",
    "mean_signed_first_order_correction",
    "mean_second_order_correction",
    "off_policy_fraction",
    "mean_off_policy_correction",
    "mean_taylor_residual",
)


def parse_setting(model):
    """'poly4:target=taylor,explore=0.2' -> {'target': 'taylor', 'explore': '0.2'}."""
    _, _, knobs = model.partition(":")
    return dict(knob.split("=", 1) for knob in knobs.split(",") if "=" in knob)


def canonical(model):
    """The same setting whatever order its knobs were written in: 'poly4:b=1,a=2' -> 'poly4:a=2,b=1'."""
    name, _, knobs = model.partition(":")
    return f"{name}:{','.join(sorted(knobs.split(',')))}" if knobs else name


def load(paths):
    rows = defaultdict(list)
    for path in paths:
        with open(path) as f:
            for row in csv.DictReader(f):
                rows[(row["problem"], canonical(row["model"]))].append(row)
    return rows


def summarize(rows, references):
    out = []
    for (problem, model), runs in sorted(rows.items()):
        reference = references.get(problem)
        costs = [float(r["final_cost"]) for r in runs]
        broke = sum(1 for c in costs if math.isnan(c))
        gaps = [100.0 * (c / reference - 1.0) for c in costs if not math.isnan(c)] if reference else []
        entry = {
            "problem": problem,
            "model": model,
            "setting": parse_setting(model),
            "runs": len(runs),
            "broke": broke,
            "gap_median": statistics.median(gaps) if gaps else None,
            "gap_worst": max(gaps) if gaps else None,
            "gaps": gaps,
            "iterations": statistics.median(int(r["iterations"]) for r in runs),
        }
        for name in DIAGNOSTICS:
            values = [float(r[name]) for r in runs if r.get(name) not in (None, "")]
            entry[name] = statistics.mean(values) if values else None
        # The share of the target the first-order correction makes up: the headline K-term usage.
        if entry["mean_first_order_correction"] is not None and entry["mean_target_magnitude"]:
            entry["k_share"] = entry["mean_first_order_correction"] / entry["mean_target_magnitude"]
        else:
            entry["k_share"] = None
        out.append(entry)
    return out


def fmt(value, spec):
    return "-" if value is None else format(value, spec)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--json")
    parser.add_argument("--reference", action="append", default=[])
    args = parser.parse_args()
    references = dict(REFERENCES)
    for item in args.reference:
        name, value = item.rsplit("=", 1)
        references[name] = float(value)
    summary = summarize(load(args.runs), references)
    problem = None
    for entry in summary:
        if entry["problem"] != problem:
            problem = entry["problem"]
            print(f"\n=== {problem}")
            print(f"{'setting':<58}{'n':>3}{'brk':>4}{'gap med':>9}{'worst':>9}{'||D||':>7}"
                  f"{'|K|/|Y|':>9}{'offpol':>8}{'|K| off':>9}{'K sign':>9}{'resid':>8}{'iters':>7}")
        print(f"{entry['model']:<58}{entry['runs']:>3}{entry['broke']:>4}"
              f"{fmt(entry['gap_median'], '+8.1f')}%{fmt(entry['gap_worst'], '+8.1f')}%"
              f"{fmt(entry['mean_girsanov_drift'], '7.3f')}{fmt(entry['k_share'], '9.4f')}"
              f"{fmt(entry['off_policy_fraction'], '8.3f')}{fmt(entry['mean_off_policy_correction'], '9.4f')}"
              f"{fmt(entry['mean_signed_first_order_correction'], '+9.4f')}{fmt(entry['mean_taylor_residual'], '8.4f')}"
              f"{entry['iterations']:>7.0f}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
