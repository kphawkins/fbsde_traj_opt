#!/usr/bin/env python3
"""Summarizes the carried-normal-equations sweep: per problem, model, and setting, how close, how
reliably, how often the line search accepted, and how fast.

    python3 tools/analyze_rls_sweep.py RUN_DIR [--json OUT.json]

RUN_DIR holds rls_<problem>.csv checkpoint files and rls_runs.csv from value_model_study.
"""

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_value_model_study as avm  # noqa: E402


def setting(label):
    """The model name and its memory setting, as (model, memory, reset)."""
    model, _, rest = label.partition(":")
    values = dict(item.split("=") for item in rest.split(",") if "=" in item)
    return model, float(values.get("memory", 0.0)), values.get("reset", "0") not in ("0", "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    checkpoint_files = sorted(args.run_dir.glob("rls_*.csv"))
    checkpoint_files = [p for p in checkpoint_files if p.name != "rls_runs.csv"]
    summary = avm.summarize(avm.load_runs(checkpoint_files), {})
    runs = defaultdict(list)
    with open(args.run_dir / "rls_runs.csv") as f:
        for row in csv.DictReader(f):
            runs[(row["problem"], row["model"])].append(row)

    table = []
    for problem, entry in summary.items():
        for label, m in entry["models"].items():
            model, memory, reset = setting(label)
            rows = runs.get((problem, label), [])
            accepted = [int(r["accepted"]) / max(int(r["iterations"]), 1) for r in rows]
            iterations = [int(r["iterations"]) for r in rows]
            table.append({
                "problem": problem, "label": label, "model": model, "memory": memory, "reset": reset,
                "runs": m["runs"], "failures": m["failures"],
                "gap_median": m["final_gap_median"], "gap_worst": m["final_gap_worst"],
                "gaps": m["final_gaps"],
                "within5_at_end": m["reach"]["0.05"]["at_end"],
                "seconds_to_5": m["reach"]["0.05"]["median_seconds"],
                "seconds_to_10": m["reach"]["0.10"]["median_seconds"],
                "accepted": statistics.median(accepted) if accepted else math.nan,
                "iterations": statistics.median(iterations) if iterations else math.nan,
                "curve": m["curve"], "times": entry["times"],
            })
    table.sort(key=lambda r: (r["problem"], r["model"], r["reset"], r["memory"]))
    pct = lambda g: "  BREAK" if g is None or (isinstance(g, float) and math.isinf(g)) else f"{100 * g:+6.1f}%"
    secs = lambda t: "never" if t is None or math.isinf(t) else f"{t:.3g}s"
    current = None
    for r in table:
        if (r["problem"], r["model"]) != current:
            current = (r["problem"], r["model"])
            print(f"\n=== {r['problem']} / {r['model']}")
            print(f"{'memory':>6} {'reset':>5} {'gap med':>8} {'worst':>8} {'broke':>5} {'<=5% end':>8} {'t to 10%':>9}"
                  f" {'t to 5%':>8} {'accepted':>8} {'iters':>6}")
        print(f"{r['memory']:6.2f} {'yes' if r['reset'] else 'no':>5} {pct(r['gap_median'])} {pct(r['gap_worst'])}"
              f" {r['failures']:>3}/{r['runs']} {r['within5_at_end']:>5}/{r['runs']} {secs(r['seconds_to_10']):>9}"
              f" {secs(r['seconds_to_5']):>8} {100 * r['accepted']:7.1f}% {r['iterations']:6.0f}")
    if args.json:
        args.json.write_text(json.dumps(avm.finite(table)))


if __name__ == "__main__":
    main()
