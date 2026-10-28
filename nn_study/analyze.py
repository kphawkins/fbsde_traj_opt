"""Summarizes the DT-NN study: median held-out cost at checkpoints, final range, and the held-out vs fixed-noise gap.

    python3 analyze.py            # full runs, from full/
"""
import csv
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = ["default", "nn", "ll", "llgrad"]
PROBLEMS = {"integrator": [1, 5, 20, 60], "pendulum": [1, 5, 20, 60], "double-pendulum": [5, 20, 60, 120]}


def at(rows, seed, seconds):
    kept = [r for r in rows if r["seed"] == seed and float(r["seconds"]) <= seconds]
    return float(kept[-1]["held_out_cost"]) if kept else None


def main():
    for problem, checkpoints in PROBLEMS.items():
        print(f"\n{problem}: median held-out cost at " + ", ".join(f"{t} s" for t in checkpoints))
        for name in NAMES:
            path = os.path.join(HERE, "full", f"{name}.{problem}.csv")
            if not os.path.exists(path):
                continue
            rows = list(csv.DictReader(open(path)))
            seeds = sorted({r["seed"] for r in rows})
            medians = [statistics.median(v for s in seeds if (v := at(rows, s, t)) is not None) for t in checkpoints]
            finals = [float([r for r in rows if r["seed"] == s][-1]["held_out_cost"]) for s in seeds]
            line = f"  {name:8s} n={len(seeds)} " + " ".join(f"{m:8.3f}" for m in medians)
            line += f"   final {min(finals):.2f}-{max(finals):.2f}"
            nn_path = os.path.join(HERE, "full", f"{name}.{problem}.nn.csv")
            if os.path.exists(nn_path):
                iterations = list(csv.DictReader(open(nn_path)))
                gaps = []
                for s in seeds:
                    batch = float([r for r in iterations if r["seed"] == s][-1]["best_batch_cost"])
                    gaps.append(100 * (finals[seeds.index(s)] - batch) / batch)
                line += "   held-out vs fixed-noise " + " ".join(f"{g:+.1f}%" for g in gaps)
            print(line)


if __name__ == "__main__":
    main()
