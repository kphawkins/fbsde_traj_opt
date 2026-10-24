#!/usr/bin/env python3
"""Builds the value-model and sigma-point report from the study outputs that exist so far.

    python3 report2/build.py OUT.html
"""

import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import analyze_value_model_study as avm  # noqa: E402

MODEL = ROOT / "model_study"
SIGMA = ROOT / "sigma_study"
DIP_REFERENCE = 2.4928  # lowest held-out cost of any policy found (method comparison, iLQR)
GRID_V0 = {"L1 double integrator": 4.2353, "L1 inverted pendulum": 2.3134}


def family(label):
    name = label.split(":")[0]
    for prefix, fam in (("poly", "Monomial polynomial"), ("leg", "Legendre polynomial"), ("rbf", "Gaussian RBF"),
                        ("spline", "Cubic B-spline"), ("softmin", "Soft-min quadratics"), ("tanh", "Tanh network")):
        if name.startswith(prefix):
            return fam
    return "Other"


def read_rows(path, skip=lambda row: False):
    if not path.exists():
        return []
    with open(path) as f:
        return [row for row in csv.DictReader(f) if not skip(row)]


def write_filtered(paths_and_skips, out):
    rows = []
    for path, skip in paths_and_skips:
        rows += read_rows(path, skip)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["problem", "model", "seed", "iteration", "seconds", "held_out_cost"])
        writer.writeheader()
        writer.writerows(rows)
    return out


def finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [finite(v) for v in value]
    return value


def strip_entries(summary, problem, value="gap"):
    entries = []
    for label, m in summary.get(problem, {}).get("models", {}).items():
        entries.append({
            "label": label,
            "family": family(label),
            "values": [None if math.isinf(g) else g for g in m["final_gaps"]],
            "median": m["final_gap_median"],
            "failures": m["failures"],
            "runs": m["runs"],
            "ms_per_iteration": 1e3 * m["seconds_per_iteration"],
            "reach": m["reach"],
        })
    entries.sort(key=lambda e: (e["median"] if math.isfinite(e["median"]) else 1e9))
    return entries


def curves(summary, problem, labels):
    entry = summary.get(problem)
    if not entry:
        return None
    return {
        "times": entry["times"],
        "budget": entry["budget"],
        "series": [{"label": l, "curve": entry["models"][l]["curve"]} for l in labels if l in entry["models"]],
    }


def main():
    out = Path(sys.argv[1])
    tmp = ROOT / "report2"
    # The 2-D model screens. Round 2's roughness rows ran a binary built before the option existed
    # and are left out; dt_rough.csv holds the rerun.
    screen = write_filtered(
        [(MODEL / "dt_screen.csv", lambda r: False),
         (MODEL / "dt_round2.csv", lambda r: "rough=" in r["model"]),
         (MODEL / "dt_rough.csv", lambda r: False)],
        tmp / "screen_2d.csv")
    screen_summary = avm.summarize(avm.load_runs([screen]), {})
    dip_summary = avm.summarize(avm.load_runs([MODEL / "dt_dip.csv"]), {"L1 double inverted pendulum": DIP_REFERENCE})
    confirm_path = MODEL / "dt_confirm.csv"
    confirm_summary = avm.summarize(avm.load_runs([confirm_path]), {}) if confirm_path.exists() and len(read_rows(confirm_path)) > 0 else {}

    oracle = []
    for row in read_rows(MODEL / "oracle_v2.csv"):
        if ":" in row["model"]:
            continue
        oracle.append({
            "problem": row["problem"], "model": row["model"], "family": family(row["model"]),
            "params": int(row["num_parameters"]),
            "gap": float(row["policy_cost"]) / float(row["grid_optimum"]) - 1,
            "value_rmse": float(row["value_rmse"]), "slope_rmse": float(row["slope_rmse"]),
            "policy_us": float(row["policy_us"]),
        })

    sigma_paths = [SIGMA / "sigma_2d.csv", SIGMA / "sigma_2d_random.csv"]
    sigma_summary = avm.summarize(avm.load_runs([p for p in sigma_paths if p.exists() and len(read_rows(p)) > 0]), {})
    sigma_runs = read_rows(SIGMA / "sigma_2d_runs.csv") + read_rows(SIGMA / "sigma_2d_random_runs.csv")
    sigma_table = {}
    for row in sigma_runs:
        key = (row["problem"], row["config"])
        sigma_table.setdefault(key, []).append(row)
    sigma_rows = []
    for (problem, config), rows in sigma_table.items():
        med = lambda xs: sorted(xs)[len(xs) // 2]
        optimum = avm.GRID_OPTIMA[problem]
        sigma_rows.append({
            "problem": problem, "config": config, "batch": int(rows[0]["batch"]),
            "gap_median": med([float(r["final_cost"]) / optimum - 1 for r in rows]),
            "gap_spread": max(float(r["final_cost"]) for r in rows) / optimum - min(float(r["final_cost"]) for r in rows) / optimum,
            "v0_error": med([abs(float(r["v0_estimate"]) / GRID_V0[problem] - 1) for r in rows]),
            "accepted": med([int(r["accepted"]) / max(int(r["iterations"]), 1) for r in rows]),
            "ms_per_iteration": med([1e3 * float(r["seconds"]) / max(int(r["iterations"]), 1) for r in rows]),
            "forward": med([float(r["forward_seconds"]) / float(r["seconds"]) for r in rows]),
            "backward": med([float(r["backward_seconds"]) / float(r["seconds"]) for r in rows]),
            "line_search": med([float(r["line_search_seconds"]) / float(r["seconds"]) for r in rows]),
            "runs": len(rows),
        })
    sigma_dip_runs = (read_rows(SIGMA / "sigma_dip_runs.csv") + read_rows(SIGMA / "sigma_dip_random_runs.csv")
                      + read_rows(SIGMA / "sigma_dip_followup_runs.csv"))

    problems_2d = ["L1 double integrator", "L1 inverted pendulum"]
    data = {
        "screen": {p: strip_entries(screen_summary, p) for p in problems_2d},
        "screen_curves": {p: curves(screen_summary, p, ["poly2", "poly4", "poly6", "rbf5", "spline5", "softmin4", "leg6:rough=1e-6"]) for p in problems_2d},
        "confirm": {p: strip_entries(confirm_summary, p) for p in problems_2d} if confirm_summary else None,
        "dip": strip_entries(dip_summary, "L1 double inverted pendulum"),
        "dip_curves": curves(dip_summary, "L1 double inverted pendulum", ["poly2", "poly4", "poly6", "softmin4", "softmin2", "tanh32"]),
        "dip_reference": DIP_REFERENCE,
        "oracle": oracle,
        "sigma": {p: strip_entries(sigma_summary, p) for p in problems_2d},
        "sigma_table": sigma_rows,
        "sigma_dip_runs": [{k: r[k] for k in ("config", "seed", "batch", "final_cost", "iterations", "accepted")} for r in sigma_dip_runs],
    }
    prose = json.loads((ROOT / "report2" / "prose.json").read_text())
    page = ("<title>DT-FBSDE Model Study</title>\n" + (ROOT / "report2" / "head.html").read_text() + "\n"
            + (ROOT / "report2" / "body.html").read_text())
    page = page.replace("/*DATA*/null", json.dumps(finite(data), separators=(",", ":")))
    page = page.replace("/*LIT*/[]", json.dumps(prose.pop("LIT_TABLE", [])))
    for key, html in prose.items():
        page = page.replace(f"<!--{key}-->", html)
    out.write_text(page)
    print(f"wrote {out} ({len(page) // 1024} KB)")
    for p in problems_2d:
        print(p, [(e["label"], round(100 * e["median"], 1) if e["median"] is not None and math.isfinite(e["median"]) else "FAIL") for e in data["screen"][p]][:8])
    print("dip", [(e["label"], e["values"]) for e in data["dip"]])


if __name__ == "__main__":
    main()
