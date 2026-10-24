#!/usr/bin/env python3
"""Summarizes the iLQR-seeding study: DT-FBSDE warm-started from iLQR against plain DT-FBSDE and iLQR.

    python3 tools/analyze_ilqr_seed_study.py ilqr_seed_study [--prior comparison_runs/full_dip]

STUDY_DIR holds one method_comparison run per subdirectory (method_comparison.csv,
method_comparison_reference.csv, log.txt). A subdirectory whose name ends in `_sNNN` ran the seeded
method at --seed-smoothing 0.NNN, and its method is renamed to say so; the rest ran it at the default.

The 2-D references are the grid DP optima. The double inverted pendulum has none: its reference is
the lowest held-out cost of any policy at any checkpoint in this study or in --prior (an earlier
method_comparison run on the same held-out rollouts, whose costs are comparable even though its
times are not).

Writes summary.json, summary.txt, and report.html into STUDY_DIR.
"""

import argparse
import contextlib
import csv
import html
import io
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_method_comparison as amc  # noqa: E402

SEED_LINE = re.compile(
    r"^(?P<problem>.+?)\s+DT-FBSDE\+iLQR\s+seed\s+(?P<seed>\d+): seeded from iLQR iteration (?P<chosen>\d+) of "
    r"(?P<total>\d+) \((?P<candidates>\d+) candidates, (?P<seconds>[0-9.]+) s\): solver cost (?P<cost>[0-9.]+), "
    r"unseeded (?P<unseeded>[0-9.]+)$"
)

# Categorical slots 1-4 of the dataviz reference palette, in a fixed order per method.
COLORS = {
    "DT-FBSDE": ("#2a78d6", "#3987e5"),
    "iLQR": ("#eb6834", "#d95926"),
    "DT-FBSDE+iLQR": ("#1baf7a", "#199e70"),
    "DT-FBSDE+iLQR (s=0.03)": ("#eda100", "#c98500"),
}
SHOWN = list(COLORS)


def method_name(method, variant):
    if method == "DT-FBSDE+iLQR" and variant is not None:
        return f"DT-FBSDE+iLQR (s={variant})"
    return method


def load_study(study_dir):
    traces = defaultdict(lambda: defaultdict(list))
    references = {}
    seeding = defaultdict(list)
    for run_dir in sorted(p for p in study_dir.iterdir() if (p / "method_comparison.csv").exists()):
        match = re.search(r"_s(\d+)$", run_dir.name)
        variant = f"0.{match.group(1)[1:]}" if match else None
        run_traces, _, run_references = amc.load(run_dir)
        for problem, methods in run_traces.items():
            for method, runs in methods.items():
                traces[problem][method_name(method, variant)].extend(runs)
        references.update(run_references)
        log = run_dir / "log.txt"
        if log.exists():
            for line in log.read_text().splitlines():
                parsed = SEED_LINE.match(line.strip())
                if parsed:
                    seeding[(parsed["problem"].strip(), method_name("DT-FBSDE+iLQR", variant))].append(
                        {key: float(value) for key, value in parsed.groupdict().items() if key != "problem"}
                    )
    return traces, references, seeding


def prior_best(prior_dir):
    best = {}
    if prior_dir is None:
        return best
    with open(prior_dir / "method_comparison.csv") as f:
        for row in csv.DictReader(f):
            cost = float(row["held_out_cost"])
            best[row["problem"]] = min(best.get(row["problem"], math.inf), cost)
    return best


def svg_panel(problem, entry, width=460, height=260):
    """Median gap to the reference against charged time, log-log, with an IQR band per method."""
    left, right, top, bottom = 52, 150, 28, 40
    times = entry["times"]
    series = []
    low = math.inf
    high = 0.0
    for method in SHOWN:
        if method not in entry["methods"]:
            continue
        curve = entry["methods"][method]["curve"]
        series.append((method, curve))
        for q25, q50, q75 in curve:
            for value in (q25, q50, q75):
                if value is not None and value > 0 and math.isfinite(value):
                    low = min(low, value)
                    high = max(high, value)
    low = max(low, 1e-3)
    y_lo = 10 ** math.floor(math.log10(low))
    y_hi = 10 ** math.ceil(math.log10(high))
    x_lo, x_hi = math.log10(times[0]), math.log10(times[-1])

    def x(t):
        return left + (math.log10(t) - x_lo) / (x_hi - x_lo) * (width - left - right)

    def y(gap):
        gap = min(max(gap, y_lo), y_hi)
        return top + (math.log10(y_hi) - math.log10(gap)) / (math.log10(y_hi) - math.log10(y_lo)) * (
            height - top - bottom
        )

    out = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(problem)}: median gap to the '
        f'reference against charged time">',
        f'<text x="{left}" y="16" class="title">{html.escape(problem)}</text>',
    ]
    decade = y_lo
    while decade <= y_hi * 1.0001:
        out.append(f'<line class="grid" x1="{left}" x2="{width - right}" y1="{y(decade):.1f}" y2="{y(decade):.1f}"/>')
        out.append(f'<text class="tick" x="{left - 6}" y="{y(decade) + 4:.1f}" text-anchor="end">'
                   f'{100 * decade:g}%</text>')
        decade *= 10
    t = 10 ** math.ceil(x_lo)
    while t <= times[-1] * 1.0001:
        label = f"{t:g} s" if t >= 1 else f"{1e3 * t:g} ms"
        out.append(f'<text class="tick" x="{x(t):.1f}" y="{height - bottom + 16}" text-anchor="middle">{label}</text>')
        t *= 10
    out.append(f'<text class="tick" x="{(left + width - right) / 2:.1f}" y="{height - 6}" text-anchor="middle">'
               "charged solver time</text>")
    labels = []
    for method, curve in series:
        points = [(x(times[i]), y(c[1])) for i, c in enumerate(curve) if c[1] is not None]
        band = [(x(times[i]), y(c[0])) for i, c in enumerate(curve) if c[0] is not None]
        band += [(x(times[i]), y(c[2])) for i, c in reversed(list(enumerate(curve))) if c[2] is not None]
        # Color follows the method, not its position among the methods present.
        slot = f"s{SHOWN.index(method) + 1}"
        out.append(f'<polygon class="band {slot}" points="{" ".join(f"{a:.1f},{b:.1f}" for a, b in band)}"/>')
        out.append(f'<polyline class="line {slot}" points="{" ".join(f"{a:.1f},{b:.1f}" for a, b in points)}">'
                   f"<title>{html.escape(method)}</title></polyline>")
        labels.append([points[-1][1], method, slot, curve[-1][1]])
    labels.sort()
    for i in range(1, len(labels)):
        labels[i][0] = max(labels[i][0], labels[i - 1][0] + 13)
    for label_y, method, slot, gap in labels:
        out.append(f'<circle class="dot {slot}" cx="{width - right + 6}" cy="{label_y - 4:.1f}" r="4"/>')
        out.append(f'<text class="label" x="{width - right + 14}" y="{label_y:.1f}">{html.escape(method)} '
                   f'{100 * gap:+.1f}%</text>')
    out.append("</svg>")
    return "\n".join(out)


def legend_classes():
    rules = []
    dark = []
    for index, method in enumerate(SHOWN):
        light_hex, dark_hex = COLORS[method]
        rules.append(f"  --s{index + 1}: {light_hex};")
        dark.append(f"    --s{index + 1}: {dark_hex};")
    marks = "\n".join(
        f".line.s{i + 1} {{ stroke: var(--s{i + 1}); }} .band.s{i + 1} {{ fill: var(--s{i + 1}); }} "
        f".dot.s{i + 1} {{ fill: var(--s{i + 1}); }}"
        for i in range(len(SHOWN))
    )
    return "\n".join(rules), "\n".join(dark), marks


def report_html(summary, seeding, text):
    light, dark, marks = legend_classes()
    panels = "\n".join(svg_panel(problem, entry) for problem, entry in summary.items())
    seed_rows = []
    for (problem, method), runs in sorted(seeding.items()):
        for run in sorted(runs, key=lambda r: r["seed"]):
            seed_rows.append(
                f"<tr><td>{html.escape(problem)}</td><td>{html.escape(method)}</td><td>{int(run['seed'])}</td>"
                f"<td>{int(run['chosen'])} of {int(run['total'])}</td><td>{int(run['candidates'])}</td>"
                f"<td>{run['seconds']:.2f}</td><td>{run['cost']:.4f}</td><td>{run['unseeded']:.2f}</td></tr>"
            )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>iLQR seeding study</title>
<style>
:root {{
  --surface: #fcfcfb; --ink: #1d1d1b; --muted: #6b6a63; --grid: #e4e3de;
{light}
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --surface: #1a1a19; --ink: #f0efec; --muted: #a9a89e; --grid: #33332f;
{dark}
  }}
}}
body {{ background: var(--surface); color: var(--ink); font: 14px/1.45 system-ui, sans-serif; margin: 0 auto;
  max-width: 980px; padding: 16px; }}
.panels {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; }}
svg {{ width: 100%; height: auto; }}
.title {{ font-weight: 600; fill: var(--ink); font-size: 13px; }}
.tick, .label {{ fill: var(--muted); font-size: 11px; }}
.label {{ fill: var(--ink); }}
.grid {{ stroke: var(--grid); stroke-width: 1; }}
.line {{ fill: none; stroke-width: 2; stroke-linejoin: round; }}
.band {{ opacity: 0.12; stroke: none; }}
{marks}
pre {{ overflow-x: auto; font-size: 11px; background: color-mix(in srgb, var(--grid) 35%, transparent); padding: 8px; }}
table {{ border-collapse: collapse; font-size: 12px; }} td, th {{ padding: 2px 8px; border-bottom: 1px solid var(--grid); }}
.scroll {{ overflow-x: auto; }}
</style></head><body>
<h1>Seeding DT-FBSDE with iLQR</h1>
<p>Median gap to the reference (line) and interquartile range (band) of the held-out cost against charged solver
time. 2-D problems: gap to the grid DP optimum. Double inverted pendulum: gap to the best policy seen in this study
or the earlier comparison. See README.md for the setup and conclusions.</p>
<div class="panels">
{panels}
</div>
<h2>Seeding</h2>
<p>Which iLQR iterate each run was seeded from, how many candidates it scored, the charged time of the whole iLQR
phase, and the seeded table's cost on DT-FBSDE's own 256 evaluation rollouts against the unseeded table's.</p>
<div class="scroll"><table><tr><th>problem</th><th>method</th><th>seed</th><th>iLQR iterate</th><th>candidates</th>
<th>seconds</th><th>seeded cost</th><th>unseeded cost</th></tr>
{"".join(seed_rows)}
</table></div>
<h2>Tables</h2>
<pre>{html.escape(text)}</pre>
</body></html>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("study_dir", type=Path)
    parser.add_argument("--prior", type=Path)
    args = parser.parse_args()
    traces, references, seeding = load_study(args.study_dir)
    for problem, best in prior_best(args.prior).items():
        if problem in traces and problem not in references:
            seen = min(cost for runs in traces[problem].values() for run in runs for cost in run["cost"])
            references[problem] = {
                "optimum": min(best, seen),
                "kind": "lowest held-out cost of any policy found, here or in the prior comparison",
            }
    summary = amc.summarize(traces, {}, references)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        amc.print_summary(summary)
    text = buffer.getvalue()
    print(text)
    (args.study_dir / "summary.txt").write_text(text)
    (args.study_dir / "summary.json").write_text(
        json.dumps(amc.finite({"summary": summary, "seeding": {f"{p} / {m}": r for (p, m), r in seeding.items()}}))
    )
    (args.study_dir / "report.html").write_text(report_html(summary, seeding, text))


if __name__ == "__main__":
    main()
