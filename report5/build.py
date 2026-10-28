#!/usr/bin/env python3
"""Builds the off-policy drift study page from whatever runs have finished.

    python3 report5/build.py OUT.html
"""

import datetime
import glob
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HERE = ROOT / "report5"
STUDY = ROOT / "drift_study"

# The settings stage 3 reran on 5 new seeds, as the analyzer names them.
CONFIRMED = [
    "poly4:explore=0.03,target=greedy",
    "poly4:explore=0.1,target=taylor,tpolicy=greedy",
    "poly4:explore=0.1,inflate=1,target=taylor",
    "poly4:explore=0.1,inflate=1,target=current",
]

# Stage-2 table groups: which settings each holds, and which stage-1 rows are its baselines.
STAGE2_GROUPS = [
    {"name": "Ablation at ε = 0.1: which orders of the K-term expansion are kept", "keys": ["order"],
     "pred": "s.explore === '0.1'", "baseline": "s.explore === '0.1' && s.target === 'current'"},
    {"name": "Ablation at ε = 0.3", "keys": ["order"],
     "pred": "s.explore === '0.3'", "baseline": "s.explore === '0.3' && s.target === 'current'"},
    {"name": "Held random arcs: a random drift kept for several stages", "keys": ["hold"],
     "baseline": "(s.explore === '0.1' || s.explore === '0.2') && (s.target === 'greedy' || s.tpolicy === 'greedy')"},
    {"name": "Girsanov-weighted fit, weight exp(−κ‖D‖²/2)", "keys": ["gweight"],
     "baseline": "(s.explore === '0.1' || s.explore === '0.2') && s.target === 'taylor'"},
    {"name": "Random drift in place of noise inflation", "keys": ["inflate"], "pred": "!('explorers' in s)",
     "baseline": "s.explore === '0.03' || s.explore === '0.1'"},
    {"name": "Exploring only some paths", "keys": ["explorers"], "baseline": "s.explore === '0.03'"},
]


def summarize(paths, extra=()):
    if not paths:
        return []
    out = STUDY / "_summary.json"
    subprocess.run([sys.executable, str(ROOT / "tools" / "analyze_drift_study.py"), *paths, *extra, "--json", str(out)],
                   check=True, capture_output=True)
    return json.loads(out.read_text())


def main():
    target = Path(sys.argv[1])
    stage1 = summarize(sorted(glob.glob(str(STUDY / "s1_*_runs.csv"))))
    stage2 = summarize(sorted(glob.glob(str(STUDY / "s2_*_runs.csv"))) + sorted(glob.glob(str(STUDY / "s3_*_runs.csv"))))
    every = sorted(glob.glob(str(STUDY / "s[123]_*_runs.csv")))
    confirm = [r for r in summarize(every) if r["model"] in CONFIRMED]
    reference = ["--reference", "L1 double inverted pendulum=2.4928"]
    dip = summarize(sorted(glob.glob(str(STUDY / "dip2_*_runs.csv"))), reference)
    dip_early = summarize(sorted(glob.glob(str(STUDY / "dip_early" / "dip_*_runs.csv"))), reference)
    prose = json.loads((HERE / "prose.json").read_text())
    data = {
        "stage1": stage1,
        "stage2": stage2,
        "confirm": confirm,
        "dip": dip,
        "dip_early": dip_early,
        "stage2_groups": STAGE2_GROUPS,
        "status": prose.pop("_status") + [["Updated", datetime.datetime.now().strftime("%-d %B %Y, %H:%M")]],
    }
    page = ("<title>Off-Policy Drift Study</title>\n<meta name=\"claude:viewport-fit-sides\" content=\"cover\">\n" +
            (HERE / "base_style.html").read_text() + (HERE / "body.html").read_text())
    page = page.replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))
    for key, html in prose.items():
        page = page.replace(f"<!--{key}-->", html)
    target.write_text(page)
    print(f"wrote {target}: {len(stage1)} stage-1, {len(stage2)} stage-2, {len(dip)} DIP rows")


if __name__ == "__main__":
    main()
