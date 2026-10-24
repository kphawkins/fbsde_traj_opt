#!/usr/bin/env python3
"""Builds the carried-normal-equations sweep page from whatever runs have finished.

    python3 report3/build.py OUT.html
"""

import datetime
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROBLEMS = ["L1 inverted pendulum", "L1 double integrator"]
SETTINGS = [(0.0, False), (0.5, False), (0.8, False), (0.9, False), (0.95, False),
            (0.5, True), (0.8, True), (0.9, True), (0.95, True), (1.0, True)]
SEEDS = 5


def main():
    out = Path(sys.argv[1])
    subprocess.run([sys.executable, str(ROOT / "tools" / "analyze_rls_sweep.py"), str(ROOT / "rls_study"),
                    "--json", str(ROOT / "report3" / "rls.json")], check=True, capture_output=True)
    table = json.loads((ROOT / "report3" / "rls.json").read_text())
    rows = {}
    for r in table:
        rows[(r["problem"], r["memory"], r["reset"])] = r
    data = {"problems": [], "seeds": SEEDS}
    done = 0
    for problem in PROBLEMS:
        entries = []
        for memory, reset in SETTINGS:
            r = rows.get((problem, memory, reset))
            entries.append({"memory": memory, "reset": reset, "result": r})
            done += r["runs"] if r else 0
        data["problems"].append({"name": problem, "entries": entries})
    data["done"] = done
    data["total"] = len(PROBLEMS) * len(SETTINGS) * SEEDS
    data["updated"] = datetime.datetime.now().strftime("%-d %B %Y, %H:%M")
    prose = json.loads((ROOT / "report3" / "prose.json").read_text())
    page = (ROOT / "report3" / "page.html").read_text()
    page = page.replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))
    for key, html in prose.items():
        page = page.replace(f"<!--{key}-->", html)
    out.write_text(page)
    print(f"wrote {out}: {done} of {data['total']} runs")


if __name__ == "__main__":
    main()
