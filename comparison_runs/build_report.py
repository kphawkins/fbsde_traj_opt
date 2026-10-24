#!/usr/bin/env python3
"""Assembles the comparison report from analysis summaries and the prose in report_prose.json."""
import json
import sys
from pathlib import Path

here = Path(__file__).parent
ANCHORS = {"L1 double integrator": "integrator", "L1 inverted pendulum": "pendulum", "L1 double inverted pendulum": "double-pendulum"}

problems = []
for path in sys.argv[1:-1]:
    for name, entry in json.loads(Path(path).read_text()).items():
        entry["name"] = name
        entry["anchor"] = ANCHORS[name]
        problems.append(entry)
problems.sort(key=lambda e: list(ANCHORS).index(e["name"]))
prose = json.loads((here / "report_prose.json").read_text())
page = (here / "report_template.html").read_text()
page = page.replace("/*DATA*/null", json.dumps({"problems": problems}, separators=(",", ":")))
page = page.replace("/*NOTES*/{}", json.dumps(prose.pop("NOTES", {})))
for key, html in prose.items():
    page = page.replace(f"<!--{key}-->", html)
Path(sys.argv[-1]).write_text(page)
print(f"wrote {sys.argv[-1]} ({len(page) // 1024} KB)")
