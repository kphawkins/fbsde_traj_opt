"""Renders summary.json (analyze.py) as the report's lead section: a stacked bar of phase shares per
configuration and the table behind it. Prints the HTML fragment; gen.py-era styles are assumed."""

import html
import json
import sys

SEGMENTS = [
    ('fw', 'Forward pass', 'var(--s1)'),
    ('bt', 'Backward targets (Q argmin)', 'var(--s2)'),
    ('bf', 'Backward fit (per stage)', 'var(--s3)'),
    ('ts', 'Time-shared joint solve', 'var(--s7)'),
    ('ls', 'Line search: screening rollouts', 'var(--s4)'),
    ('lf', 'Line search: full rollouts', 'var(--s5)'),
    ('lo', 'Other', 'var(--s6)'),
]


def pct(x, digits=0):
    return f'{100 * x:.{digits}f}%'


def ms(x):
    return f'{x:,.0f} ms' if x >= 100 else f'{x:.1f} ms'


def result(e):
    if 'median_gap' in e:
        return f"gap {100 * e['median_gap']:+.2f}%"
    if e['median_cost'] is not None:
        return f"cost {e['median_cost']:.1f}"
    return ''


def chart(summary):
    width, bar_h, label_h, gap, group_gap = 1000, 30, 18, 12, 26
    parts = []
    y = 0
    problem = None
    for e in summary:
        if e['problem'] != problem:
            if problem is not None:
                y += group_gap
            problem = e['problem']
            parts.append(f'<text x="0" y="{y + 14}" class="group-label">{html.escape(problem)}</text>')
            y += 26
        parts.append(f'<text x="0" y="{y + 12}" class="row-label">{html.escape(e["name"])} '
                     f'<tspan class="row-sub">· {ms(1e3 * e["s_per_it"])} per iteration · {result(e)}</tspan></text>')
        x = 0.0
        top = y + label_h
        for key, name, color in SEGMENTS:
            w = e['share'][key] * width
            if w <= 0.4:
                continue
            tip = (f'{e["problem"]}, {e["name"]} · {name}: {pct(e["share"][key], 1)} of solver time '
                   f'({ms(e["ms_per_it"][key])} per iteration)')
            parts.append(f'<rect x="{x:.2f}" y="{top}" width="{max(w - 2, 0.5):.2f}" height="{bar_h}" rx="3" '
                         f'fill="{color}" data-tip="{html.escape(tip)}"/>')
            if w > 52:
                parts.append(f'<text x="{x + 6:.2f}" y="{top + bar_h / 2 + 4.5}" class="seg-label">'
                             f'{pct(e["share"][key])}</text>')
            x += w
        y = top + bar_h + gap
    return (f'<svg viewBox="0 0 {width} {y}" class="chart" role="img" '
            f'aria-label="Share of solver time by phase, per value-function model">{"".join(parts)}</svg>')


def legend():
    return '<ul class="legend">' + ''.join(
        f'<li><span class="sw" style="background:{c}"></span>{n}</li>' for _, n, c in SEGMENTS) + '</ul>'


def table(summary):
    head = ''.join(f'<th>{n}</th>' for _, n, _ in SEGMENTS[:-1])
    rows = []
    for e in summary:
        cells = ''.join(f'<td>{pct(e["share"][k], 1)} <span class="ms">{ms(e["ms_per_it"][k])}</span></td>'
                        for k, _, _ in SEGMENTS[:-1])
        acc = pct(e['accept_rate'], 1) if e['accept_rate'] is not None else 'n/a'
        rows.append(f'<tr><th scope="row">{html.escape(e["problem"])} · {html.escape(e["name"])}</th>'
                    f'<td><b>{ms(1e3 * e["s_per_it"])}</b></td>{cells}'
                    f'<td>{e["iterations"]:,} / {e["seeds"]}</td><td>{acc}</td><td>{result(e)}</td></tr>')
    return ('<div class="tw"><table class="wide"><thead><tr><th>Configuration</th><th>Per iteration</th>'
            f'{head}<th>Iterations / seeds</th><th>Accepted</th><th>Median final</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


def main():
    summary = json.load(open(sys.argv[1]))
    print(f'<div class="figure">{legend()}{chart(summary)}</div>')
    print(table(summary))


if __name__ == '__main__':
    main()
