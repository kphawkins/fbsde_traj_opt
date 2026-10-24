# Builds report.html from results/agg.json, results/perf.json, report_prose.json and template.html.
# Usage: python3 gen.py
import json, os, sys, html

HERE = os.path.dirname(os.path.abspath(__file__))
AGG = json.load(open(os.path.join(HERE, 'results', 'agg.json')))
PERF = json.load(open(os.path.join(HERE, 'results', 'perf.json')))
EXTRA = json.load(open(os.path.join(HERE, 'report_prose.json')))
OUT = os.path.join(HERE, 'report.html')

LABEL = {'integrator': 'Double integrator', 'pendulum': 'Pendulum swing-up',
         'dip': 'Double inverted pendulum', 'dip-sweep': 'Double inverted pendulum (2-D recipe)'}

SEGMENTS = [
    ('fw', 'Forward pass', 'var(--s1)'),
    ('bt', 'Backward targets (Q argmin)', 'var(--s2)'),
    ('bf', 'Backward fit', 'var(--s3)'),
    ('ls', 'Line search: screening rollouts', 'var(--s4)'),
    ('lf', 'Line search: full rollouts', 'var(--s5)'),
    ('lo', 'Other', 'var(--s6)'),
]


def segments(v):
    t = v['seconds']
    ph = v['phases']
    g = lambda k: ph.get(k, {}).get('seconds', 0.0)
    fw = v['forward']
    bt = g('backward.targets')
    bf = v['backward'] - bt
    ls = g('line_search.screen_eval')
    lf = g('line_search.full_eval')
    lo = t - fw - bt - bf - ls - lf
    return {'fw': fw / t, 'bt': bt / t, 'bf': bf / t, 'ls': ls / t, 'lf': lf / t, 'lo': max(lo, 0) / t}


def pct(x, digits=0):
    return f'{100 * x:.{digits}f}%'


def stacked_chart(problems):
    row_h, gap, left, width = 34, 22, 0, 1000
    rows = []
    y = 0
    for p in problems:
        v = AGG[p]
        seg = segments(v)
        x = 0.0
        rects = []
        for key, name, color in SEGMENTS:
            w = seg[key] * width
            if w <= 0:
                continue
            tip = f'{LABEL[p]} · {name}: {pct(seg[key], 1)} of solver time ({seg[key] * v["s_per_it"] * 1e3:.1f} ms per iteration)'
            inner = max(w - 2, 0.5)
            rects.append(f'<rect x="{x:.2f}" y="{y + 18}" width="{inner:.2f}" height="{row_h}" rx="3" fill="{color}" data-tip="{html.escape(tip)}"/>')
            if w > 60:
                rects.append(f'<text x="{x + 6:.2f}" y="{y + 18 + row_h / 2 + 4}" class="seg-label">{pct(seg[key])}</text>')
            x += w
        rows.append(f'<text x="0" y="{y + 12}" class="row-label">{LABEL[p]} <tspan class="row-sub">· {v["s_per_it"] * 1e3:.0f} ms per iteration</tspan></text>' + ''.join(rects))
        y += row_h + 18 + gap
    h = y
    return f'<svg viewBox="0 0 {width} {h}" class="chart" role="img" aria-label="Share of solver time by phase">{"".join(rows)}</svg>'


def legend():
    return '<ul class="legend">' + ''.join(
        f'<li><span class="sw" style="background:{c}"></span>{n}</li>' for _, n, c in SEGMENTS) + '</ul>'


def candidates_chart(p):
    v = AGG[p]
    c = {int(k): n for k, n in v['candidates'].items()}
    total = sum(c.values())
    keys = list(range(1, 9))
    width, height, bottom = 420, 170, 26
    bw = width / len(keys)
    mx = max(c.values())
    bars = []
    for i, k in enumerate(keys):
        n = c.get(k, 0)
        hgt = (height - bottom - 16) * n / mx
        x = i * bw + 4
        yy = height - bottom - hgt
        tip = f'{LABEL[p]}: {n} iterations ({pct(n / total, 1)}) tried {k} candidate step{"s" if k > 1 else ""}'
        bars.append(f'<rect x="{x:.1f}" y="{yy:.1f}" width="{bw - 8:.1f}" height="{max(hgt, 1):.1f}" rx="3" fill="var(--s4)" data-tip="{html.escape(tip)}"/>')
        bars.append(f'<text x="{x + (bw - 8) / 2:.1f}" y="{height - 8}" class="tick" text-anchor="middle">{k}</text>')
        if n / total >= 0.08:
            bars.append(f'<text x="{x + (bw - 8) / 2:.1f}" y="{yy - 4:.1f}" class="val" text-anchor="middle">{pct(n / total)}</text>')
    base = f'<line x1="0" x2="{width}" y1="{height - bottom}" y2="{height - bottom}" class="axis"/>'
    return f'<svg viewBox="0 0 {width} {height}" class="chart small" role="img" aria-label="Candidates per iteration, {LABEL[p]}">{base}{"".join(bars)}</svg>'


def phase_table(problems):
    head = ''.join(f'<th>{LABEL[p]}</th>' for p in problems)
    rows = []
    for key, name, _ in SEGMENTS:
        cells = ''.join(f'<td>{pct(segments(AGG[p])[key], 1)} <span class="ms">{segments(AGG[p])[key] * AGG[p]["s_per_it"] * 1e3:.1f} ms</span></td>' for p in problems)
        rows.append(f'<tr><th scope="row">{name}</th>{cells}</tr>')
    tot = ''.join(f'<td><b>{AGG[p]["s_per_it"] * 1e3:.1f} ms</b></td>' for p in problems)
    its = ''.join(f'<td>{AGG[p]["iterations"]:,} over {len(AGG[p]["per_seed"])} seeds</td>' for p in problems)
    return f'<div class="tw"><table><thead><tr><th>Phase (wall clock)</th>{head}</tr></thead><tbody>{"".join(rows)}<tr class="tot"><th scope="row">Per iteration</th>{tot}</tr><tr><th scope="row">Iterations profiled</th>{its}</tr></tbody></table></div>'


def inner_table(problems):
    def us(p, k):
        ph = AGG[p]['phases'].get(k)
        return ph['seconds'] / ph['calls'] * 1e6 if ph and ph['calls'] else 0.0

    def share(p, k, parent):
        ph = AGG[p]['phases']
        return ph[k]['seconds'] / ph[parent]['seconds'] if ph.get(parent, {}).get('seconds') else 0.0

    lines = [
        ('One policy decision in a rollout (µs)', lambda p: f'{us(p, "rollout.policy"):.1f}'),
        ('Rollout thread time spent choosing controls', lambda p: pct(share(p, 'rollout.policy', 'rollout.sample'), 1)),
        ('Rollout thread time stepping the dynamics', lambda p: pct(share(p, 'rollout.dynamics', 'rollout.sample'), 1)),
        ('Backward target time in the chooser’s argmin', lambda p: pct(share(p, 'backward.argmin', 'backward.target_sample'), 1)),
        ('Backward target time in the slope target', lambda p: pct(share(p, 'backward.slope_target', 'backward.target_sample'), 1)),
        ('Backward pass time in the least-squares / SGD fit', lambda p: pct(AGG[p]['phases']['backward.fit']['seconds'] / AGG[p]['phases']['backward_pass']['seconds'], 1)),
        ('Iterations that accepted a step', lambda p: pct(AGG[p]['accepted'] / AGG[p]['iterations'], 1)),
        ('Screening evaluations per iteration', lambda p: f'{AGG[p]["phases"]["line_search.screen_eval"]["calls"] / AGG[p]["iterations"]:.2f}'),
        ('Full evaluations per iteration', lambda p: f'{AGG[p]["phases"]["line_search.full_eval"]["calls"] / AGG[p]["iterations"]:.2f}'),
        ('One screening evaluation (ms)', lambda p: f'{AGG[p]["phases"]["line_search.screen_eval"]["seconds"] / max(AGG[p]["phases"]["line_search.screen_eval"]["calls"], 1) * 1e3:.1f}'),
        ('One full evaluation (ms)', lambda p: f'{AGG[p]["phases"]["line_search.full_eval"]["seconds"] / max(AGG[p]["phases"]["line_search.full_eval"]["calls"], 1) * 1e3:.1f}'),
        ('Uncharged held-out evaluation, 1024 rollouts (ms)', lambda p: f'{AGG[p]["phases"]["held_out_eval"]["seconds"] / AGG[p]["phases"]["held_out_eval"]["calls"] * 1e3:.0f}'),
    ]
    head = ''.join(f'<th>{LABEL[p]}</th>' for p in problems)
    body = ''.join(f'<tr><th scope="row">{n}</th>' + ''.join(f'<td>{f(p)}</td>' for p in problems) + '</tr>' for n, f in lines)
    return f'<div class="tw"><table><thead><tr><th>Measure</th>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def perf_table(p):
    rows = ''.join(f'<tr><td class="fn">{html.escape(name)}</td><td>{share:.1f}%</td></tr>' for name, share in PERF[p][:8])
    return f'<div class="tw"><table class="perf"><thead><tr><th>Function (self time)</th><th>CPU samples</th></tr></thead><tbody>{rows}</tbody></table></div>'


two_d = ['integrator', 'pendulum']
dip = [p for p in ['dip', 'dip-sweep'] if p in AGG]

T = open(os.path.join(HERE, 'template.html')).read()
subs = {
    '{{STACK_2D}}': stacked_chart(two_d),
    '{{LEGEND}}': legend(),
    '{{TABLE_2D}}': phase_table(two_d),
    '{{INNER_2D}}': inner_table(two_d),
    '{{CAND_INT}}': candidates_chart('integrator'),
    '{{CAND_PEN}}': candidates_chart('pendulum'),
    '{{PERF_INT}}': perf_table('integrator'),
    '{{PERF_PEN}}': perf_table('pendulum'),
}
if dip:
    subs['{{STACK_DIP}}'] = stacked_chart(dip)
    subs['{{TABLE_DIP}}'] = phase_table(dip)
    subs['{{INNER_DIP}}'] = inner_table(dip)
    subs['{{PERF_DIP}}'] = perf_table('dip') if 'dip' in PERF else ''
for k, val in EXTRA.items():
    T = T.replace('{{' + k + '}}', str(val))
for k, val in subs.items():
    T = T.replace(k, val)
open(OUT, 'w').write(T)
