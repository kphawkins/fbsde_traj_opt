"""Summarizes run_queue.sh's profiles: per configuration, the wall-clock share of each phase, the
seconds per iteration, the acceptance rate, and the final held-out cost (gap to the grid optimum in
2-D). Run from the directory holding the CSVs; writes summary.json and prints a table."""

import csv
import json
import re
import statistics
from collections import defaultdict

SHARED = 'rank=8,time=linear,knots=16,balance=1,sweeps=2'
COMMON = 'batch=128,inflation=6,uphill=1'
# (file stem, model label in the CSVs, display name)
CONFIGS = [
    ('integrator', 'poly6', 'per stage'),
    ('integrator', 'poly6:time=linear,knots=16', '16 knots'),
    ('integrator', f'poly6:{SHARED}', 'rank 8 + 16 knots'),
    ('pendulum', 'poly6', 'per stage'),
    ('pendulum', 'poly6:time=linear,knots=16', '16 knots'),
    ('pendulum', f'poly6:{SHARED}', 'rank 8 + 16 knots'),
    ('dip', f'poly4:{COMMON}', 'per stage'),
    ('dip', f'poly4:{COMMON},time=linear,knots=16', '16 knots'),
    ('dip', f'poly4:{COMMON},{SHARED}', 'rank 8 + 16 knots'),
    ('dip_softmin', 'DT-FBSDE', 'soft-min, SGD (default)'),
]
PROBLEM = {'integrator': 'Double integrator', 'pendulum': 'Pendulum swing-up', 'dip': 'Double inverted pendulum',
           'dip_softmin': 'Double inverted pendulum'}


def read_profile(stem):
    totals = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
    seeds = defaultdict(set)
    with open(f'{stem}_profile.csv') as f:
        for row in csv.DictReader(f):
            model = row.get('model') or row.get('method')
            totals[model][row['phase']][0] += float(row['seconds'])
            totals[model][row['phase']][1] += int(row['calls'])
            seeds[model].add(row['seed'])
    return totals, seeds


def final_costs(stem):
    """Each run's final held-out cost, keyed by model, and the grid optimum if there is one."""
    costs = defaultdict(list)
    optimum = None
    if stem == 'dip_softmin':
        last = {}
        with open('dip_softmin.csv') as f:
            for row in csv.DictReader(f):
                if row['method'] == 'DT-FBSDE':
                    last[row['seed']] = float(row['held_out_cost'])
        costs['DT-FBSDE'] = list(last.values())
        return costs, optimum
    with open(f'{stem}_runs.csv') as f:
        for row in csv.DictReader(f):
            costs[row['model']].append(float(row['final_cost']))
    with open(f'{stem}.log') as f:
        match = re.search(r'grid optimum ([0-9.]+)', f.read())
        optimum = float(match.group(1)) if match else None
    return costs, optimum


def accepted(stem):
    out = defaultdict(lambda: [0, 0])
    if stem == 'dip_softmin':
        return out
    with open(f'{stem}_runs.csv') as f:
        for row in csv.DictReader(f):
            out[row['model']][0] += int(row['accepted'])
            out[row['model']][1] += int(row['iterations'])
    return out


def main():
    summary = []
    cache = {}
    for stem, model, name in CONFIGS:
        if stem not in cache:
            try:
                cache[stem] = (read_profile(stem), final_costs(stem), accepted(stem))
            except FileNotFoundError:
                cache[stem] = None
        if cache[stem] is None:
            continue
        (totals, seeds), (costs, optimum), acc = cache[stem]
        if model not in totals:
            continue
        ph = totals[model]
        g = lambda key: ph[key][0]
        charged, iterations = ph['charged_total']
        targets = g('backward.targets')
        share = g('backward.time_share')
        segments = {
            'fw': g('forward_pass'),
            'bt': targets,
            'bf': g('backward_pass') - targets - share,
            'ts': share,
            'ls': g('line_search.screen_eval'),
            'lf': g('line_search.full_eval'),
        }
        segments['lo'] = max(charged - sum(segments.values()), 0.0)
        run_costs = costs.get(model, [])
        entry = {
            'problem': PROBLEM[stem], 'stem': stem, 'model': model, 'name': name,
            'seeds': len(seeds[model]), 'iterations': iterations, 'charged': charged,
            's_per_it': charged / iterations,
            'share': {k: v / charged for k, v in segments.items()},
            'ms_per_it': {k: 1e3 * v / iterations for k, v in segments.items()},
            'fit_share_of_backward': g('backward.fit') / g('backward_pass') if g('backward_pass') else 0.0,
            'decision_us': 1e6 * g('rollout.policy') / max(ph['rollout.policy'][1], 1),
            'screen_per_it': ph['line_search.screen_eval'][1] / iterations,
            'full_per_it': ph['line_search.full_eval'][1] / iterations,
            'accept_rate': acc[model][0] / acc[model][1] if acc[model][1] else None,
            'final_costs': run_costs,
            'median_cost': statistics.median(run_costs) if run_costs else None,
            'optimum': optimum,
        }
        if optimum and run_costs:
            entry['median_gap'] = statistics.median(c / optimum - 1 for c in run_costs)
        summary.append(entry)
    with open('summary.json', 'w') as f:
        json.dump(summary, f, indent=1)
    for e in summary:
        sh = e['share']
        result = f"gap {100 * e['median_gap']:+.2f}%" if 'median_gap' in e else (
            f"cost {e['median_cost']:.2f}" if e['median_cost'] is not None else '')
        print(f"{e['problem'][:18]:<18} {e['name']:<24} {1e3 * e['s_per_it']:8.1f} ms/it {e['iterations']:6d} it "
              + ' '.join(f'{k} {100 * v:4.1f}%' for k, v in sh.items()) + f'  {result}')


if __name__ == '__main__':
    main()
