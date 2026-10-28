# Summarizes the DIP slope-target ablation: held-out cost against charged time and iterations, and
# the phase shares from --profile-csv. Usage: python3 analyze.py
import collections
import csv
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))


def last_at(series, index, limit):
    value = None
    for point in series:
        if point[index] <= limit:
            value = point[2]
    return value


for weight in ['10', '0']:
    runs = collections.defaultdict(list)
    for row in csv.DictReader(open(os.path.join(HERE, f'slope_w{weight}.csv'))):
        runs[row['seed']].append((float(row['seconds']), int(row['iteration']), float(row['held_out_cost'])))
    phases = collections.defaultdict(float)
    iterations = 0
    for row in csv.DictReader(open(os.path.join(HERE, f'profile_w{weight}.csv'))):
        phases[row['phase']] += float(row['seconds'])
        if row['phase'] == 'charged_total':
            iterations += int(row['calls'])
    total = phases['charged_total']
    print(f'slope weight {weight}: {total / iterations:.2f} s per iteration over {iterations} iterations')
    for seconds in [10, 30, 60, 120]:
        costs = [last_at(runs[seed], 0, seconds) for seed in sorted(runs)]
        print(f'  {seconds:>3} s: median {statistics.median(costs):6.2f}  mean {statistics.mean(costs):6.2f}  '
              + ' '.join(f'{cost:5.1f}' for cost in costs))
    for iteration in [20, 40, 80]:
        costs = [last_at(runs[seed], 1, iteration) for seed in sorted(runs)]
        print(f'  iteration {iteration:>3}: median {statistics.median(costs):6.2f}')
    print('  wall shares: ' + ', '.join(f'{name} {phases[name] / total:.0%}' for name in
                                       ['forward_pass', 'backward.targets', 'backward.fit', 'line_search']))
