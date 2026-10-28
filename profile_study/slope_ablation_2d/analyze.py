# Summarizes the 2-D slope ablation: final gap to the grid DP optimum, time to within 5% and 10%, and
# seconds per iteration, for DT-FBSDE and DT-FBSDE+iLQR with slopes on and off. Usage: python3 analyze.py
import collections
import csv
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
optimum = {row['problem']: float(row['grid_optimum']) for row in csv.DictReader(open(os.path.join(HERE, 'reference.csv')))}


def time_to(points, target):
    for seconds, cost in points:
        if cost <= target:
            return seconds
    return None


print(f'{"problem":<28}{"method":<16}{"slopes":<8}{"median gap":>11}{"worst":>8}{"t<10% med":>11}{"t<5% med":>10}'
      f'{"s/iter":>9}')
for problem in optimum:
    for method in ['DT-FBSDE', 'DT-FBSDE+iLQR']:
        for arm in ['on', 'off']:
            runs = collections.defaultdict(list)
            for row in csv.DictReader(open(os.path.join(HERE, f'slopes_{arm}.csv'))):
                if row['problem'] == problem and row['method'] == method:
                    runs[row['seed']].append((float(row['seconds']), float(row['held_out_cost']), int(row['iteration'])))
            reference = optimum[problem]
            gaps = [runs[seed][-1][1] / reference - 1 for seed in runs]
            per_iteration = statistics.mean(runs[seed][-1][0] / runs[seed][-1][2] for seed in runs)
            def median_time(fraction):
                times = [time_to([(s, c) for s, c, _ in runs[seed]], reference * (1 + fraction)) for seed in runs]
                reached = [t for t in times if t is not None]
                text = f'{statistics.median(reached):.2f}' if len(reached) == len(times) else f'{len(reached)}/{len(times)}'
                return text
            print(f'{problem:<28}{method:<16}{arm:<8}{100 * statistics.median(gaps):>10.2f}%{100 * max(gaps):>7.2f}%'
                  f'{median_time(0.10):>11}{median_time(0.05):>10}{1e3 * per_iteration:>8.1f}ms')
