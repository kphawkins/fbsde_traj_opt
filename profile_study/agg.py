# Aggregates the profiling driver's per-iteration and per-phase CSVs into results/agg.json.
# Usage: python3 agg.py integrator pendulum dip dip-sweep
import csv, json, os, sys, statistics as st, collections
d=os.path.join(os.path.dirname(os.path.abspath(__file__)),'results')+'/'
out={}
for prob in sys.argv[1:]:
    its=list(csv.DictReader(open(d+prob+'_it.csv')))
    ph=list(csv.DictReader(open(d+prob+'_ph.csv')))
    P=collections.defaultdict(float); C=collections.defaultdict(int)
    for r in ph: P[r['phase']]+=float(r['seconds']); C[r['phase']]+=int(r['calls'])
    n=len(its)
    tot=sum(float(r['seconds']) for r in its)
    fw=sum(float(r['forward']) for r in its); bw=sum(float(r['backward']) for r in its); ls=sum(float(r['line_search']) for r in its)
    seeds=sorted(set(r['seed'] for r in its))
    per_seed=[]
    for s in seeds:
        rs=[r for r in its if r['seed']==s]
        per_seed.append(dict(seed=int(s),iterations=len(rs),seconds=sum(float(r['seconds']) for r in rs),final_cost=float(rs[-1]['cost'])))
    cands=collections.Counter(int(r['candidates']) for r in its)
    accepted=sum(1 for r in its if float(r['accepted_step'])>0)
    # trend: per seed, bin iterations into deciles of iteration index
    trend=[]
    s0=[r for r in its if r['seed']==seeds[0]]
    B=max(1,len(s0)//20)
    for i in range(0,len(s0),B):
        ch=s0[i:i+B]
        trend.append(dict(it=int(ch[0]['iteration']),fw=st.mean(float(r['forward']) for r in ch),bw=st.mean(float(r['backward']) for r in ch),ls=st.mean(float(r['line_search']) for r in ch),cand=st.mean(int(r['candidates']) for r in ch)))
    out[prob]=dict(iterations=n,seconds=tot,s_per_it=tot/n,forward=fw,backward=bw,line_search=ls,other=tot-fw-bw-ls,
        phases={k:dict(seconds=P[k],calls=C[k]) for k in P},per_seed=per_seed,candidates=dict(sorted(cands.items())),accepted=accepted,trend=trend)
json.dump(out,open(d+'agg.json','w'),indent=1)
