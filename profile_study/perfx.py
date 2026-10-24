# Summarizes `perf record -g` profiles into results/perf.json: the top functions by self time.
# Usage: python3 perfx.py integrator pendulum dip   (reads <problem>.perf.data from the current directory)
import subprocess, json, re, sys, os
P=os.environ.get('PERF','perf')
HERE=os.path.dirname(os.path.abspath(__file__))
out={}
def short(s):
    s=s.strip().split('   ')[0].strip()
    s=re.sub(r'^void ','',s)
    hm=re.search(r'Derivatives<(true|false)>',s)
    poly=re.match(r'.*PolynomialValueFunctionApprox<(\d+), (\d+)',s)
    # strip args
    depth=0; res=''
    i=0
    # remove template args recursively
    for ch in s:
        if ch=='<': depth+=1; 
        if depth==0: res+=ch
        if ch=='>': depth-=1; 
        if ch=='>' and depth==0: res+='<…>'
    res=res.split('(')[0]
    res=res.replace('fbsde_traj_opt::','').replace('examples::','')
    if hm: res=res.replace('Derivatives<…>','Derivatives<'+hm.group(1)+'>')
    if poly: res=res.replace('PolynomialValueFunctionApprox<…>','PolynomialValueFunctionApprox<'+poly.group(1)+','+poly.group(2)+'>')
    if res.endswith('::operator'): res+='()'
    return res
for p in sys.argv[1:]:
    f=f'{p}.perf.data'
    if not os.path.exists(f): continue
    r=subprocess.run([P,'report','-i',f,'--no-children','--sort','symbol','--stdio','-g','none'],capture_output=True,text=True).stdout
    rows=[]
    for line in r.splitlines():
        m=re.match(r'\s+([\d.]+)%\s+\[[.k]\]\s+(.*)',line)
        if m: rows.append((short(m.group(2)),float(m.group(1))))
    out[p]=rows[:12]
json.dump(out,open(os.path.join(HERE,'results','perf.json'),'w'),indent=1)
for p,v in out.items(): print(p,v[:8])
