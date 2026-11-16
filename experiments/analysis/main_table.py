"""The main table: per cell and fee, test tracking error / cost of the single policy and the Actor-Inactor (mean and
standard deviation over the seeds), with the model band and no hedge. Per seed, the single is the validation-best of
the cell's two rates and the Actor-Inactor is the validation-best of the doubles started from that single. Runs of the
follow-up experiments (band net, parity pair, gate, transfer) are left out. Reads the table written by collect_runs.py.
usage: python experiments/analysis/main_table.py [runs.json]     (default: experiments/results/collected/runs.json)"""
import json,statistics as st,re,sys,os
R=json.load(open(sys.argv[1] if len(sys.argv)>1 else os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),'experiments','results','collected','runs.json')))
main=[r for r in R if r.get('table') and r['run']!='baseline' and not any(k in r['run'] for k in ('_gate_','_xfer_','_pair_','band_lr'))]   # main recipe only: no gate/xfer/parity/band runs
band={(r['dynamics'],r['product'],r['fee']):r for r in R if r['group']=='baseline' and 'Whalley' in r['label']}
nhm={(r['dynamics'],r['product'],r['fee']):r for r in R if r['group']=='baseline' and 'no hedge' in r['label'].lower()}
MAIN={('bsm','call','0.1%'):('1e-4','3e-5'),('bsm','call','1%'):('1e-4','3e-5'),('bsm','doc','0.1%'):('1e-3','3e-3'),('bsm','doc','1%'):('3e-4','1e-4'),
      ('bsm','con','0.1%'):('3e-3','1e-2'),('bsm','con','1%'):('1e-3','3e-5'),('heston','call','0.1%'):('3e-3','1e-3'),('heston','call','1%'):('3e-3','1e-3'),
      ('heston','doc','0.1%'):('3e-3','1e-3'),('heston','doc','1%'):('3e-4','3e-3'),('heston','con','0.1%'):('3e-3','1e-3'),('heston','con','1%'):('3e-3','1e-3')}
cells=[('bsm','call','BSM call'),('bsm','doc','BSM down-and-out call'),('bsm','con','BSM cash-or-nothing'),('heston','call','Heston call'),('heston','doc','Heston down-and-out call'),('heston','con','Heston cash-or-nothing')]
def fromlr(r):   # returns the source rate in actor_lr notation (tag 3e5 -> 3e-5)
    m=re.search(r'_fromlr(\d)e(\d)_s\d',r['run']); return f"{m.group(1)}e-{m.group(2)}" if m else None
def m(xs): return (f"{st.mean(xs):.2f} ± {st.pstdev(xs):.2f}" if len(xs)>1 else f"{xs[0]:.2f}") if xs else '—'
def c(xs): return f"{st.mean(xs):.1f}" if xs else '—'
out=[]
hdr="| Cell | Fee | Single: test track / cost | Double: test track / cost | Model band | No hedge | Seeds S/D | Running |\n|---|---|---|---|---|---|---|---|"
print(hdr)
for dyn,prod,name in cells:
  for fee in ['0.1%','1%']:
    cr=[r for r in main if r['dynamics']==dyn and r['product']==prod and r['fee']==fee]
    r1,r2=MAIN[(dyn,prod,fee)]; rates={r1,r2}
    S,D,SC,DC=[],[],[],[]
    for seed in range(1,6):
        sg=[r for r in cr if r['net']=='single' and r['seed']==seed and r['status']=='done' and r.get('test_track') is not None and (r['actor_lr'] in rates or not rates)]
        if not sg: continue
        best=min(sg,key=lambda r:r['best_val_track'])
        S.append(best['test_track']); SC.append(best['test_cost'])
        dd=[r for r in cr if r['net']=='double' and r['seed']==seed and r['status']=='done' and r.get('test_track') is not None]
        # a tagged double comes from the second-rate single; an untagged one from the main-rate single (seed 1: the tuning pick)
        # untagged doubles come from the first-rate single (seed 1: the tuning pick); tagged ones from the second-rate single
        pick=[r for r in dd if not fromlr(r)] if best['actor_lr']==r1 else [r for r in dd if fromlr(r)==best['actor_lr']]
        if pick:
            bd=min(pick,key=lambda r:r['best_val_track']); D.append(bd['test_track']); DC.append(bd['test_cost'])
    b=band.get((dyn,prod,fee)); nh=nhm.get((dyn,prod,fee))
    bs=('%.2f / %.1f'%(b['test_track'],b['test_cost'])) if b else 'running'
    run=[r for r in cr if r['status']!='done']
    print(f"| {name} | {fee} | {m(S)} / {c(SC)} | {m(D)} / {c(DC)} | {bs} | {nh['test_track']:.1f} | {len(S)}/{len(D)} | {len(run)} |")
