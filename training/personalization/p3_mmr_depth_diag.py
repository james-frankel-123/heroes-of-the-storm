"""P3 #2 diagnostics: MMR leakage test (this-game vs lagged MMR) and lift by history depth. Run from training/: python3 personalization/p3_mmr_depth_diag.py"""
import sys, os, json, datetime
sys.path.insert(0,'personalization'); sys.path.insert(0,'.')
import numpy as np
import p3_nested_lift as P
games, rows = P.load()
n=len(games['y']); days=games['date_days']
e0=(datetime.date.fromisoformat(P.E_START)-datetime.date(1970,1,1)).days
in_e=games['in_sample']&(days>=e0); post=~games['in_sample']; med=np.median(days[post])
v1,v2=post&(days<med),post&(days>=med)
feats,meta=P.personal_features(games,rows,in_e)
g=rows['g']; y=games['y']
# MMR leak test: previous-game MMR of the same player
order=np.lexsort((games['replay_ids'][g], days[g], rows['blizz_ids']))
b=rows['blizz_ids'][order]; m=rows['mmr'][order]
prev=np.full(len(order),np.nan,np.float32); same=np.r_[False,b[1:]==b[:-1]]
prev[1:][same[1:]]=m[:-1][same[1:]]
prev_mmr=np.empty_like(prev); prev_mmr[order]=prev
fill=lambda a: np.nan_to_num(a,nan=np.nanmean(a))/100.0
cnt=np.bincount(g,minlength=n); full=cnt==10
has_prev=np.bincount(g,weights=~np.isnan(prev_mmr),minlength=n)==10
lo=np.log(games['wp0']/(1-games['wp0']))
D={k:P.team_diff(v,rows,n) for k,v in feats.items() if k!='n_p_E'}
D['mmr_this']=P.team_diff(fill(rows['mmr']),rows,n)
D['mmr_prev']=P.team_diff(fill(prev_mmr),rows,n)
fit,test=v1&full&has_prev, v2&full&has_prev
print('games with prev MMR for all 10: test',test.sum())
for name,cols in {'M0':[],'this-game MMR':['mmr_this'],'previous-game MMR':['mmr_prev'],'M4':['raw','resid_ph'],'M4+prev MMR':['raw','resid_ph','mmr_prev']}.items():
    X=np.column_stack([lo]+[D[c] for c in cols]); w=P.fit_logistic(X[fit],y[fit]); p=P.predict(w,X)
    mt=P.metrics(p[test],y[test]); print(f'{name:20s} acc {mt["acc"]:.4f} ll {mt["logloss"]:.5f} auc {mt["auc"]:.4f}')
# coverage & history-depth subsets (M0 vs M4)
nE=np.bincount(g,weights=np.minimum(feats['n_p_E'],1000),minlength=n)/np.maximum(cnt,1)
seen=np.bincount(g,weights=feats['n_p_E']>0,minlength=n)
t=v2&full
print('V2 slots with any E history: %.3f; mean E games per slot %.0f'%( (seen[t]/10).mean(), nE[t].mean()))
res={}
for nm,cols in {'M0':[],'M4':['raw','resid_ph']}.items():
    X=np.column_stack([lo]+[D[c] for c in cols]); w=P.fit_logistic(X[v1&full],y[v1&full]); res[nm]=P.predict(w,X)
qs=np.percentile(nE[t],[25,50,75])
edges=[-1,qs[0],qs[1],qs[2],1e9]
for i in range(4):
    s=t&(nE>edges[i])&(nE<=edges[i+1])
    a0=P.metrics(res['M0'][s],y[s]); a4=P.metrics(res['M4'][s],y[s])
    print(f'quartile {i+1} mean E games/slot ({edges[i]:.0f},{edges[i+1]:.0f}]: n {s.sum()} acc {a0["acc"]:.4f}->{a4["acc"]:.4f} ll gain {a0["logloss"]-a4["logloss"]:+.5f}')
# MMR from the player's most recent game at least G days earlier
ds=days[g][order]
for G in (1,7,30):
    out=np.full(len(order),np.nan,np.float32)
    # per player pointer walk
    start=0
    bb=b; 
    idx=np.arange(len(order))
    # for each row, last index j in same player with ds[j] <= ds[i]-G
    brk=np.flatnonzero(np.r_[True,bb[1:]!=bb[:-1]]); ends=np.r_[brk[1:],len(bb)]
    for s,e in zip(brk,ends):
        d=ds[s:e]; j=np.searchsorted(d, d-G, side='right')-1
        ok=j>=0; seg=np.full(e-s,np.nan,np.float32); seg[ok]=m[s:e][j[ok]]; out[s:e]=seg
    lag=np.empty_like(out); lag[order]=out
    hl=np.bincount(g,weights=~np.isnan(lag),minlength=n)==10
    D['lag']=P.team_diff(fill(lag),rows,n)
    fit,test=v1&full&hl, v2&full&hl
    for name,cols in {'M0':[],'lag MMR':['lag'],'M4':['raw','resid_ph'],'M4+lag':['raw','resid_ph','lag']}.items():
        X=np.column_stack([lo]+[D[c] for c in cols]); w=P.fit_logistic(X[fit],y[fit]); p=P.predict(w,X)
        mt=P.metrics(p[test],y[test]); print(f'lag>={G}d n={test.sum()} {name:10s} acc {mt["acc"]:.4f} ll {mt["logloss"]:.5f}')
