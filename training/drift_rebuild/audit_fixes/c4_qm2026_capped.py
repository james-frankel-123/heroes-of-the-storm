"""Consolidated audit C4: the capped QM-2026 judge (pool ending at the 2026
cutoff, 2026-02-11) vs the uncapped pool, 3 seeds each, scoring the W6
drafts. Copied from the second audit's scratch (qm2026_capped.py) so the
0.513-0.537 numbers have a committed artifact. Run from training/.
Output: drift_rebuild/results/c4_qm2026_capped.json"""
import json,sys,os,numpy as np,torch,torch.nn as nn
sys.path.insert(0,'.')
torch.set_num_threads(4)
from qm2026.train_qm_wp import MLP, featurize
R=json.load(open('drift2026/results/w6_head2head.json'))['records']
allg=[json.loads(l) for l in open('qm2026/data/qm_games.jsonl')]
def run(lo,hi,seed):
    rng=np.random.default_rng(seed); torch.manual_seed(seed)
    games=[g for g in allg if lo<=(g.get('game_date') or '')<hi]
    idx=rng.permutation(len(games)); test=set(idx[:len(games)//10].tolist())
    def build(split):
        X,y=[],[]
        for i,g in enumerate(games):
            if (i in test)!=(split=='test'): continue
            X.append(featurize(g['team0_heroes'],g['team1_heroes'],g['game_map'],g['skill_tier'])); y.append(1.0 if g['winner']==0 else 0.0)
            if split=='train':
                X.append(featurize(g['team1_heroes'],g['team0_heroes'],g['game_map'],g['skill_tier'])); y.append(1-y[-1])
        return torch.tensor(np.stack(X)),np.array(y,dtype=np.float32)
    Xtr,ytr=build('train');Xte,yte=build('test');ytr=torch.tensor(ytr)
    m=MLP();opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=30);lf=nn.BCELoss()
    best,bs,pat=0,None,0
    for ep in range(30):
        m.train();perm=torch.randperm(len(Xtr))
        for i in range(0,len(perm),4096):
            b=perm[i:i+4096];opt.zero_grad();l=lf(m(Xtr[b]),ytr[b]);l.backward();opt.step()
        sch.step();m.eval()
        with torch.no_grad(): p=m(Xte).numpy().ravel()
        a=float(((p>.5)==(yte>.5)).mean())
        if a>best: best,bs,pat=a,{k:v.clone() for k,v in m.state_dict().items()},0
        else: pat+=1
        if pat>=5: break
    m.load_state_dict(bs);m.eval()
    Xf=torch.tensor(np.stack([featurize(r['maintained'],r['frozen'],r['game_map'],r['tier']) for r in R]))
    Xr=torch.tensor(np.stack([featurize(r['frozen'],r['maintained'],r['game_map'],r['tier']) for r in R]))
    with torch.no_grad(): w=(0.5*(m(Xf)+(1-m(Xr)))).numpy().ravel()
    print(lo,hi,seed,len(games),'acc %.4f wp %.4f pairSE %.4f'%(best,w.mean(),w.reshape(25,80).mean(1).std(ddof=1)/5),flush=True)
    RES.append({'from':lo,'to':hi,'seed':seed,'n_games':len(games),'acc':best,'maintained_wp':float(w.mean()),'pair_se':float(w.reshape(25,80).mean(1).std(ddof=1)/5)})
RES=[]
for seed in (20260721,7,991):
    run('2025-07-01','9999',seed)
    run('2025-07-01','2026-02-11',seed)

json.dump(RES,open('drift_rebuild/results/c4_qm2026_capped.json','w'),indent=1)
