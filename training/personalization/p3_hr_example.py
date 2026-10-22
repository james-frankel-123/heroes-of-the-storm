"""Worked example for P3_HERO_RECS_AND_REPLAYS: one heavy held-out-style
player at the 2026-02-10 cutoff (anonymized; no ids written)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

import p3_hr_common as HC  # noqa: E402

z = np.load(os.path.join(HC.HR_CACHE, "backtest_t0_n5.npz"))
names = HC.load()["hero_names"]
N, n_post, cand, act = z["N"], z["o_n"], z["cand"], z["active"]
took = cand & (n_post >= 20)
tot = N.sum(1)
ok = act & (tot >= 1500) & (took.sum(1) >= 3)
cands = np.flatnonzero(ok)
print("eligible players", len(cands))
# the one with most taken-up heroes, ties by volume
p = cands[np.lexsort((-tot[cands], -took[cands].sum(1)))[0]]
e1 = z["o_e1_s"][p] / np.maximum(z["o_e1_c"][p], 1)
e2 = z["o_e2_s"][p] / np.maximum(z["o_e2_c"][p], 1)
lines = [f"games before cutoff {int(tot[p])}, heroes with 20+ games {(N[p] >= 20).sum()}, candidates {cand[p].sum()}"]
top = np.argsort(-N[p])[:6]
lines.append("most played: " + "; ".join(f"{names[h]} {int(N[p, h])} g, model {z['main'][p, h]:+.1f}pp" for h in top))
for nm in ("main", "pref", "combo", "imp2", "sbsim"):
    key = "_pct_" + nm
    if key not in z.files:
        continue
    pct = z[key][p]
    order = [h for h in np.argsort(pct) if cand[p, h]][:5]
    lines.append(f"top-5 {nm}: " + "; ".join(f"{names[h]} ({z[nm][p, h]:+.1f})" for h in order))
lines.append("taken up after cutoff (games after; early 1-10; later 11-50; rank pct main / pref / imp2):")
for h in np.flatnonzero(cand[p] & (n_post[p] >= 10)):
    lines.append(f"  {names[h]}: {int(n_post[p, h])} g, early {e1[h]:+.1f}, later "
                 f"{(e2[h] if n_post[p, h] >= 20 else float('nan')):+.1f}, pct main {z['_pct_main'][p, h]:.2f} "
                 f"pref {z['_pct_pref'][p, h]:.2f} imp2 {z['_pct_imp2'][p, h]:.2f}, main score {z['main'][p, h]:+.1f}, before {int(N[p, h])} g")
print("\n".join(lines))
open(os.path.join(HC.HR_RES, "example_player.txt"), "w").write("\n".join(lines) + "\n")
