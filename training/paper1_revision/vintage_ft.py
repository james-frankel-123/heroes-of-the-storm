"""
Upkeep option (c), step 1: fine-tune a stale vintage's leaf WP on the paper-1
train games played AFTER its cutoff (through the snapshot end, 2026-05-22).

  start     the base vintage's selected enriched WP (site/vint/<base>/models)
  rows      current site train_oof features (paper-1 out-of-fold statistics:
            a fold-k row uses oof{k} of the CURRENT corpus), only games on
            builds after the base cutoff
  stopping  the post-cutoff games of the current site val split (deploy
            statistics): the arm adapts to recent games, so recent games judge
            it. At least one epoch is taken (the arm never equals the stale WP).
  optimizer AdamW lr 1e-4, wd 5e-3 (train_wp.py's), batch 4096, constant lr,
            <= 30 epochs, patience 5
  eval      current site test and NODRIFT features (train_wp.evaluate), and
            the post-cutoff val and test games, before and after

Output (train_wp.py schema, so train_mcts.py reads it unchanged):
  site/vint/<name>/models/enriched.json, enriched_s<seed>.pt

Usage (from training/): P1_TIERS=site P1_VINTAGE=s4m_ft python paper1_revision/vintage_ft.py
"""
import os
import sys
import json
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from paper1_revision import core, vintage, train_wp  # noqa: E402
from overfit2026 import data as odata  # noqa: E402

import numpy as np  # noqa: E402

LR, MAX_EPOCHS, PATIENCE, BATCH = 1e-4, 30, 5, 4096


def eval_arrays(model, Xf, Xs, y, cols, device):
    """train_wp.evaluate on in-memory rows (both team orders, symmetrized)."""
    import torch
    out = []
    with torch.no_grad():
        for X in (Xf, Xs):
            ps = [model(torch.tensor(X[i:i + 65536][:, cols], device=device)).float().view(-1).cpu().numpy()
                  for i in range(0, len(X), 65536)]
            out.append(np.concatenate(ps))
    pf, ps_ = out
    acc_both = float(0.5 * (np.mean((pf > .5) == (y > .5)) + np.mean((ps_ > .5) == (y < .5))))
    p = np.clip(0.5 * (pf + 1 - ps_), 1e-6, 1 - 1e-6)
    return {"n": int(len(y)), "acc": acc_both, "acc_sym": float(np.mean((p > .5) == (y > .5))),
            "ll": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
            "slope": train_wp.fit_slope(p, y)}


def main():
    import torch
    import torch.nn as nn
    from sweep_enriched_wp import WinProbEnrichedModel
    assert core.VINTAGE and vintage.is_ft(), "run with P1_VINTAGE=<vintage>_ft"
    t_start = time.time()
    base = vintage.spec()["base"]
    out_json = os.path.join(core.MODEL_DIR, "enriched.json")
    if os.path.exists(out_json):
        print(f"skip ({out_json} exists)")
        return
    bmeta = json.load(open(odata.art(HERE, "vint", base, "models", "enriched.json")))
    seed = int(bmeta["selected_seed"])
    bpt = odata.art(HERE, "vint", base, "models", f"enriched_s{seed}.pt")
    cols = core.group_cols(bmeta["spec"]["groups"])
    dev = torch.device("cuda")

    Xf, Xs, y, rid, tier = core.load_features("train_oof")
    b, d = vintage.build_info(rid)
    cut = vintage.cutoff_idx(base)
    sel = np.where(b > cut)[0]
    assert len(sel) > 0 and int(d[sel].max()) <= vintage._day("2026-05-22")
    print(f"fine-tune rows: {len(sel):,} train games after {vintage.spec(base)['cutoff']} "
          f"(of {len(rid):,})", flush=True)
    tX = torch.tensor(np.concatenate([Xf[sel][:, cols], Xs[sel][:, cols]]), device=dev)
    tY = torch.tensor(np.concatenate([y[sel], 1 - y[sel]]), device=dev)
    del Xf, Xs
    vXf, vXs, vy, vrid, _ = core.load_features("val")
    vsel = np.where(vintage.build_info(vrid)[0] > cut)[0]
    vXf, vXs, vy = vXf[vsel], vXs[vsel], vy[vsel]
    vX = torch.tensor(np.concatenate([vXf[:, cols], vXs[:, cols]]), device=dev)
    vY = torch.tensor(np.concatenate([vy, 1 - vy]), device=dev)
    tXf, tXs, ty, trid, _ = core.load_features("test")
    tsel = np.where(vintage.build_info(trid)[0] > cut)[0]
    tXf, tXs, ty = tXf[tsel], tXs[tsel], ty[tsel]
    print(f"stopping on {len(vsel):,} post-cutoff val games; post-cutoff test games {len(tsel):,}", flush=True)

    def post_metrics():
        model.eval()
        return {"val_post": eval_arrays(model, vXf, vXs, vy, cols, dev),
                "test_post": eval_arrays(model, tXf, tXs, ty, cols, dev)}

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = WinProbEnrichedModel(len(cols), bmeta["spec"]["arch"], dropout=0.3).to(dev)
    model.load_state_dict(torch.load(bpt, map_location="cpu", weights_only=True))
    crit = nn.BCELoss()

    def val_loss():
        model.eval()
        with torch.no_grad():
            vp = model(vX).view(-1)
            return crit(vp, vY).item(), ((vp > 0.5).float() == vY).float().mean().item()

    res = {"name": "enriched", "spec": bmeta["spec"], "input_dim": int(len(cols)),
           "fine_tune": {"base": base, "base_wp": bpt, "lr": LR, "max_epochs": MAX_EPOCHS,
                         "patience": PATIENCE, "batch": BATCH, "after_build": vintage.spec(base)["cutoff"],
                         "n_games": int(len(sel)), "stats": "current site deploy/oof",
                         "stopping": "post-cutoff games of the current val split, >= 1 epoch",
                         "n_val_post_games": int(len(vsel)), "n_test_post_games": int(len(tsel))},
           "n_train_rows": int(len(tX)), "n_val_rows": int(len(vX))}
    r0 = {"val_loss": val_loss()[0]}
    for fs in ("test", "NODRIFT"):
        r0[fs] = train_wp.evaluate(model, cols, fs, dev)
    r0.update(post_metrics())
    res["before"] = r0
    print(f"before: val_post ll={r0['val_post']['ll']:.5f} acc={r0['val_post']['acc']:.4f} | test_post "
          f"ll={r0['test_post']['ll']:.5f} acc={r0['test_post']['acc']:.4f} | test acc={r0['test']['acc']:.4f} "
          f"slope={r0['test']['slope']:.3f}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=5e-3)
    best, best_state, pat, hist = float("inf"), None, 0, []
    n = len(tX)
    t0 = time.time()
    for ep in range(MAX_EPOCHS):
        model.train()
        pm = torch.randperm(n, device=dev)
        for i in range(0, n, BATCH):
            idx = pm[i:i + BATCH]
            loss = crit(model(tX[idx]).view(-1), tY[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        vl, va = val_loss()
        hist.append((ep + 1, round(vl, 6), round(va, 5)))
        print(f"  ep {ep + 1}: val_ll={vl:.5f} val_acc={va:.4f}", flush=True)
        if vl < best:
            best, pat = vl, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            pat += 1
            if pat >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    train_s = time.time() - t0
    torch.save(model.state_dict(), os.path.join(core.MODEL_DIR, f"enriched_s{seed}.pt"))
    r = {"val_loss": best, "epochs": len(hist), "history": hist,
         "best_epoch": int(min(hist, key=lambda h: h[1])[0])}
    for fs in ("test", "NODRIFT"):
        r[fs] = train_wp.evaluate(model, cols, fs, dev)
    r.update(post_metrics())
    res["seeds"] = {str(seed): r}
    res["selected_seed"] = seed
    res["wall_s"] = {"total": round(time.time() - t_start, 1), "train_loop": round(train_s, 1)}
    json.dump(res, open(out_json, "w"), indent=1)
    print(f"after: val_post ll={r['val_post']['ll']:.5f} acc={r['val_post']['acc']:.4f} | test_post "
          f"ll={r['test_post']['ll']:.5f} acc={r['test_post']['acc']:.4f}", flush=True)
    print(f"after: ep={len(hist)} best_ep={r['best_epoch']} val_ll={best:.5f} test acc={r['test']['acc']:.4f} "
          f"slope={r['test']['slope']:.3f} | NODRIFT acc={r['NODRIFT']['acc']:.4f} "
          f"slope={r['NODRIFT']['slope']:.3f} (train {train_s:.0f}s, total {time.time() - t_start:.0f}s)",
          flush=True)


if __name__ == "__main__":
    main()
