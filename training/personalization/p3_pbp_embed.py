"""
P3 play-by-play study (results/P3_PLAY_BY_PLAY.md): a small learned per-game
"style" embedding from the play-by-play sequence, to compare with the
hand-built features.

Stages
  seq    (this box) per player-game sequence of 10 s bins from gates open,
         first 20 minutes, 12 channels: actions, ability casts, alive share,
         position on the core-to-core axis, distance to the team centroid,
         enemies and allies within 12 units, died, takedown, enemy minion
         deaths nearby, camera away from the hero, in-game mask. Each
         channel is centred on the hero's mean sequence (rows before
         2026-10-16). -> cache/pbp/seq.npz (row order = cache/pbp/rows.npz)
  train  (5090 or CPU) 1D CNN encoder, 16-d output, supervised-contrastive
         loss with "same player" as the positive (no outcomes used), trained
         on rows before 2026-10-16 only. -> cache/pbp/emb.npz (16 dims per row)
The embedding then enters p3_pbp_eval.py as family `emb` (as-of player means,
same normalization and windows as the hand-built families).
"""
import gzip
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

CACHE = os.path.join(HERE, "cache", "pbp")
SEQ = os.path.join(CACHE, "seq.npz")
EMB = os.path.join(CACHE, "emb.npz")
NB = 120
NC = 12


def game_seqs(path):
    import p3_pbp_features as PF
    d = json.load(gzip.open(path))
    g0 = d.get("gates_open_s")
    pb = d.get("pbp")
    if g0 is None or pb is None or len(d.get("players") or []) != 10:
        return None
    players = d["players"]
    team = {p["pid"]: p["team"] for p in players}
    pids = [p["pid"] for p in players]
    deaths = sorted(d.get("deaths", []), key=lambda e: e["t"])
    revives = pb.get("revives", [])
    dead = {pid: [] for pid in pids}
    for de in deaths:
        v = de["victim"]
        if v in dead:
            nxt = [r[0] for r in revives if r[1] == v and r[0] > de["t"]]
            dead[v].append((de["t"], min(nxt) if nxt else de["t"] + 60.0))
    tr = {p["pid"]: PF.Track(p["positions"], dead[p["pid"]]) for p in players}
    core = {}
    for e in d.get("notable_units", []):
        if e["ev"] == "born" and e["type"] == "KingsCore" and e.get("team") in (0, 1):
            core.setdefault(e["team"], np.array([e["x"], e["y"]], float))
    mid = g0 + 10.0 * np.arange(NB) + 5.0
    inb = mid < d["length_s"]
    P = {pid: tr[pid].at(mid) for pid in pids}
    minions = np.asarray(pb.get("minions") or np.zeros((0, 5)), float).reshape(-1, 5)
    out = {}
    for p in players:
        pid, tm = p["pid"], p["team"]
        X = np.zeros((NB, NC), np.float32)
        pp = pb["players"].get(str(pid), {})
        sec = np.frombuffer(pp.get("sec", "").encode("latin-1"), np.uint8).astype(np.float32)
        nb = min(len(sec) // 10, NB)
        X[:nb, 0] = np.log1p(sec[:nb * 10].reshape(nb, 10).sum(1))
        ab = np.asarray(pp.get("abil_bins", []), np.float32)[:NB]
        X[:len(ab), 1] = np.log1p(ab)
        X[:, 2] = tr[pid].alive(mid)
        me = P[pid]
        ok = np.isfinite(me[:, 0])
        if len(core) == 2:
            c0, c1 = core[tm], core[1 - tm]
            ax = c1 - c0
            X[ok, 3] = (me[ok] - c0) @ ax / max(ax @ ax, 1e-9)
        al = np.stack([P[q] for q in pids if team[q] == tm and q != pid])
        with np.errstate(invalid="ignore"):
            cxy = np.nanmean(al, 0)
        okc = ok & np.isfinite(cxy[:, 0])
        X[okc, 4] = np.hypot(*(me[okc] - cxy[okc]).T) / 40.0
        for q in pids:
            if q == pid:
                continue
            dq = np.nan_to_num(np.hypot(*(P[q] - me).T), nan=1e9) <= 12
            X[:, 5 if team[q] != tm else 6] += dq
        for de in deaths:
            b = int((de["t"] - g0) // 10)
            if 0 <= b < NB:
                if de["victim"] == pid:
                    X[b, 7] = 1
                elif team.get(de["victim"]) == 1 - tm and pid in de["killers"]:
                    X[b, 8] = 1
        if len(minions):
            em = minions[minions[:, 4] == 1 - tm]
            hp = tr[pid].at(em[:, 0])
            near = np.nan_to_num(np.hypot(hp[:, 0] - em[:, 1], hp[:, 1] - em[:, 2]), nan=1e9) <= 15
            b = ((em[near, 0] - g0) // 10).astype(int)
            b = b[(b >= 0) & (b < NB)]
            X[:, 9] = np.log1p(np.bincount(b, minlength=NB)[:NB])
        cam = np.asarray(pp.get("cam", []), float).reshape(-1, 2) / 2.0
        if len(cam):
            tb = g0 + 5.0 * np.arange(len(cam)) + 2.5
            hp = tr[pid].at(tb)
            away = np.nan_to_num(np.hypot(cam[:, 0] - hp[:, 0], cam[:, 1] - hp[:, 1]), nan=0) > 25
            b = (np.arange(len(cam)) // 2)
            m = b < NB
            X[:, 10] = np.bincount(b[m], weights=away[m], minlength=NB)[:NB] / 2.0
        X[:, 11] = inb
        X[~inb, :11] = 0
        out[p["toon_id"]] = X
    return int(d["replay_id"] or os.path.basename(path).split(".")[0]), out


def seq(procs=4):
    from multiprocessing import Pool
    import p3_pbp_features as PF
    t0 = time.time()
    z = dict(np.load(os.path.join(CACHE, "rows.npz")))
    paths = []
    for b in sorted(os.listdir(PF.PBP)):
        dd = os.path.join(PF.PBP, b)
        if os.path.isdir(dd):
            paths += [os.path.join(dd, fn) for fn in os.listdir(dd) if fn.endswith(".json.gz")]
    sealed = PF.sealed_ids()
    assert not (set(int(os.path.basename(p).split(".")[0]) for p in paths) & sealed)
    want = {int(r) for r in z["replay_id"]}
    paths = [p for p in paths if int(os.path.basename(p).split(".")[0]) in want]
    with Pool(procs) as pool:
        res = pool.map(game_seqs, paths, chunksize=20)
    lut = {}
    for r in res:
        if r:
            for toon, X in r[1].items():
                lut[(r[0], toon)] = X
    S = np.zeros((len(z["replay_id"]), NB, NC), np.float16)
    found = np.zeros(len(S), bool)
    for i, (rid, toon) in enumerate(zip(z["replay_id"], z["toon_id"])):
        X = lut.get((int(rid), int(toon)))
        if X is not None:
            S[i] = X
            found[i] = True
    # hero-centring on rows before the holdout start
    import p3_pbp_eval as E
    pre = (np.floor(z["unix"] / 86400) < E.HOLD0) & found
    Sf = S.astype(np.float32)
    for h in np.unique(z["hero"]):
        m = z["hero"] == h
        mu = Sf[m & pre].mean(0) if (m & pre).sum() >= 20 else Sf[pre].mean(0)
        Sf[m, :, :11] -= mu[None, :, :11] * Sf[m, :, 11:12]
    sd = Sf[pre].reshape(-1, NC).std(0) + 1e-6
    sd[11] = 1.0
    Sf /= sd
    np.savez(SEQ, S=Sf.astype(np.float16), found=found)
    print(f"seq: {found.sum():,} of {len(found):,} rows; {time.time() - t0:.0f}s", flush=True)


def train(device="cuda", epochs=12, P=256, K=4, dim=16, seed=0):
    import torch
    import torch.nn as nn
    import torch.nn.functional as Fn
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    z = np.load(os.path.join(CACHE, "rows_keys.npz")) if os.path.exists(os.path.join(CACHE, "rows_keys.npz")) else None
    S = np.load(SEQ)["S"].astype(np.float32)
    found = np.load(SEQ)["found"]
    if z is None:
        raise SystemExit("need cache/pbp/rows_keys.npz (pkey, day) from p3_pbp_eval prep")
    pk, day, hold0 = z["pkey"], z["day"], int(z["hold0"])
    tr = found & (day < hold0)
    up, inv = np.unique(pk[tr], return_inverse=True)
    idx_tr = np.flatnonzero(tr)
    by = [[] for _ in range(len(up))]
    for j, p in zip(idx_tr, inv):
        by[p].append(j)
    elig = [b for b in by if len(b) >= 2]
    print(f"train rows {tr.sum():,}, players with 2+ games {len(elig):,}", flush=True)

    class Enc(nn.Module):
        def __init__(self):
            super().__init__()
            self.c = nn.Sequential(nn.Conv1d(NC, 48, 5, padding=2), nn.GELU(),
                                   nn.Conv1d(48, 48, 5, padding=2, dilation=1), nn.GELU(),
                                   nn.Conv1d(48, 64, 5, padding=4, dilation=2), nn.GELU())
            self.o = nn.Linear(128, dim)
            self.h = nn.Sequential(nn.Linear(dim, 64), nn.GELU(), nn.Linear(64, 32))

        def forward(self, x):            # x (B, NB, NC)
            m = x[:, :, 11:12].transpose(1, 2)
            h = self.c(x.transpose(1, 2))
            mean = (h * m).sum(2) / m.sum(2).clamp(min=1)
            mx = (h - 1e4 * (1 - m)).max(2).values
            return self.o(torch.cat([mean, mx], 1))
    net = Enc().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    St = torch.tensor(S, device=device)
    steps = max(len(elig) // P, 1)
    for ep in range(epochs):
        net.train()
        perm = rng.permutation(len(elig))
        tot = 0.0
        for s in range(steps):
            ps = perm[s * P:(s + 1) * P]
            ids, lab = [], []
            for a, p in enumerate(ps):
                g = elig[p]
                ch = rng.choice(g, K, replace=len(g) < K)
                ids += list(ch)
                lab += [a] * K
            x = St[torch.tensor(ids, device=device)]
            lab = torch.tensor(lab, device=device)
            e = Fn.normalize(net.h(net(x)), dim=1)
            sim = e @ e.T / 0.1
            eye = torch.eye(len(lab), device=device, dtype=torch.bool)
            pos = (lab[:, None] == lab[None, :]) & ~eye
            sim = sim.masked_fill(eye, -1e9)
            logp = sim - torch.logsumexp(sim, 1, keepdim=True)
            loss = -(logp * pos).sum(1) / pos.sum(1).clamp(min=1)
            loss = loss.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
        # identification accuracy on held-out pairs (rows on or after the holdout start)
        print(f"epoch {ep} loss {tot / steps:.3f}", flush=True)
    net.eval()
    out = np.zeros((len(S), dim), np.float32)
    with torch.no_grad():
        for k in range(0, len(S), 8192):
            out[k:k + 8192] = net(St[k:k + 8192]).cpu().numpy()
    out[~found] = np.nan
    np.savez(EMB, E=out, epochs=epochs, dim=dim)
    print(f"wrote {EMB}", flush=True)


if __name__ == "__main__":
    st = sys.argv[1]
    if st == "seq":
        seq(int(sys.argv[2]) if len(sys.argv) > 2 else 4)
    elif st == "train":
        train(device=sys.argv[2] if len(sys.argv) > 2 else "cuda")
