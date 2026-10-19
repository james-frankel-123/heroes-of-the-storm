"""
P3 tournament baseline: DraftRec (Lee et al., WWW 2022, "DraftRec:
Personalized Draft Recommendation for Winning in Multi-Player Online Battle
Arena Games"; code github.com/dojeon-ai/DraftRec) rebuilt for Storm League.

Architecture (as the paper and its released code, templates/draftrec.yaml):
  player network  a Transformer over each player's last L games before the
                  game day; token = hero + hero role + win/loss + Linear(in-game
                  stats) + sinusoidal position (left-padded, most recent game
                  last); the player representation is the output at the last
                  position (after a final LayerNorm)
  match network   a Transformer over the ten players in pick order; token =
                  hero picked so far (zero for not-yet-picked: the code's mask)
                  + team + sinusoidal turn position + player representation
                  (zeroed for the other team's players, as the code's
                  user_masks: histories of the opponents are not used)
  heads           pick: Linear-GELU then dot product with the (tied) hero
                  embeddings + bias, unavailable heroes masked;
                  outcome: Linear-GELU-Linear on (team-0 tokens - team-1
                  tokens) / 10, sigmoid = P(first-pick team wins)
  loss            per match, the 10 pick turns plus the two post-draft states
                  (one per team's perspective, value only):
                  lambda * CE(pick) + (1 - lambda) * BCE(outcome), lambda 0.1
  training        Adam (lr 1e-3, L2 1e-5), grad-norm clip 5, cosine annealing
                  over 10 epochs, dropout 0.1, d 128, 2 blocks per network,
                  2 heads (head dim 64), FF 4d, pre-LN, GELU, ~512 samples
                  per batch (43 matches x 12 states)
Storm League adaptations (deviations, stated in the report):
  - no pre-draft role token in the match network (HotS has no assigned
    positions; the paper drops roles for Dota2 for the same reason); the hero's
    fine role stays in history tokens;
  - map and tier embeddings added to every match token (one LoL map, top-0.1%
    LoL players only);
  - stats: the 15 scoreboard z-metrics of the rich-CF export (sb_a_shrink) plus a
    has-stats flag (HP history games without our replay have none);
  - unavailable heroes in the pick head = bans and picks made before the step
    (LoL bans all come first);
  - the team picks (tournament rule): see act_draftrec.

Data (TRAINING DATA RULE): games of the pinned snapshot with a full draft
(pick order: pickorder_2024q2 + x_bans; 2024-04-01 .. 2026-05-22), never the
sealed build; validation = the last three weeks (2026-05-02 .. 05-22), the
train split ends 2026-05-01; the deployed (final) model trains on all games
through 2026-05-22 with the validation-chosen epoch. History tokens: slot rows
plus the merged Heroes Profile SL history rows (merged_rows_full_1017), as of
the day before the game (rows on days < game day), table capped at
2026-06-29 (the last day a tournament-window history can read). OOT
tournament-window games (2026-05-23 .. 06-30, p3t/lobbies.npz) are evaluation
only.

Stages (from training/, env of remote/run_rcf_full.sh for P3_CACHE):
  data   (3090 CPU)  $P3_CACHE/p3t/draftrec/{tok_*.npy, games_train.npz, games_oot.npz, meta.json}
  hist   (CPU)       $P3_CACHE/p3t/draftrec/hist_tourney.npz (tournament lobbies, pick-index order)
  train  (GPU)       [--final] draftrec_val.pt / draftrec_final.pt and results json
  teameval (GPU)     team-mode pick accuracy on OOT window games, GD comparison

Inference: act_draftrec(ctx) (see its docstring). Needs on the host:
  $P3_CACHE/p3t/draftrec/draftrec_final.pt and hist_tourney.npz (and the GD
  vintage files for bans, as every system).
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
os.environ.setdefault("P1_TIERS", "site")
import numpy as np

import p3_hs_core as C

H = 90
DR_DIR = os.path.join(C.CACHE, "p3t", "draftrec")
SEALED = "2.55.17.98025"
TRAIN_START = C.day_of("2024-04-01")
VAL_START = C.day_of("2026-05-02")
POP_END = C.day_of("2026-05-22")
LOBBY_END = C.day_of("2026-06-30")
HIST_END = C.day_of("2026-06-29")
DRAFT_TEAM = [0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1]
IS_PICK = [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1]
PICK_STEPS = [4, 5, 6, 7, 8, 11, 12, 13, 14, 15]
BAN_STEPS = [0, 1, 2, 3, 9, 10]
PICK_TEAM = [DRAFT_TEAM[k] for k in PICK_STEPS]          # 0,1,1,0,0,1,1,0,0,1
BAN_TEAM = [DRAFT_TEAM[k] for k in BAN_STEPS]            # 0,1,0,1,1,0
NB_BEFORE = [4] * 5 + [6] * 5                            # bans made before pick turn j
NM = 15                                                  # scoreboard metrics

CFG = dict(L=50, D=128, blocks=2, heads=2, dropout=0.1, lr=1e-3, wd=1e-5, lmbda=0.1, clip=5.0,
           epochs=10, batch_matches=43, seed=0)


def _dr_file(name):
    return os.path.join(DR_DIR, name)


def fine_roles_shared():
    from shared import HEROES
    return C.hero_meta(HEROES)["fine"].astype(np.int64)


def site_tier_idx(league_tier, avg_mmr):
    """shared.SKILL_TIERS index of overfit2026.data.site_tier; -1 unknown.
    x_tiers stores NULL league_tier (Master) as a negative value."""
    lt = np.asarray(league_tier).astype(np.int64)
    mm = np.asarray(avg_mmr, np.float64)
    out = np.where(lt <= 3, 0, np.where(lt <= 5, 1, 2))
    out = np.where(lt < 0, np.where(np.isfinite(mm), 2, -1), out)
    return out


# ------------------------------------------------------------------ data

def _join_sorted(keys_sorted, q):
    j = np.minimum(np.searchsorted(keys_sorted, q), len(keys_sorted) - 1)
    return j, keys_sorted[j] == q


def build_tokens(d, sh):
    """History token table: slot rows plus merged HP rows with day <=
    HIST_END, sorted by (pid, day, replay_id). Returns pid start offsets and
    the sorted (pid, day) keys for as-of pointers."""
    import p3_c_merged as M
    from p3_keys import key_blizz
    rows_path = os.environ.get("P3_RCF_ROWS") or os.path.join(C.CACHE, "merged_rows_full_1017.npz")
    ex, _ = M.load_rows(rows_path, d)
    vs = [str(x) for x in d["versions"]]
    sv = vs.index(SEALED) if SEALED in vs else -999
    s_ok = (d["day"] <= HIST_END) & (d["version"] != sv)
    e_ok = (ex["end_day"] <= HIST_END) & (ex["hero"] >= 0)
    pid = np.r_[d["pid"][s_ok], ex["pid"][e_ok]].astype(np.int64)
    day = np.r_[d["day"][s_ok], ex["end_day"][e_ok]].astype(np.int32)
    hero = sh[np.r_[d["hero"][s_ok], ex["hero"][e_ok]].astype(np.int64)].astype(np.uint8)
    win = (np.r_[d["y"][s_ok], ex["y"][e_ok]] > 0.5).astype(np.uint8)
    rid = np.r_[d["replay_id"][s_ok], ex["replay_id"][e_ok]].astype(np.int64)
    n_slot = int(s_ok.sum())
    del ex
    o = np.lexsort((rid, day, pid))
    pid, day, hero, win, rid = pid[o], day[o], hero[o], win[o], rid[o]
    # dedupe (pid, replay_id): extras already exclude slot rows; guard anyway
    dup = np.zeros(len(pid), bool)
    dup[1:] = (pid[1:] == pid[:-1]) & (rid[1:] == rid[:-1])
    if dup.any():
        k = ~dup
        pid, day, hero, win, rid = pid[k], day[k], hero[k], win[k], rid[k]
    print(f"tokens: {len(pid):,} rows ({n_slot:,} slot rows before dedupe, {int(dup.sum()):,} duplicates dropped)",
          flush=True)
    # scoreboard metrics joined on (replay_id, blizz_id)
    z = np.load(os.path.join(C.CACHE, "rcf", "sb_a_shrink.npz"))
    comp = (z["replay_id"].astype(np.int64) << 40) | z["blizz_id"].astype(np.int64)
    so = np.argsort(comp)
    cs = comp[so]
    bl = key_blizz(d["player_keys"])[pid].astype(np.int64)
    j, hit = _join_sorted(cs, (rid << 40) | bl)
    Zs = z["Z"]
    Z = np.zeros((len(pid), NM), np.float16)
    zf = np.zeros(len(pid), np.uint8)
    zi = so[j[hit]]
    Zh = Zs[zi].astype(np.float32)
    zf[hit] = (~np.isnan(Zh).all(1)).astype(np.uint8)
    Z[hit] = np.clip(np.nan_to_num(Zh, nan=0.0), -5, 5).astype(np.float16)
    del Zh
    metrics = [str(m) for m in z["metrics"]]
    del z, Zs, comp, cs, so
    print(f"  stats: {zf.mean():.3f} of rows have scoreboard metrics; {metrics}", flush=True)
    npl = int(d["n_players"])
    start = np.searchsorted(pid, np.arange(npl + 1)).astype(np.int64)
    key = pid * 100000 + day
    return dict(hero=hero, win=win, Z=Z, zf=zf, start=start, key=key, metrics=metrics)


def pointers(tok, pid, day):
    """As-of history pointers: end = first token of (pid, day >= game day);
    n = number of earlier tokens of the player (all days < game day)."""
    pid = np.asarray(pid, np.int64)
    he = np.searchsorted(tok["key"], pid * 100000 + np.asarray(day, np.int64), side="left")
    hn = he - tok["start"][pid]
    assert (hn >= 0).all()
    return he.astype(np.int64), np.minimum(hn, 1000).astype(np.int16)


def snapshot_games(d, sh):
    """Snapshot games with a standard 16-step draft (as p3_pgd_data): pick
    order from pickorder_2024q2, bans from x_bans, map from x_side_games, site
    tier from x_tiers; picks joined to slot rows by (replay_id, hero)."""
    from shared import MAPS
    from p3_heroes import HKEY
    names = [str(h) for h in d["hero_names"]]
    hidx = {n: i for i, n in enumerate(names)}
    snap = ~d["post"]
    rkey = d["replay_id"].astype(np.int64) * HKEY + d["hero"].astype(np.int64)
    ro = np.flatnonzero(snap)[np.argsort(rkey[snap])]
    rks = rkey[ro]

    def rows_of(rid, hero):
        j, ok = _join_sorted(rks, rid.astype(np.int64) * HKEY + hero.astype(np.int64))
        return np.where(ok, ro[j], -1)

    po = np.load(os.path.join(C.CACHE, "pickorder_2024q2.npz"))
    uh, inv = np.unique(po["hero"], return_inverse=True)
    ph = np.array([hidx.get(str(h), -1) for h in uh])[inv]
    prid = po["replay_ids"].astype(np.int64)
    prow = np.where(ph >= 0, rows_of(prid, np.maximum(ph, 0)), -1)
    o = np.lexsort((po["pick_rank"], prid))
    prid, ph, prank, prow = prid[o], ph[o], po["pick_rank"][o], prow[o]
    ug, gs, gc = np.unique(prid, return_index=True, return_counts=True)
    good = gc == 10
    sel = (gs[good][:, None] + np.arange(10)[None]).ravel()
    P_h, P_row, P_rank = ph[sel].reshape(-1, 10), prow[sel].reshape(-1, 10), prank[sel].reshape(-1, 10)
    rid = ug[good]
    okp = (P_h >= 0).all(1) & (P_row >= 0).all(1) & (P_rank == np.arange(10)).all(1)
    bz = np.load(os.path.join(C.CACHE, "x_bans.npz"))
    bo = np.argsort(bz["replay_ids"])
    j, okb = _join_sorted(bz["replay_ids"][bo], rid)
    bi = bo[j]
    B_h, B_t, B_pn, first = bz["hero"][bi], bz["team"][bi], bz["pick_number"][bi], bz["first_pick_team"][bi]
    okb &= (bz["n_bans"][bi] == 6) & (B_h >= 0).all(1) & np.isin(first, [0, 1])
    bs = np.argsort(np.where(B_pn < 0, 99, B_pn), axis=1)
    B_h, B_t = np.take_along_axis(B_h, bs, 1), np.take_along_axis(B_t, bs, 1)
    sm = np.load(os.path.join(C.CACHE, "x_side_games.npz"), allow_pickle=True)
    so = np.argsort(sm["replay_ids"])
    j, oks = _join_sorted(sm["replay_ids"][so], rid)
    mp = np.array([MAPS.index(str(n)) if str(n) in MAPS else -1 for n in sm["map_names"]])[sm["map"][so][j]]
    xt = np.load(os.path.join(C.CACHE, "x_tiers.npz"))
    to = np.argsort(xt["replay_ids"])
    j2, okt_ = _join_sorted(xt["replay_ids"][to], rid)
    tier = site_tier_idx(xt["league_tier"][to][j2], xt["avg_mmr"][to][j2])
    # real teams of picks; kernel label: first-pick team = 0
    T_p = d["team"][np.maximum(P_row, 0)].astype(np.int64)
    ft = first.astype(np.int64)
    exp_p = np.where(np.array(PICK_TEAM)[None] == 0, ft[:, None], 1 - ft[:, None])
    exp_b = np.where(np.array(BAN_TEAM)[None] == 0, ft[:, None], 1 - ft[:, None])
    oko = (T_p == exp_p).all(1) & (B_t == exp_b).all(1)
    day = d["day"][np.maximum(P_row[:, 0], 0)]
    vs = [str(x) for x in d["versions"]]
    keep = okp & okb & oks & okt_ & oko & (mp >= 0) & (tier >= 0) & (day >= TRAIN_START) & (day <= POP_END)
    print(f"snapshot: {len(rid):,} games with 10 ranked picks; kept {keep.sum():,} (picks {okp.mean():.3f}, "
          f"bans {okb.mean():.3f}, map {oks.mean():.3f}, tier {okt_.mean():.3f}, order {oko[okp & okb].mean():.4f})",
          flush=True)
    assert SEALED not in vs or not (d["version"][P_row[keep].ravel()] == vs.index(SEALED)).any()
    rows = P_row[keep]
    y0 = (d["y"][rows[:, 0]] > 0.5).astype(np.int8)          # pick 0 is the first-pick team
    bans_sh = sh[B_h[keep].astype(np.int64)]
    return dict(rid=rid[keep], rows=rows, picks=sh[P_h[keep]].astype(np.uint8), bans=bans_sh.astype(np.uint8),
                map=mp[keep].astype(np.int8), tier=tier[keep].astype(np.int8), y0=y0, day=day[keep].astype(np.int32))


def oot_games(d):
    """Tournament-window OOT games (2026-05-23 .. 06-30) from p3t/lobbies.npz,
    in pick-index order (evaluation only)."""
    from shared import MAPS, SKILL_TIERS
    import p3t_data as TD
    L = TD.load_lobbies()
    gi = np.flatnonzero(L["day"] <= LOBBY_END)
    st = L["steps"][gi]
    first = L["first_team"][gi].astype(np.int64)
    pk = st[:, PICK_STEPS]
    kt = pk[:, :, 2] ^ first[:, None]
    assert (kt == np.array(PICK_TEAM)[None]).all()
    rows = pk[:, :, 3]
    return dict(gi=gi, rid=L["replay_id"][gi], rows=rows, picks=pk[:, :, 0].astype(np.uint8),
                bans=st[:, BAN_STEPS, 0].astype(np.uint8),
                map=np.array([MAPS.index(str(m)) for m in L["map"][gi]], np.int8),
                tier=np.array([SKILL_TIERS.index(str(t)) for t in L["tier"][gi]], np.int8),
                y0=(L["winner"][gi] == first).astype(np.int8), day=L["day"][gi].astype(np.int32))


def stage_data():
    import p3t_data as TD
    t0 = time.time()
    os.makedirs(DR_DIR, exist_ok=True)
    z = np.load(os.path.join(C.CACHE, "x_slots_ext.npz"))
    d = {k: z[k] for k in ("replay_id", "pid", "hero", "team", "y", "day", "post", "version", "player_keys",
                           "hero_names", "versions", "n_players")}
    del z
    sh = TD.to_shared(d["hero_names"])
    print(f"slot table loaded ({time.time() - t0:.0f}s)", flush=True)
    G = snapshot_games(d, sh)
    O = oot_games(d)
    tok = build_tokens(d, sh)
    print(f"tokens built ({time.time() - t0:.0f}s)", flush=True)
    for name, g in (("train", G), ("oot", O)):
        pid = d["pid"][g["rows"]]
        he, hn = pointers(tok, pid.ravel(), np.repeat(g["day"], 10))
        g["he"], g["hn"] = he.reshape(-1, 10), hn.reshape(-1, 10)
        g["pid"] = pid.astype(np.int64)
        # as-of check: the last history token of every player is from an earlier day
        last = np.maximum(g["he"] - 1, 0)
        has = g["hn"] > 0
        tday = tok["key"][last] % 100000
        assert (tday[has] < np.repeat(g["day"], 10).reshape(-1, 10)[has]).all()
        del g["rows"]
        np.savez(_dr_file(f"games_{name}.npz"), **g)
        print(f"games_{name}: {len(g['day']):,} games; median earlier games per player {np.median(g['hn']):.0f}; "
              f"share with >= 50: {(g['hn'] >= 50).mean():.3f}", flush=True)
    for k in ("hero", "win", "Z", "zf", "start", "key"):
        np.save(_dr_file(f"tok_{k}.npy"), tok[k])
    import datetime
    meta = dict(n_tokens=int(len(tok["hero"])), metrics=tok["metrics"], hist_end="2026-06-29",
                train=[str(C.EPOCH + datetime.timedelta(int(G["day"].min()))), "2026-05-22"],
                val_start="2026-05-02", n_train_games=int(len(G["day"])),
                n_val_games=int((G["day"] >= VAL_START).sum()), n_oot_games=int(len(O["day"])))
    json.dump(meta, open(_dr_file("meta.json"), "w"), indent=1)
    print(f"data done: {meta} ({time.time() - t0:.0f}s)", flush=True)
    stage_hist()


def load_tokens(mmap=True):
    m = "r" if mmap else None
    return {k: np.load(_dr_file(f"tok_{k}.npy"), mmap_mode=m) for k in ("hero", "win", "Z", "zf", "start", "key")}


def stage_hist(lobby_file=None, out=None):
    """Raw history tensors (the 100 most recent earlier games) for the
    tournament lobbies, rows in pick-index order j (p3t/lobbies.npz
    slot_rows order); kernel slot sl of lobby i is pick index src[i, sl].
    Uses the as-of pointers of games_oot.npz (every tournament-window game)."""
    import p3t_data as TD
    lobby_file = lobby_file or os.path.join(TD.OUT, "tourney", "lobbies.npz")
    gi = np.load(lobby_file)["gi"]
    O = load_games("oot")
    pos = {int(g): i for i, g in enumerate(O["gi"])}
    r = np.array([pos[int(g)] for g in gi])
    he, hn = O["he"][r].ravel(), O["hn"][r].ravel().astype(np.int64)
    tok = load_tokens()
    Lm = 100
    idx = he[:, None] - Lm + np.arange(Lm)[None]
    valid = np.arange(Lm)[None] >= Lm - np.minimum(hn, Lm)[:, None]
    idx = np.where(valid, idx, 0)
    hero = np.where(valid, np.asarray(tok["hero"][idx.ravel()]).reshape(idx.shape).astype(np.int16) + 1, 0)
    win = np.where(valid, np.asarray(tok["win"][idx.ravel()]).reshape(idx.shape).astype(np.int16) + 1, 0)
    Z = np.where(valid[..., None], np.asarray(tok["Z"][idx.ravel()]).reshape(idx.shape + (NM,)), 0).astype(np.float16)
    zf = np.where(valid, np.asarray(tok["zf"][idx.ravel()]).reshape(idx.shape), 0).astype(np.uint8)
    n = len(gi)
    out = out or _dr_file("hist_tourney.npz")
    np.savez(out, gi=gi, hero=hero.reshape(n, 10, Lm).astype(np.uint8), win=win.reshape(n, 10, Lm).astype(np.uint8),
             Z=Z.reshape(n, 10, Lm, NM), zf=zf.reshape(n, 10, Lm), hn=hn.reshape(n, 10), day=O["day"][r])
    print(f"hist: {n} lobbies -> {out}; earlier games per player median {np.median(hn):.0f}, min {hn.min()}",
          flush=True)


# ------------------------------------------------------------------ model

def _torch():
    import torch
    return torch


def build_model(cfg, n_roles):
    torch = _torch()
    nn = torch.nn
    F = torch.nn.functional
    import math

    class LayerNorm(nn.Module):
        def __init__(self, d, eps=1e-6):
            super().__init__()
            self.a = nn.Parameter(torch.ones(d))
            self.b = nn.Parameter(torch.zeros(d))
            self.eps = eps

        def forward(self, x):
            m = x.mean(-1, keepdim=True)
            s = x.std(-1, keepdim=True)
            return self.a * (x - m) / (s + self.eps) + self.b

    def gelu(x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

    class Block(nn.Module):
        """Pre-LN Transformer block (DraftRec code: SublayerConnection)."""

        def __init__(self, d, h, p):
            super().__init__()
            self.h, self.dk = h, d // h
            self.qkv = nn.ModuleList([nn.Linear(d, d) for _ in range(3)])
            self.o = nn.Linear(d, d)
            self.w1, self.w2 = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)
            self.n1, self.n2 = LayerNorm(d), LayerNorm(d)
            self.p = p
            self.drop = nn.Dropout(p)

        def forward(self, x):
            B, T, D = x.shape
            y = self.n1(x)
            q, k, v = [l(y).view(B, T, self.h, self.dk).transpose(1, 2) for l in self.qkv]
            a = F.scaled_dot_product_attention(q, k, v, dropout_p=self.p if self.training else 0.0)
            x = x + self.drop(self.o(a.transpose(1, 2).reshape(B, T, D)))
            y = self.n2(x)
            return x + self.drop(self.w2(self.drop(gelu(self.w1(y)))))

    def sinus(n, d):
        pe = torch.zeros(n, d)
        pos = torch.arange(0, n).unsqueeze(1).float()
        div = torch.pow(torch.full((d // 2,), 10000.0), torch.arange(0, d, 2).float() / d)
        pe[:, 0::2] = torch.sin(pos / div)
        pe[:, 1::2] = torch.cos(pos / div)
        return pe

    class DraftRec(nn.Module):
        def __init__(self):
            super().__init__()
            D, p = cfg["D"], cfg["dropout"]
            self.cfg = cfg
            self.hero = nn.Embedding(H + 1, D, padding_idx=0)       # 0 pad / masked, 1..90 heroes
            self.role = nn.Embedding(n_roles + 1, D, padding_idx=0)
            self.outcome = nn.Embedding(3, D, padding_idx=0)         # pad, loss, win
            self.stat = nn.Linear(NM + 1, D)
            self.team = nn.Embedding(2, D)
            self.map = nn.Embedding(14, D)
            self.tier = nn.Embedding(3, D)
            self.register_buffer("pe_user", sinus(cfg["L"], D))
            self.register_buffer("pe_match", sinus(10, D))
            self.drop = nn.Dropout(p)
            self.ublocks = nn.ModuleList([Block(D, cfg["heads"], p) for _ in range(cfg["blocks"])])
            self.mblocks = nn.ModuleList([Block(D, cfg["heads"], p) for _ in range(cfg["blocks"])])
            self.unorm, self.mnorm = LayerNorm(D), LayerNorm(D)
            self.pol = nn.Linear(D, D)
            self.pol_bias = nn.Parameter(torch.zeros(H))
            self.v1, self.v2 = nn.Linear(D, D), nn.Linear(D, 1)
            s = 1 / math.sqrt(D)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    m.weight.data.normal_(0, s)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Embedding):
                    m.weight.data.normal_(0, s)
                    if m.padding_idx is not None:
                        m.weight.data[m.padding_idx].zero_()

        def players(self, hh, rr, ww, Z, zf):
            """hh, rr, ww (N, L) long (0 = pad), Z (N, L, NM), zf (N, L) -> (N, D)."""
            x = self.hero(hh) + self.role(rr) + self.outcome(ww)
            x = x + self.stat(torch.cat([Z, zf.unsqueeze(-1)], -1).to(x.dtype))
            x = self.drop(x + self.pe_user[: x.shape[1]])
            for b in self.ublocks:
                x = b(x)
            return self.unorm(x[:, -1])

        def match(self, U, champ, umask, mp, tr):
            """U (B, 10, D) player representations in turn order; champ (B, 10)
            0 = masked / not yet picked else hero + 1; umask (B, 10) 1 = hide
            the history; mp, tr (B,). Returns final tokens (B, 10, D)."""
            team = torch.tensor(PICK_TEAM, device=U.device)
            x = self.hero(champ) + self.team(team)[None] + U * (1 - umask.unsqueeze(-1).to(U.dtype))
            x = x + (self.map(mp) + self.tier(tr)).unsqueeze(1)
            x = self.drop(x + self.pe_match[None])
            for b in self.mblocks:
                x = b(x)
            return self.mnorm(x)

        def pick_logits(self, P, turn):
            q = P[torch.arange(len(P), device=P.device), turn]
            return gelu(self.pol(q)) @ self.hero.weight[1:].T + self.pol_bias

        def value_logit(self, P):
            t0 = (torch.tensor(PICK_TEAM, device=P.device) == 0).to(P.dtype)[None, :, None]
            diff = (P * t0).sum(1) / 10 - (P * (1 - t0)).sum(1) / 10
            return self.v2(gelu(self.v1(diff))).squeeze(-1)

    return DraftRec()


# ------------------------------------------------------------------ batches (GPU)

class TokStore:
    """The history token table on the device; gathers (N, L) histories."""

    def __init__(self, device, L):
        torch = _torch()
        t = load_tokens(mmap=True)
        self.dev, self.L = device, L
        for k, a in (("hero", "hero"), ("win", "win"), ("Z", "Z"), ("zf", "zf")):
            setattr(self, k, torch.from_numpy(np.array(t[a])).to(device))
        self.role = torch.from_numpy(fine_roles_shared() + 1).to(device)

    def gather(self, he, hn):
        torch = _torch()
        L = self.L
        ar = torch.arange(L, device=self.dev)
        idx = he.unsqueeze(-1) - L + ar
        valid = ar >= (L - torch.clamp(hn.long(), max=L)).unsqueeze(-1)
        idx = torch.where(valid, idx, torch.zeros_like(idx))
        hh = torch.where(valid, self.hero[idx].long() + 1, 0)
        rr = torch.where(valid, self.role[torch.clamp(hh - 1, min=0)], 0)
        ww = torch.where(valid, self.win[idx].long() + 1, 0)
        Z = torch.where(valid.unsqueeze(-1), self.Z[idx], torch.zeros((), dtype=self.Z.dtype, device=self.dev))
        zf = torch.where(valid, self.zf[idx], 0)
        return hh, rr, ww, Z, zf


def to_device(G, device):
    torch = _torch()
    keys = ("he", "hn", "picks", "bans", "map", "tier", "y0")
    return {k: torch.from_numpy(G[k].astype(np.float32 if k == "y0" else np.int64)).to(device) for k in keys}


def match_states(Gt, b):
    """The 12 training states of each match in batch rows b (tensor): turns
    1..10 (pick query at turn s) and the two post-draft states (perspective
    team 0, team 1). Returns tensors for model.match plus targets."""
    torch = _torch()
    device = b.device
    picks = Gt["picks"][b]                                                       # (B, 10) shared idx
    bans = Gt["bans"][b]                                                         # (B, 6)
    B = len(b)
    j = torch.arange(10, device=device)
    turn = torch.arange(12, device=device)
    vis = (j[None, :] < torch.clamp(turn, max=10)[:, None])                      # (12, 10) visible champions
    champ = torch.where(vis[None], picks[:, None, :] + 1, 0)                     # (B, 12, 10)
    pt = torch.tensor(PICK_TEAM, device=device)
    persp = torch.cat([pt, torch.tensor([0, 1], device=device)])                 # (12,)
    umask = (pt[None, :] != persp[:, None]).long()                               # (12, 10) hide the other team
    unav = torch.zeros(B, 10, H, dtype=torch.bool, device=device)
    for s in range(10):
        unav[:, s].scatter_(1, bans[:, :NB_BEFORE[s]], True)
        if s:
            unav[:, s].scatter_(1, picks[:, :s], True)
    return dict(champ=champ, umask=umask, picks=picks, unav=unav)


def forward_batch(model, ts, Gt, b):
    """Matches b (tensor). Returns pick logits (B, 10, H) with unavailable
    heroes masked, targets (B, 10), value logits (B, 12)."""
    torch = _torch()
    device = b.device
    B = len(b)
    hh, rr, ww, Z, zf = ts.gather(Gt["he"][b].view(-1), Gt["hn"][b].view(-1))
    U = model.players(hh, rr, ww, Z, zf).view(B, 10, -1)
    S = match_states(Gt, b)
    U12 = U.unsqueeze(1).expand(B, 12, 10, U.shape[-1]).reshape(B * 12, 10, -1)
    P = model.match(U12, S["champ"].reshape(B * 12, 10), S["umask"][None].expand(B, 12, 10).reshape(B * 12, 10),
                    Gt["map"][b].repeat_interleave(12), Gt["tier"][b].repeat_interleave(12))
    P = P.view(B, 12, 10, -1)
    Pq = P[:, :10].reshape(B * 10, 10, -1)
    lg = model.pick_logits(Pq, torch.arange(10, device=device).repeat(B)).view(B, 10, H)
    lg = lg.float().masked_fill(S["unav"], -1e9)
    v = model.value_logit(P.reshape(B * 12, 10, -1)).view(B, 12).float()
    return lg, S["picks"], v


# ------------------------------------------------------------------ metrics

def auc(p, y):
    o = np.argsort(p, kind="mergesort")
    r = np.empty(len(p))
    r[o] = np.arange(1, len(p) + 1)
    # average ranks for ties
    ps = p[o]
    i = 0
    while i < len(ps):
        k = i
        while k + 1 < len(ps) and ps[k + 1] == ps[i]:
            k += 1
        if k > i:
            r[o[i:k + 1]] = (i + k + 2) / 2
        i = k + 1
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def logloss(p, y):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def evaluate(model, ts, G, Gt, idx, device, bs=512):
    """Player-mode metrics (the paper's setting: real pick order)."""
    torch = _torch()
    model.eval()
    ranks, ce, vp = [], [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for s in range(0, len(idx), bs):
            b = torch.as_tensor(idx[s:s + bs], device=device)
            lg, tgt, v = forward_batch(model, ts, Gt, b)
            tl = lg.gather(2, tgt.unsqueeze(-1))
            ranks.append(((lg > tl).sum(2) + 1).cpu().numpy())
            ce.append((torch.logsumexp(lg, 2) - tl.squeeze(-1)).cpu().numpy())
            vp.append(torch.sigmoid(v).cpu().numpy())
    model.train()
    r = np.concatenate(ranks)
    ce = np.concatenate(ce)
    vp = np.concatenate(vp)
    y = G["y0"][idx].astype(np.float64)
    post = vp[:, 10:].mean(1)
    out = {"games": int(len(idx)), "pick_top1": float((r == 1).mean()), "pick_top5": float((r <= 5).mean()),
           "pick_top10": float((r <= 10).mean()),
           "pick_ndcg10": float(np.where(r <= 10, 1 / np.log2(r + 1), 0).mean()), "pick_ce": float(ce.mean()),
           "pick_top1_by_turn": [round(float((r[:, j] == 1).mean()), 4) for j in range(10)],
           "wp_post_logloss": logloss(post, y), "wp_post_auc": auc(post, y),
           "wp_post_acc": float(((post > 0.5) == (y == 1)).mean()),
           "wp_pre_logloss": logloss(vp[:, 0], y), "wp_pre_auc": auc(vp[:, 0], y),
           "wp_all_states_logloss": logloss(vp.ravel(), np.repeat(y, 12)),
           "base_logloss": logloss(np.full(len(y), y.mean()), y)}
    return out


def pop_baseline(G, tr_idx, ev_idx):
    """Popularity (the paper's POP over all training picks) for scale."""
    cnt = np.bincount(G["picks"][tr_idx].ravel().astype(np.int64), minlength=H).astype(float)
    order = np.argsort(-cnt)
    rank_of = np.empty(H, np.int64)
    top1 = top5 = 0
    n = 0
    for g in ev_idx:
        av = np.ones(H, bool)
        av[G["bans"][g][:4]] = False
        for j in range(10):
            if j == 5:
                av[G["bans"][g][4:]] = False
            o = order[av[order]]
            rank_of[o] = np.arange(1, len(o) + 1)
            r = rank_of[G["picks"][g][j]]
            top1 += r == 1
            top5 += r <= 5
            n += 1
            av[G["picks"][g][j]] = False
    return {"pick_top1": top1 / n, "pick_top5": top5 / n}


# ------------------------------------------------------------------ train

def load_games(name):
    z = np.load(_dr_file(f"games_{name}.npz"))
    return {k: z[k] for k in z.files}


def stage_train(final=False, epochs=None, keep_epoch=None, max_games=None, device_id=0):
    torch = _torch()
    cfg = dict(CFG)
    if epochs:
        cfg["epochs"] = epochs
    t0 = time.time()
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["seed"])
    rng = np.random.RandomState(cfg["seed"])
    G = load_games("train")
    O = load_games("oot")
    val = np.flatnonzero(G["day"] >= VAL_START)
    tr = np.arange(len(G["day"])) if final else np.flatnonzero(G["day"] < VAL_START)
    if max_games:
        tr = tr[-max_games:]
    oot = np.arange(len(O["day"]))
    ts = TokStore(device, cfg["L"])
    Gt, Ot = to_device(G, device), to_device(O, device)
    print(f"tokens on {device} ({time.time() - t0:.0f}s); train {len(tr):,} games, val {len(val):,}, "
          f"OOT window {len(oot):,}", flush=True)
    model = build_model(cfg, int(fine_roles_shared().max()) + 1).to(device)
    nparam = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.999), weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    tag = "final" if final else "val"
    res = {"cfg": cfg, "n_params": nparam, "train_games": int(len(tr)), "val_games": int(len(val)),
           "oot_games": int(len(oot)), "final": final, "epochs": []}
    best = (np.inf, -1)
    bm = cfg["batch_matches"]
    lam = cfg["lmbda"]
    for ep in range(cfg["epochs"]):
        te = time.time()
        perm = tr[rng.permutation(len(tr))]
        tot = np.zeros(3)
        nb = 0
        for s in range(0, len(perm) - bm + 1, bm):
            b = torch.as_tensor(perm[s:s + bm], device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                lg, tgt, v = forward_batch(model, ts, Gt, b)
            y = Gt["y0"][b]
            lp = torch.nn.functional.cross_entropy(lg.reshape(-1, H), tgt.reshape(-1))
            lv = torch.nn.functional.binary_cross_entropy_with_logits(v, y[:, None].expand_as(v))
            loss = lam * lp + (1 - lam) * lv
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
            opt.step()
            tot += [loss.item(), lp.item(), lv.item()] if nb % 50 == 0 else 0
            nb += 1
            if nb % 5000 == 0:
                print(f"  ep {ep} step {nb}: {(time.time() - te) / nb * 1000:.1f} ms/step", flush=True)
        sched.step()
        ent = {"epoch": ep + 1, "train_loss_sampled": (tot / max(1, (nb + 49) // 50)).round(4).tolist(),
               "secs": round(time.time() - te)}
        ev = evaluate(model, ts, G, Gt, val, device)
        ent["val"] = ev
        ent["val_objective"] = lam * ev["pick_ce"] + (1 - lam) * float(np.mean([ev["wp_all_states_logloss"]]))
        ent["oot"] = evaluate(model, ts, O, Ot, oot, device)
        res["epochs"].append(ent)
        print(f"ep {ep + 1}: {ent['secs']}s train {ent['train_loss_sampled']} | val top1 {ev['pick_top1']:.4f} "
              f"top5 {ev['pick_top5']:.4f} wp-post ll {ev['wp_post_logloss']:.4f} auc {ev['wp_post_auc']:.4f} "
              f"obj {ent['val_objective']:.4f} | OOT top1 {ent['oot']['pick_top1']:.4f} "
              f"ll {ent['oot']['wp_post_logloss']:.4f}", flush=True)
        sd = {"cfg": cfg, "state": model.state_dict(), "epoch": ep + 1}
        torch.save(sd, _dr_file(f"draftrec_{tag}_ep{ep + 1}.pt"))
        if not final and ent["val_objective"] < best[0]:
            best = (ent["val_objective"], ep + 1)
        res["gpu_hours"] = (time.time() - t0) / 3600
        json.dump(res, open(_dr_file(f"train_{tag}.json"), "w"), indent=1)
    keep = keep_epoch or (best[1] if not final else cfg["epochs"])
    res["kept_epoch"] = keep
    import shutil
    shutil.copy(_dr_file(f"draftrec_{tag}_ep{keep}.pt"), _dr_file(f"draftrec_{tag}.pt"))
    res["pop_baseline"] = {"val": pop_baseline(G, tr, val[:20000]), "oot": pop_baseline(G, tr, oot[:20000])}
    res["gpu_hours"] = (time.time() - t0) / 3600
    json.dump(res, open(_dr_file(f"train_{tag}.json"), "w"), indent=1)
    print(f"TRAIN DONE {tag}: kept epoch {keep}; {res['gpu_hours']:.2f} h", flush=True)


# ------------------------------------------------------------------ inference (team mode)

_INF = {}


def load_inference(device=None, model_file=None, hist_file=None):
    torch = _torch()
    key = (str(device), model_file, hist_file)
    if key in _INF:
        return _INF[key]
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    mf = model_file or (_dr_file("draftrec_final.pt") if os.path.exists(_dr_file("draftrec_final.pt"))
                        else _dr_file("draftrec_val.pt"))
    sd = torch.load(mf, map_location="cpu", weights_only=False)
    model = build_model(sd["cfg"], int(fine_roles_shared().max()) + 1)
    model.load_state_dict(sd["state"])
    model.to(device).eval()
    if hist_file == "none":
        ent = {"model": model, "device": device, "file": mf}
        _INF[key] = ent
        return ent
    hf = hist_file or _dr_file("hist_tourney.npz")
    z = np.load(hf)
    L = sd["cfg"]["L"]
    role = fine_roles_shared() + 1
    hero = z["hero"][:, :, -L:].astype(np.int64)
    rr = np.where(hero > 0, role[np.maximum(hero - 1, 0)], 0)
    with torch.no_grad():
        n = len(z["gi"])
        U = model.players(torch.from_numpy(hero.reshape(n * 10, L)).to(device),
                          torch.from_numpy(rr.reshape(n * 10, L)).to(device),
                          torch.from_numpy(z["win"][:, :, -L:].astype(np.int64).reshape(n * 10, L)).to(device),
                          torch.from_numpy(z["Z"][:, :, -L:].astype(np.float32).reshape(n * 10, L, NM)).to(device),
                          torch.from_numpy(z["zf"][:, :, -L:].astype(np.float32).reshape(n * 10, L)).to(device))
    ent = {"model": model, "device": device, "U": U.view(n, 10, -1), "pos": {int(g): i for i, g in enumerate(z["gi"])},
           "file": mf}
    _INF[key] = ent
    return ent


def _src_from_steps(gi):
    import p3t_data as TD
    L = TD.load_lobbies()
    out = np.zeros((len(gi), 10), np.int64)
    for i, g in enumerate(gi):
        cnt = [0, 0]
        j = 0
        for k in range(16):
            h, ty, tm, r = L["steps"][g][k]
            if ty == 1:
                kt = tm ^ L["first_team"][g]
                out[i, 5 * kt + cnt[kt]] = j
                cnt[kt] += 1
                j += 1
    return out


class TeamState:
    """Team-mode replay of a draft through DraftRec: each earlier team pick
    is assigned to the team's not-yet-assigned player with the highest pick
    probability for that hero at that state (the model's own reading of who
    took it). Players keep kernel slots; turn positions: assigned players at
    their pick turn, the query player at the current turn, the other
    unassigned players at their team's remaining turns in slot order."""

    def __init__(self, model, U, mp, tr, device):
        torch = _torch()
        self.m, self.U, self.dev = model, U, device
        self.mp = torch.as_tensor(np.asarray(mp, np.int64), device=device)
        self.tr = torch.as_tensor(np.asarray(tr, np.int64), device=device)
        self.B = U.shape[0]
        self.assigned = np.full((self.B, 10), -1, np.int64)     # hero per kernel slot
        self.aturn = np.full((self.B, 10), -1, np.int64)        # pick turn per kernel slot
        self.taken = np.zeros((self.B, H), bool)

    def _layout(self, j, team, cand):
        """Turn layout with candidate slot cand (B,) as the query at turn j:
        perm (B, 10) kernel slot at each turn, champ (B, 10)."""
        B = self.B
        perm = np.full((B, 10), -1, np.int64)
        champ = np.zeros((B, 10), np.int64)
        for i in range(B):
            used = set()
            for sl in range(10):
                if self.aturn[i, sl] >= 0:
                    perm[i, self.aturn[i, sl]] = sl
                    champ[i, self.aturn[i, sl]] = self.assigned[i, sl] + 1
                    used.add(sl)
            perm[i, j] = cand[i]
            used.add(cand[i])
            for t in (0, 1):
                free_turns = [jj for jj in range(10) if PICK_TEAM[jj] == t and perm[i, jj] < 0]
                free_slots = [sl for sl in range(5 * t, 5 * t + 5) if sl not in used]
                assert len(free_turns) == len(free_slots), (i, j, t, free_turns, free_slots)
                for jj, sl in zip(free_turns, free_slots):
                    perm[i, jj] = sl
        return perm, champ

    def probs(self, j, team, persp=None):
        """Pick distributions (B, 5, H) of the five players of `team` as the
        query at turn j (unavailable heroes not masked), and a (B, 5) mask of
        players still unassigned."""
        torch = _torch()
        persp = team if persp is None else persp
        out = torch.zeros(self.B, 5, H, device=self.dev)
        free = self.aturn[:, 5 * team:5 * team + 5] < 0
        Ps = []
        for u in range(5):
            cand = np.full(self.B, 5 * team + u)
            ok = free[:, u]
            cand = np.where(ok, cand, -1)
            # rows where u is assigned: use any free slot as a placeholder (masked later)
            fb = np.argmax(free, 1) + 5 * team
            cand = np.where(ok, cand, fb)
            perm, champ = self._layout(j, team, cand)
            Ut = self.U[torch.arange(self.B, device=self.dev)[:, None], torch.as_tensor(perm, device=self.dev)]
            umask = torch.as_tensor(np.array(PICK_TEAM)[None].repeat(self.B, 0) != persp, device=self.dev).long()
            P = self.m.match(Ut, torch.as_tensor(champ, device=self.dev), umask, self.mp, self.tr)
            lg = self.m.pick_logits(P, torch.full((self.B,), j, device=self.dev)).float()
            out[:, u] = torch.softmax(lg.masked_fill(torch.as_tensor(self.taken, device=self.dev), -1e9), 1)
            Ps.append(P)
        return out, free

    def mixture(self, j, team, persp=None):
        p, free = self.probs(j, team, persp)
        fr = _torch().as_tensor(free, device=self.dev).float()
        mix = (p * fr.unsqueeze(-1)).sum(1) / fr.sum(1, keepdim=True).clamp(min=1)
        return mix, p, free

    def apply_pick(self, j, team, hero, p=None, free=None):
        if p is None:
            _, p, free = self.mixture(j, team)
        ph = p[np.arange(self.B), :, hero].float().cpu().numpy()          # (B, 5)
        ph = np.where(free, ph, -1.0)
        u = ph.argmax(1)
        sl = 5 * team + u
        self.assigned[np.arange(self.B), sl] = hero
        self.aturn[np.arange(self.B), sl] = j
        self.taken[np.arange(self.B), hero] = True

    def apply_ban(self, hero):
        self.taken[np.arange(self.B), hero] = True


def replay_state(inf, ctx):
    """TeamState after the actions acts[:, :k]."""
    torch = _torch()
    gi = np.asarray(ctx["gi"])
    rows = np.array([inf["pos"][int(g)] for g in gi])
    src = np.asarray(ctx["src"]) if "src" in ctx else _src_from_steps(gi)
    U = inf["U"][torch.as_tensor(rows, device=inf["device"])[:, None], torch.as_tensor(src, device=inf["device"])]
    S = TeamState(inf["model"], U, ctx["map"], ctx["tier"], inf["device"])
    acts = np.asarray(ctx["acts"])
    j = 0
    for kk in range(ctx["k"]):
        h = acts[:, kk]
        if IS_PICK[kk]:
            S.apply_pick(j, DRAFT_TEAM[kk], h)
            j += 1
        else:
            S.apply_ban(h)
    return S, j


def act_draftrec(ctx, mode=None, ban=None, device=None):
    """DraftRec actor for the P3 tournament.

    ctx: p3t_tourney.ctx_for dict plus ctx["gi"] (B,) lobby ids (rows of
    p3t/lobbies.npz) and ctx["src"] (B, 10) pick index of each kernel slot
    (computed from the lobby steps if absent). Returns (B,) hero indices
    (shared.HEROES order).

    Picks (team mode): replay the draft so far (TeamState), then score each
    hero by the mixture of the pick-head probabilities of the team's
    not-yet-assigned players (mean of probabilities = log-mean-exp of
    log-probabilities), argmax over p3t_systems.legal_mask (free heroes in
    the team's pool union).
      mode "p"  (default; the request): argmax of the mixture.
      mode "pv" (the paper's DraftRec_{p+v}): among legal heroes with mixture
                probability > tau = 0.02, the highest outcome-head win
                probability for the team with the hero filled in (by the
                player most likely to take it); argmax of p if none.
    Bans: the paper has no ban model. ban "gd" (default): p3t_systems.act_gd
    with the cur vintage (behavioral argmax, the same as GD); ban "deny":
    the free hero with the highest mixture probability for the opponent's
    next pick (DraftRec's pick head read for the other team).
    Env overrides: P3T_DRAFTREC_MODE, P3T_DRAFTREC_BAN."""
    torch = _torch()
    import p3t_systems as SY
    mode = mode or os.environ.get("P3T_DRAFTREC_MODE", "p")
    ban = ban or os.environ.get("P3T_DRAFTREC_BAN", "gd")
    k = ctx["k"]
    if not IS_PICK[k] and ban == "gd":
        return SY.act_gd(SY.vintage("cur"), ctx)
    inf = load_inference(device)
    with torch.no_grad():
        S, j = replay_state(inf, ctx)
        team = DRAFT_TEAM[k]
        acts = np.asarray(ctx["acts"])
        if not IS_PICK[k]:
            o = 1 - team
            jn = next(jj for jj in range(10) if PICK_STEPS[jj] > k and PICK_TEAM[jj] == o)
            mix, _, _ = S.mixture(jn, o)
            legal = SY.legal_mask(acts, k, ctx["team_pool"])
            return np.where(legal, mix.cpu().numpy(), -1.0).argmax(1)
        mix, p, free = S.mixture(j, team)
        legal = SY.legal_mask(acts, k, ctx["team_pool"])
        m = mix.cpu().numpy()
        out = np.where(legal, m, -1.0).argmax(1)
        if mode == "pv":
            out = _pv_choice(S, j, team, m, p, free, legal, out)
        return out


def _pv_choice(S, j, team, m, p, free, legal, fallback, tau=0.02):
    torch = _torch()
    B = S.B
    cand = legal & (m > tau)
    out = fallback.copy()
    pn = p.float().cpu().numpy()
    for i in range(B):
        hs = np.flatnonzero(cand[i])
        if len(hs) < 2:
            continue
        best, bv = out[i], -1.0
        for h in hs:
            ph = np.where(free[i], pn[i, :, h], -1.0)
            sl = 5 * team + int(ph.argmax())
            T = TeamState.__new__(TeamState)
            T.__dict__.update({k_: v_ for k_, v_ in S.__dict__.items()})
            T.U, T.mp, T.tr, T.B = S.U[i:i + 1], S.mp[i:i + 1], S.tr[i:i + 1], 1
            T.assigned, T.aturn, T.taken = S.assigned[i:i + 1].copy(), S.aturn[i:i + 1].copy(), S.taken[i:i + 1].copy()
            T.assigned[0, sl], T.aturn[0, sl] = h, j
            nxt = j + 1 if j < 9 else None
            if nxt is None:
                perm, champ = T._layout(j, team, np.array([sl]))
            else:
                tn = PICK_TEAM[nxt]
                fr = [s_ for s_ in range(5 * tn, 5 * tn + 5) if T.aturn[0, s_] < 0]
                perm, champ = T._layout(nxt, tn, np.array([fr[0]]))
            Ut = T.U[torch.zeros(1, 10, dtype=torch.long, device=S.dev), torch.as_tensor(perm, device=S.dev)]
            umask = torch.as_tensor((np.array(PICK_TEAM) != team)[None], device=S.dev).long()
            P = S.m.match(Ut, torch.as_tensor(champ, device=S.dev), umask, T.mp, T.tr)
            v = float(torch.sigmoid(S.m.value_logit(P))[0])
            v = v if team == 0 else 1 - v
            if v > bv:
                best, bv = h, v
        out[i] = best
    return out


# ------------------------------------------------------------------ team-mode evaluation

def stage_teameval(n_games=20000, seed=0, gd=True):
    """Team-mode next-pick accuracy on OOT tournament-window games: at every
    real pick step, rank the real hero in the team mixture (free heroes, no
    pool rule), replaying the real draft. GD (cur vintage, 5-model average)
    on the same steps for comparison."""
    torch = _torch()
    t0 = time.time()
    O = load_games("oot")
    rng = np.random.RandomState(seed)
    sel = np.sort(rng.choice(len(O["day"]), min(n_games, len(O["day"])), replace=False))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import p3t_data as TD
    inf = load_inference(device, hist_file="none")
    ts = TokStore(device, inf["model"].cfg["L"])
    with torch.no_grad():
        Uall = []
        for s0 in range(0, len(sel), 2000):
            r = sel[s0:s0 + 2000]
            hh, rr, ww, Z, zf = ts.gather(torch.as_tensor(O["he"][r].ravel(), device=device),
                                          torch.as_tensor(O["hn"][r].ravel().astype(np.int64), device=device))
            Uall.append(inf["model"].players(hh, rr, ww, Z.float(), zf.float()).view(len(r), 10, -1))
        Uall = torch.cat(Uall)
    del ts
    L = TD.load_lobbies()
    gi = O["gi"][sel]
    real = L["steps"][gi][:, :, 0].astype(np.int64)
    src = _src_from_steps(gi)
    res = {"games": int(len(sel))}
    ranks_dr, ranks_gd = [], []
    if gd:
        import p3t_systems as SY
        V = SY.vintage("cur")
    bs = 1000
    for s in range(0, len(sel), bs):
        b = np.arange(s, min(s + bs, len(sel)))
        ctx = {"gi": gi[b], "src": src[b], "map": O["map"][sel][b].astype(np.int64),
               "tier": O["tier"][sel][b].astype(np.int64)}
        with torch.no_grad():
            Uq = Uall[torch.as_tensor(b, device=device)[:, None], torch.as_tensor(src[b], device=device)]
            S = TeamState(inf["model"], Uq, ctx["map"], ctx["tier"], device)
            rk = np.zeros((len(b), 10), np.int64)
            rg = np.zeros((len(b), 10), np.int64)
            j = 0
            for k in range(16):
                h = real[b, k]
                if IS_PICK[k]:
                    t = DRAFT_TEAM[k]
                    mix, p, free = S.mixture(j, t)
                    m = mix.cpu().numpy()
                    free_h = ~S.taken
                    m = np.where(free_h, m, -1.0)
                    rk[:, j] = (m > m[np.arange(len(b)), h][:, None]).sum(1) + 1
                    if gd:
                        X = SY.features(real[b], k, ctx["map"], ctx["tier"], t)
                        g = SY.gd_probs(V, X[:, :289], free_h)
                        rg[:, j] = (g > g[np.arange(len(b)), h][:, None]).sum(1) + 1
                    S.apply_pick(j, t, h, p, free)
                    j += 1
                else:
                    S.apply_ban(h)
        ranks_dr.append(rk)
        ranks_gd.append(rg)
    rd = np.concatenate(ranks_dr)
    res["draftrec_team"] = {"top1": float((rd == 1).mean()), "top5": float((rd <= 5).mean()),
                            "top1_by_turn": [round(float((rd[:, j] == 1).mean()), 4) for j in range(10)]}
    if gd:
        rgg = np.concatenate(ranks_gd)
        res["gd_cur"] = {"top1": float((rgg == 1).mean()), "top5": float((rgg <= 5).mean()),
                         "top1_by_turn": [round(float((rgg[:, j] == 1).mean()), 4) for j in range(10)]}
    res["model_file"] = inf["file"]
    res["secs"] = round(time.time() - t0)
    json.dump(res, open(_dr_file("teameval.json"), "w"), indent=1)
    print(json.dumps(res), flush=True)


def stage_gdplayer(n_games=20000, seed=0):
    """GD (cur vintage) on the player-mode targets of the val and OOT games
    (same heroes as the paper's metric: the real pick at each pick step)."""
    import p3t_systems as SY
    V = SY.vintage("cur")
    out = {}
    for name in ("train", "oot"):
        G = load_games(name)
        idx = np.flatnonzero(G["day"] >= VAL_START) if name == "train" else np.arange(len(G["day"]))
        rng = np.random.RandomState(seed)
        idx = np.sort(rng.choice(idx, min(n_games, len(idx)), replace=False))
        acts = np.zeros((len(idx), 16), np.int64)
        acts[:, PICK_STEPS] = G["picks"][idx]
        acts[:, BAN_STEPS] = G["bans"][idx]
        r = np.zeros((len(idx), 10), np.int64)
        for j, k in enumerate(PICK_STEPS):
            X = SY.features(acts, k, G["map"][idx].astype(np.int64), G["tier"][idx].astype(np.int64), DRAFT_TEAM[k])
            free = SY.legal_mask(acts, k, np.ones((len(idx), 2, H), bool))
            g = SY.gd_probs(V, X[:, :289], free)
            h = acts[:, k]
            r[:, j] = (g > g[np.arange(len(idx)), h][:, None]).sum(1) + 1
        out["val" if name == "train" else "oot"] = {"games": int(len(idx)), "top1": float((r == 1).mean()),
                                                    "top5": float((r <= 5).mean())}
    json.dump(out, open(_dr_file("gd_player.json"), "w"), indent=1)
    print(json.dumps(out), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["data", "hist", "train", "teameval", "gdplayer"])
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--keep-epoch", default=None, help="epoch checkpoint to keep; 'auto' = the validation run's choice")
    ap.add_argument("--max-games", type=int)
    ap.add_argument("--lobbies", default=None, help="hist: npz with gi (default p3t/tourney/lobbies.npz)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--games", type=int, default=20000)
    ap.add_argument("--no-gd", action="store_true")
    a = ap.parse_args()
    if a.stage == "data":
        stage_data()
    elif a.stage == "hist":
        stage_hist(a.lobbies, a.out)
    elif a.stage == "train":
        ke = a.keep_epoch
        if ke == "auto":
            ke = json.load(open(_dr_file("train_val.json")))["kept_epoch"]
        stage_train(a.final, a.epochs, int(ke) if ke else None, a.max_games)
    elif a.stage == "teameval":
        stage_teameval(a.games, gd=not a.no_gd)
    else:
        stage_gdplayer(a.games)


if __name__ == "__main__":
    main()
