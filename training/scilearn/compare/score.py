"""Fair comparison: every model scored on the same games, paired SEs.

  python -m training.scilearn.compare.score      -> paper/hypothesis_system/results/fair_comparison.{json,md tables}

Sets (no C, F, R or sealed rows; these are the harness extracts' Dpre, G and Dpost):
  dev    Dpost, 173,394 games
  rare   O5 on Dpost: either team's role multiset holds at most 0.2% of S teams
  onset  O2 on Dpost: the first 10K (and 1K) games of each 2.55.17 build, by game time
Paired differences use the week x player-cluster block bootstrap of evaluator.block_se.
Writes results/fair_comparison.json and prints the tables that fair_comparison.md quotes.
"""
import json

import numpy as np
from scipy.special import logit

from training.scilearn import config as C
from training.scilearn import data as D
from training.scilearn import evaluator as E
from training.scilearn import tranches as T
from training.scilearn.baselines.report import rare_mask
from training.scilearn.fit import nll

JOBS = C.PRIVATE / "jobs"
FAIR = C.PRIVATE / "fair"
NEURAL = C.PRIVATE / "fits" / "neural_n"
OUT = C.DOCS / "results"

# label -> (kind, source); kind: job (dir, model) | neural (file pattern) | p3 (tag)
MODELS = {
    # draft-only
    "A0 draft-only": ("fair", "A0d_SDG"),
    "K0 draft-only": ("fair", "K0d_SDG"),
    "w2-K-b draft-only": ("fair", "wKbd_SDG"),
    "w2-K-blinded draft-only": ("fair", "wKbld_SDG"),
    "N frozen (to 2.55.14)": ("neural", {"Dpost": FAIR / "nscore" / "n_frozen_strict_Dpost.npz",
                                         "Dpre": NEURAL / "n_drift_Dpre.npz", "G": NEURAL / "n_drift_G.npz"}),
    "N matched (S+Dpre+G)": ("neural", {"Dpost": FAIR / "nscore" / "n_matched_strict_Dpost.npz"}),
    "N frozen, build-clock stats": ("neural", {"Dpost": NEURAL / "n_drift_Dpost.npz"}),
    "N matched, build-clock stats": ("neural", {"Dpost": FAIR / "nscore" / "n_drift_Dpost.npz"}),
    "A0 draft-only, fit S": ("fair", "A0d_S"),
    "K0 draft-only, fit S": ("fair", "K0d_S"),
    # with player data
    "A0": ("job", ("a0_fitSDG", "A0")),
    "K0": ("job", ("inc_b40b452c1e061c38", "INC")),
    "w2-K-b": ("job", ("inc_be8151b00daa906a", "INC")),
    "w2-K-blinded": ("job", ("inc_1a768a53b59ba3dc", "INC")),
    "P3 on N frozen": ("p3", "frozen"),
    "P3 on N matched": ("p3", "sdg"),
    "A0, fit S": ("job", ("a0_fitS", "A0")),
    "K0, fit S": ("job", ("k0_fitS3090", "K0")),
}
DRAFT_ONLY = ["A0 draft-only", "K0 draft-only", "w2-K-b draft-only", "w2-K-blinded draft-only",
              "N frozen (to 2.55.14)", "N matched (S+Dpre+G)"]
WITH_PLAYER = ["A0", "K0", "w2-K-b", "w2-K-blinded", "P3 on N frozen", "P3 on N matched"]
S_ONLY = ["A0 draft-only, fit S", "K0 draft-only, fit S", "A0, fit S", "K0, fit S",
          "N frozen (to 2.55.14)", "P3 on N frozen"]
REF_DRAFT, REF_PLAYER = "N matched (S+Dpre+G)", "P3 on N matched"
PAIRS = {  # model -> references it is paired against
    **{m: [REF_DRAFT, "N frozen (to 2.55.14)"] for m in DRAFT_ONLY[:4]},
    **{m: [REF_PLAYER, "P3 on N frozen"] for m in WITH_PLAYER[:4]},
    "N matched (S+Dpre+G)": ["N frozen (to 2.55.14)"],
    "N frozen, build-clock stats": ["N frozen (to 2.55.14)"],
    "N matched, build-clock stats": ["N matched (S+Dpre+G)"],
    "P3 on N matched": ["P3 on N frozen", REF_DRAFT],
    "P3 on N frozen": ["N frozen (to 2.55.14)"],
    "A0 draft-only, fit S": ["N frozen (to 2.55.14)"],
    "K0 draft-only, fit S": ["N frozen (to 2.55.14)"],
    "A0, fit S": ["P3 on N frozen", "A0 draft-only, fit S"],
    "K0, fit S": ["P3 on N frozen", "K0 draft-only, fit S"],
    "A0": [REF_PLAYER, "P3 on N frozen", "A0 draft-only"],
    "K0": [REF_PLAYER, "P3 on N frozen", "K0 draft-only"],
    "w2-K-b": [REF_PLAYER, "P3 on N frozen", "w2-K-b draft-only"],
    "w2-K-blinded": [REF_PLAYER, "P3 on N frozen", "w2-K-blinded draft-only"],
}


def load_eta(label, tr):
    kind, src = MODELS[label]
    if kind in ("fair", "job"):
        if kind == "fair":
            f = C.PRIVATE / "fair_jobs.json"
            reg = json.loads(f.read_text()) if f.exists() else {}
            if src not in reg:
                return None
            job, model = reg[src]["job"], reg[src]["model"]
        else:
            job, model = src
        f = JOBS / job / "out" / f"eta_{model}_{tr}.npz"
        if not f.exists():
            return None
        z = np.load(f)
        return z["replay_id"], z["eta"].astype(float)
    if kind == "neural":
        f = src.get(tr)
        if f is None or not f.exists():
            return None
        z = np.load(f)
        return z["replay_id"], logit(np.clip(z["p"].astype(float), 1e-6, 1 - 1e-6))
    if kind == "p3":
        if tr != "Dpost":     # the combiner is fit on Dpre+G
            return None
        f = FAIR / "p3" / f"eta_P3_{src}_{tr}.npz"
        if not f.exists():
            return None
        z = np.load(f)
        return z["replay_id"], z["eta"].astype(float)


def align(ids, got):
    rid, eta = got
    o = np.argsort(rid)
    j = np.clip(np.searchsorted(rid[o], ids), 0, len(rid) - 1)
    if not np.array_equal(rid[o][j], ids):
        return None
    return eta[o][j]


def score_set(etas, y, blocks, sel):
    out = {}
    for lab, eta in etas.items():
        mt = E.metrics(eta[sel], y[sel])
        mt["logloss_se"], _ = E.block_se(nll(eta[sel], y[sel]), blocks[sel])
        mt["vs"] = {}
        for ref in PAIRS.get(lab, []):
            if ref in etas:
                pg = E.paired_gain(etas[ref][sel], eta[sel], y[sel], blocks[sel])
                acc_d = 100 * (mt["acc"] - float(((etas[ref][sel] > 0) == (y[sel] > 0)).mean()))
                mt["vs"][ref] = {"ll_diff": -pg["gain"], "se": pg["se"], "acc_pp": acc_d}
        out[lab] = mt
    return out


def main():
    g, meta, y_all, pl = D.load(tranches=["S", "Dpre", "G", "Dpost"], labels=True, players=True)
    S = {k: v[g["tranche"] == T.CODES["S"]] for k, v in g.items()}
    res = {"sets": {}, "n": {}}
    for tr in ("Dpost", "Dpre", "G"):
        m = g["tranche"] == T.CODES[tr]
        ids = g["replay_id"][m]
        y = y_all[m].astype(float)
        blocks = E.clusters(g["game_ts"][m], pl["player_key"][m])
        etas = {}
        for lab in MODELS:
            got = load_eta(lab, tr)
            if got is not None:
                e = align(ids, got)
                if e is None:
                    print(f"{lab} {tr}: rows do not match, skipped")
                    continue
                etas[lab] = e
        sets = {tr: np.ones(m.sum(), bool)}
        if tr == "Dpost":
            gm = {k: v[m] for k, v in g.items()}
            sets["rare"] = rare_mask(gm, meta, S)
            b, ts = gm["build"], gm["game_ts"]
            on10, on1 = np.zeros(len(b), bool), np.zeros(len(b), bool)
            for bb in np.unique(b):
                bi = np.flatnonzero(b == bb)
                bi = bi[np.argsort(ts[bi], kind="stable")]
                on10[bi[:10000]] = True
                on1[bi[:1000]] = True
                sets[f"onset {meta['builds'][bb]} first 10K"] = np.isin(np.arange(len(b)), bi[:10000])
                sets[f"onset {meta['builds'][bb]} first 1K"] = np.isin(np.arange(len(b)), bi[:1000])
            sets["onset first 10K"] = on10
            sets["onset first 1K"] = on1
        for name, sel in sets.items():
            res["sets"][name] = score_set(etas, y, blocks, sel)
            res["n"][name] = int(sel.sum())
        print(tr, {k: round(v["logloss"], 5) for k, v in res["sets"][tr].items()}, flush=True)
    reg = C.PRIVATE / "fair_jobs.json"
    res["fair_jobs"] = json.loads(reg.read_text()) if reg.exists() else {}
    for f in sorted((FAIR / "p3").glob("combiner_*.json")):
        res.setdefault("p3_combiners", {})[f.stem] = json.loads(f.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fair_comparison.json").write_text(json.dumps(res, indent=1, default=float))
    print_tables(res)


def fmt_vs(mt, ref):
    v = mt["vs"].get(ref)
    return f"{v['ll_diff']:+.5f} ({v['se']:.5f})" if v else "ref" if ref is None else "-"


def print_tables(res):
    S_ = res["sets"]
    for title, rows, ref in (("draft-only", DRAFT_ONLY, REF_DRAFT), ("with player data", WITH_PLAYER, REF_PLAYER)):
        print(f"\n### {title}")
        print("| Model | dev LL | dev acc | dev vs ref | rare LL | rare vs ref | onset10K LL | onset10K vs ref | onset1K LL | onset1K vs ref | slope |")
        for lab in rows:
            if lab not in S_["Dpost"]:
                continue
            c = [S_[k][lab] for k in ("Dpost", "rare", "onset first 10K", "onset first 1K")]
            print(f"| {lab} | {c[0]['logloss']:.5f} | {100 * c[0]['acc']:.2f}% | {fmt_vs(c[0], ref)} | "
                  f"{c[1]['logloss']:.5f} | {fmt_vs(c[1], ref)} | {c[2]['logloss']:.5f} | {fmt_vs(c[2], ref)} | "
                  f"{c[3]['logloss']:.5f} | {fmt_vs(c[3], ref)} | {c[0]['cal_slope']:.3f} |")


if __name__ == "__main__":
    main()
