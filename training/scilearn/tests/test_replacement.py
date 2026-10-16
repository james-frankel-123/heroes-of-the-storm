"""Replacement cards: swap a term in the incumbent for a new one in one test (Max, 2026-10-15;
CYCLE3_PREP.md section 3).

  PYTHONPATH=. python -m training.scilearn.tests.test_replacement

Worker (run_q2), on a synthetic world with planted ally-pair effects (real hero and kit
tables, 60,000 games, 5 builds):
  1. without `replaces`, run_q2 returns exactly the cycle-2 Q2 numbers (e1 = e0 + term);
  2. replacing a sparse pair table (high support floor) by a dense one gains, and passes z 3;
  3. the reverse swap (dense replaced by sparse) loses, so the gate rejects it;
  4. replacing the pair table by a null term (keyed-hash coin) loses;
  5. replacing a term by the same spec is neutral (gain under 1e-4, not accepted) when the
     incumbent and the term are both fit to convergence, as in cycle 3;
  6. the result reports the new term's k and the replaced term's k.
Gate (apply_gate):
  7. an accepted replacement swaps the terms, records `replaced`, charges 1.5 bits once to the
     union of both terms' groups, and releases which card it replaced;
  8. the parameter cap is checked on the result: w2-K-blinded's p03 (2,553) for c10 (1,595) on
     17,668 fits (18,626); a swap that would exceed 20,000 is refused, free;
  9. a replacement whose target left the incumbent while it ran is refused, free;
 10. a losing replacement is rejected and the incumbent is unchanged.
Broker (card_submit):
 11. a card with `replaces:` registers with both terms' groups and the target recorded;
 12. refusals: unknown card id, a backbone term, an explanation card with `replaces`.
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from training.scilearn import broker as BR
from training.scilearn import config as C
from training.scilearn import fit as F
from training.scilearn import harness as HN
from training.scilearn import terms as TM
from training.scilearn import tranches as T
from training.scilearn import worker as W
from training.scilearn.ledger import Ledger

EXTRACT = C.PRIVATE / "extracts" / "snap_20261013_real_kited64c8e6"
BUILDS = [95, 96, 97, 98, 99]


def world(n=60000, seed=11):
    meta = json.loads(EXTRACT.with_suffix(".json").read_text())
    kit = np.load(EXTRACT.with_suffix(".npz"))["kit"]
    rng = np.random.default_rng(seed)
    H = len(meta["heroes"])
    # a skewed hero pick distribution, so pair supports range from a few to hundreds of games
    w = rng.gamma(0.7, 1.0, H)
    w /= w.sum()
    heroes = np.stack([rng.choice(H, 10, replace=False, p=w) for _ in range(n)])
    ts = np.sort(rng.integers(1.6e9, 1.7e9, n))
    g = {"heroes": heroes, "map": rng.integers(0, len(meta["maps"]), n), "tier": rng.integers(0, 4, n),
         "build": np.repeat(BUILDS, n // len(BUILDS))[:n], "game_ts": ts, "first_seen_ts": ts + 86400,
         "replay_id": np.arange(n) + 1}
    players = {"player_key": rng.integers(0, 3000, (n, 10)), "party": np.zeros((n, 10), np.int64)}
    ctx = TM.Ctx(g, meta, kit=kit, players=players)
    th = rng.normal(0, 0.15, H)
    pair = rng.normal(0, 0.30, (H, H))
    pair = np.triu(pair, 1) + np.triu(pair, 1).T
    A, B = heroes[:, :5], heroes[:, 5:]
    ia, ib = np.triu_indices(5, 1)
    eta = th[A].sum(1) - th[B].sum(1) + pair[A[:, ia], A[:, ib]].sum(1) - pair[B[:, ia], B[:, ib]].sum(1)
    y = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(np.float64)
    # every 5th game is dev (Dpost), the rest fit (S); time order kept within each
    trc = np.where(np.arange(n) % 5 == 4, T.CODES["Dpost"], T.CODES["S"])
    return g, ctx, y, players, trc


HERO = {"type": "lookup", "keys": ["hero"], "name": "h"}
DENSE = {"type": "pair_family", "min_games": 40, "name": "card_dense"}
SPARSE = {"type": "pair_family", "min_games": 400, "name": "card_sparse"}
NULL = {"type": "hash_null", "_key": "replacement-test", "name": "card_null"}


def worker_cases():
    out = []
    g, full, y, players, trc = world()
    fit_idx = np.flatnonzero(trc == T.CODES["S"])
    games = {**g, "tranche": trc}

    def inc(specs, converge=None):
        return F.fit(specs, full.subset(fit_idx), y[fit_idx], converge=converge)

    def q2(model, term, drop=None, conv=None):
        with tempfile.TemporaryDirectory() as td:
            spec = {"term": term, "dev_tranche": "Dpost", "recent_s_builds": BUILDS}
            if conv:
                spec["term_fit_options"] = conv
            if drop:
                spec["replaces"] = drop
            return W.run_q2(spec, Path(td), model, full, fit_idx, trc, games, y, players)

    inc_sparse = inc([HERO, SPARSE])
    inc_dense = inc([HERO, DENSE])
    # 1. plain Q2 unchanged: e1 = e0 + term, as the cycle-2 worker computed it
    r = q2(inc_sparse, DENSE)
    cand = F.fit([DENSE], full.subset(fit_idx), y[fit_idx], offset=inc_sparse.logit(full.subset(fit_idx)))
    ev = np.flatnonzero(trc == T.CODES["Dpost"])
    from training.scilearn import evaluator as E
    e0 = inc_sparse.logit(full.subset(ev))
    ref = E.paired_gain(e0, e0 + cand.logit(full.subset(ev)), y[ev],
                        E.clusters(g["game_ts"][ev], players["player_key"][ev]))
    out.append(("plain Q2 gives the cycle-2 numbers", abs(r["paired"]["gain"] - ref["gain"]) < 1e-15
                and abs(r["paired"]["z"] - ref["z"]) < 1e-9 and "replaced" not in r))
    # 2. sparse -> dense
    r2 = q2(inc_sparse, DENSE, drop="card_sparse")
    out.append((f"replacing the sparse pair table by the dense one gains: gain {r2['paired']['gain']:+.5f}, "
                f"z {r2['paired']['z']:.2f}", r2["paired"]["gain"] > 0 and r2["paired"]["z"] >= 3.0))
    # 3. dense -> sparse
    r3 = q2(inc_dense, SPARSE, drop="card_dense")
    out.append((f"the reverse swap loses: gain {r3['paired']['gain']:+.5f}, z {r3['paired']['z']:.2f}",
                r3["paired"]["gain"] < 0))
    # 4. dense -> null
    r4 = q2(inc_dense, NULL, drop="card_dense")
    out.append((f"replacing the pair table by a null term loses: gain {r4['paired']['gain']:+.5f}, "
                f"z {r4['paired']['z']:.2f}", r4["paired"]["gain"] < 0 and r4["paired"]["z"] < -3))
    # 5. dense -> dense, against a converged incumbent (cycle 3 incumbents are converged; on
    # the legacy 400-iteration incumbent the same swap gains a little from the refit alone)
    same = dict(DENSE, name="card_dense2")
    r5l = q2(inc_dense, same, drop="card_dense")
    inc_c = inc([HERO, DENSE], {"mode": "cold"})
    r5m = q2(inc_c, same, drop="card_dense")
    r5 = q2(inc_c, same, drop="card_dense", conv={"mode": "cold"})
    # The paired z of two nearly identical models stays near 2 whatever the tolerance (its SE
    # shrinks with the gain: at pgtol 1e-5 the gain is -1.5e-8 and z is -1.95), so neutrality is
    # judged on the gain, and the gate must not accept it.
    out.append((f"replacing a term by the same spec is neutral when both fits converge (cycle 3): gain "
                f"{r5['paired']['gain']:+.1e} (z {r5['paired']['z']:.2f}), against {r2['paired']['gain']:+.4f} for a "
                f"real swap; legacy incumbent and term fit: gain {r5l['paired']['gain']:+.1e} (z {r5l['paired']['z']:.2f}); "
                f"converged incumbent, legacy term fit: gain {r5m['paired']['gain']:+.1e}",
                abs(r5["paired"]["gain"]) < 1e-4 and r5["paired"]["z"] < 3.0))
    # 6. sizes
    kd = int(inc_dense.offs[2] - inc_dense.offs[1])
    ks = int(inc_sparse.offs[2] - inc_sparse.offs[1])
    out.append((f"sizes reported: new k {r2['term_k']} (dense table {kd}), replaced k {r2['replaced']['k']} "
                f"(sparse table {ks})", r2["term_k"] == kd and r2["replaced"] == {"term": "card_sparse", "k": ks}))
    return out


# --------------------------------------------------------------------------- gate

class GateBroker:
    def __init__(self, terms, card, groups_bits=None):
        self.recs, self.dev = [], "Dpost"
        self.keys = {"noise": "test"}
        self.meta = {}
        self.state = {"accepted_terms": [dict(t) for t in terms], "cards": {"r01": card}}
        self.gb = groups_bits or {}

    def log(self, kind, **kw):
        self.recs.append((kind, kw))

    def save_state(self):
        pass

    def budget(self):
        return {"groups": dict(self.gb), "dev_bits": 10.0}


def gate_cases():
    out = []
    saved = (HN.incumbent_params, HN.safety_probes, HN.F.Model.load)
    from training.scilearn import data as D
    saved_kit = D.load_kit
    try:
        HN.safety_probes = lambda *a, **k: {"five_tank": (0.2, 0.2), "no_healer": (0.3, 0.3)}
        HN.F.Model.load = classmethod(lambda cls, *a, **k: None)
        D.load_kit = lambda meta, path=None: (None, [])
        c10 = {"type": "pair_family", "min_games": 10000, "name": "card_c10_ally_synergy_common"}
        p03 = {"type": "pair_family", "min_games": 5000, "name": "card_r01"}
        others = [{"type": "multiset", "min_teams": 200, "name": "card_c01_role_multiset"},
                  {"type": "decay", "half_life_days": 4, "name": "card_p02_warm_hands"}]

        def card():
            return {"family": "composition", "hash": "h", "job": "q2_r01", "status": "testing",
                    "status_before_test": "registered", "term": dict(p03), "expected": {"sign": "+"},
                    "groups": ["family:pair_family:"],
                    "replaces": {"card_id": "c10_ally_synergy_common", "term_name": c10["name"]}}

        def res(gain, z, k=2553, k_old=1595):
            return {"term_k": k, "replaced": {"term": c10["name"], "k": k_old},
                    "paired": {"gain": gain, "se": abs(gain / z) if z else 1e-5, "z": z, "sum2": 50.0},
                    "invariance": {"ok": True}, "term_coefs": {"x": 0.1}}

        # 7 + 8: accepted, under the cap on the result
        HN.incumbent_params = lambda inc: 17668
        b = GateBroker(others + [c10], card(), {"family:pair_family:": 3.0})
        HN.apply_gate(b, "r01", res(0.00013, 5.9), {"job": "inc_x", "name": "INC"})
        names = [t["name"] for t in b.state["accepted_terms"]]
        tests = [r for r in b.recs if r[0] == "test"]
        rel = b.state["cards"]["r01"]["released"]
        out.append(("an accepted replacement swaps c10 for the new term and records it",
                    b.state["cards"]["r01"]["status"] == "accepted" and c10["name"] not in names
                    and names[-1] == "card_r01" and len(names) == 3
                    and b.state["replaced"] == {"c10_ally_synergy_common": "r01"}))
        out.append(("it is charged 1.5 bits once, to its groups; the release names the replaced card",
                    len(tests) == 1 and tests[0][1]["bits_charged"] == C.Q2_BITS
                    and rel["replaces"] == "c10_ally_synergy_common"))
        out.append((f"cap on the result: 17,668 - 1,595 + 2,553 = {rel['parameters_after']:,} <= 20,000, accepted",
                    rel["parameters_after"] == 18626))
        # 8b: over the cap on the result
        b = GateBroker(others + [c10], card())
        HN.apply_gate(b, "r01", res(0.00013, 5.9, k=4000), {"job": "inc_x", "name": "INC"})
        c = b.state["cards"]["r01"]
        out.append(("a swap that would exceed the cap (17,668 - 1,595 + 4,000) is refused, free",
                    [r[0] for r in b.recs] == ["refusal"] and b.recs[0][1]["bits_charged"] == 0.0
                    and c["status"] == "registered" and c10["name"] in [t["name"] for t in b.state["accepted_terms"]]))
        # 9: target gone
        b = GateBroker(others, card())
        HN.apply_gate(b, "r01", res(0.00013, 5.9), {"job": "inc_x", "name": "INC"})
        out.append(("a replacement whose target left the incumbent is refused, free",
                    [r[0] for r in b.recs] == ["refusal"] and b.state["cards"]["r01"]["status"] == "registered"))
        # 10: losing swap
        b = GateBroker(others + [c10], card())
        HN.apply_gate(b, "r01", res(-0.0002, -4.0), {"job": "inc_x", "name": "INC"})
        out.append(("a losing replacement is rejected and the incumbent is unchanged",
                    b.state["cards"]["r01"]["status"] == "rejected"
                    and [t["name"] for t in b.state["accepted_terms"]] == [t["name"] for t in others + [c10]]))
    finally:
        HN.incumbent_params, HN.safety_probes, HN.F.Model.load = saved
        D.load_kit = saved_kit
    return out


# --------------------------------------------------------------------------- broker

def broker_cases():
    out = []
    meta = json.loads(EXTRACT.with_suffix(".json").read_text())
    kit = np.load(EXTRACT.with_suffix(".npz"))["kit"]
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        b = BR.Broker.__new__(BR.Broker)
        b.sid, b.priv, b.meta, b.kit, b.axes = "t", tmp, meta, kit, meta["kit_axes"]
        b.names = BR.Names(meta, None)
        b.ledger = Ledger("t", root=tmp / "ledger")
        b.code_hash = "test"
        c10 = {"type": "pair_family", "min_games": 10000, "name": "card_c10_ally_synergy_common"}
        b.state = {"session_id": "t", "arm": "blinded", "variant": "K", "cycle": 3, "snapshot": "s",
                   "incumbent": "K0", "cards": {}, "accepted_terms": [c10],
                   "inherited_cards": {"c10_ally_synergy_common": {"status": "accepted", "term": c10, "cycle": 1}}}
        card = ("id: r01\nfamily: composition\ntheory: t\nexpected_effect: {sign: '+'}\n"
                "replaces: c10_ally_synergy_common\nterm: {type: pair_family, min_games: 5000}\n")
        r = b.card_submit(card)
        c = b.state["cards"]["r01"]
        out.append(("a replacement card registers, with the target recorded and both terms' groups",
                    r["status"] == "registered" and r["replaces"] == "c10_ally_synergy_common"
                    and c["replaces"]["term_name"] == c10["name"] and c["groups"] == ["family:pair_family:"]))
        refused = []
        for bad in ("replaces: c99\nterm: {type: pair_family, min_games: 5000}\n",
                    "replaces: hero_build\nterm: {type: pair_family, min_games: 5000}\n",
                    "replaces: c10_ally_synergy_common\nterm: {type: explain, table: c10_ally_synergy_common, "
                    "features: [{kit_sum: burst}]}\n"):
            try:
                b.card_submit(f"id: rx{len(refused)}\nfamily: composition\ntheory: t\nexpected_effect: {{sign: '+'}}\n"
                              + bad)
                refused.append(False)
            except BR.Refusal:
                refused.append(True)
        out.append(("refused: an unknown card id, a backbone term, an explanation card with replaces", all(refused)))
    return out


def run():
    res = broker_cases() + gate_cases() + worker_cases()
    for name, ok in res:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    n = sum(ok for _, ok in res)
    print(f"{n}/{len(res)} passed")
    return n == len(res)


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
