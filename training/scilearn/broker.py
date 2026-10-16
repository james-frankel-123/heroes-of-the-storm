"""Query broker (FRAMING 5). The only path from a proposer session to outcome data.

  python -m training.scilearn.broker serve <session_id>
  python -m training.scilearn.broker open <session_id> --arm b --variant K [--cycle 1]

One broker process per session, listening on SESSIONS/<id>/broker.sock (bound into that
session's sandbox and nowhere else, so the socket is the session's identity). It holds the
labels, applies coarsening and noise, charges budgets, caches answers and writes the
session's hash-chained ledger. Refusals are logged too.

Protocol: one JSON object per line in each direction. Ops: help, budget, query, card_submit,
card_list, test, test_status, ledger_mine, score_incumbent, fit_sandbox.
"""
import argparse
import hashlib
import json
import math
import os
import socketserver
import threading
import traceback

import numpy as np
import yaml

from training.scilearn import config as C
from training.scilearn import data as D
from training.scilearn import evaluator as E
from training.scilearn import terms as TM
from training.scilearn import tranches as T
from training.scilearn.ledger import Ledger, canonical

ARMS = {
    "a": {"dev": "Dpre", "labels": "real", "names": True},
    "b": {"dev": "Dpost", "labels": "real", "names": True},
    "c": {"dev": "Dpre", "labels": "real", "names": True},
    "blinded": {"dev": "Dpost", "labels": "real", "names": False},
    "shuffled": {"dev": "Dpost", "labels": "shuffled", "names": True},
}
QUERYABLE = {"S", "Dpre", "Dpost"}
GROUP_BY = {"map", "tier", "build", "map_class", "hero", "enemy_hero", "first_pick"}
CARD_FIELDS = {"id", "family", "tag", "theory", "mechanism", "expected_effect", "term", "sources",
               "alternatives_considered", "replaces"}
FAMILIES = {"composition", "strength", "solo_lane", "macro", "team_fight", "ganking", "scaling", "others"}


class Refusal(Exception):
    pass


def session_dir(sid):
    return C.SESSIONS / sid


def private_session_dir(sid):
    d = C.PRIVATE / "sessions" / sid
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------- naming

class Names:
    """Real names, or per-session H01..H91 / M01..M15 labels for the blinded arm."""

    def __init__(self, meta, blind_key=None):
        self.heroes, self.maps = meta["heroes"], meta["maps"]
        self.blind = blind_key is not None
        if self.blind:
            rng = np.random.default_rng(int(hashlib.sha256(blind_key.encode()).hexdigest()[:12], 16))
            hp, mp = rng.permutation(len(self.heroes)), rng.permutation(len(self.maps))
            self.h_out = {self.heroes[i]: f"H{int(hp[i]) + 1:02d}" for i in range(len(self.heroes))}
            self.m_out = {self.maps[i]: f"M{int(mp[i]) + 1:02d}" for i in range(len(self.maps))}
        else:
            self.h_out = {h: h for h in self.heroes}
            self.m_out = {m: m for m in self.maps}
        self.h_in = {v: k for k, v in self.h_out.items()}
        self.m_in = {v: k for k, v in self.m_out.items()}

    def hero_idx(self, name):
        if name not in self.h_in:
            raise Refusal(f"unknown hero {name!r}")
        return self.heroes.index(self.h_in[name])

    def map_idx(self, name):
        if name not in self.m_in:
            raise Refusal(f"unknown map {name!r}")
        return self.maps.index(self.m_in[name])

    def translate_term_in(self, obj):
        """Rewrite blinded labels in a term spec to real names (harness side only)."""
        if isinstance(obj, dict):
            return {k: self.translate_term_in(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.translate_term_in(v) for v in obj]
        if isinstance(obj, str):
            return self.h_in.get(obj, self.m_in.get(obj, obj))
        return obj

    def term_name_out(self, s):
        for real, lab in sorted(self.h_out.items(), key=lambda kv: -len(kv[0])):
            s = s.replace(real, lab)
        for real, lab in self.m_out.items():
            s = s.replace(real, lab)
        return s


# --------------------------------------------------------------------------- broker

class Broker:
    def __init__(self, sid):
        self.sid = sid
        self.priv = private_session_dir(sid)
        st = json.loads((self.priv / "state.json").read_text())
        self.state = st
        self.arm = ARMS[st["arm"]]
        self.dev = self.arm["dev"]
        self.ledger = Ledger(sid)
        self.keys = T.load_keys()
        self.lock = threading.RLock()
        g, meta, y, pl = D.load(st["snapshot"], ["S", "Dpre", "Dpost"], labels=True, players=False)
        if self.arm["labels"] == "shuffled":
            y = E.shuffle_labels(y, g["game_ts"], g["map"], g["tier"], g["build"], st["label_key"], g["replay_id"])
        self.meta, self.g, self.y = meta, g, y
        kit, axes = D.load_kit(meta)
        self.kit, self.axes = kit, axes
        self.names = Names(meta, st.get("blind_key") if not self.arm["names"] else None)
        self.hero_roles = np.asarray(meta["hero_roles"])
        self.tables = {}
        self.cache_path = self.priv / "cache.json"
        self.cache = json.loads(self.cache_path.read_text()) if self.cache_path.exists() else {}
        self.code_hash = code_hash()
        self.m = None              # M feature store, loaded on first Q3 / inspect / mtest
        self.preparing = set()     # background table fits started by `table`
        self.stages = {}           # card id -> where its test is waiting (shown by test-status)

    # ---- budget, recomputed from the ledger (single source of truth)
    def budget(self):
        b = {"sandbox_numbers": 0, "dev_bits": 0.0, "mediator_bits": 0.0, "cards": 0, "q4": 0,
             "q6": 0, "inspect": 0, "groups": {}}
        cyc = self.state["cycle"]
        for r in self.ledger.records():
            if r.get("note") and f"cycle={cyc}" not in r["note"]:
                continue
            if r["kind"] == "query" and r.get("template") in ("Q1", "Q6T") and r.get("tranche") == "S":
                b["sandbox_numbers"] += len(r.get("cells") or [])
            if r.get("bits_charged"):
                pj = json.loads(r.get("params_json") or "{}")
                if pj.get("_budget") == "mediator":
                    b["mediator_bits"] += r["bits_charged"]
                else:
                    b["dev_bits"] += r["bits_charged"]
                for gname, bits in (pj.get("_group_charges") or {}).items():
                    b["groups"][gname] = b["groups"].get(gname, 0.0) + bits
            if r["kind"] == "inspect":
                b["inspect"] += 1
            if r["kind"] == "card":
                b["cards"] += 1
            if r["kind"] == "query" and r.get("template") == "Q4":
                b["q4"] += 1
            if r["kind"] == "fit":
                b["q6"] += 1
        # Bits a predecessor session already spent on the same dev tranche stay on its groups:
        # the tranche is the same data, so the gate keeps paying for them (CYCLE2_PREP.md).
        for gname, bits in (self.state.get("prior_group_bits") or {}).items():
            b["groups"][gname] = b["groups"].get(gname, 0.0) + bits
        return b

    def log(self, kind, **kw):
        kw.setdefault("note", "")
        kw["note"] = (kw["note"] + f" cycle={self.state['cycle']}").strip()
        return self.ledger.append(kind, session_id=self.sid, arm=self.state["arm"],
                                  variant=self.state["variant"], snapshot_id=self.state["snapshot"],
                                  code_hash=self.code_hash, **kw)

    # ---- team table per tranche
    def team_table(self, tr):
        if tr not in self.tables:
            m = self.g["tranche"] == T.CODES[tr]
            h = self.g["heroes"][m].astype(np.int64)
            y = self.y[m].astype(np.float64)
            n = len(y)
            team = np.concatenate([h[:, :5], h[:, 5:]])
            opp = np.concatenate([h[:, 5:], h[:, :5]])
            win = np.concatenate([y, 1 - y])
            mp = np.tile(self.g["map"][m], 2).astype(np.int64)
            tier = np.tile(self.g["tier"][m], 2).astype(np.int64)
            build = np.tile(self.g["build"][m], 2).astype(np.int64)
            fp = np.zeros(2 * n, np.int64)
            first = self.g["pick_team"][m][:, 0]
            fp[:n] = first == 0
            fp[n:] = first == 1
            self.tables[tr] = {"team": team, "opp": opp, "win": win, "map": mp, "tier": tier,
                               "build": build, "first_pick": fp, "n_games": n}
        return self.tables[tr]

    # ---- Q0 / Q1
    def run_cells(self, q, tr, table=None):
        t = table if table is not None else self.team_table(tr)
        f = q.get("filter", {}) or {}
        allowed_f = {"hero", "ally", "enemy", "map", "tier", "build", "role_count"}
        bad = set(f) - allowed_f
        if bad:
            raise Refusal(f"filter keys {sorted(bad)} not allowed; use {sorted(allowed_f)}")
        mask = np.ones(len(t["win"]), bool)
        team_heroes, enemy_heroes, structure = [], [], False
        for k in ("hero", "ally"):
            for hn in _aslist(f.get(k)):
                hi = self.names.hero_idx(hn)
                mask &= (t["team"] == hi).any(1)
                team_heroes.append(hn)
        for hn in _aslist(f.get("enemy")):
            hi = self.names.hero_idx(hn)
            mask &= (t["opp"] == hi).any(1)
            enemy_heroes.append(hn)
        if f.get("map") is not None:
            mask &= t["map"] == self.names.map_idx(f["map"])
        if f.get("tier") is not None:
            mask &= t["tier"] == self.meta["tiers"].index(f["tier"])
        if f.get("build") is not None:
            mask &= t["build"] == self.meta["builds"].index(f["build"])
        for role, cnt in (f.get("role_count") or {}).items():
            ri = self.meta["roles"].index(role)
            mask &= (self.hero_roles[t["team"]] == ri).sum(1) == int(cnt)
            structure = True
        gb = q.get("group_by", []) or []
        if len(gb) > C.Q1_MAX_DIMS:
            raise Refusal(f"at most {C.Q1_MAX_DIMS} grouping dimensions")
        for d in gb:
            if d not in GROUP_BY and not d.startswith("role_count:"):
                raise Refusal(f"group_by {d!r} not allowed; use {sorted(GROUP_BY)} or role_count:<Role>")
        idx = np.flatnonzero(mask)
        keys = [()]
        cols = []
        for d in gb:
            cols.append(d)
        # expand rows for hero / enemy_hero grouping
        rows_key, rows_idx = [], []
        if not gb:
            rows_key, rows_idx = [np.zeros(len(idx), np.int64)], idx
            labels = [()]
        else:
            rows_idx = idx
            parts = []
            for d in gb:
                if d in ("hero", "enemy_hero"):
                    src = t["team"] if d == "hero" else t["opp"]
                    rows_idx_new = np.repeat(rows_idx, 5)
                    parts = [p_.repeat(5) for p_ in parts]
                    parts.append(src[rows_idx].ravel())
                    rows_idx = rows_idx_new
                elif d.startswith("role_count:"):
                    ri = self.meta["roles"].index(d.split(":", 1)[1])
                    parts.append((self.hero_roles[t["team"][rows_idx]] == ri).sum(1))
                    structure = True
                elif d == "map_class":
                    mc = np.array([1 - int(m in TM.LANES2) for m in self.meta["maps"]])
                    parts.append(mc[t["map"][rows_idx]])
                else:
                    parts.append(t[d][rows_idx])
            rows_key = parts
        stacked = np.stack(rows_key, 1) if gb else np.zeros((len(rows_idx), 0), np.int64)
        uniq, inv = (np.unique(stacked, axis=0, return_inverse=True) if gb else (np.zeros((1, 0), np.int64), np.zeros(len(rows_idx), np.int64)))
        inv = np.asarray(inv).ravel()
        cnt = np.bincount(inv, minlength=len(uniq))
        wins = np.bincount(inv, weights=t["win"][rows_idx], minlength=len(uniq))
        cells = []
        for ci, keyv in enumerate(uniq):
            lab, th, eh, struct = {}, list(team_heroes), list(enemy_heroes), structure
            for d, v in zip(gb, keyv):
                v = int(v)
                if d == "hero":
                    lab[d] = self.names.h_out[self.meta["heroes"][v]]; th.append(lab[d])
                elif d == "enemy_hero":
                    lab[d] = self.names.h_out[self.meta["heroes"][v]]; eh.append(lab[d])
                elif d == "map":
                    lab[d] = self.names.m_out[self.meta["maps"][v]]
                elif d == "tier":
                    lab[d] = self.meta["tiers"][v]
                elif d == "build":
                    lab[d] = self.meta["builds"][v]
                elif d == "map_class":
                    lab[d] = TM.MAP_CLASSES[v]
                else:
                    lab[d] = v
            cells.append({"cell": lab, "n": int(cnt[ci]), "wins": float(wins[ci]),
                          "group": cell_group(th, eh, struct, f, lab)})
        return cells

    def query(self, q):
        cls = q.get("class")
        tr = q.get("tranche")
        if cls == "Q3":
            return self.query_q3(q)
        if cls not in ("Q0", "Q1", "Q4"):
            raise Refusal(f"query class {cls!r} not available here (Q2 is `test`, Q3 is a query on M-test, "
                          "Q5 is `inspect`, fitted tables are `table`)")
        if tr not in QUERYABLE:
            raise Refusal(f"tranche {tr!r} has no query class (queryable: S, Dpre, Dpost)")
        if cls != "Q0" and tr in ("Dpre", "Dpost") and tr != self.dev:
            raise Refusal(f"outcome queries on {tr} are not part of this session's contract (dev tranche: {self.dev})")
        if cls == "Q4":
            return self.score_incumbent()
        norm = {"class": cls, "tranche": tr, "stat": q.get("stat"), "filter": q.get("filter") or {},
                "group_by": q.get("group_by") or []}
        ch = hashlib.sha256((self.sid + canonical(norm)).encode()).hexdigest()
        if ch in self.cache:
            ans = self.cache[ch]
            self.log("query", template=cls, tranche=tr, params_json=canonical(norm), canonical_hash=ch,
                     released=ans, bits_charged=0.0, note="cache hit")
            return {"cached": True, **ans}
        cells = self.run_cells(norm, tr)
        b = self.budget()
        released, charges, refused = [], {}, 0
        if cls == "Q0":
            if norm["stat"] not in ("count", "share"):
                raise Refusal("Q0 stats: count, share")
            tot = sum(c["n"] for c in cells)
            for c in cells:
                if c["n"] < C.Q0_MIN_CELL:
                    refused += 1
                    continue
                v = c["n"] if norm["stat"] == "count" else round(c["n"] / max(tot, 1), 4)
                released.append({**c["cell"], "teams": c["n"], "value": v})
            bits = 0.0
        else:
            if norm["stat"] != "win_rate":
                raise Refusal("Q1 stat: win_rate (residual tables open once the session has a fitted incumbent)")
            if len(cells) > C.Q1_MAX_CELLS:
                raise Refusal(f"{len(cells)} cells; Q1 allows at most {C.Q1_MAX_CELLS}. Coarsen or filter.")
            if tr == "S":
                if b["sandbox_numbers"] + len(cells) > C.BUDGET["sandbox_numbers"]:
                    raise Refusal("sandbox number budget exhausted for this cycle")
                for c in cells:
                    if c["n"] < C.Q1_S_MIN_CELL:
                        refused += 1
                        continue
                    wr = round(c["wins"] / c["n"] / C.Q1_S_ROUND) * C.Q1_S_ROUND
                    released.append({**c["cell"], "teams": c["n"], "win_rate": round(wr, 4)})
                bits = 0.0
            else:
                rng = np.random.default_rng(int(hashlib.sha256(f"{self.keys['noise']}:{ch}".encode()).hexdigest()[:12], 16))
                per = 0.5 * math.log2(1 + 1 / C.Q1_DEV_KAPPA ** 2)
                for c in cells:
                    if c["n"] < C.Q1_DEV_MIN_CELL:
                        refused += 1
                        continue
                    g_ = c["group"]
                    if b["groups"].get(g_, 0) + charges.get(g_, 0) + per > C.BUDGET["group_bits"]:
                        refused += 1
                        continue
                    p = c["wins"] / c["n"]
                    se = math.sqrt(max(p * (1 - p), 1e-6) / c["n"])
                    noisy = p + rng.normal(0, C.Q1_DEV_KAPPA * se)
                    rel = round(noisy / C.Q1_DEV_ROUND) * C.Q1_DEV_ROUND
                    released.append({**c["cell"], "teams": c["n"], "win_rate": round(rel, 4),
                                     "noise_sd": round(C.Q1_DEV_KAPPA * se, 4)})
                    charges[g_] = charges.get(g_, 0) + per
                bits = sum(charges.values())
                if b["dev_bits"] + bits > C.BUDGET["dev_bits"]:
                    raise Refusal("dev bit budget exhausted for this cycle")
        ans = {"class": cls, "tranche": tr, "cells": released, "refused_cells": refused,
               "bits_charged": round(bits, 4)}
        if not released and refused:
            raise Refusal(f"all {refused} cells refused (cells too small or groups over their bit cap). "
                          "Pool cells (e.g. a kit-level or role-level query) or explore on S.")
        params = dict(norm)
        params["_group_charges"] = charges
        cells_log = [[c["group"], c["n"]] for c in cells]
        nb = self.budget()
        self.log("query", template=cls, tranche=tr, params_json=canonical(params), canonical_hash=ch,
                 cells=cells_log, released=ans, raw_ref=f"cells:{len(cells)}", noise_seed=ch[:12],
                 bits_charged=round(bits, 4),
                 session_bits_after=round(nb["dev_bits"] + bits, 4))
        self.cache[ch] = ans
        self.cache_path.write_text(json.dumps(self.cache))
        return ans

    # ---- cards
    def card_submit(self, text):
        try:
            card = yaml.safe_load(text)
        except yaml.YAMLError as e:
            raise Refusal(f"card is not valid YAML: {e}")
        if not isinstance(card, dict):
            raise Refusal("card must be a mapping")
        missing = {"id", "family", "theory", "expected_effect", "term"} - set(card)
        if missing:
            raise Refusal(f"card missing fields {sorted(missing)}")
        extra = set(card) - CARD_FIELDS
        if extra:
            raise Refusal(f"unknown card fields {sorted(extra)}")
        if card["family"] not in FAMILIES:
            raise Refusal(f"family must be one of {sorted(FAMILIES)}")
        cid = str(card["id"])
        cards = self.state.setdefault("cards", {})
        if cid in cards or cid in (self.state.get("inherited_cards") or {}):
            raise Refusal(f"card id {cid} already registered (this cycle or the previous one); use a new id")
        if len(cards) >= C.BUDGET["cards"]:
            raise Refusal("card budget (40 per cycle) exhausted")
        term = card["term"]
        if not isinstance(term, dict):
            raise Refusal("term must be a mapping")
        term = {k: v for k, v in term.items() if not str(k).startswith("_")}   # harness-only keys
        if card.get("replaces") is not None and term.get("type") in ("explain", "mechanism"):
            raise Refusal("replaces: only a predictive term can replace a term in the incumbent")
        if term.get("type") == "explain":
            return self.explain_submit(card, cid, cards, term)
        if term.get("type") == "mechanism":
            return self.mechanism_submit(card, cid, cards, term)
        if term.get("type") in TM.HARNESS_ONLY:
            raise Refusal(f"term type {term['type']!r} is reserved for the harness")
        rep = None
        if card.get("replaces") is not None:
            rep = resolve_replacement(self.accepted_term_names(), self.state.get("accepted_terms", []),
                                      card["replaces"])
        term = self.names.translate_term_in(term)
        term.setdefault("name", f"card_{cid}")
        term["name"] = f"card_{_slug(cid)}"
        # Stage 2: compile and run the fixtures on a small sample (FRAMING 7.1).
        from training.scilearn import fixtures
        fx = fixtures.check_term(term, self.meta, self.kit, self.axes)
        h = hashlib.sha256(canonical(card).encode()).hexdigest()
        groups = term_groups(term)
        if rep:
            # one card, one charge: the test touches the new term's groups and the old one's
            groups = sorted(set(groups) | set(term_groups(rep["term"])))
        if not fx["ok"]:
            self.log("reject", card_id=cid, family=card["family"], canonical_hash=h,
                     decision="reject", note=f"fixture: {fx['reason']}")
            cards[cid] = {"status": "rejected_fixture", "hash": h, "reason": fx["reason"]}
            self.save_state()
            return {"card_id": cid, "status": "rejected", "reason": fx["reason"]}
        self.log("card", card_id=cid, family=card["family"], canonical_hash=h,
                 params_json=canonical({"card": card, "compiled_term": term, "groups": groups}))
        cards[cid] = {"status": "registered", "hash": h, "family": card["family"], "term": term,
                      "groups": groups, "expected": card.get("expected_effect"), "tag": card.get("tag"),
                      "mechanism": card.get("mechanism")}
        out = {"card_id": cid, "status": "registered", "hash": h, "cell_groups": groups}
        if rep:
            cards[cid]["replaces"] = {"card_id": rep["card_id"], "term_name": rep["term_name"]}
            out["replaces"] = rep["card_id"]
        self.save_state()
        return out

    def save_state(self):
        tmp = self.priv / "state.json.tmp"
        tmp.write_text(json.dumps(self.state, indent=1))
        os.replace(tmp, self.priv / "state.json")

    def test(self, cid):
        from training.scilearn import harness
        cards = self.state.get("cards", {})
        if cid not in cards:
            raise Refusal(f"unknown card {cid}")
        c = cards[cid]
        if c["status"] not in ("registered", "near_miss", "error"):
            raise Refusal(f"card {cid} is {c['status']}")
        if c["status"] == "error":
            c["status"] = c.pop("status_before_error", "registered")   # a harness error charged nothing
        b = self.budget()
        cost = C.Q2_BITS * (2 if c["status"] == "near_miss" else 1)
        if b["dev_bits"] + cost > C.BUDGET["dev_bits"]:
            raise Refusal("dev bit budget exhausted for this cycle")
        job = harness.submit_q2(self, cid)
        c["status_before_test"] = c["status"]
        c["status"] = "testing"
        c["job"] = job
        self.save_state()
        threading.Thread(target=self._finish_test, args=(cid,), daemon=True).start()
        return {"card_id": cid, "status": "testing", "note": "poll with `hyp test-status`"}

    def _finish_test(self, cid):
        from training.scilearn import harness
        try:
            if self.state["cards"][cid]["term"].get("type") == "explain":
                harness.finish_explain(self, cid)
            else:
                harness.finish_q2(self, cid)
        except Exception as e:
            with self.lock:
                c = self.state["cards"][cid]
                if c["status"] == "testing":
                    c["status_before_error"] = c.get("status_before_test", "registered")
                c["status"] = "error"
                c["error"] = str(e)[:500]
                self.save_state()
                try:
                    self.log("error", card_id=cid, family=c.get("family"), canonical_hash=c.get("hash"),
                             bits_charged=0.0, raw_ref=c.get("job"),
                             note=f"harness error, no bits charged, card may be retested: {str(e)[:200]}")
                except Exception:
                    traceback.print_exc()
            traceback.print_exc()
        finally:
            self.stages.pop(cid, None)

    def test_status(self):
        out = {}
        for cid, c in self.state.get("cards", {}).items():
            out[cid] = {k: c.get(k) for k in ("status", "released", "reason", "refusal", "mtest") if c.get(k) is not None}
            if c.get("status") in ("testing", "mtesting") and self.stages.get(cid):
                out[cid]["stage"] = self.stages[cid]
            if c.get("status") == "error":
                out[cid]["reason"] = "harness error (not your card); no bits charged; `hyp test` it again"
        return out

    def ledger_mine(self):
        recs = []
        for r in self.ledger.records():
            recs.append({k: r.get(k) for k in ("seq", "ts", "kind", "card_id", "template", "tranche",
                                               "released", "bits_charged", "decision", "note")})
        return recs

    def score_incumbent(self):
        b = self.budget()
        if b["q4"] >= C.BUDGET["q4_per_cycle"]:
            raise Refusal("Q4 budget (3 per cycle) exhausted")
        raise Refusal("Q4 opens once the session's incumbent has changed from the backbone")

    # ---- cycle 2: explanation cards
    def explain_submit(self, card, cid, cards, term):
        from training.scilearn import explain as X
        from training.scilearn import harness
        tables = harness.accepted_tables(self.state)
        tid = str(term.get("table", ""))
        if tid not in tables:
            raise Refusal(f"explain term needs `table: <card id>` of an accepted table in your incumbent; "
                          f"tables: {sorted(tables)}")
        tspec = tables[tid]
        kind = X.table_kind(tspec)
        feats = self.names.translate_term_in(term.get("features"))
        ctx = tspec["keys"][1] if kind == "hero_ctx" else None
        try:
            k = X.validate_theory({"features": feats}, kind, self.axes, self.meta["roles"], ctx=ctx)
        except X.ExplainError as e:
            raise Refusal(f"explain term: {e}")
        term = {"type": "explain", "table": tid, "features": feats, "name": f"card_{_slug(cid)}"}
        h = hashlib.sha256(canonical(card).encode()).hexdigest()
        groups = [f"explain:{tid}"]
        self.log("card", card_id=cid, family=card["family"], canonical_hash=h,
                 params_json=canonical({"card": card, "compiled_term": term, "groups": groups, "max_params": k}))
        cards[cid] = {"status": "registered", "hash": h, "family": card["family"], "term": term,
                      "groups": groups, "expected": card.get("expected_effect"), "tag": card.get("tag"),
                      "mechanism": card.get("mechanism")}
        self.save_state()
        return {"card_id": cid, "status": "registered", "hash": h, "cell_groups": groups,
                "table_kind": kind, "max_params": k}

    def accepted_term_names(self):
        """{card id: term name} for every term in the incumbent that came from a card."""
        from training.scilearn import harness
        names = {t.get("name") for t in self.state.get("accepted_terms", [])}
        out = {}
        for cid in list(self.state.get("inherited_cards") or {}) + list(self.state.get("cards", {})):
            n = harness.table_term_name(cid)
            if n in names:
                out[cid] = n
        return out

    def mechanism_submit(self, card, cid, cards, term):
        """A mechanism card: a declared mediator claim about a term already in the incumbent."""
        from training.scilearn import replay_features as RF
        acc = self.accepted_term_names()
        of = str(term.get("of", ""))
        if of not in acc:
            raise Refusal(f"mechanism term needs `of: <card id>` of a term in your incumbent; options: {sorted(acc)}")
        mech = card.get("mechanism") or {}
        if mech.get("mediator") not in RF.MEDIATORS or mech.get("negative_control") not in RF.MEDIATORS:
            raise Refusal(f"a mechanism card needs mechanism.mediator and mechanism.negative_control from {RF.MED_NAMES}")
        if str(mech.get("direction", "+")).strip() not in ("+", "-"):
            raise Refusal("mechanism.direction is + or -")
        term = {"type": "mechanism", "of": of, "name": acc[of]}
        h = hashlib.sha256(canonical(card).encode()).hexdigest()
        self.log("card", card_id=cid, family=card["family"], canonical_hash=h,
                 params_json=canonical({"card": card, "compiled_term": term, "groups": [f"mech:{mech['mediator']}"]}))
        cards[cid] = {"status": "registered", "hash": h, "family": card["family"], "term": term,
                      "groups": [f"mech:{mech['mediator']}"], "expected": card.get("expected_effect"),
                      "tag": card.get("tag"), "mechanism": mech}
        self.save_state()
        return {"card_id": cid, "status": "registered", "hash": h, "next": f"hyp mtest {cid}"}

    # ---- cycle 2: the M tranche (parsed replays)
    def m_data(self):
        from training.scilearn import replay_features as RF
        mp = RF.STORE / RF.VERSION / "manifest.json"
        if not mp.exists():
            raise Refusal("no parsed replays yet: M is empty")
        man = json.loads(mp.read_text())
        if self.m is None or self.m["man"]["built"] != man["built"]:
            f, y, man = RF.load()
            perm = np.arange(len(y))
            if self.arm["labels"] == "shuffled":
                # whole records (trajectory plus outcome) permuted within map x tier x build
                perm = RF.shuffle_index(f["map"], f["tier"], f["build"], self.state["label_key"])
                f["med"], y = f["med"][perm], y[perm]
            f["traj_index"] = perm
            self.m = {"f": f, "y": y, "man": man,
                      "paths": json.loads((RF.STORE / RF.VERSION / "paths.json").read_text())}
        return self.m

    def m_team_table(self, mediator, power=1):
        from training.scilearn import replay_features as RF
        m = self.m_data()
        f = m["f"]
        test = f["side"] == RF.SIDES["M-test"]
        j = RF.MED_NAMES.index(mediator)
        h = f["heroes"][test]
        med = f["med"][test][:, :, j]
        x = np.concatenate([med[:, 0], med[:, 1]])
        keep = ~np.isnan(x)
        return {"team": np.concatenate([h[:, :5], h[:, 5:]])[keep], "opp": np.concatenate([h[:, 5:], h[:, :5]])[keep],
                "win": (x ** power)[keep], "map": np.tile(f["map"][test], 2)[keep],
                "tier": np.tile(f["tier"][test], 2)[keep], "build": np.tile(f["build"][test], 2)[keep],
                "first_pick": np.zeros(int(keep.sum()), np.int64), "n_games": int(test.sum())}

    def query_q3(self, q):
        """Q3 mediator cells on M-test (FRAMING 5.2): cells of 200+ team rows, noise sd = 2 SE,
        0.16 rho^2 bits per number from the mediator budget, 8 bits per cell group."""
        from training.scilearn import replay_features as RF
        med = q.get("mediator")
        if med not in RF.MEDIATORS:
            raise Refusal(f"Q3 needs `mediator`, one of {RF.MED_NAMES}")
        if q.get("tranche", "M-test") not in ("M", "M-test"):
            raise Refusal("Q3 runs on M-test only (M-explore is for `inspect`)")
        gb = q.get("group_by") or []
        if "first_pick" in gb or "build" in gb:
            raise Refusal("Q3 group_by takes map, tier, map_class, hero, enemy_hero or role_count:<Role>")
        norm = {"class": "Q3", "tranche": "M-test", "mediator": med, "filter": q.get("filter") or {}, "group_by": gb}
        ch = hashlib.sha256((self.sid + canonical(norm)).encode()).hexdigest()
        m = self.m_data()
        ckey = ch + ":" + m["man"]["built"]
        if ckey in self.cache:
            ans = self.cache[ckey]
            self.log("query", template="Q3", tranche="M-test", params_json=canonical(norm), canonical_hash=ch,
                     released=ans, bits_charged=0.0, note="cache hit")
            return {"cached": True, **ans}
        c1 = self.run_cells(norm, "M", table=self.m_team_table(med, 1))
        c2 = self.run_cells(norm, "M", table=self.m_team_table(med, 2))
        if len(c1) > C.Q1_MAX_CELLS:
            raise Refusal(f"{len(c1)} cells; Q3 allows at most {C.Q1_MAX_CELLS}")
        rho = float(m["man"]["rho"].get(med, 0.5))
        per = 0.5 * math.log2(1 + 1 / C.Q1_DEV_KAPPA ** 2) * rho ** 2
        step = 0.001 if RF.MEDIATORS[med][0] in ("share", "flag") else 0.01
        b = self.budget()
        rng = np.random.default_rng(int(hashlib.sha256(f"{self.keys['noise']}:{ckey}".encode()).hexdigest()[:12], 16))
        released, charges, refused = [], {}, 0
        for a, a2 in zip(c1, c2):
            g_ = f"med:{med}|{a['group']}"
            if a["n"] < C.Q3_MIN_CELL or b["groups"].get(g_, 0) + charges.get(g_, 0) + per > C.BUDGET["group_bits"]:
                refused += 1
                continue
            mean = a["wins"] / a["n"]
            var = max(a2["wins"] / a2["n"] - mean ** 2, 1e-12)
            se = math.sqrt(var / a["n"])
            noisy = mean + rng.normal(0, C.Q1_DEV_KAPPA * se)
            released.append({**a["cell"], "teams": a["n"], "mean": round(round(noisy / step) * step, 4),
                             "noise_sd": round(C.Q1_DEV_KAPPA * se, 4)})
            charges[g_] = charges.get(g_, 0) + per
        bits = sum(charges.values())
        if b["mediator_bits"] + bits > C.BUDGET["mediator_bits"]:
            raise Refusal("mediator bit budget exhausted for this cycle")
        if not released:
            raise Refusal(f"all {refused} cells refused (cells under {C.Q3_MIN_CELL} team rows, or groups over "
                          f"their bit cap). M-test has {self.m_team_table(med)['n_games']} games; pool cells.")
        ans = {"class": "Q3", "tranche": "M-test", "mediator": med, "cells": released, "refused_cells": refused,
               "bits_charged": round(bits, 4), "m_test_games": self.m_team_table(med)["n_games"]}
        self.log("query", template="Q3", tranche="M-test",
                 params_json=canonical({**norm, "_group_charges": charges, "_budget": "mediator"}),
                 canonical_hash=ch, cells=[[f"med:{med}|{a['group']}", a["n"]] for a in c1], released=ans,
                 noise_seed=ch[:12], bits_charged=round(bits, 4))
        self.cache[ckey] = ans
        self.cache_path.write_text(json.dumps(self.cache))
        return ans

    def inspect(self, req):
        """Q5: one M-explore game's record. The game is spent; it never enters a test."""
        from training.scilearn import replay_features as RF
        import gzip
        b = self.budget()
        if b["inspect"] >= C.BUDGET["inspect"]:
            raise Refusal("inspection budget (300 games per cycle) exhausted")
        m = self.m_data()
        f = m["f"]
        flt = req.get("filter") or {}
        cand = f["side"] == RF.SIDES["M-explore"]
        if flt.get("hero"):
            hi = self.names.hero_idx(flt["hero"])
            cand &= (f["heroes"] == hi).any(1)
        if flt.get("map"):
            cand &= f["map"] == self.names.map_idx(flt["map"])
        if flt.get("cohort") in ("interesting", "random"):
            cand &= f["cohort"] == (RF.COHORT_CODES["surprise_v1"] if flt["cohort"] == "interesting" else RF.COHORT_CODES["stratified_b"])
        spent = set(self.state.get("spent_games", []))
        idx = [i for i in np.flatnonzero(cand) if int(f["replay_id"][i]) not in spent]
        if not idx:
            raise Refusal("no unspent M-explore game matches that filter")
        idx.sort(key=lambda i: T.khash(self.keys["noise"], int(f["replay_id"][i])))
        i = idx[0]
        traj = int(f["traj_index"][i])
        rid_traj = int(f["replay_id"][traj])
        d = json.load(gzip.open(C.REPLAYS_ROOT / m["paths"][str(rid_traj)]))
        rec = self.sanitize_record(d, f["heroes"][i], int(f["map"][i]), shuffled=(traj != i))
        token = "G" + hashlib.sha256(f"{self.keys['noise']}:{int(f['replay_id'][i])}".encode()).hexdigest()[:10]
        rec["game"] = token
        rec["cohort"] = "interesting" if int(f["cohort"][i]) == RF.COHORT_CODES["surprise_v1"] else "random"
        self.state.setdefault("spent_games", []).append(int(f["replay_id"][i]))
        self.save_state()
        self.log("inspect", template="Q5", tranche="M-explore", params_json=canonical({"filter": flt}),
                 raw_ref=f"replay:{int(f['replay_id'][i])} traj:{rid_traj}", released={"game": token},
                 note=f"game spent {token}")
        return rec

    def sanitize_record(self, d, draft_heroes, map_idx, shuffled=False):
        """No player identifiers, no action rates. Blinded: labels only, no talent, map-event or
        unit names. Shuffled: the trajectory of another game in the same map x tier x build
        block, shown with this game's draft."""
        blind = self.names.blind
        players = sorted(d["players"], key=lambda p: (int(p["team"]), int(p["pid"])))
        slot = {int(p["pid"]): k for k, p in enumerate(players)}
        out = {"map": self.names.m_out[self.meta["maps"][map_idx]], "length_s": d["length_s"],
               "gates_open_s": d.get("gates_open_s"), "winner_team": d["winner_team"],
               "team_levels": d["team_levels"], "xp_periodic": d["xp_periodic"], "deaths": d["deaths"],
               "structures": d["structures"], "mercs": d["mercs"], "players": []}
        if not blind:
            out["build"] = d.get("version")
        for p in players:
            k = slot[int(p["pid"])]
            hero = self.meta["heroes"][int(draft_heroes[k])]
            q = {"pid": p["pid"], "team": p["team"], "hero": self.names.h_out[hero], "win": p.get("win"),
                 "levels": p.get("levels"), "positions": p.get("positions")}
            if not (blind or shuffled):
                q["talents"] = [{"tier": t.get("tier"), "name": t.get("name"), "t": t.get("t")} for t in p.get("talents", [])]
                q["end"] = p.get("end")
            else:
                q["talents"] = [{"tier": t.get("tier"), "t": t.get("t")} for t in p.get("talents", [])]
                q["end"] = {k2: v for k2, v in (p.get("end") or {}).items() if k2 != "talent_choices"}
            out["players"].append(q)
        if not (blind or shuffled):
            out["map_events"] = d.get("map_events")
            out["notable_units"] = d.get("notable_units")
            out["picks"] = d.get("picks")
            out["bans"] = d.get("bans")
        return out

    # ---- cycle 2: fitted tables (an S-only fit of the session's incumbent; Q6-style release)
    def table_view(self, req):
        from training.scilearn import explain as X
        from training.scilearn import fit as F
        from training.scilearn import harness
        cid = str(req.get("card_id"))
        tables = harness.accepted_tables(self.state)
        if cid not in tables:
            raise Refusal(f"no accepted table {cid!r}; tables: {sorted(tables)}")
        key = harness.incumbent_key(self.state, self.dev, harness.EXPLORE_FIT)
        reg = harness._registry()
        if key not in reg:
            if key not in self.preparing:
                self.preparing.add(key)
                st = json.loads(json.dumps(self.state))
                threading.Thread(target=self._prepare_explore, args=(st, key), daemon=True).start()
            return {"status": "preparing", "note": "the S-only fit of your incumbent is running (30 to 90 min); "
                    "ask again later. Nothing was charged."}
        top = max(1, min(int(req.get("top", 40)), 200))
        flt = req.get("hero")
        norm = {"op": "table", "card_id": cid, "top": top, "hero": flt}
        ch = hashlib.sha256((self.sid + canonical(norm) + key).encode()).hexdigest()
        if ch in self.cache:
            return {"cached": True, **self.cache[ch]}
        meta = dict(self.meta)
        meta["kit_axes"] = self.axes
        model = F.Model.load(C.PRIVATE / "jobs" / reg[key]["job"] / "out" / "INC", meta)
        tname = tables[cid]["name"]
        cells = X.table_cells(model, tname, meta)
        P = X.Projector(X.nuisance_design(cells, len(meta["heroes"])), cells["w"])
        r = P(cells["y"])
        se = np.sqrt(cells["s"])
        S = X.signal_ss(r, cells["w"], P.rank)
        tot = float((cells["w"] * r ** 2).sum())
        idx = np.arange(len(r))
        if flt:
            hi = self.names.hero_idx(flt)
            idx = idx[(cells["a"][idx] == hi) | ((cells["kind"] != "hero_ctx") & (cells["b"][idx] == hi))]
        idx = idx[np.argsort(-np.abs(r[idx]) / se[idx])][:top]
        H = self.meta["heroes"]
        out = []
        for j in idx:
            a = self.names.h_out[H[int(cells["a"][j])]]
            if cells["kind"] == "hero_ctx":
                lab = cells["ctx_labels"][int(cells["b"][j])]
                lab = self.names.m_out.get(lab, lab)
                key_ = {"hero": a, cells["ctx"]: lab}
            elif cells["kind"] == "pair":
                key_ = {"pair": [a, self.names.h_out[H[int(cells["b"][j])]]]}
            else:
                key_ = {"team_with": a, "against": self.names.h_out[H[int(cells["b"][j])]]}
            out.append({**key_, "value": round(float(r[j]), 3), "se": round(float(se[j]), 3)})
        ans = {"table": cid, "kind": cells["kind"], "fit": "S only", "n_cells": int(len(r)),
               "signal_share_of_weighted_ss": round(S / tot, 3) if tot > 0 else None,
               "values": "interaction part: hero and context main effects projected out (logit units)",
               "cells": out}
        b = self.budget()
        if b["sandbox_numbers"] + len(out) > C.BUDGET["sandbox_numbers"]:
            raise Refusal("sandbox number budget exhausted for this cycle")
        self.log("query", template="Q6T", tranche="S", params_json=canonical(norm), canonical_hash=ch,
                 cells=[[f"table:{cid}", 1] for _ in out], released={"n": len(out)}, bits_charged=0.0,
                 raw_ref=reg[key]["job"])
        self.cache[ch] = ans
        self.cache_path.write_text(json.dumps(self.cache))
        return ans

    def _prepare_explore(self, st, key):
        from training.scilearn import harness
        try:
            harness.ensure_fit(st, self.dev, fit=harness.EXPLORE_FIT)
        except Exception:
            traceback.print_exc()
        finally:
            self.preparing.discard(key)

    # ---- cycle 2: mechanism tests on M-test
    def mtest(self, cid):
        from training.scilearn import harness
        from training.scilearn import replay_features as RF
        cards = self.state.get("cards", {})
        if cid not in cards:
            raise Refusal(f"unknown card {cid}")
        c = cards[cid]
        t = c.get("term") or {}
        if t.get("type") == "mechanism":
            if c.get("status") != "registered":
                raise Refusal(f"card {cid} is {c.get('status')}")
            src = {"kind": "incumbent", "term_name": t["name"]}
        elif c.get("status") in ("accepted", "rejected", "near_miss") and c.get("job"):
            if t.get("type") in ("explain", "decay"):
                raise Refusal("mechanism tests take draft-composition terms (not explain or decay cards)")
            src = {"kind": "incumbent" if c["status"] == "accepted" else "q2", "term_name": t["name"]}
        else:
            raise Refusal("`hyp mtest` takes a mechanism card, or a card whose term `hyp test` has fitted")
        if c.get("mtest"):
            raise Refusal("one mechanism test per card per cycle")
        mech = c.get("mechanism") or {}
        if mech.get("mediator") not in RF.MEDIATORS or mech.get("negative_control") not in RF.MEDIATORS:
            raise Refusal(f"the card's `mechanism` needs `mediator` and `negative_control` from {RF.MED_NAMES}")
        if str(mech.get("direction", "+")).strip() not in ("+", "-"):
            raise Refusal("mechanism.direction is + or -")
        m = self.m_data()
        n_test = int((m["f"]["side"] == RF.SIDES["M-test"]).sum())
        if n_test < harness.M_TEST_MIN_GAMES:
            raise Refusal(f"M-test has {n_test} parsed games; mechanism tests open at {harness.M_TEST_MIN_GAMES}")
        b = self.budget()
        if b["mediator_bits"] + harness.MTEST_BITS > C.BUDGET["mediator_bits"]:
            raise Refusal("mediator bit budget exhausted for this cycle")
        c["status_for_mtest"] = c["status"]
        c["mtest_source"] = src
        c["status"] = "mtesting"
        self.save_state()
        threading.Thread(target=self._finish_mtest, args=(cid,), daemon=True).start()
        return {"card_id": cid, "status": "mtesting", "m_test_games": n_test}

    def _finish_mtest(self, cid):
        from training.scilearn import harness
        try:
            harness.finish_mtest(self, cid)
        except Exception as e:
            with self.lock:
                c = self.state["cards"][cid]
                c["status"] = c.pop("status_for_mtest", "rejected")
                self.save_state()
                self.log("error", card_id=cid, bits_charged=0.0,
                         note=f"mechanism test harness error, nothing charged: {str(e)[:200]}")
            traceback.print_exc()
        finally:
            self.stages.pop(cid, None)

    def incumbent_summary(self):
        from training.scilearn import harness
        base = json.loads((C.CODE / "models" / f"{self.state['incumbent'].lower()}.json").read_text())
        tables = harness.accepted_tables(self.state)
        by_name = {harness.table_term_name(c): c for c in
                   list(self.state.get("inherited_cards") or {}) + list(self.state.get("cards", {}))}
        terms = []
        for t in self.state.get("accepted_terms", []):
            cid = by_name.get(t.get("name"))
            terms.append({"card_id": cid, "type": t["type"],
                          **{k: v for k, v in t.items() if k in ("keys", "min_games", "min_teams", "axis", "by")},
                          "table": cid in tables})
        return {"backbone": self.state["incumbent"], "backbone_doc": base.get("_doc"), "accepted_terms": terms,
                "explanations": [{k: e[k] for k in ("card_id", "table", "features")}
                                 for e in self.state.get("explanations", [])]}

    def mediator_summary(self):
        from training.scilearn import replay_features as RF
        try:
            m = self.m_data()
            f = m["f"]
            n = {s: int((f["side"] == c).sum()) for s, c in RF.SIDES.items()}
        except Refusal:
            n = {"M-explore": 0, "M-test": 0}
        return {"games": n, "definitions": {k: v[1] for k, v in RF.MEDIATORS.items()}}

    def handle(self, req):
        op = req.get("op")
        if op == "help":
            return {"ops": ["budget", "query", "card_submit", "card_list", "test", "test_status",
                            "ledger_mine", "meta", "table", "inspect", "mtest"], "dev_tranche": self.dev}
        if op == "meta":
            return {"heroes": [self.names.h_out[h] for h in self.meta["heroes"]],
                    "maps": [self.names.m_out[m] for m in self.meta["maps"]],
                    "tiers": self.meta["tiers"], "roles": self.meta["roles"],
                    "hero_roles": {self.names.h_out[h]: self.meta["roles"][r] for h, r in zip(self.meta["heroes"], self.meta["hero_roles"])},
                    "builds": {tr: sorted({self.meta["builds"][b] for b in np.unique(self.g["build"][self.g["tranche"] == T.CODES[tr]])})
                               for tr in ("S", "Dpre", "Dpost")} if self.arm["names"] else "withheld",
                    "kit_axes": self.axes, "dev_tranche": self.dev, "cycle": self.state["cycle"],
                    "term_types": [t for t in TM.TERM_TYPES if t not in TM.HARNESS_ONLY] + ["explain (card only)"],
                    "incumbent": self.incumbent_summary(), "mediators": self.mediator_summary()}
        if op == "budget":
            b = self.budget()
            return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in b.items() if k != "groups"} | {
                "groups": {self.names.term_name_out(k): round(v, 3) for k, v in b["groups"].items()},
                "caps": C.BUDGET}
        if op == "query":
            return self.query(req.get("query") or {})
        if op == "card_submit":
            return self.card_submit(req.get("card", ""))
        if op == "card_list":
            return {cid: {"status": c["status"], "groups": c.get("groups")} for cid, c in self.state.get("cards", {}).items()}
        if op == "test":
            return self.test(str(req.get("card_id")))
        if op == "test_status":
            return self.test_status()
        if op == "ledger_mine":
            return self.ledger_mine()
        if op == "table":
            return self.table_view(req)
        if op == "inspect":
            return self.inspect(req)
        if op == "mtest":
            return self.mtest(str(req.get("card_id")))
        raise Refusal(f"unknown op {op!r}")


def resolve_replacement(term_names, accepted_terms, target):
    """A replacement card names the card id of a term now in the incumbent. Returns
    {"card_id", "term_name", "term"}; refuses anything else (backbone terms, explanations,
    mechanism cards, terms already replaced)."""
    rid = str(target)
    if rid not in term_names:
        raise Refusal(f"replaces: needs the card id of a term in your incumbent; options: {sorted(term_names)}")
    name = term_names[rid]
    spec = next((t for t in accepted_terms if t.get("name") == name), None)
    if spec is None:
        raise Refusal(f"replaces: {rid} is not in your incumbent")
    return {"card_id": rid, "term_name": name, "term": spec}


def _aslist(v):
    if v is None:
        return []
    return v if isinstance(v, list) else [v]


def _slug(s):
    return "".join(ch if ch.isalnum() else "_" for ch in str(s))[:40]


def cell_group(team_heroes, enemy_heroes, structure, f, lab):
    th, eh = sorted(set(team_heroes)), sorted(set(enemy_heroes))
    if len(th) >= 2:
        return "pair:" + "|".join(th[:2])
    if th and eh:
        return f"vs:{th[0]}|{eh[0]}"
    if th:
        return f"hero:{th[0]}"
    if eh:
        return f"hero:{eh[0]}"
    if structure:
        return "structure:roles"
    m = f.get("map") or lab.get("map")
    return f"map:{m}" if m else "global"


def term_groups(term):
    """Cell groups a term touches, for bit accounting (FRAMING 5.3)."""
    groups = set()

    def walk(o):
        if isinstance(o, dict):
            if o.get("field") == "hero_in_team":
                groups.add(f"hero:{o['value']}")
            if o.get("field") == "kit_sum":
                groups.add(f"axis:{o['key']}")
            if o.get("field") == "role_count":
                groups.add("structure:roles")
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(term)
    t = term.get("type")
    if t in ("attr_sum", "product"):
        groups.add(f"axis:{term['axis']}" if t == "attr_sum" else f"map_axis:{term['axis']}")
    if t == "bilinear":
        for e in term.get("entries", []):
            groups.add(f"axis_pair:{e['a']}x{e['b']}")
    if t == "multiset":
        groups.add("structure:roles")
    if t in ("pair_family", "cross_family", "lookup", "state", "decay"):
        groups.add(f"family:{t}:{'x'.join(term.get('keys', []))}")
    return sorted(groups) or [f"term:{t}"]


def code_hash():
    h = hashlib.sha256()
    for p in sorted(C.CODE.glob("*.py")):
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        for line in self.rfile:
            try:
                req = json.loads(line)
                with self.server.broker.lock:
                    resp = {"ok": True, "result": self.server.broker.handle(req)}
            except Refusal as e:
                resp = {"ok": False, "refused": str(e)}
                try:
                    with self.server.broker.lock:
                        q = req.get("query") if isinstance(req, dict) else None
                        self.server.broker.log(
                            "refusal", template=(q or {}).get("class") or req.get("op"),
                            tranche=(q or {}).get("tranche"), card_id=req.get("card_id"),
                            params_json=canonical(req)[:4000], note=str(e)[:300])
                except Exception:
                    traceback.print_exc()
            except Exception as e:
                traceback.print_exc()
                resp = {"ok": False, "error": f"{type(e).__name__}: {e}"[:500]}
            self.wfile.write((json.dumps(resp, default=_np_default) + "\n").encode())
            self.wfile.flush()


def _np_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


class Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


def ledger_group_bits(sid):
    """Bits charged to each cell group in a session's ledger (all cycles)."""
    out = {}
    for r in Ledger(sid).records():
        if r.get("bits_charged"):
            pj = json.loads(r.get("params_json") or "{}")
            for g, bits in (pj.get("_group_charges") or {}).items():
                out[g] = out.get(g, 0.0) + bits
    return out


def open_session(sid, arm, variant, cycle=1, incumbent="K0", snapshot=None, predecessor=None, fit_policy=None):
    """predecessor: a closed session of the same arm and dev tranche whose incumbent, shuffle
    and blinding keys, and per-group bits carry over (cycle 2 continues on the same Dpost)."""
    keys = T.load_keys()
    priv = private_session_dir(sid)
    if (priv / "state.json").exists():
        raise SystemExit(f"session {sid} already exists")
    st = {"session_id": sid, "arm": arm, "variant": variant, "cycle": cycle,
          "snapshot": snapshot or D.latest_snapshot(), "incumbent": incumbent, "cards": {},
          "accepted_terms": []}
    if arm == "shuffled":
        st["label_key"] = hashlib.sha256(f"{keys['canary']}:shuffle:{sid}".encode()).hexdigest()
    if arm == "blinded":
        st["blind_key"] = hashlib.sha256(f"{keys['blind']}:{sid}".encode()).hexdigest()
    if predecessor:
        prev = json.loads((private_session_dir(predecessor) / "state.json").read_text())
        if prev["arm"] != arm or prev["snapshot"] != st["snapshot"] or prev["incumbent"] != incumbent:
            raise SystemExit("predecessor must share arm, snapshot and backbone")
        from training.scilearn import harness
        st["predecessor"] = predecessor
        st["accepted_terms"] = list(prev.get("accepted_terms", []))
        st["inherited_tables"] = harness.accepted_tables(prev)
        # Cards of every earlier cycle: their ids stay reserved, their accepted terms stay
        # addressable (mechanism and replacement cards name them by card id).
        inh = {cid: dict(c, cycle=c.get("cycle", prev["cycle"] - 1))
               for cid, c in (prev.get("inherited_cards") or {}).items()}
        for cid, c in prev.get("cards", {}).items():
            inh[cid] = {"status": c.get("status"), "family": c.get("family"), "term": c.get("term"),
                        "released": c.get("released"), "cycle": prev["cycle"],
                        **({"voided_release": c["voided_release"]} if c.get("voided_release") else {}),
                        **({"replaces": c["replaces"]} if c.get("replaces") else {})}
        st["inherited_cards"] = inh
        # Bits on the same dev tranche from every earlier cycle stay on their groups.
        pg = dict(prev.get("prior_group_bits") or {})
        for g, bits in ledger_group_bits(predecessor).items():
            pg[g] = pg.get(g, 0.0) + bits
        st["prior_group_bits"] = pg
        for k in ("label_key", "blind_key"):
            if k in prev:
                st[k] = prev[k]
        # inspected games stay spent; replacements and earlier results stay on record
        if prev.get("spent_games"):
            st["spent_games"] = list(prev["spent_games"])
        if prev.get("replaced"):
            st["replaced"] = dict(prev["replaced"])
        for k in ("explanations", "mechanism_results"):
            old_ = list(prev.get("inherited_" + k, [])) + list(prev.get(k, []))
            if old_:
                st["inherited_" + k] = old_
    if fit_policy if fit_policy is not None else cycle >= 3:
        # Max, 2026-10-15: the session starts from a cold fit to convergence of its incumbent;
        # refits after an accept are warm; the end-of-cycle refit is cold (CYCLE3_PREP.md).
        from training.scilearn import harness
        st["fit_policy"] = {"start": "cold", "mid_cycle": "warm", "end_of_cycle": "cold",
                            "start_terms_key": harness._terms_key(st)}
    (priv / "state.json").write_text(json.dumps(st, indent=1))
    Ledger(sid).append("session_open", session_id=sid, arm=arm, variant=variant,
                       snapshot_id=st["snapshot"], code_hash=code_hash(),
                       params_json=canonical({"predecessor": predecessor,
                                              "accepted_terms": [t.get("name") for t in st["accepted_terms"]],
                                              "prior_group_bits": st.get("prior_group_bits", {}),
                                              "fit_policy": st.get("fit_policy")}),
                       note=f"cycle={cycle} incumbent={incumbent} budget={json.dumps(C.BUDGET)}")
    (C.SESSIONS / sid).mkdir(parents=True, exist_ok=True)
    return st


def serve(sid):
    b = Broker(sid)
    sock = session_dir(sid) / "broker.sock"
    sock.parent.mkdir(parents=True, exist_ok=True)
    if sock.exists():
        sock.unlink()
    srv = Server(str(sock), Handler)
    os.chmod(sock, 0o660)
    srv.broker = b
    print(f"broker for {sid} on {sock} (arm {b.state['arm']}, dev {b.dev})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("serve"); s.add_argument("sid")
    o = sp.add_parser("open"); o.add_argument("sid"); o.add_argument("--arm", required=True, choices=sorted(ARMS))
    o.add_argument("--variant", default="K"); o.add_argument("--cycle", type=int, default=1)
    o.add_argument("--incumbent", default="K0"); o.add_argument("--from", dest="predecessor")
    a = ap.parse_args()
    if a.cmd == "serve":
        serve(a.sid)
    else:
        print(json.dumps(open_session(a.sid, a.arm, a.variant, a.cycle, a.incumbent, predecessor=a.predecessor), indent=1))
