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
               "alternatives_considered"}
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

    # ---- budget, recomputed from the ledger (single source of truth)
    def budget(self):
        b = {"sandbox_numbers": 0, "dev_bits": 0.0, "mediator_bits": 0.0, "cards": 0, "q4": 0,
             "q6": 0, "inspect": 0, "groups": {}}
        cyc = self.state["cycle"]
        for r in self.ledger.records():
            if r.get("note") and f"cycle={cyc}" not in r["note"]:
                continue
            if r["kind"] == "query" and r.get("template") == "Q1" and r.get("tranche") == "S":
                b["sandbox_numbers"] += len(r.get("cells") or [])
            if r.get("bits_charged"):
                b["dev_bits"] += r["bits_charged"]
                for gname, bits in (json.loads(r.get("params_json") or "{}").get("_group_charges") or {}).items():
                    b["groups"][gname] = b["groups"].get(gname, 0.0) + bits
            if r["kind"] == "card":
                b["cards"] += 1
            if r["kind"] == "query" and r.get("template") == "Q4":
                b["q4"] += 1
            if r["kind"] == "fit":
                b["q6"] += 1
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
    def run_cells(self, q, tr):
        t = self.team_table(tr)
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
        if cls not in ("Q0", "Q1", "Q4"):
            raise Refusal(f"query class {cls!r} not available here (Q2 is `test`, Q3/Q5 need parsed replays, Q6 is `fit_sandbox`)")
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
        if cid in cards:
            raise Refusal(f"card id {cid} already registered; use a new id")
        if len(cards) >= C.BUDGET["cards"]:
            raise Refusal("card budget (40 per cycle) exhausted")
        term = card["term"]
        if not isinstance(term, dict):
            raise Refusal("term must be a mapping")
        term = {k: v for k, v in term.items() if not str(k).startswith("_")}   # harness-only keys
        if term.get("type") in TM.HARNESS_ONLY:
            raise Refusal(f"term type {term['type']!r} is reserved for the harness")
        term = self.names.translate_term_in(term)
        term.setdefault("name", f"card_{cid}")
        term["name"] = f"card_{_slug(cid)}"
        # Stage 2: compile and run the fixtures on a small sample (FRAMING 7.1).
        from training.scilearn import fixtures
        fx = fixtures.check_term(term, self.meta, self.kit, self.axes)
        h = hashlib.sha256(canonical(card).encode()).hexdigest()
        groups = term_groups(term)
        if not fx["ok"]:
            self.log("reject", card_id=cid, family=card["family"], canonical_hash=h,
                     decision="reject", note=f"fixture: {fx['reason']}")
            cards[cid] = {"status": "rejected_fixture", "hash": h, "reason": fx["reason"]}
            self.save_state()
            return {"card_id": cid, "status": "rejected", "reason": fx["reason"]}
        self.log("card", card_id=cid, family=card["family"], canonical_hash=h,
                 params_json=canonical({"card": card, "compiled_term": term, "groups": groups}))
        cards[cid] = {"status": "registered", "hash": h, "family": card["family"], "term": term,
                      "groups": groups, "expected": card.get("expected_effect"), "tag": card.get("tag")}
        self.save_state()
        return {"card_id": cid, "status": "registered", "hash": h, "cell_groups": groups}

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
        if c["status"] not in ("registered", "near_miss"):
            raise Refusal(f"card {cid} is {c['status']}")
        b = self.budget()
        cost = C.Q2_BITS * (2 if c["status"] == "near_miss" else 1)
        if b["dev_bits"] + cost > C.BUDGET["dev_bits"]:
            raise Refusal("dev bit budget exhausted for this cycle")
        job = harness.submit_q2(self, cid)
        c["status"] = "testing"
        c["job"] = job
        self.save_state()
        threading.Thread(target=self._finish_test, args=(cid,), daemon=True).start()
        return {"card_id": cid, "status": "testing", "note": "poll with `hyp test-status`"}

    def _finish_test(self, cid):
        from training.scilearn import harness
        try:
            harness.finish_q2(self, cid)
        except Exception as e:
            with self.lock:
                self.state["cards"][cid]["status"] = "error"
                self.state["cards"][cid]["error"] = str(e)[:500]
                self.save_state()
            traceback.print_exc()

    def test_status(self):
        out = {}
        for cid, c in self.state.get("cards", {}).items():
            out[cid] = {k: c.get(k) for k in ("status", "released", "reason")}
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

    def handle(self, req):
        op = req.get("op")
        if op == "help":
            return {"ops": ["budget", "query", "card_submit", "card_list", "test", "test_status",
                            "ledger_mine", "meta"], "dev_tranche": self.dev}
        if op == "meta":
            return {"heroes": [self.names.h_out[h] for h in self.meta["heroes"]],
                    "maps": [self.names.m_out[m] for m in self.meta["maps"]],
                    "tiers": self.meta["tiers"], "roles": self.meta["roles"],
                    "hero_roles": {self.names.h_out[h]: self.meta["roles"][r] for h, r in zip(self.meta["heroes"], self.meta["hero_roles"])},
                    "builds": {tr: sorted({self.meta["builds"][b] for b in np.unique(self.g["build"][self.g["tranche"] == T.CODES[tr]])})
                               for tr in ("S", "Dpre", "Dpost")} if self.arm["names"] else "withheld",
                    "kit_axes": self.axes, "dev_tranche": self.dev,
                    "term_types": list(TM.TERM_TYPES)}
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
        raise Refusal(f"unknown op {op!r}")


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


def open_session(sid, arm, variant, cycle=1, incumbent="K0", snapshot=None):
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
    (priv / "state.json").write_text(json.dumps(st, indent=1))
    Ledger(sid).append("session_open", session_id=sid, arm=arm, variant=variant,
                       snapshot_id=st["snapshot"], code_hash=code_hash(),
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
    o.add_argument("--incumbent", default="K0")
    a = ap.parse_args()
    if a.cmd == "serve":
        serve(a.sid)
    else:
        print(json.dumps(open_session(a.sid, a.arm, a.variant, a.cycle, a.incumbent), indent=1))
