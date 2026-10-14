"""Harness side of a Q2 test: incumbent fits, remote screening job, gate, release, ledger,
canaries and safety probes (FRAMING 5.4, 7.1, 8). Never mounted into a sandbox.
"""
import hashlib
import json
import math

import numpy as np
from scipy.special import expit

from training.scilearn import config as C
from training.scilearn import evaluator as E
from training.scilearn import fit as F
from training.scilearn import remote as R
from training.scilearn import terms as TM
from training.scilearn.ledger import Ledger, canonical

REGISTRY = C.PRIVATE / "incumbents.json"
HOST = "max-windows-3090"
REMOTE_SLOTS = 2      # concurrent scilearn jobs on the 3090 (about 16 GB peak each; our share is 60 GB)


class remote_slot:
    """Cross-process semaphore over the broker processes (one flock file per slot)."""

    def __enter__(self):
        import fcntl
        import time as _t
        d = C.PRIVATE / "locks"
        d.mkdir(exist_ok=True)
        while True:
            for i in range(REMOTE_SLOTS):
                f = open(d / f"remote_slot_{i}", "w")
                try:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    self.f = f
                    return self
                except BlockingIOError:
                    f.close()
            _t.sleep(10)

    def __exit__(self, *exc):
        import fcntl
        fcntl.flock(self.f, fcntl.LOCK_UN)
        self.f.close()


def fit_tranches(dev):
    return ["S"] if dev == "Dpre" else ["S", "Dpre", "G"]


def label_mode(state):
    return f"shuffled:{state['label_key']}" if state["arm"] == "shuffled" else "real"


def incumbent_specs(state):
    base = json.loads((C.CODE / "models" / f"{state['incumbent'].lower()}.json").read_text())["terms"]
    return base + list(state.get("accepted_terms", []))


LOCKS = C.PRIVATE / "locks"


class key_lock:
    """Exclusive flock on LOCKS/<name>.lock. Blocks across threads (each acquisition opens
    its own file description) and across broker processes."""

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        import fcntl
        LOCKS.mkdir(parents=True, exist_ok=True)
        self.f = open(LOCKS / f"{self.name}.lock", "w")
        fcntl.flock(self.f, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        import fcntl
        fcntl.flock(self.f, fcntl.LOCK_UN)
        self.f.close()


def _registry():
    return json.loads(REGISTRY.read_text()) if REGISTRY.exists() else {}


def _registry_put(key, entry):
    """Read-modify-write of the shared registry under its own lock, atomic replace."""
    import os
    with key_lock("registry"):
        reg = _registry()
        reg[key] = entry
        tmp = REGISTRY.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(reg, indent=1))
        os.replace(tmp, REGISTRY)


def incumbent_key(state, dev, fit=None):
    return hashlib.sha256(canonical({"specs": incumbent_specs(state), "fit": fit or fit_tranches(dev),
                                     "labels": label_mode(state), "snap": state["snapshot"]}).encode()).hexdigest()[:16]


def ensure_fit(state, dev, fit=None, prefix="inc"):
    """Fit (once) the session's incumbent on `fit` tranches (default: its fit tranches) under
    its label mode, and register it.

    Concurrency: wave 1 lost three cards (w1-K-blinded c03, c05, c13) because two test
    threads that finished after an accept both found the new incumbent missing from the
    registry and both launched `inc_<key>`; hotsjob refused the second launch ("already
    running") and the card ended in "error". Now one fitter per key holds a lock while it
    fits; everyone else waits on the lock and then reads the registry. remote.submit also
    attaches to a job that is already running or finished instead of failing."""
    fit = fit or fit_tranches(dev)
    lm = label_mode(state)
    key = incumbent_key(state, dev, fit)
    reg = _registry()
    if key in reg:
        return reg[key]
    with key_lock(f"{prefix}_{key}"):
        reg = _registry()
        if key in reg:
            return reg[key]
        job = f"{prefix}_{key}"
        evals = [dev] if fit == fit_tranches(dev) else []
        with remote_slot():
            R.submit({"job_id": job, "kind": "fit_eval", "fit_tranches": fit, "eval_tranches": evals,
                      "models": {"INC": incumbent_specs(state)}}, HOST, label_mode=lm)
            res = R.wait(job, HOST)
        entry = {"job": job, "name": "INC", "labels": lm, "dev": dev, "fit": fit,
                 "dev_metrics_private": res["models"]["INC"]["eval"].get(dev)}
        _registry_put(key, entry)
        Ledger("harness").append("fit", session_id=state["session_id"], tranche=",".join(fit),
                                 snapshot_id=state["snapshot"], canonical_hash=key,
                                 note=f"{prefix} fit {job} labels={lm.split(':')[0]}")
    return entry


def ensure_incumbent(state, dev):
    return ensure_fit(state, dev)


def recent_s_builds(meta, k=6):
    from training.scilearn import data as D
    g, _, _, _ = D.load(meta["snap_id"], ["S"])
    b, cnt = np.unique(g["build"], return_counts=True)
    b = b[cnt >= 5000]
    return [int(x) for x in sorted(b)[-k:]]


def submit_q2(broker, cid):
    """Called under the broker lock; the remote work runs in finish_q2's thread."""
    return f"q2_{broker.sid}_{cid}_{hashlib.sha256(cid.encode()).hexdigest()[:6]}"


def finish_q2(broker, cid):
    with broker.lock:   # a stable copy: other test threads may append accepted terms
        st = json.loads(json.dumps(broker.state))
    card = st["cards"][cid]
    dev = broker.dev
    inc = ensure_incumbent(st, dev)
    job = card["job"]
    spec = {"job_id": job, "kind": "q2", "fit_tranches": fit_tranches(dev), "eval_tranches": [],
            "models": {}, "incumbent_job": inc["job"], "incumbent": inc["name"], "term": card["term"],
            "dev_tranche": dev, "recent_s_builds": recent_s_builds(broker.meta)}
    with remote_slot():
        R.submit(spec, HOST, label_mode=label_mode(st))
        res = R.wait(job, HOST)
    with broker.lock:
        apply_gate(broker, cid, res, inc)


def safety_probes(model_inc, term_model, meta, kit=None, n=400, seed=7):
    """Five-tank and no-healer probes (read no outcomes). Returns P(probe team wins)."""
    rng = np.random.default_rng(seed)
    roles = np.asarray(meta["hero_roles"])
    by = {r: np.flatnonzero(roles == i) for i, r in enumerate(meta["roles"])}

    def standard():
        t = [rng.choice(by["Tank"]), rng.choice(by["Bruiser"]), rng.choice(by["Healer"])]
        t += list(rng.choice(np.setdiff1d(by["Ranged Assassin"], t), 2, replace=False))
        return t

    out = {}
    for probe in ("five_tank", "no_healer"):
        H = []
        for _ in range(n):
            b = standard()
            pool_a = by["Tank"] if probe == "five_tank" else np.concatenate([by["Tank"], by["Bruiser"], by["Ranged Assassin"], by["Melee Assassin"]])
            a = list(rng.choice(np.setdiff1d(pool_a, b), 5, replace=False))
            H.append(a + b)
        H = np.array(H)
        g = {"heroes": H, "map": rng.integers(0, len(meta["maps"]), n), "tier": np.ones(n, np.int64),
             "build": np.full(n, len(meta["builds"]) - 1), "game_ts": np.full(n, 1.79e9, np.int64),
             "first_seen_ts": np.full(n, 1.79e9, np.int64), "replay_id": -np.arange(1, n + 1)}
        m = dict(meta)
        ctx = TM.Ctx(g, m, kit=kit, players={"player_key": np.full((n, 10), -1), "party": np.zeros((n, 10), np.int64)})
        e0 = model_inc.logit(ctx)
        e1 = e0 + (term_model.logit(TM.Ctx(g, m, kit=kit, players=ctx.players)) if term_model is not None else 0)
        out[probe] = (float(expit(e0).mean()), float(expit(e1).mean()))
    return out


def apply_gate(broker, cid, res, inc):
    st = broker.state
    card = st["cards"][cid]
    b = broker.budget()
    groups = card["groups"]
    I_g = max([b["groups"].get(g, 0.0) for g in groups] + [0.0])
    pr = res["paired"]
    k = int(res.get("term_k", 1))
    decision, thr, alt, near = E.gate_decisions(pr["z"], I_g, k_params=k, sum2=pr["sum2"])
    # card["status"] is "testing" here; the retry rule needs the status before the test
    retry = card.get("status_before_test") == "near_miss"
    reasons = []
    if not decision:
        reasons.append("z below threshold")
    exp_sign = str((card.get("expected") or {}).get("sign", "+")).strip()
    if pr["gain"] <= 0:
        reasons.append("no gain")
    if k == 1 and exp_sign in ("+", "-"):
        coef = list(res["term_coefs"].values())[0]
        # the card's sign refers to the term's direction for the team that has the feature
        if (coef > 0) != (exp_sign == "+"):
            reasons.append("sign opposite to the card")
    inv = res.get("invariance", {})
    if not inv.get("ok"):
        reasons.append(f"invariance failed ({inv.get('reason') or 'sign or Cochran Q'})")
    # safety probes on the fitted models
    try:
        from training.scilearn import data as D
        kit, axes = D.load_kit(broker.meta)
        meta = dict(broker.meta)
        meta["kit_axes"] = axes
        inc_m = F.Model.load(C.PRIVATE / "jobs" / inc["job"] / "out" / inc["name"], meta)
        term_m = F.Model.load(C.PRIVATE / "jobs" / card["job"] / "out" / "term", meta)
        probes = safety_probes(inc_m, term_m, meta, kit)
        for pname, (p0, p1) in probes.items():
            if p1 > p0 + 0.01:
                reasons.append(f"safety probe {pname} worse ({p0:.3f} -> {p1:.3f})")
    except Exception as e:  # probes must not silently pass
        probes = {"error": str(e)[:200]}
        reasons.append("safety probes could not run")
    flags = ["unexplained"]   # no parsed replays yet, so no mechanism test can pass (M is empty)
    accept = not reasons
    # Release: decision plus the gain rounded to 1e-4 with noise sd = SE (seeded, cached).
    rng = np.random.default_rng(int(hashlib.sha256(f"{broker.keys['noise']}:{card['hash']}".encode()).hexdigest()[:12], 16))
    released = {"card_id": cid, "decision": "accept" if accept else "reject",
                "gain": round((pr["gain"] + rng.normal(0, pr["se"])) / C.Q2_ROUND) * C.Q2_ROUND,
                "near_miss": bool(near and not accept and not retry),
                "flags": flags if accept else []}
    cost = C.Q2_BITS * (2 if retry else 1)
    charges = {g: cost for g in groups}
    broker.log("test", card_id=cid, family=card["family"], template="Q2", tranche=broker.dev,
               params_json=canonical({"_group_charges": charges, "term": card["term"]}),
               canonical_hash=card["hash"], released=released, raw_ref=f"jobs/{card['job']}/out/result.json",
               bits_charged=cost, group_bits_after={g: round(b["groups"].get(g, 0) + cost, 3) for g in groups},
               session_bits_after=round(b["dev_bits"] + cost, 3), gate_z=round(pr["z"], 3),
               threshold=round(thr, 3), decision=released["decision"],
               alt_decisions={**alt, "invariance": inv, "probes": probes, "reasons": reasons, "k": k})
    broker.log("accept" if accept else "reject", card_id=cid, family=card["family"],
               canonical_hash=card["hash"], decision=released["decision"],
               note="; ".join(reasons) if reasons else "accepted (unexplained: no mechanism data)")
    card["released"] = released
    card["status"] = "accepted" if accept else ("near_miss" if released["near_miss"] else "rejected")
    if accept:
        st.setdefault("accepted_terms", []).append(card["term"])
    broker.save_state()


# --------------------------------------------------------------------------- canaries

def canary_cards(sid, axes):
    """6 harness-written cards per session per cycle (FRAMING 8): 4 null, 2 known-true."""
    from training.scilearn import tranches as T
    key = T.load_keys()["canary"]
    k = hashlib.sha256(f"{key}:{sid}".encode()).hexdigest()
    rng = np.random.default_rng(int(k[:12], 16))
    nulls = [{"type": "hash_null", "name": "canary_null_hash_1", "_key": k[:20]},
             {"type": "hash_null", "name": "canary_null_hash_2", "_key": k[20:40]}]
    for i, ax in enumerate(rng.choice(axes, 2, replace=False) if axes else []):
        nulls.append({"type": "attr_sum", "name": f"canary_null_perm_{i + 1}", "axis": str(ax),
                      "shape": "free", "n_knots": 4, "_perm_key": f"{k[40:56]}:{i}"})
    # Known-true: structure the K0 backbone lacks and that is stable across S builds.
    # (No-healer flags are absorbed by K0's healing_throughput shape, so they cannot serve.)
    true = [{"type": "multiset", "name": "canary_true_role_multiset", "min_teams": 200},
            {"type": "pair_family", "name": "canary_true_ally_pairs", "min_games": 2000}]
    return [("null", t) for t in nulls] + [("true", t) for t in true]


def run_canaries(broker, wait=True):
    """Test the session's canaries on its own contract. Logged to the harness ledger only;
    never charged to the session and never shown to it."""
    st = broker.state
    dev = broker.dev
    inc = ensure_incumbent(st, dev)
    # Known-true canaries are scored against the starting backbone, which lacks them; against a
    # cycle-2 incumbent that already holds role-multiset and pair terms they would be redundant.
    inc_true = ensure_incumbent({**st, "accepted_terms": []}, dev)
    hl = Ledger("harness")
    out = []
    jobs = []
    for kind, term in canary_cards(broker.sid, broker.axes):
        job = f"canary_{broker.sid}_{term['name']}"
        ref = inc_true if kind == "true" else inc
        spec = {"job_id": job, "kind": "q2", "fit_tranches": fit_tranches(dev), "eval_tranches": [],
                "models": {}, "incumbent_job": ref["job"], "incumbent": ref["name"], "term": term,
                "dev_tranche": dev, "recent_s_builds": recent_s_builds(broker.meta)}
        jobs.append((kind, term, job, spec))
    for kind, term, job, spec in jobs:
        # one at a time: keeps scilearn RAM on the 3090 well under its 60 GB share
        with remote_slot():
            R.submit(spec, HOST, label_mode=label_mode(st))
            res = R.wait(job, HOST)
        pr = res["paired"]
        k = int(res.get("term_k", 1))
        decision, thr, alt, near = E.gate_decisions(pr["z"], 0.0, k_params=k, sum2=pr["sum2"])
        inv = res.get("invariance", {})
        accept = bool(decision and pr["gain"] > 0 and inv.get("ok"))
        hl.append("canary", session_id=broker.sid, arm=st["arm"], variant=st["variant"], card_id=term["name"],
                  tranche=dev, snapshot_id=st["snapshot"], gate_z=round(pr["z"], 3), threshold=round(thr, 3),
                  decision="accept" if accept else "reject",
                  alt_decisions={**alt, "invariance_ok": inv.get("ok"), "gain": pr["gain"], "se": pr["se"]},
                  note=f"{kind} canary")
        out.append({"canary": term["name"], "kind": kind, "z": pr["z"], "gain": pr["gain"],
                    "invariance_ok": inv.get("ok"), "accept": accept})
    return out


# --------------------------------------------------------------------------- cycle 2: tables

# Table fits for explanation cards. Sessions explore on an S-only fit of their incumbent
# (`hyp table`); explanation cards are scored on a confirm fit on data disjoint from S and
# from the session's dev tranche.
EXPLORE_FIT = ["S"]
CONFIRM_FIT = {"Dpost": ["Dpre", "G"], "Dpre": ["G"]}


def table_term_name(card_id):
    return "card_" + "".join(ch if ch.isalnum() else "_" for ch in str(card_id))[:40]


def accepted_tables(state):
    """{card_id: term spec} for accepted cards whose term is an explainable table."""
    from training.scilearn import explain as X
    names = {t.get("name"): t for t in state.get("accepted_terms", [])}
    out = {}
    for cid, c in state.get("cards", {}).items():
        t = names.get(table_term_name(cid))
        if c.get("status") == "accepted" and t is not None:
            try:
                X.table_kind(t)
                out[cid] = t
            except X.ExplainError:
                pass
    for cid, t in (state.get("inherited_tables") or {}).items():
        if t.get("name") in names:
            out.setdefault(cid, t)
    return out


def _derived_key(*parts):
    from training.scilearn import tranches as T
    k = T.load_keys()
    return hashlib.sha256(":".join([k["canary"], *map(str, parts)]).encode()).hexdigest()[:32]


def finish_explain(broker, cid):
    from training.scilearn import tranches as T
    with broker.lock:
        st = json.loads(json.dumps(broker.state))
    card = st["cards"][cid]
    dev = broker.dev
    inc = ensure_incumbent(st, dev)
    conf = ensure_fit(st, dev, fit=CONFIRM_FIT[dev])
    term = card["term"]
    spec = {"job_id": card["job"], "kind": "explain", "fit_tranches": [], "eval_tranches": [], "models": {},
            "inc_job": inc["job"], "confirm_job": conf["job"], "table_name": table_term_name(term["table"]),
            "theory": {"features": term["features"]}, "dev_tranche": dev,
            "fold_key": _derived_key("explain-fold"), "perm_key": _derived_key("explain-perm", card["hash"]),
            "o3_key": T.load_keys()["o3_heroes"], "n_perm": 200, "lanes2": sorted(TM.LANES2)}
    with remote_slot():
        R.submit(spec, HOST, label_mode=label_mode(st))
        res = R.wait(card["job"], HOST)
    with broker.lock:
        apply_explain_gate(broker, cid, res["explain_result"])


def apply_explain_gate(broker, cid, res):
    from training.scilearn import explain as X
    st = broker.state
    card = st["cards"][cid]
    b = broker.budget()
    groups = card["groups"]
    I_g = max([b["groups"].get(g, 0.0) for g in groups] + [0.0])
    accept, reasons, thr = X.gate(res, I_g)
    rng = np.random.default_rng(int(hashlib.sha256(f"{broker.keys['noise']}:{card['hash']}".encode()).hexdigest()[:12], 16))
    ex, tr, on, pm = res["explain"], res.get("transfer") or {}, res.get("onset") or {}, res.get("permutation") or {}

    def noisy(v, se, step):
        if v is None or not np.isfinite(v):
            return None
        return round(round((v + rng.normal(0, se if np.isfinite(se) else 0)) / step) * step, 4)
    released = {
        "card_id": cid, "decision": "accept" if accept else "reject",
        "explained_share": noisy(ex["share"], ex.get("share_se", 0.0), 0.01),
        "transfer_share": noisy(tr.get("share"), (tr.get("se", 0) / max(abs(tr.get("sum_d", 1e-9)), 1e-9)) * abs(tr.get("share", 0) or 0), 0.01),
        "compresses": bool(ex["bits_saved"] > ex["param_bits"]),
        "onset_gain": noisy(on.get("gain"), on.get("se", 0.0), C.Q2_ROUND),
        "beats_kit_permutations": (None if pm.get("p") is None else bool(pm["p"] <= X.GATE["p_perm"])),
        "reasons": reasons,
    }
    cost = C.Q2_BITS
    charges = {g: cost for g in groups}
    broker.log("test", card_id=cid, family=card["family"], template="Q2E", tranche=broker.dev,
               params_json=canonical({"_group_charges": charges, "term": card["term"]}),
               canonical_hash=card["hash"], released=released, raw_ref=f"jobs/{card['job']}/out/result.json",
               bits_charged=cost, group_bits_after={g: round(b["groups"].get(g, 0) + cost, 3) for g in groups},
               session_bits_after=round(b["dev_bits"] + cost, 3), gate_z=round(ex["z"], 3),
               threshold=round(thr, 3), decision=released["decision"],
               alt_decisions={"explain": ex, "transfer": tr, "onset": on, "permutation": pm,
                              "coefs": res.get("coefs"), "k": res.get("k"), "cells": res.get("cells"),
                              "signal_ss": res.get("signal_ss"), "reasons": reasons})
    broker.log("accept" if accept else "reject", card_id=cid, family=card["family"],
               canonical_hash=card["hash"], decision=released["decision"],
               note=("; ".join(reasons) if reasons else "accepted explanation") + f" table={card['term']['table']}")
    card["released"] = released
    card["status"] = "accepted" if accept else "rejected"
    if accept:
        st.setdefault("explanations", []).append({"card_id": cid, "table": card["term"]["table"],
                                                  "features": card["term"]["features"],
                                                  "coefs": res.get("coefs"), "share": ex["share"],
                                                  "transfer_share": tr.get("share")})
    broker.save_state()


# --------------------------------------------------------------------------- cycle 2: mechanisms

MTEST_BITS = 1.5           # mediator bits per mechanism test (FRAMING 5.3: 60 per cycle)
M_TEST_MIN_GAMES = 2000    # below this M-test is too small for a mechanism test


def m_extract_name():
    from training.scilearn import replay_features as RF
    man = json.loads((RF.STORE / RF.VERSION / "manifest.json").read_text())
    return f"M_{RF.VERSION}_{man['code_hash']}_{man['n_games']}.npz", man


def ship_m_extract(host=HOST):
    """Mediators and draft covariates for M games. No outcomes are shipped: a mechanism
    test regresses mediators on draft-time term values only."""
    from training.scilearn import replay_features as RF
    name, man = m_extract_name()
    p = C.PRIVATE / "extracts" / name
    if not p.exists():
        f = dict(np.load(RF.STORE / RF.VERSION / "features.npz"))
        np.savez(p, **f)
    R.rsync_to(host, p, f"{C.REMOTE_ROOT}/data/")
    return name, man


def finish_mtest(broker, cid):
    with broker.lock:
        st = json.loads(json.dumps(broker.state))
    card = st["cards"][cid]
    mech = card["mechanism"]
    dev = broker.dev
    ms = card["mtest_source"]
    if ms["kind"] == "incumbent":
        inc = ensure_incumbent(st, dev)
        src = {"job": inc["job"], "model": "INC", "term_name": ms["term_name"]}
    else:
        src = {"job": card["job"], "model": "term", "term_name": ms["term_name"]}
    name, _ = ship_m_extract()
    job = f"mtest_{broker.sid}_{cid}_{hashlib.sha256(canonical(mech).encode()).hexdigest()[:6]}"
    spec = {"job_id": job, "kind": "mtest", "fit_tranches": [], "eval_tranches": [], "models": {},
            "m_data": name, "term_source": src, "mediator": mech["mediator"],
            "negative_control": mech.get("negative_control"),
            "shuffle_key": st.get("label_key") if st["arm"] == "shuffled" else None}
    with remote_slot():
        R.submit(spec, HOST, label_mode=label_mode(st))
        res = R.wait(job, HOST)
    with broker.lock:
        apply_mtest_gate(broker, cid, res["mtest_result"], job)


def apply_mtest_gate(broker, cid, res, job):
    st = broker.state
    card = st["cards"][cid]
    mech = card["mechanism"]
    b = broker.budget()
    group = f"mech:{mech['mediator']}"
    I_g = b["groups"].get(group, 0.0)
    thr = E.threshold(I_g)
    md, nc = res.get("mediator", {}), res.get("negative_control")
    direction = str(mech.get("direction", "+")).strip()
    z = float(md.get("z", 0.0))
    signed = z if direction == "+" else -z
    reasons = []
    if signed < thr:
        reasons.append(f"mediator z {signed:.2f} (in the declared direction) below {thr:.2f}")
    if nc is not None and abs(nc.get("z", 0.0)) >= 2.0:
        reasons.append(f"negative control moved (|z| {abs(nc['z']):.2f} >= 2)")
    ok = not reasons
    rng = np.random.default_rng(int(hashlib.sha256(f"{broker.keys['noise']}:mtest:{card['hash']}".encode()).hexdigest()[:12], 16))
    se_std = abs(md.get("std_effect", 0.0)) / max(abs(z), 1.0)
    released = {"card_id": cid, "decision": "mechanism_ok" if ok else "mechanism_not_shown",
                "mediator": mech["mediator"], "n_m_test": res.get("n_m_test"),
                "std_effect": round(round((md.get("std_effect", 0.0) + rng.normal(0, se_std)) / 0.01) * 0.01, 2),
                "negative_control_ok": None if nc is None else bool(abs(nc.get("z", 0.0)) < 2.0),
                "reasons": reasons}
    broker.log("test", card_id=cid, family=card["family"], template="Q3T", tranche="M-test",
               params_json=canonical({"_group_charges": {group: MTEST_BITS}, "_budget": "mediator",
                                      "mechanism": mech}),
               canonical_hash=card["hash"], released=released, raw_ref=f"jobs/{job}/out/result.json",
               bits_charged=MTEST_BITS, gate_z=round(signed, 3), threshold=round(thr, 3),
               decision=released["decision"], alt_decisions=res)
    card["mtest"] = released
    prev = card.pop("status_for_mtest", card.get("status"))
    if card["term"].get("type") == "mechanism":
        card["status"] = released["decision"]
        target = st["cards"].get(card["term"]["of"])
        st.setdefault("mechanism_results", []).append({"card_id": cid, "of": card["term"]["of"], **released})
    else:
        card["status"] = prev
        target = card
    if ok and target is not None and target.get("status") == "accepted" and target.get("released"):
        fl = [f for f in target["released"].get("flags", []) if f != "unexplained"]
        target["released"]["flags"] = fl + ["mechanism_ok"]
    broker.save_state()


# --------------------------------------------------------------------------- cycle 2: canaries

# One fixed kit theory per table type; the canary runs it on a within-role permuted kit
# against the session's real table. It must be rejected (FRAMING 8, null canaries).
CANARY_THEORIES = {
    ("hero_ctx", "map"): [{"kit": "waveclear", "by": "map_class"}, {"kit": "global_presence", "by": "map_class"}],
    ("hero_ctx", "map_class"): [{"kit": "waveclear", "by": "map_class"}],
    ("hero_ctx", "tier"): [{"kit": "early_vs_late", "by": "tier"}],
    ("pair", None): [{"kit_pair": ["hard_cc", "burst"]}, {"kit_sum": "healing_throughput"}],
    ("cross", None): [{"kit_pair": ["hard_cc", "mobility"]}, {"kit_pair": ["burst", "self_sustain_shields"]}],
}


def run_explain_canaries(state, dev, n_per_table=1):
    from training.scilearn import explain as X
    from training.scilearn import tranches as T
    sid = state["session_id"]
    inc = ensure_incumbent(state, dev)
    conf = ensure_fit(state, dev, fit=CONFIRM_FIT[dev])
    out = []
    for cid, t in sorted(accepted_tables(state).items()):
        kind = X.table_kind(t)
        theory = {"features": CANARY_THEORIES[(kind, t["keys"][1] if kind == "hero_ctx" else None)]}
        for i in range(n_per_table):
            job = f"xcanary_{sid}_{cid}_{i}"
            spec = {"job_id": job, "kind": "explain", "fit_tranches": [], "eval_tranches": [], "models": {},
                    "inc_job": inc["job"], "confirm_job": conf["job"], "table_name": t["name"],
                    "theory": theory, "dev_tranche": dev, "fold_key": _derived_key("explain-fold"),
                    "perm_key": _derived_key("xcanary-perm", sid, cid, i), "o3_key": T.load_keys()["o3_heroes"],
                    "n_perm": 200, "lanes2": sorted(TM.LANES2),
                    "kit_perm_key": _derived_key("xcanary-kit", sid, cid, i)}
            with remote_slot():
                R.submit(spec, HOST, label_mode=label_mode(state))
                res = R.wait(job, HOST)["explain_result"]
            accept, reasons, thr = X.gate(res, 0.0)
            Ledger("harness").append(
                "canary", session_id=sid, arm=state["arm"], variant=state["variant"], card_id=f"xcanary_{cid}_{i}",
                tranche=dev, snapshot_id=state["snapshot"], gate_z=round(res["explain"]["z"], 3),
                threshold=round(thr, 3), decision="accept" if accept else "reject",
                alt_decisions={"explain": res["explain"], "transfer": res.get("transfer"), "onset": res.get("onset"),
                               "permutation": res.get("permutation"), "reasons": reasons, "theory": theory},
                note="null explain canary (within-role permuted kit)")
            out.append({"canary": f"xcanary_{cid}_{i}", "z": res["explain"]["z"], "share": res["explain"]["share"],
                        "transfer_z": (res.get("transfer") or {}).get("z"), "perm_p": res["permutation"].get("p"),
                        "onset_z": (res.get("onset") or {}).get("z"), "accept": accept})
    return out
