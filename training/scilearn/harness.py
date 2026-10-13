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


def fit_tranches(dev):
    return ["S"] if dev == "Dpre" else ["S", "Dpre", "G"]


def label_mode(state):
    return f"shuffled:{state['label_key']}" if state["arm"] == "shuffled" else "real"


def incumbent_specs(state):
    base = json.loads((C.CODE / "models" / f"{state['incumbent'].lower()}.json").read_text())["terms"]
    return base + list(state.get("accepted_terms", []))


def _registry():
    return json.loads(REGISTRY.read_text()) if REGISTRY.exists() else {}


def ensure_incumbent(state, dev, wait=True):
    """Fit (once) the session's incumbent on its fit tranches under its label mode."""
    specs = incumbent_specs(state)
    lm = label_mode(state)
    key = hashlib.sha256(canonical({"specs": specs, "fit": fit_tranches(dev), "labels": lm,
                                    "snap": state["snapshot"]}).encode()).hexdigest()[:16]
    reg = _registry()
    if key in reg:
        return reg[key]
    job = f"inc_{key}"
    R.submit({"job_id": job, "kind": "fit_eval", "fit_tranches": fit_tranches(dev), "eval_tranches": [dev],
              "models": {"INC": specs}}, HOST, label_mode=lm)
    res = R.wait(job, HOST) if wait else None
    reg = _registry()
    reg[key] = {"job": job, "name": "INC", "labels": lm, "dev": dev,
                "dev_metrics_private": res["models"]["INC"]["eval"].get(dev) if res else None}
    REGISTRY.write_text(json.dumps(reg, indent=1))
    Ledger("harness").append("fit", session_id=state["session_id"], tranche=",".join(fit_tranches(dev)),
                             snapshot_id=state["snapshot"], canonical_hash=key,
                             note=f"incumbent fit {job} labels={lm.split(':')[0]}")
    return reg[key]


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
    st = broker.state
    card = st["cards"][cid]
    dev = broker.dev
    inc = ensure_incumbent(st, dev)
    job = card["job"]
    spec = {"job_id": job, "kind": "q2", "fit_tranches": fit_tranches(dev), "eval_tranches": [],
            "models": {}, "incumbent_job": inc["job"], "incumbent": inc["name"], "term": card["term"],
            "dev_tranche": dev, "recent_s_builds": recent_s_builds(broker.meta)}
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
                "near_miss": bool(near and not accept and card["status"] != "near_miss"),
                "flags": flags if accept else []}
    cost = C.Q2_BITS * (2 if card["status"] == "near_miss" else 1)
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
    true = [{"type": "indicator", "name": "canary_true_no_healer", "sign": "-",
             "predicates": [{"field": "role_count", "key": "Healer", "op": "==", "value": 0}]},
            {"type": "indicator", "name": "canary_true_no_healer_lanes3", "sign": "-",
             "predicates": [{"field": "role_count", "key": "Healer", "op": "==", "value": 0},
                            {"field": "map_class", "value": "lanes3"}]}]
    return [("null", t) for t in nulls] + [("true", t) for t in true]


def run_canaries(broker, wait=True):
    """Test the session's canaries on its own contract. Logged to the harness ledger only;
    never charged to the session and never shown to it."""
    st = broker.state
    dev = broker.dev
    inc = ensure_incumbent(st, dev)
    hl = Ledger("harness")
    out = []
    jobs = []
    for kind, term in canary_cards(broker.sid, broker.axes):
        job = f"canary_{broker.sid}_{term['name']}"
        spec = {"job_id": job, "kind": "q2", "fit_tranches": fit_tranches(dev), "eval_tranches": [],
                "models": {}, "incumbent_job": inc["job"], "incumbent": inc["name"], "term": term,
                "dev_tranche": dev, "recent_s_builds": recent_s_builds(broker.meta)}
        R.submit(spec, HOST, label_mode=label_mode(st))
        jobs.append((kind, term, job))
    for kind, term, job in jobs:
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
