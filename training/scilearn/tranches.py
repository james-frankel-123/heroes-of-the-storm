"""Tranche manager (FRAMING 4.1, 4.2, 11.2). `tranche(replay_id)` and the keyed hashes.

Keys live only in PRIVATE/keys.json (broker side). The rules never read outcomes.
"""
import datetime as dt
import hashlib
import json
import secrets

from training.scilearn import config as C

CODES = {"S": 0, "Dpre": 1, "G": 2, "Dpost": 3, "M": 4, "pool": 5, "R": 6, "C": 7, "F": 8, "audit": 9}
NAMES = {v: k for k, v in CODES.items()}


def load_keys():
    if not C.KEYS_FILE.exists():
        C.PRIVATE.mkdir(parents=True, exist_ok=True)
        keys = {k: secrets.token_hex(32) for k in ("pool", "m_split", "audit", "o3_heroes", "canary", "blind", "noise")}
        keys["created"] = dt.datetime.now(dt.timezone.utc).isoformat()
        C.KEYS_FILE.write_text(json.dumps(keys, indent=1))
        C.KEYS_FILE.chmod(0o600)
    return json.loads(C.KEYS_FILE.read_text())


def khash(key, rid):
    """Uniform in [0, 1) from a keyed sha256."""
    h = hashlib.sha256(f"{key}:{rid}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def cohort_b_ids():
    ids = set()
    if C.REPLAY_MANIFEST.exists():
        for line in open(C.REPLAY_MANIFEST):
            r = json.loads(line)
            if r.get("cohort") == "stratified_b" and r.get("status") in ("ok", "linked"):
                ids.add(int(r["replay_id"]))
    return ids


def _ts(day):
    return int(dt.datetime.fromisoformat(day).replace(tzinfo=dt.timezone.utc).timestamp())


C_END_TS = _ts(C.C_END_DATE)
F_START_TS = _ts(C.F_BLOCK_START)


def f_block(game_ts):
    return 1 + (game_ts - F_START_TS) // (C.F_BLOCK_DAYS * 86400)


def assign(rid, version, game_ts, *, sealed, cohort_b, keys):
    if version.startswith("2.57."):
        if rid in sealed:
            return "R"
        if khash(keys["pool"], rid) < C.POOL_FRAC:
            return "pool"
        if rid in cohort_b:
            return "M"
        if game_ts < C_END_TS:
            return "C"
        return "audit" if khash(keys["audit"], rid) < C.AUDIT_FRAC else "F"
    if version.startswith("2.55.17."):
        return "Dpost"
    if version in C.G_BUILDS:
        return "G"
    if version.startswith("2.55.15.") or version in C.DPRE_EXTRA_BUILDS:
        return "Dpre"
    parts = tuple(int(x) for x in version.split("."))
    if parts <= (2, 55, 14, 99999):
        return "S"
    raise ValueError(f"build {version} has no tranche rule")


def m_side(rid, keys):
    return "M-explore" if khash(keys["m_split"], rid) < C.M_EXPLORE_FRAC else "M-test"
