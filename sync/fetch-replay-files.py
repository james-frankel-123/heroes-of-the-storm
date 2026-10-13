"""
Download full .StormReplay files from Heroes Profile into cohorts on the local RAID.

Layout under /home/max/hots-replays (override with HOTS_REPLAYS_ROOT):

  raw/<build>/<replayID>.StormReplay      every non-sealed cohort
  sealed/files/<build>/<replayID>.StormReplay   cohort a, fetched and checksummed ONLY
  sealed/universe.jsonl, sealed/sealed_ids.txt  frozen at freeze time
  manifest.jsonl     append-only: one line per download attempt or cohort link
  state.json         last seen quota headers, per-window counters
  backup_state.json  which manifest lines are already in a NAS bundle

Rules for every cohort live in training/replays/cohorts/ (SELECTION_RULES.md is the
frozen text, cohorts.json is the live config). Sealed IDs are excluded from every
other cohort in code, not by convention.

Commands (source .env first; DATABASE_URL_RESEARCH = local research role):
  python3 sync/fetch-replay-files.py freeze-sealed   one-time; refuses if already frozen
  python3 sync/fetch-replay-files.py plan            show what the next run would fetch
  python3 sync/fetch-replay-files.py run [--cron] [--max N]
  python3 sync/fetch-replay-files.py status
  python3 sync/fetch-replay-files.py verify [--sealed]   re-hash files against the manifest
  python3 sync/fetch-replay-files.py backup           weekly tar bundle to /archive-backup
  python3 sync/fetch-replay-files.py probe-retention  find the oldest downloadable replayID

Quota (replay_download pool): 25,000 per rolling week, 120 per minute. Every response
carries x-hp-quota-remaining / x-hp-quota-reset; we stop when remaining <= margin.
403 (outside retention / unavailable) and 404 are recorded as terminal and never retried.
"""
import argparse
import datetime as dt
import hashlib
import heapq
import json
import os
import signal
import sys
import tarfile
import time

import requests

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.environ.get("HOTS_REPLAYS_ROOT", "/home/max/hots-replays")
COHORT_DIR = os.path.join(REPO, "training", "replays", "cohorts")
CONFIG = os.path.join(COHORT_DIR, "cohorts.json")
FROZEN = os.path.join(COHORT_DIR, "sealed_rule.json")
SEALED_IDS_REPO = os.path.join(COHORT_DIR, "sealed_ids.txt")
MANIFEST = os.path.join(ROOT, "manifest.jsonl")
STATE = os.path.join(ROOT, "state.json")
SEALED_DIR = os.path.join(ROOT, "sealed")
NAS = "/archive-backup"
NAS_ROOT = os.path.join(NAS, "hots-replays")

API = "https://www.heroesprofile.com/api/external/v1"
MIN_INTERVAL = 0.55          # ~109/min, under the 120/min limit
TERMINAL = {"http_403", "http_404"}
MAGIC = b"MPQ\x1b"           # MPQ user-data header that starts every .StormReplay

_stop = False


def _on_term(signum, frame):
    global _stop
    _stop = True


signal.signal(signal.SIGTERM, _on_term)
signal.signal(signal.SIGINT, _on_term)


def log(msg):
    print(f"[{dt.datetime.now(dt.timezone.utc):%Y-%m-%dT%H:%M:%SZ}] {msg}", flush=True)


def now_iso():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def salted_rank(salt, replay_id):
    """Deterministic pseudo-random rank of a replayID under a salt (sha256 hex)."""
    return hashlib.sha256(f"{salt}:{int(replay_id)}".encode()).hexdigest()


def build_of(version):
    return str(version).rsplit(".", 1)[-1] if version else "unknown"


def load_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


def save_json(path, obj):
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(path + ".tmp", path)


# ---------------------------------------------------------------- manifest

def read_manifest():
    rows = []
    if os.path.exists(MANIFEST):
        with open(MANIFEST) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def append_manifest(row):
    with open(MANIFEST, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


class Ledger:
    """Folded view of manifest.jsonl."""

    def __init__(self, rows):
        self.ok = {}            # replay_id -> row of the verified download
        self.terminal = {}      # replay_id -> status (403/404)
        self.members = {}       # cohort -> set(replay_id) (downloaded or linked)
        self.window = {}        # window_id -> cohort -> count of quota-charged calls
        for r in rows:
            rid = r["replay_id"]
            if r["status"] == "ok":
                self.ok[rid] = r
                self.members.setdefault(r["cohort"], set()).add(rid)
            elif r["status"] == "linked":
                self.members.setdefault(r["cohort"], set()).add(rid)
            elif r["status"] in TERMINAL:
                self.terminal[rid] = r["status"]
            if r.get("charged") and r.get("window"):
                w = self.window.setdefault(r["window"], {})
                w[r["cohort"]] = w.get(r["cohort"], 0) + 1


# ---------------------------------------------------------------- API

class Api:
    def __init__(self):
        key = os.environ.get("HEROES_PROFILE_V2_API_KEY")
        if not key:
            sys.exit("HEROES_PROFILE_V2_API_KEY not set (source .env)")
        self.s = requests.Session()
        self.s.headers["Authorization"] = f"Bearer {key}"
        self.last = 0.0
        self.quota_remaining = None
        self.quota_reset = None   # seconds until the rolling window frees calls
        self.last_429 = ""

    def _pace(self, interval):
        wait = self.last + interval - time.time()
        if wait > 0:
            time.sleep(wait)
        self.last = time.time()

    def listing(self, after, **params):
        for attempt in range(6):
            self._pace(0.35)
            r = self.s.get(f"{API}/replays", params={"after": after, **params}, timeout=60)
            if r.status_code == 429:
                time.sleep(int(r.headers.get("Retry-After", "30")) + 1)
                continue
            if r.status_code >= 500:
                time.sleep(10 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError(f"listing after={after} failed repeatedly")

    def download(self, replay_id, dest_part):
        """Stream one replay to dest_part. Returns (status, charged, size)."""
        self._pace(MIN_INTERVAL)
        r = self.s.get(f"{API}/download/replay", params={"replayID": replay_id},
                       timeout=120, stream=True)
        h = r.headers
        if "x-hp-quota-remaining" in h:
            self.quota_remaining = int(h["x-hp-quota-remaining"])
        if "x-hp-quota-reset" in h:
            self.quota_reset = int(h["x-hp-quota-reset"])
        if r.status_code == 200:
            n = 0
            with open(dest_part, "wb") as f:
                for chunk in r.iter_content(1 << 16):
                    f.write(chunk)
                    n += len(chunk)
            return "ok", True, n
        body = r.content[:300]
        r.close()
        if r.status_code == 429:
            self.last_429 = body.decode("utf-8", "replace")[:200]
            return "http_429", False, int(h.get("Retry-After", "60"))
        # 403/404/500 observed uncharged (quota header did not move on 403s)
        return f"http_{r.status_code}", False, body.decode("utf-8", "replace")


def window_key(state, reset=None):
    """Identify the quota window by its end time. The pool is a 7-day window that starts at
    the first call, so now + x-hp-quota-reset is stable within a window; jitter under an
    hour keeps the stored key."""
    now = dt.datetime.now(dt.timezone.utc)
    stored = state.get("window_end")
    if reset is not None:
        end = now + dt.timedelta(seconds=reset)
        if not stored or abs((dt.datetime.fromisoformat(stored) - end).total_seconds()) > 3600:
            state["window_end"] = end.isoformat(timespec="seconds")
        return state["window_end"]
    if stored and dt.datetime.fromisoformat(stored) > now:
        return stored
    return "pending"


# ---------------------------------------------------------------- DB

def db():
    import psycopg2
    url = os.environ.get("DATABASE_URL_RESEARCH")
    if not url or "neon" in url:
        sys.exit("DATABASE_URL_RESEARCH must point at the LOCAL store (never Neon)")
    c = psycopg2.connect(url)
    c.set_session(readonly=True)
    return c


# ---------------------------------------------------------------- cohorts

def load_config():
    cfg = load_json(CONFIG)
    if cfg is None:
        sys.exit(f"missing {CONFIG}")
    today = dt.date.today().isoformat()
    caps = dict(cfg.get("window_caps", {}))
    total = cfg["window_total_cap"]
    for s in cfg.get("schedule", []):
        if today < s["until"]:
            caps.update(s.get("window_caps", {}))
            total = s.get("window_total_cap", total)
            break
    return cfg, caps, total


def sealed_ids():
    p = os.path.join(SEALED_DIR, "sealed_ids.txt")
    src = p if os.path.exists(p) else SEALED_IDS_REPO
    if not os.path.exists(src):
        return set()
    with open(src) as f:
        return {int(x) for x in f if x.strip()}


def candidates_sealed(ledger):
    """Sealed IDs not yet fetched, oldest first (they expire first)."""
    meta = {}
    with open(os.path.join(SEALED_DIR, "universe.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            meta[r["replayID"]] = r
    for rid in sorted(sealed_ids()):
        if rid in ledger.ok or rid in ledger.terminal:
            continue
        m = meta[rid]
        yield {"replay_id": rid, "game_version": m["game_version"], "game_map": m["game_map"],
               "game_date": m["game_date"], "region": m["region"], "league_tier": None,
               "skill_tier": None, "reason": "sealed: lowest salted hash in frozen universe"}


def candidates_stratified(cohort, ledger, exclude):
    """Balanced over (league_tier, map) cells: always take the next item from the cell
    with the fewest members so far; inside a cell, order by salted hash."""
    conn = db()
    cur = conn.cursor()
    cur.execute("""SELECT replay_id, game_version, game_map, game_date, region, league_tier, skill_tier
                   FROM replay_draft_data WHERE game_version LIKE %s""", (cohort["patch_prefix"] + "%",))
    rows = cur.fetchall()
    conn.close()
    cells = {}
    have = ledger.members.get(cohort["name"], set())
    counts = {}
    for rid, ver, mp, date, region, lt, st in rows:
        key = (str(lt) if lt is not None else "NULL", mp)
        if rid in have:
            counts[key] = counts.get(key, 0) + 1
            continue
        if rid in exclude or rid in ledger.terminal:
            continue
        cells.setdefault(key, []).append(
            (salted_rank(cohort["salt"], rid),
             {"replay_id": rid, "game_version": ver, "game_map": mp, "game_date": str(date),
              "region": region, "league_tier": lt, "skill_tier": st}))
    heap = []
    for key, items in cells.items():
        items.sort(key=lambda x: x[0], reverse=True)   # pop() from the end = lowest hash
        tie = hashlib.sha256(f"{cohort['salt']}:{key[0]}:{key[1]}".encode()).hexdigest()
        heapq.heappush(heap, (counts.get(key, 0), tie, key))
    while heap:
        n, tie, key = heapq.heappop(heap)
        items = cells[key]
        if not items:
            continue
        _, c = items.pop()
        c["reason"] = f"stratified cell tier={key[0]} map={key[1]} rank={n}"
        yield c
        if items:
            heapq.heappush(heap, (n + 1, tie, key))


def candidates_query(cohort, ledger, exclude):
    """Cohort c: a SQL query (must return replay_id, may return a 'reason' column) or a
    file of replayIDs, one per line, optional tab-separated reason."""
    meta_sql = """SELECT replay_id, game_version, game_map, game_date, region, league_tier, skill_tier
                  FROM replay_draft_data WHERE replay_id = ANY(%s)"""
    picks = []
    if "sql" in cohort:
        conn = db()
        cur = conn.cursor()
        cur.execute(cohort["sql"])
        cols = [d[0] for d in cur.description]
        for row in cur.fetchall():
            r = dict(zip(cols, row))
            picks.append((int(r["replay_id"]), r.get("reason")))
        conn.close()
    else:
        path = cohort["ids_file"]
        if not os.path.isabs(path):
            path = os.path.join(COHORT_DIR, path)
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    parts = line.split("\t", 1)
                    picks.append((int(parts[0]), parts[1] if len(parts) > 1 else None))
    have = ledger.members.get(cohort["name"], set())
    picks = [(rid, why) for rid, why in picks
             if rid not in have and rid not in exclude and rid not in ledger.terminal]
    conn = db()
    cur = conn.cursor()
    ids = [p[0] for p in picks]
    meta = {}
    for i in range(0, len(ids), 10000):
        cur.execute(meta_sql, (ids[i:i + 10000],))
        for rid, ver, mp, date, region, lt, st in cur.fetchall():
            meta[rid] = {"game_version": ver, "game_map": mp, "game_date": str(date),
                         "region": region, "league_tier": lt, "skill_tier": st}
    conn.close()
    for rid, why in picks:
        m = meta.get(rid, {"game_version": None, "game_map": None, "game_date": None,
                           "region": None, "league_tier": None, "skill_tier": None})
        yield {"replay_id": rid, **m, "reason": why or cohort.get("reason", cohort["name"])}


def cohort_iter(cohort, ledger, sealed):
    if cohort["kind"] == "sealed":
        return candidates_sealed(ledger)
    if cohort["kind"] == "stratified":
        return candidates_stratified(cohort, ledger, sealed)
    if cohort["kind"] in ("sql", "ids"):
        return candidates_query(cohort, ledger, sealed)
    raise ValueError(f"unknown cohort kind {cohort['kind']}")


# ---------------------------------------------------------------- run

def file_path(cohort_kind, game_version, replay_id):
    base = os.path.join(SEALED_DIR, "files") if cohort_kind == "sealed" else os.path.join(ROOT, "raw")
    return os.path.join(base, build_of(game_version), f"{replay_id}.StormReplay")


def run(args):
    os.makedirs(ROOT, exist_ok=True)
    cfg, caps, total_cap = load_config()
    margin = cfg["quota_margin"]
    ledger = Ledger(read_manifest())
    sealed = sealed_ids()
    api = Api()
    state = load_json(STATE, {})
    fetched = 0
    t_start = time.time()
    deadline = t_start + args.max_minutes * 60 if args.max_minutes else None

    # one cheap probe call isn't available for this pool, so the first download tells us
    # the quota; until then trust the last saved state.
    if state.get("quota_remaining") is not None and state.get("quota_seen_at"):
        seen = dt.datetime.fromisoformat(state["quota_seen_at"])
        freed = (dt.datetime.now(dt.timezone.utc) - seen).total_seconds() >= state.get("quota_reset", 0)
        if not freed and state["quota_remaining"] <= margin:
            log(f"quota benched: remaining {state['quota_remaining']} <= margin {margin}; "
                f"window frees in {state['quota_reset']}s from {state['quota_seen_at']}")
            return

    cohorts = sorted(cfg["cohorts"], key=lambda c: c["priority"])
    for cohort in cohorts:
        if _stop:
            break
        if not cohort.get("enabled", True):
            continue
        name = cohort["name"]
        if cohort["kind"] == "sealed" and not os.path.exists(FROZEN):
            log("sealed cohort not frozen yet; run freeze-sealed first")
            return
        exclude = sealed if cohort["kind"] != "sealed" else set()
        log(f"cohort {name}: building candidate list")
        for c in cohort_iter(cohort, ledger, exclude):
            if _stop:
                break
            if deadline and time.time() > deadline:
                log("time budget reached")
                return finish(state, api, fetched)
            if args.max and fetched >= args.max:
                log(f"--max {args.max} reached")
                return finish(state, api, fetched)
            wid = window_key(state, api.quota_reset)
            used = ledger.window.get(wid, {})
            if sum(used.values()) >= total_cap:
                log(f"window total cap {total_cap} reached")
                return finish(state, api, fetched)
            cap = caps.get(name)
            if cap is not None and used.get(name, 0) >= cap:
                log(f"cohort {name}: window cap {cap} reached")
                break
            if cohort.get("target") and len(ledger.members.get(name, ())) >= cohort["target"]:
                log(f"cohort {name}: target {cohort['target']} reached")
                break
            if api.quota_remaining is not None and api.quota_remaining <= margin:
                log(f"quota margin reached: remaining {api.quota_remaining}")
                return finish(state, api, fetched)

            rid = c["replay_id"]
            if rid in ledger.ok:                       # fetched under another cohort
                if cohort["kind"] == "sealed":
                    log(f"WARNING sealed id {rid} already fetched as non-sealed; skipping link")
                    continue
                row = {**{k: c[k] for k in c}, "cohort": name, "status": "linked",
                       "path": ledger.ok[rid]["path"], "sha256": ledger.ok[rid]["sha256"],
                       "size": ledger.ok[rid]["size"], "fetched_at": now_iso()}
                append_manifest(row)
                ledger.members.setdefault(name, set()).add(rid)
                continue

            dest = file_path(cohort["kind"], c["game_version"], rid)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            part = dest + ".part"
            for attempt in range(4):
                try:
                    status, charged, info = api.download(rid, part)
                except requests.RequestException as e:
                    status, charged, info = "net_error", False, str(e)[:200]
                if status == "http_429":
                    log(f"429: sleeping {info}s (quota remaining {api.quota_remaining}): {api.last_429}")
                    if "quota" in api.last_429.lower() and "rate" not in api.last_429.lower():
                        return finish(state, api, fetched)
                    if api.quota_remaining is not None and api.quota_remaining <= 0:
                        return finish(state, api, fetched)
                    time.sleep(int(info) + 1)
                    continue
                if status in ("http_500", "http_502", "http_503", "http_504", "net_error"):
                    time.sleep(15 * (attempt + 1))
                    continue
                break
            wid = window_key(state, api.quota_reset)
            row = {**c, "cohort": name, "status": status, "charged": charged,
                   "window": wid, "fetched_at": now_iso(), "build": build_of(c["game_version"])}
            if status == "ok":
                size = info
                with open(part, "rb") as f:
                    head = f.read(4)
                if head != MAGIC or size < 10_000:
                    row["status"] = "bad_file"
                    row["size"] = size
                    os.remove(part)
                else:
                    os.replace(part, dest)
                    row.update(size=size, sha256=sha256_file(dest), path=os.path.relpath(dest, ROOT))
                    ledger.ok[rid] = row
                    ledger.members.setdefault(name, set()).add(rid)
                    fetched += 1
            else:
                row["detail"] = info if isinstance(info, str) else None
                if os.path.exists(part):
                    os.remove(part)
                if status in TERMINAL:
                    ledger.terminal[rid] = status
            if charged:
                w = ledger.window.setdefault(wid, {})
                w[name] = w.get(name, 0) + 1
            append_manifest(row)
            if fetched and fetched % 250 == 0:
                rate = fetched / (time.time() - t_start) * 60
                log(f"{fetched} fetched this run ({rate:.0f}/min), quota remaining {api.quota_remaining}")
                save_quota(state, api)
    return finish(state, api, fetched)


def save_quota(state, api):
    if api.quota_remaining is not None:
        state["quota_remaining"] = api.quota_remaining
        state["quota_reset"] = api.quota_reset
        state["quota_seen_at"] = now_iso()
    save_json(STATE, state)


def finish(state, api, fetched):
    save_quota(state, api)
    log(f"run done: {fetched} files fetched, quota remaining {api.quota_remaining}, "
        f"window frees in {api.quota_reset}s")


# ---------------------------------------------------------------- freeze

def freeze_sealed(args):
    if os.path.exists(FROZEN):
        sys.exit(f"{FROZEN} exists: the sealed set is frozen and cannot be redrawn")
    cfg, _, _ = load_config()
    sc = next(c for c in cfg["cohorts"] if c["kind"] == "sealed")
    api = Api()
    os.makedirs(SEALED_DIR, exist_ok=True)
    prefix = sc["patch_prefix"]
    after = sc["universe_scan_from"]
    first = api.listing(after, game_type="Storm League")
    freeze_max = first["max_replay_id"]
    universe = []
    page = first
    calls = 1
    while True:
        for r in page["replays"]:
            if r["replayID"] > freeze_max:
                continue
            if (r["game_type"] == "Storm League" and str(r["game_version"]).startswith(prefix)
                    and not r["deleted"] and r["downloadable"]):
                universe.append({k: r[k] for k in ("replayID", "region", "game_version",
                                                   "game_map", "game_date")})
        nxt = page.get("next_after")
        if not nxt or nxt >= freeze_max or not page["replays"]:
            break
        page = api.listing(nxt, game_type="Storm League")
        calls += 1
        if calls % 20 == 0:
            log(f"listing: {calls} pages, after={nxt}, universe so far {len(universe)}")
    universe.sort(key=lambda r: r["replayID"])
    salt = os.urandom(32).hex()
    ranked = sorted(universe, key=lambda r: salted_rank(salt, r["replayID"]))
    chosen = sorted(r["replayID"] for r in ranked[:sc["target"]])
    with open(os.path.join(SEALED_DIR, "universe.jsonl"), "w") as f:
        for r in universe:
            f.write(json.dumps(r) + "\n")
    ids_txt = "".join(f"{i}\n" for i in chosen)
    for p in (os.path.join(SEALED_DIR, "sealed_ids.txt"), SEALED_IDS_REPO):
        with open(p, "w") as f:
            f.write(ids_txt)
    uni_txt = "".join(f"{r['replayID']}\n" for r in universe)
    frozen = {
        "frozen_at": now_iso(),
        "salt": salt,
        "rule": ("universe = every replay in the Heroes Profile /replays listing with "
                 f"game_type == 'Storm League', game_version starting '{prefix}', deleted == 0, "
                 f"downloadable == true, and replayID in ({sc['universe_scan_from']}, {freeze_max}]. "
                 "rank(id) = sha256(f'{salt}:{id}') as hex; sealed = the "
                 f"{sc['target']} universe members with the lowest rank."),
        "universe_scan_from_exclusive": sc["universe_scan_from"],
        "freeze_max_replay_id": freeze_max,
        "listing_calls": calls,
        "universe_size": len(universe),
        "universe_ids_sha256": hashlib.sha256(uni_txt.encode()).hexdigest(),
        "sealed_size": len(chosen),
        "sealed_ids_sha256": hashlib.sha256(ids_txt.encode()).hexdigest(),
    }
    save_json(FROZEN, frozen)
    save_json(os.path.join(SEALED_DIR, "sealed_rule.json"), frozen)
    log(f"frozen: universe {len(universe)} games, sealed {len(chosen)}; rule in {FROZEN}")


# ---------------------------------------------------------------- status / verify

def status(args):
    rows = read_manifest()
    ledger = Ledger(rows)
    state = load_json(STATE, {})
    by = {}
    for r in rows:
        k = (r["cohort"], r["status"])
        by[k] = by.get(k, 0) + 1
    for (c, s), n in sorted(by.items()):
        print(f"{c:24s} {s:10s} {n:7d}")
    gb = sum(r.get("size", 0) for r in ledger.ok.values()) / 1e9
    print(f"files on disk: {len(ledger.ok)} ({gb:.2f} GB); terminal failures: {len(ledger.terminal)}")
    print(f"quota: {state.get('quota_remaining')} remaining as of {state.get('quota_seen_at')}, "
          f"frees in {state.get('quota_reset')}s from then")
    for w, d in sorted(ledger.window.items())[-3:]:
        print(f"window ending ~{w}: {d}")


def verify(args):
    ledger = Ledger(read_manifest())
    bad = 0
    n = 0
    for rid, r in sorted(ledger.ok.items()):
        is_sealed = r["cohort"] == "sealed"
        if is_sealed != bool(args.sealed) and not args.all:
            continue
        p = os.path.join(ROOT, r["path"])
        n += 1
        if not os.path.exists(p) or os.path.getsize(p) != r["size"] or sha256_file(p) != r["sha256"]:
            bad += 1
            print(f"MISMATCH {rid} {p}")
    log(f"verified {n} files, {bad} mismatches")
    if bad:
        sys.exit(1)


def plan(args):
    cfg, caps, total = load_config()
    ledger = Ledger(read_manifest())
    sealed = sealed_ids()
    print(f"window total cap {total}, caps {caps}, margin {cfg['quota_margin']}")
    for cohort in sorted(cfg["cohorts"], key=lambda c: c["priority"]):
        if not cohort.get("enabled", True):
            print(f"{cohort['name']}: disabled")
            continue
        if cohort["kind"] == "sealed" and not os.path.exists(FROZEN):
            print("sealed: not frozen")
            continue
        it = cohort_iter(cohort, ledger, sealed if cohort["kind"] != "sealed" else set())
        cand = []
        for c in it:
            cand.append(c)
            if len(cand) >= args.n:
                break
        have = len(ledger.members.get(cohort["name"], ()))
        print(f"{cohort['name']}: have {have}, next {len(cand)} (showing up to {args.n})")
        cells = {}
        for c in cand:
            k = (c["league_tier"], c["game_map"])
            cells[k] = cells.get(k, 0) + 1
        if cohort["kind"] == "stratified":
            print(f"  cells touched {len(cells)}, min/max per cell {min(cells.values(), default=0)}/"
                  f"{max(cells.values(), default=0)}")


# ---------------------------------------------------------------- backup

def backup(args):
    """One tar per run (weekly) with every file fetched since the last bundle, plus the
    full manifest and frozen files. Sealed files go into a separate sealed_*.tar."""
    if not os.path.ismount(NAS):
        sys.exit(f"{NAS} is not mounted")
    bstate_p = os.path.join(ROOT, "backup_state.json")
    bstate = load_json(bstate_p, {"bundled": []})
    done = set(bstate["bundled"])
    ledger = Ledger(read_manifest())
    todo = {"raw": [], "sealed": []}
    for rid, r in sorted(ledger.ok.items()):
        if r["path"] in done:
            continue
        todo["sealed" if r["cohort"] == "sealed" else "raw"].append(r)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    os.makedirs(NAS_ROOT, exist_ok=True)
    meta_files = [MANIFEST, STATE, os.path.join(SEALED_DIR, "sealed_rule.json"),
                  os.path.join(SEALED_DIR, "sealed_ids.txt"), os.path.join(SEALED_DIR, "universe.jsonl"),
                  os.path.join(SEALED_DIR, "README.md")]
    for kind, rows in todo.items():
        if not rows and kind == "sealed":
            continue
        final = os.path.join(NAS_ROOT, f"{kind}_{stamp}.tar")
        partial = final + ".partial"
        manifest = {"kind": kind, "stamp": stamp, "files": len(rows),
                    "bytes": sum(r["size"] for r in rows),
                    "members": [{"path": r["path"], "replay_id": r["replay_id"], "cohort": r["cohort"],
                                 "sha256": r["sha256"], "size": r["size"]} for r in rows]}
        mpath = os.path.join(ROOT, f".bundle_manifest_{kind}.json")
        save_json(mpath, manifest)
        with tarfile.open(partial, "w") as tar:      # .StormReplay is already compressed
            tar.add(mpath, arcname="bundle_manifest.json")
            for p in meta_files:
                if os.path.exists(p):
                    tar.add(p, arcname="meta/" + os.path.relpath(p, ROOT))
            for r in rows:
                src = os.path.join(ROOT, r["path"])
                if sha256_file(src) != r["sha256"]:      # verify each file as it is bundled
                    raise RuntimeError(f"sha256 mismatch for {src}; bundle aborted")
                tar.add(src, arcname=r["path"])
        os.rename(partial, final)
        os.remove(mpath)
        side = {"archive": os.path.basename(final), "sha256": sha256_file(final),
                "size": os.path.getsize(final), "files": len(rows), "stamp": stamp}
        save_json(final + ".json", side)
        done.update(r["path"] for r in rows)
        bstate["bundled"] = sorted(done)
        save_json(bstate_p, bstate)
        log(f"bundle {final}: {len(rows)} files, {side['size']/1e9:.2f} GB, sha256 {side['sha256'][:12]}")


# ---------------------------------------------------------------- retention probe

def probe_retention(args):
    """Binary search on the listing's `downloadable` flag (index pool, no download quota)."""
    api = Api()
    top = api.listing(0)["max_replay_id"]
    lo, hi = args.lo, top
    while hi - lo > 1000:
        mid = (lo + hi) // 2
        rows = api.listing(mid)["replays"]
        frac = sum(r["downloadable"] for r in rows) / max(1, len(rows))
        if frac > 0.5:
            hi = mid
        else:
            lo = mid
    rows = api.listing(lo)["replays"]
    first = next((r for r in rows if r["downloadable"]), None)
    out = {"probed_at": now_iso(), "max_replay_id": top,
           "oldest_downloadable_id": first and first["replayID"],
           "oldest_downloadable_neighbour_game_date": first and first["game_date"]}
    print(json.dumps(out))
    os.makedirs(ROOT, exist_ok=True)
    with open(os.path.join(ROOT, "retention_probes.jsonl"), "a") as f:
        f.write(json.dumps(out) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--cron", action="store_true", help="non-interactive; same behaviour")
    r.add_argument("--max", type=int, default=0, help="stop after N files")
    r.add_argument("--max-minutes", type=float, default=0)
    sub.add_parser("freeze-sealed")
    sub.add_parser("status")
    v = sub.add_parser("verify")
    v.add_argument("--sealed", action="store_true")
    v.add_argument("--all", action="store_true")
    p = sub.add_parser("plan")
    p.add_argument("-n", type=int, default=20000)
    sub.add_parser("backup")
    pr = sub.add_parser("probe-retention")
    pr.add_argument("--lo", type=int, default=60_000_000)
    args = ap.parse_args()
    {"run": run, "freeze-sealed": freeze_sealed, "status": status, "verify": verify,
     "plan": plan, "backup": backup, "probe-retention": probe_retention}[args.cmd](args)


if __name__ == "__main__":
    main()
