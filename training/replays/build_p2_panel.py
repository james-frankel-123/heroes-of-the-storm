"""
Daily builder for replay-file cohort p2_panel_v1 (SELECTION_RULES.md C.2).

Picks new Storm League games (2.57 or later) whose lobbies hold the most panel players with
a full Heroes Profile history pulled, and appends them to interesting/p2_panel_v1.txt, which
sync/fetch-replay-files.py downloads in file order. Selection uses panel coverage only:
no outcome, MMR, tier, map or length is read. Sealed IDs are excluded inside the SQL, so
their rows are never read.

  python3 training/replays/build_p2_panel.py            add today's batch (once per UTC day)
  python3 training/replays/build_p2_panel.py --dry-run  show what today's batch would be

Cron (daily, nice 19, cores 48-63; DATABASE_URL_RESEARCH = local research role):
  45 7 * * * cd /home/max/heroes-of-the-storm && set -a && . ./.env && set +a && nice -n 19 \
    taskset -c 48-51 /home/linuxbrew/.linuxbrew/bin/python3 training/replays/build_p2_panel.py \
    >> sync/logs/p2-panel-build.log 2>&1
"""
import argparse
import collections
import datetime as dt
import glob
import hashlib
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COHORT_DIR = os.path.join(REPO, "training", "replays", "cohorts")
IDS_FILE = os.path.join(COHORT_DIR, "interesting", "p2_panel_v1.txt")
BATCH_LOG = os.path.join(COHORT_DIR, "interesting", "p2_panel_v1_batches.jsonl")
PANEL_DIR = os.path.join(REPO, "training", "personalization", "cache", "mmr_history")
ROOT = os.environ.get("HOTS_REPLAYS_ROOT", "/home/max/hots-replays")
MANIFEST = os.path.join(ROOT, "manifest.jsonl")
SEALED_FILES = [os.path.join(ROOT, "sealed", "sealed_ids.txt"), os.path.join(COHORT_DIR, "sealed_ids.txt")]

# Frozen parameters (SELECTION_RULES.md C.2). Changing any of them means a new cohort name.
COHORT = "p2_panel_v1"
MIN_VERSION = (2, 57)
LOOKBACK_DAYS = 21       # game_date in [day - 21, day): upload >= game date, so >= 7 days of retention left
MIN_K = 3                # at least 3 of the lobby's players covered
DAILY = 3300             # about 23,000 a week (full replay quota from 2026-10-24; was 1,140)
FIRST_BATCH = 2400       # first batch also uses what is left of the window ending 2026-10-20
MAX_CATCHUP_DAYS = 3


def log(msg):
    print(f"[{dt.datetime.now(dt.timezone.utc):%Y-%m-%dT%H:%M:%SZ}] {msg}", flush=True)


def version_ok(v):
    try:
        parts = tuple(int(x) for x in str(v).split(".")[:2])
    except ValueError:
        return False
    return parts >= MIN_VERSION


def full_history_keys():
    """(region, blizz_id) in any panel*.tsv with a status 'ok' row in done.jsonl."""
    panel = set()
    files = sorted(glob.glob(os.path.join(PANEL_DIR, "panel*.tsv")))
    for f in files:
        with open(f) as fh:
            head = fh.readline().rstrip("\n").split("\t")
            ri, bi = head.index("region"), head.index("blizz_id")
            for line in fh:
                p = line.rstrip("\n").split("\t")
                if len(p) > max(ri, bi):
                    panel.add((int(p[ri]), int(p[bi])))
    ok = set()
    n_lines = 0
    with open(os.path.join(PANEL_DIR, "done.jsonl")) as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:      # a line being appended right now
                continue
            n_lines += 1
            if r.get("status") == "ok":
                ok.add((int(r["region"]), int(r["blizz_id"])))
    keys = panel & ok
    digest = hashlib.sha256("".join(f"{r}:{b}\n" for r, b in sorted(keys)).encode()).hexdigest()
    meta = {"panel_files": [os.path.basename(f) for f in files], "panel_accounts": len(panel),
            "done_lines": n_lines, "done_ok": len(ok), "full_history": len(keys),
            "full_history_sha256": digest}
    return keys, meta


def sealed_ids():
    out = set()
    for p in SEALED_FILES:
        if os.path.exists(p):
            with open(p) as f:
                out |= {int(x) for x in f if x.strip()}
    if not out:
        sys.exit("no sealed_ids.txt found; refusing to select without the sealed exclusion")
    return out


def manifest_ids():
    """Replay IDs already on disk (any cohort) or terminal (403/404)."""
    seen = set()
    if os.path.exists(MANIFEST):
        with open(MANIFEST) as f:
            for line in f:
                r = json.loads(line)
                if r["status"] in ("ok", "linked", "http_403", "http_404"):
                    seen.add(int(r["replay_id"]))
    return seen


def selected_ids():
    out = set()
    if os.path.exists(IDS_FILE):
        with open(IDS_FILE) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    out.add(int(line.split("\t", 1)[0]))
    return out


def batches():
    if not os.path.exists(BATCH_LOG):
        return []
    with open(BATCH_LOG) as f:
        return [json.loads(x) for x in f if x.strip()]


def pool(day, sealed, keys):
    """{replay_id: k} for eligible games. Reads only ids, version, date and player keys."""
    import psycopg2
    url = os.environ.get("DATABASE_URL_RESEARCH")
    if not url or "neon" in url:
        sys.exit("DATABASE_URL_RESEARCH must point at the LOCAL store (never Neon)")
    conn = psycopg2.connect(url)
    conn.set_session(readonly=True)
    cur = conn.cursor()
    lo = day - dt.timedelta(days=LOOKBACK_DAYS)
    cur.execute("""SELECT d.replay_id, d.game_version, COALESCE(p.region, d.region), p.blizz_id
                   FROM replay_draft_data d JOIN replay_players p USING (replay_id)
                   WHERE d.game_date >= %s AND d.game_date < %s
                     AND NOT (d.replay_id = ANY(%s))""",
                (lo, day, sorted(sealed)))
    k = collections.Counter()
    seen = set()
    for rid, ver, region, blizz in cur.fetchall():
        if rid in sealed or not version_ok(ver):
            continue
        seen.add(rid)
        if region is not None and blizz is not None and (int(region), int(blizz)) in keys:
            k[rid] += 1
    conn.close()
    return {rid: k.get(rid, 0) for rid in seen}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--budget", type=int, help="override today's batch size (one-off catch-up)")
    a = ap.parse_args()

    today = dt.datetime.now(dt.timezone.utc).date()
    day = dt.datetime(today.year, today.month, today.day)       # naive UTC midnight, like game_date
    prev = batches()
    if any(b["day"] == today.isoformat() for b in prev):
        log(f"batch for {today} already built; nothing to do")
        return
    if prev:
        last = dt.date.fromisoformat(prev[-1]["day"])
        n = DAILY * min(MAX_CATCHUP_DAYS, (today - last).days)
    else:
        n = FIRST_BATCH

    if a.budget:
        n = a.budget

    sealed = sealed_ids()
    keys, kmeta = full_history_keys()
    ks = pool(day, sealed, keys)
    taken = manifest_ids() | selected_ids()
    cand = [(rid, k) for rid, k in ks.items() if rid not in taken and k >= MIN_K]
    cand.sort(key=lambda x: (-x[1], x[0]))
    pick = cand[:n]

    hist = lambda pairs: dict(sorted(collections.Counter(k for _, k in pairs).items()))  # noqa: E731
    rec = {"day": today.isoformat(), "built_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "rule": "SELECTION_RULES.md C.2", "window": [str(day - dt.timedelta(days=LOOKBACK_DAYS)), str(day)],
           "budget": n, "selected": len(pick), "pool_games": len(ks),
           "pool_k": hist(ks.items()), "eligible": len(cand), "selected_k": hist(pick),
           "min_id": min((r for r, _ in pick), default=None), "max_id": max((r for r, _ in pick), default=None),
           "code_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:16], **kmeta}
    log(json.dumps({x: rec[x] for x in ("day", "budget", "selected", "pool_games", "eligible", "selected_k",
                                        "full_history")}))
    if a.dry_run:
        return

    lines = []
    if os.path.exists(IDS_FILE):
        with open(IDS_FILE) as f:
            lines = f.read().splitlines()
    else:
        lines = [f"# {COHORT}: daily panel-coverage batches; rule in SELECTION_RULES.md C.2",
                 "# replay_id<TAB>batch day, k = covered players of 10, rank in batch"]
    tag = today.strftime("%Y%m%d")
    lines += [f"{rid}\tb={tag} k={k} r={i + 1}" for i, (rid, k) in enumerate(pick)]
    tmp = IDS_FILE + ".tmp"
    with open(tmp, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(tmp, IDS_FILE)
    with open(BATCH_LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")
    log(f"appended {len(pick)} ids to {os.path.relpath(IDS_FILE, REPO)}")


if __name__ == "__main__":
    main()
