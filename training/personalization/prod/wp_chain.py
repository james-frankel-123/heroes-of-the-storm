"""
Versioned causal WP scoring chain for the personal layer (review 5.1).

Every residual r = y - wp that feeds the personal state must come from a WP
model that never saw the game. Production retrains the WP monthly, so we keep
an append-only archive of WP "vintages" and score each game with the latest
vintage whose training coverage ends before the game. A retrain then adds a
vintage; it never changes the residual of a game that an earlier vintage
already scores causally.

A vintage is a dict:
  vintage_id           unique string, e.g. "2026-11-01"
  weights              list of model weight paths (seeds)
  stats_path           the statistics file the features were built from
  stats_cutoff_date    last UTC day (inclusive) of games inside the statistics
  build_cutoff         last build (full version string, e.g. "2.55.17.98025")
                       whose games are in the statistics or the weights
  trained_through_date last UTC day (inclusive) of games inside the weights
  created_at           ISO timestamp when the vintage was registered
  files_sha256         optional {path: sha256} filled at registration

Clocks. Two clocks can say a game is "after" a vintage:
  date clock   game start >= end of max(stats_cutoff_date, trained_through_date)
  build clock  game build > build_cutoff
The DATE clock is authoritative for the personal layer: it is the clock of the
visibility contract (fetched_at before the game's start), of the production
stats decay and of the nightly state. The build clock is enforced as well,
because the research OOT WP was causal by build only and statistics for build
B+1 can contain games of build B dated after B+1 shipped. A vintage may score
a game only when BOTH clocks put the game after its coverage (the stricter of
the two). A game that no vintage precedes on both clocks is left unscored.

Game start = end_ts - game_length; an unknown length is taken as
DEFAULT_GAME_LENGTH (an hour), which moves the start earlier and so can only
make the check stricter.

Invariants enforced here:
  - the registry is append-only (ids unique, earlier entries never change,
    both cutoffs non-decreasing along the list);
  - scoring a game inside a vintage's coverage raises InSampleError;
  - for out-of-sample scores, the residual noise ratio
    mean((y - wp)^2) / mean(wp (1 - wp)) per month stays inside NOISE_BAND.
    In-sample WP scores give 0.96 to 0.98 (review 5.1), which is what the
    band is meant to catch.
"""
import datetime
import hashlib
import json
import os

import numpy as np

DEFAULT_GAME_LENGTH = 3600
NOISE_BAND = (0.98, 1.02)
NOISE_MIN_GAMES = 2000  # months with fewer games are reported, not judged
SCHEMA = 1
REQUIRED = ("vintage_id", "weights", "stats_path", "stats_cutoff_date", "build_cutoff",
            "trained_through_date", "created_at")


class InSampleError(RuntimeError):
    """A game was about to be scored by a vintage whose training covers it."""


class NoiseBandError(RuntimeError):
    """A month's residual noise ratio fell outside the band."""


# ------------------------------------------------------------------ clocks

def build_key(version):
    """Comparable int64 for a version string "a.b.c.build" (missing parts = 0)."""
    parts = [int(p) for p in str(version).strip().split(".") if p != ""]
    parts = (parts + [0, 0, 0, 0])[:4]
    a, b, c, d = parts
    if not (0 <= b < 1000 and 0 <= c < 1000 and 0 <= d < 10 ** 7):
        raise ValueError(f"version out of range for build_key: {version!r}")
    return ((a * 1000 + b) * 1000 + c) * 10 ** 7 + d


def build_keys(versions):
    cache = {}
    out = np.empty(len(versions), np.int64)
    for i, v in enumerate(versions):
        v = str(v)
        if v not in cache:
            cache[v] = build_key(v)
        out[i] = cache[v]
    return out


def day_end_ts(iso_day):
    """Epoch second at the END of a UTC day (start of the next day)."""
    d = datetime.date.fromisoformat(str(iso_day))
    return int((d - datetime.date(1970, 1, 1)).days + 1) * 86400


def coverage_end_ts(v):
    """First epoch second NOT covered by the vintage on the date clock."""
    return max(day_end_ts(v["stats_cutoff_date"]), day_end_ts(v["trained_through_date"]))


def game_start_ts(end_ts, game_length=None):
    end_ts = np.asarray(end_ts, np.int64)
    if game_length is None:
        return end_ts - DEFAULT_GAME_LENGTH
    gl = np.asarray(game_length, np.int64)
    return end_ts - np.where(gl > 0, gl, DEFAULT_GAME_LENGTH)


def after_coverage(v, builds, start_ts):
    """Bool per game: the vintage precedes the game on BOTH clocks."""
    bk = build_keys(builds) if not np.issubdtype(np.asarray(builds).dtype, np.integer) \
        else np.asarray(builds, np.int64)
    return (np.asarray(start_ts, np.int64) >= coverage_end_ts(v)) & (bk > build_key(v["build_cutoff"]))


# ------------------------------------------------------------------ registry

def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def load_registry(path):
    """Registry dict {schema, authoritative_clock, vintages: [...]}; an absent
    file is an empty registry."""
    if not os.path.exists(path):
        return {"schema": SCHEMA, "authoritative_clock": "date", "enforced": "date and build",
                "vintages": []}
    with open(path) as f:
        reg = json.load(f)
    if reg.get("schema") != SCHEMA:
        raise ValueError(f"{path}: registry schema {reg.get('schema')} != {SCHEMA}")
    validate_registry(reg)
    return reg


def validate_registry(reg):
    """Ids unique, required fields present, cutoffs non-decreasing."""
    seen = set()
    prev = None
    for v in reg["vintages"]:
        miss = [k for k in REQUIRED if k not in v]
        if miss:
            raise ValueError(f"vintage {v.get('vintage_id')!r} lacks {miss}")
        if v["vintage_id"] in seen:
            raise ValueError(f"duplicate vintage_id {v['vintage_id']!r}")
        seen.add(v["vintage_id"])
        if prev is not None:
            if coverage_end_ts(v) < coverage_end_ts(prev):
                raise ValueError(f"vintage {v['vintage_id']!r}: date coverage goes backwards")
            if build_key(v["build_cutoff"]) < build_key(prev["build_cutoff"]):
                raise ValueError(f"vintage {v['vintage_id']!r}: build_cutoff goes backwards")
        prev = v


def append_vintage(path, vintage, hash_files=True):
    """Append one vintage to the registry at path (atomic write). Refuses a
    duplicate id or a vintage whose cutoffs precede the last one. Existing
    entries are re-read after the write and must be unchanged."""
    reg = load_registry(path)
    before = json.dumps(reg["vintages"], sort_keys=True)
    v = dict(vintage)
    v.setdefault("created_at", datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"))
    if isinstance(v.get("weights"), str):
        v["weights"] = [v["weights"]]
    if hash_files and "files_sha256" not in v:
        files = list(v["weights"]) + [v["stats_path"]]
        v["files_sha256"] = {p: _sha256(p) for p in files if os.path.exists(p)}
    reg["vintages"].append(v)
    validate_registry(reg)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(reg, f, indent=1)
    os.replace(tmp, path)
    after = load_registry(path)["vintages"]
    if json.dumps(after[:-1], sort_keys=True) != before:
        raise RuntimeError("registry prefix changed during append")
    return v


def vintage_for(registry, game_build, game_end_ts, game_length=None):
    """The latest vintage that precedes the game on both clocks, or None."""
    i = assign_vintages(registry, [game_build], [game_end_ts],
                        None if game_length is None else [game_length])[0]
    return None if i < 0 else registry["vintages"][i]


def assign_vintages(registry, builds, end_ts, game_length=None):
    """Index into registry['vintages'] per game (-1 = no causal vintage)."""
    vs = registry["vintages"]
    start = game_start_ts(end_ts, game_length)
    bk = build_keys(builds)
    out = np.full(len(start), -1, np.int64)
    for i, v in enumerate(vs):  # later vintages overwrite: the latest wins
        out[after_coverage(v, bk, start)] = i
    return out


# ------------------------------------------------------------------ scoring

def assert_out_of_sample(v, builds, start_ts):
    bad = ~after_coverage(v, builds, start_ts)
    if bad.any():
        raise InSampleError(f"vintage {v['vintage_id']!r} would score {int(bad.sum())} game(s) "
                            f"inside its training coverage")


def score_with(v, games, score_fn):
    """Score games with one vintage. Raises InSampleError if any game is inside
    the vintage's coverage. score_fn(vintage, games) -> wp for team 0."""
    start = game_start_ts(games["end_ts"], games.get("game_length"))
    assert_out_of_sample(v, games["build"], start)
    wp = np.asarray(score_fn(v, games), np.float64)
    if wp.shape != (len(start),) or not np.all(np.isfinite(wp)) or wp.min() <= 0 or wp.max() >= 1:
        raise ValueError(f"vintage {v['vintage_id']!r}: score_fn returned invalid probabilities")
    return wp


def _take(games, idx):
    return {k: (np.asarray(a)[idx] if np.ndim(a) else a) for k, a in games.items()}


def rescore(games, registry, score_fn, since_vintage=None, band=NOISE_BAND,
            min_games=NOISE_MIN_GAMES, strict=False):
    """Score the games whose causal vintage is newer than since_vintage.

    games: dict of equal-length arrays replay_id, build (version strings),
      end_ts, game_length (optional), y (1 = team 0 won).
    since_vintage: id of the vintage the caller already scored with (None =
      score everything). Games whose latest causal vintage is that one or an
      older one keep their existing residuals and are skipped.
    Returns (rows, report). rows: replay_id, y, wp, vintage_id, start_ts,
      end_ts for the scored games only. report: counts, unscorable games, and
      the per-month noise ratio. strict=True raises NoiseBandError when a
      judged month is outside the band.
    """
    vs = registry["vintages"]
    ids = [v["vintage_id"] for v in vs]
    k_since = -1 if since_vintage is None else ids.index(since_vintage)
    end_ts = np.asarray(games["end_ts"], np.int64)
    gl = games.get("game_length")
    a = assign_vintages(registry, games["build"], end_ts, gl)
    start = game_start_ts(end_ts, gl)
    parts = []
    for i in range(k_since + 1, len(vs)):
        idx = np.flatnonzero(a == i)
        if not len(idx):
            continue
        sub = _take(games, idx)
        wp = score_with(vs[i], sub, score_fn)
        parts.append((idx, wp, i))
    if parts:
        idx = np.concatenate([p[0] for p in parts])
        wp = np.concatenate([p[1] for p in parts])
        vid = np.concatenate([np.full(len(p[0]), p[2]) for p in parts])
        o = np.argsort(idx, kind="stable")
        idx, wp, vid = idx[o], wp[o], vid[o]
    else:
        idx, wp, vid = np.zeros(0, np.int64), np.zeros(0), np.zeros(0, np.int64)
    rows = {"replay_id": np.asarray(games["replay_id"], np.int64)[idx],
            "y": np.asarray(games["y"], np.float64)[idx], "wp": wp,
            "vintage_id": np.array([ids[i] for i in vid], dtype="U64"),
            "start_ts": start[idx], "end_ts": end_ts[idx]}
    noise = noise_ratio_report(rows, band=band, min_games=min_games)
    report = {"scored": int(len(idx)), "skipped_older_vintage": int(((a >= 0) & (a <= k_since)).sum()),
              "unscorable_no_causal_vintage": int((a < 0).sum()),
              "by_vintage": {ids[i]: int((vid == i).sum()) for i in np.unique(vid)},
              "noise": noise}
    if strict and not noise["ok"]:
        bad = {m: r["ratio"] for m, r in noise["months"].items() if r["judged"] and not r["in_band"]}
        raise NoiseBandError(f"residual noise ratio outside {band}: {bad}")
    return rows, report


# ------------------------------------------------------------------ invariant

def month_of(ts):
    return np.asarray(ts, "datetime64[s]").astype("datetime64[M]").astype(str)


def noise_ratio_report(rows, band=NOISE_BAND, min_games=NOISE_MIN_GAMES):
    """Per UTC month of game start: mean((y-wp)^2) / mean(wp(1-wp)).

    Out-of-sample calibrated scores give 1.00 up to sampling noise; in-sample
    scores give less than 1 (the model has partly fit the outcome). Months
    with fewer than min_games are reported with judged=False."""
    y, wp = np.asarray(rows["y"], float), np.asarray(rows["wp"], float)
    months = month_of(rows["start_ts"]) if len(y) else np.zeros(0, "U7")
    out = {}
    ok = True
    for m in np.unique(months):
        s = months == m
        ratio = float(np.mean((y[s] - wp[s]) ** 2) / np.mean(wp[s] * (1 - wp[s])))
        judged = bool(s.sum() >= min_games)
        inb = bool(band[0] <= ratio <= band[1])
        ok &= inb or not judged
        out[str(m)] = {"games": int(s.sum()), "ratio": round(ratio, 5), "judged": judged, "in_band": inb}
    return {"band": list(band), "min_games": int(min_games), "ok": bool(ok), "months": out}
