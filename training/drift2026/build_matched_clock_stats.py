"""
Q12 — MATCHED-CLOCK per-signal-class decayed stats (reviewer follow-up to
SIGNAL_DECAY + PARTIAL_REFRESH: decay each signal class at its own measured
rate).

Composes the existing Q7 decayed stats files per build, at the JSON-section
level (exactly the attribute split load_patch_stats consumes, and the same
hero-family vs pair/comp split W3(d)/Q5 use):

  decayedmc     matched clock: hero family FAST, pair/comp SLOW
                  hero_stats + hero_map_stats  <- patch_stats/decayed90/
                  pairwise_stats + comp_stats  <- patch_stats/decayed365/
  decayedmcrev  reverse control: hero SLOW, pair/comp FAST
                  hero_stats + hero_map_stats  <- decayed365
                  pairwise_stats + comp_stats  <- decayed90

Signal classes (identical to phase_q5_partial_refresh / w3_build_hybrid):
  hero family = PatchStats.{hero_wr, hero_meta, hero_map_wr}
              <- JSON sections hero_stats, hero_map_stats
  pair/comp   = PatchStats.{pairwise, comp_data}
              <- JSON sections pairwise_stats, comp_stats

The composed kinds are drop-ins for common.load_patch_stats, so the existing
`decayed*_prev` branches of build_drift_features.py (feature pass) and
train_drift_wp.deploy_stats (sanity suite) handle them with no further code.

VERIFY: after writing, one build per kind is loaded via load_patch_stats and
its five attribute dicts are asserted equal to the corresponding parents'.

Usage:
    python drift2026/build_matched_clock_stats.py
"""
import os
import sys
import json
import gzip
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drift2026 import common

HERO_SECTIONS = ("hero_stats", "hero_map_stats")
PAIR_SECTIONS = ("pairwise_stats", "comp_stats")
KINDS = {
    "decayedmc": ("decayed90", "decayed365"),     # (hero_src, pair_src)
    "decayedmcrev": ("decayed365", "decayed90"),
}


def read(kind, build):
    with gzip.open(common.stats_path(kind, build), "rt") as f:
        return json.load(f)


def compose(build, hero_kind, pair_kind, out_kind):
    h, p = read(hero_kind, build), read(pair_kind, build)
    out = {"_meta": {"build": build, "kind": out_kind,
                     "hero_family_source": hero_kind,
                     "pair_comp_source": pair_kind,
                     "hero_meta": h.get("_meta"), "pair_meta": p.get("_meta")}}
    for s in HERO_SECTIONS:
        out[s] = h[s]
    for s in PAIR_SECTIONS:
        out[s] = p[s]
    path = common.stats_path(out_kind, build)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wt") as f:
        json.dump(out, f)


def verify(build):
    for out_kind, (hero_kind, pair_kind) in KINDS.items():
        got = common.load_patch_stats(out_kind, build)
        hs = common.load_patch_stats(hero_kind, build)
        ps = common.load_patch_stats(pair_kind, build)
        assert got.hero_wr == hs.hero_wr, out_kind
        assert got.hero_meta == hs.hero_meta, out_kind
        assert got.hero_map_wr == hs.hero_map_wr, out_kind
        assert got.pairwise == ps.pairwise, out_kind
        assert got.comp_data == ps.comp_data, out_kind
        print(f"  verify {out_kind}/{build}: hero family == {hero_kind}, "
              f"pair/comp == {pair_kind} (exact)")


def main():
    common.setup()
    d90 = os.path.join(common.STATS_DIR, "decayed90")
    d365 = os.path.join(common.STATS_DIR, "decayed365")
    builds = sorted(set(os.listdir(d90)) & set(os.listdir(d365)))
    builds = [b[:-len(".json.gz")] for b in builds if b.endswith(".json.gz")]
    builds.sort(key=common.build_sort_key)
    print(f"composing {len(builds)} builds x {len(KINDS)} kinds")
    t0 = time.time()
    for b in builds:
        for out_kind, (hk, pk) in KINDS.items():
            if not os.path.exists(common.stats_path(out_kind, b)):
                compose(b, hk, pk, out_kind)
    print(f"done in {time.time()-t0:.0f}s")
    verify(common.TRAIN_CUTOFF_BUILD)
    verify(builds[-1])


if __name__ == "__main__":
    main()
