"""
Meta-ness analysis: NGS (organized amateur league) drafts vs the paper's
ladder snapshot (rerun2026 dataset of record).

Inputs:
  ngs2026/ngs_drafts.json        (fetch_ngs.py --assemble)
  ngs2026/ladder_aggregates.json (build_ladder_aggregates.py)

Outputs:
  ngs2026/meta_analysis.json  — all metrics + per-draft off-meta scores
  ngs2026/META_ANALYSIS.md    — short human-readable summary

Metrics:
  - hero pick-rate distribution: Shannon entropy (bits), effective #heroes,
    top-10 / top-20 pick concentration
  - Spearman rank correlation of hero pick rates (NGS vs ladder)
  - off-meta rate: share of NGS picks outside the ladder top-N (N=20,30,40)
  - role-composition distribution: distinct comps, JS divergence vs ladder,
    coverage vs HP compositions.json observed sets (164/127/121 per tier)
  - ban overlap: Jaccard of top-10/top-20 ban sets, Spearman of ban rates
  - per-draft off-meta score: mean ladder-pick surprisal (-log2 ladder pick
    share) over the 10 picked heroes  (used by eval_wp.py quartiles)
"""
import os
import sys
import json
import math
from collections import Counter

import numpy as np

NGS_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(NGS_DIR)
sys.path.insert(0, TRAINING_DIR)
from shared import HEROES, HERO_TO_IDX, HERO_ROLE_FINE, FINE_TO_BLIZZ_ROLE  # noqa: E402

NGS_JSON = os.path.join(NGS_DIR, "ngs_drafts.json")
# NGS_TAG=<tag> (e.g. p1site) reads/writes tagged copies, leaving the originals
_TAG = ("_" + os.environ["NGS_TAG"]) if os.environ.get("NGS_TAG") else ""
LADDER_JSON = os.path.join(NGS_DIR, f"ladder_aggregates{_TAG}.json")
OUT_JSON = os.path.join(NGS_DIR, f"meta_analysis{_TAG}.json")
OUT_MD = os.path.join(NGS_DIR, f"META_ANALYSIS{_TAG}.md")
COMP_JSON = os.environ.get("NGS_COMP_JSON") or os.path.join(TRAINING_DIR, "..", "src", "lib", "data",
                                                             "compositions.json")


def entropy_bits(counts):
    tot = sum(counts.values())
    ps = [c / tot for c in counts.values() if c > 0]
    return -sum(p * math.log2(p) for p in ps)


def top_share(counts, n):
    tot = sum(counts.values())
    top = sorted(counts.values(), reverse=True)[:n]
    return sum(top) / tot


def spearman(xs, ys):
    def rank(v):
        order = np.argsort(v)
        r = np.empty(len(v))
        r[order] = np.arange(1, len(v) + 1)
        # average ties
        vals = {}
        for x, ri in zip(v, r):
            vals.setdefault(x, []).append(ri)
        return np.array([np.mean(vals[x]) for x in v])
    rx, ry = rank(np.asarray(xs)), rank(np.asarray(ys))
    rx -= rx.mean(); ry -= ry.mean()
    return float((rx * ry).sum() / math.sqrt((rx**2).sum() * (ry**2).sum()))


def js_divergence(p_counts, q_counts):
    keys = set(p_counts) | set(q_counts)
    p = np.array([p_counts.get(k, 0) for k in keys], dtype=float)
    q = np.array([q_counts.get(k, 0) for k in keys], dtype=float)
    p /= p.sum(); q /= q.sum()
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return float((a[mask] * np.log2(a[mask] / b[mask])).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def role_comp(team):
    return ",".join(sorted(FINE_TO_BLIZZ_ROLE.get(HERO_ROLE_FINE.get(h, ""),
                                                  "Ranged Assassin")
                           for h in team))


def main():
    ngs = json.load(open(NGS_JSON))
    drafts = ngs["drafts"]
    ladder = json.load(open(LADDER_JSON))

    # ── NGS counts ──
    ngs_picks, ngs_bans, ngs_comps = Counter(), Counter(), Counter()
    unknown_heroes = Counter()
    for d in drafts:
        for h in d["team0_heroes"] + d["team1_heroes"]:
            ngs_picks[h] += 1
            if h not in HERO_TO_IDX:
                unknown_heroes[h] += 1
        for h in d["team0_bans"] + d["team1_bans"]:
            ngs_bans[h] += 1
            if h not in HERO_TO_IDX:
                unknown_heroes[h] += 1
        ngs_comps[role_comp(d["team0_heroes"])] += 1
        ngs_comps[role_comp(d["team1_heroes"])] += 1

    lad_picks = Counter(ladder["pick_counts"])
    lad_bans = Counter(ladder["ban_counts"])
    lad_comps = Counter(ladder["role_comp_counts"])

    # ── Pick-rate distribution ──
    dist = {}
    for name, counts in (("ngs", ngs_picks), ("ladder", lad_picks)):
        h = entropy_bits(counts)
        dist[name] = {
            "n_picks": sum(counts.values()),
            "n_heroes_picked": len(counts),
            "entropy_bits": h,
            "effective_heroes": 2 ** h,
            "top10_share": top_share(counts, 10),
            "top20_share": top_share(counts, 20),
        }

    # Spearman over the full 90-hero roster (0 for unpicked)
    ngs_tot = sum(ngs_picks.values())
    lad_tot = sum(lad_picks.values())
    ngs_rates = [ngs_picks.get(h, 0) / ngs_tot for h in HEROES]
    lad_rates = [lad_picks.get(h, 0) / lad_tot for h in HEROES]
    rho_picks = spearman(ngs_rates, lad_rates)

    # ── Off-meta rate: NGS picks outside ladder top-N heroes ──
    lad_rank = [h for h, _ in lad_picks.most_common()]
    off_meta = {}
    for N in (20, 30, 40):
        topN = set(lad_rank[:N])
        outside = sum(c for h, c in ngs_picks.items() if h not in topN)
        lad_outside = sum(c for h, c in lad_picks.items() if h not in topN)
        off_meta[f"top{N}"] = {"ngs_offmeta_pick_share": outside / ngs_tot,
                               "ladder_offmeta_pick_share": lad_outside / lad_tot}

    # ── Role compositions ──
    hp_sets = {}
    if os.path.exists(COMP_JSON):
        raw = json.load(open(COMP_JSON))
        for tier, comps in raw.items():
            hp_sets[tier] = set(",".join(sorted(c["roles"])) for c in comps)
    all_hp = set().union(*hp_sets.values()) if hp_sets else set()
    ngs_team_total = sum(ngs_comps.values())
    in_hp = sum(c for k, c in ngs_comps.items() if k in all_hp)
    in_ladder_observed = sum(c for k, c in ngs_comps.items() if k in lad_comps)
    comps = {
        "ngs_distinct": len(ngs_comps),
        "ladder_distinct": len(lad_comps),
        "hp_distinct_per_tier": {t: len(s) for t, s in hp_sets.items()},
        "ngs_teams_total": ngs_team_total,
        "ngs_share_in_hp_observed_sets": in_hp / ngs_team_total,
        "ngs_share_in_ladder_observed": in_ladder_observed / ngs_team_total,
        "js_divergence_ngs_vs_ladder_bits": js_divergence(ngs_comps, lad_comps),
        "ngs_top5": ngs_comps.most_common(5),
        "ladder_top5": lad_comps.most_common(5),
        "ngs_comp_entropy_bits": entropy_bits(ngs_comps),
        "ladder_comp_entropy_bits": entropy_bits(lad_comps),
    }

    # ── Bans ──
    ngs_ban_tot = sum(ngs_bans.values())
    lad_ban_tot = sum(lad_bans.values())
    def jacc(n):
        a = set(h for h, _ in ngs_bans.most_common(n))
        b = set(h for h, _ in lad_bans.most_common(n))
        return len(a & b) / len(a | b)
    bans = {
        "ngs_n_bans": ngs_ban_tot,
        "ngs_bans_per_game": ngs_ban_tot / len(drafts),
        "jaccard_top10": jacc(10),
        "jaccard_top20": jacc(20),
        "spearman_ban_rates": spearman(
            [ngs_bans.get(h, 0) / max(ngs_ban_tot, 1) for h in HEROES],
            [lad_bans.get(h, 0) / max(lad_ban_tot, 1) for h in HEROES]),
        "ngs_top10_bans": ngs_bans.most_common(10),
        "ladder_top10_bans": lad_bans.most_common(10),
    }

    # ── Per-draft off-meta score: mean -log2(ladder pick share) of 10 picks ──
    # Laplace-smoothed ladder pick share so unpicked heroes are finite.
    lad_share = {h: (lad_picks.get(h, 0) + 1) / (lad_tot + len(HEROES))
                 for h in HEROES}
    floor = 1 / (lad_tot + len(HEROES))
    per_draft = {}
    for d in drafts:
        s = [-math.log2(lad_share.get(h, floor))
             for h in d["team0_heroes"] + d["team1_heroes"]]
        per_draft[str(d["replay_id"])] = float(np.mean(s))
    # ladder baseline of the same score: expected surprisal under ladder dist
    lad_expected = -sum((lad_picks[h] / lad_tot) * math.log2(lad_share[h])
                        for h in lad_picks)

    verdict = {
        "ngs_less_concentrated": dist["ngs"]["top10_share"] < dist["ladder"]["top10_share"]
                                 and dist["ngs"]["entropy_bits"] > dist["ladder"]["entropy_bits"],
        "note": "verdict field: True = NGS drafts are LESS meta-concentrated than ladder",
    }

    out = {
        "n_ngs_drafts": len(drafts),
        "n_ladder_replays": ladder["n_replays"],
        "unknown_heroes": dict(unknown_heroes),
        "pick_distribution": dist,
        "spearman_pick_rates": rho_picks,
        "off_meta": off_meta,
        "role_comps": comps,
        "bans": bans,
        "offmeta_score_mean_ngs": float(np.mean(list(per_draft.values()))),
        "offmeta_score_ladder_expected": lad_expected,
        "per_draft_offmeta_score": per_draft,
        "verdict": verdict,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=1)

    md = f"""# NGS vs Ladder Meta-ness Analysis

NGS drafts: **{len(drafts)}** (patch 2.55 era) vs ladder snapshot: **{ladder['n_replays']:,}** replays.
Unknown heroes in NGS data: {dict(unknown_heroes) or 'none — all within the 90-hero roster'}.

## Pick-rate distribution
| metric | NGS | Ladder |
|---|---|---|
| entropy (bits, max {math.log2(90):.2f}) | {dist['ngs']['entropy_bits']:.3f} | {dist['ladder']['entropy_bits']:.3f} |
| effective # heroes | {dist['ngs']['effective_heroes']:.1f} | {dist['ladder']['effective_heroes']:.1f} |
| top-10 hero share of picks | {dist['ngs']['top10_share']*100:.1f}% | {dist['ladder']['top10_share']*100:.1f}% |
| top-20 hero share of picks | {dist['ngs']['top20_share']*100:.1f}% | {dist['ladder']['top20_share']*100:.1f}% |

Spearman rank correlation of hero pick rates (NGS vs ladder): **{rho_picks:.3f}**

## Off-meta rate (share of picks outside ladder top-N heroes)
| N | NGS | Ladder |
|---|---|---|
""" + "\n".join(
        f"| {N} | {off_meta[f'top{N}']['ngs_offmeta_pick_share']*100:.1f}% | "
        f"{off_meta[f'top{N}']['ladder_offmeta_pick_share']*100:.1f}% |"
        for N in (20, 30, 40)) + f"""

## Role compositions
- distinct comps (of 252 possible): NGS **{comps['ngs_distinct']}**, ladder **{comps['ladder_distinct']}** (HP observed sets: {comps['hp_distinct_per_tier']})
- NGS team-comps found in HP observed sets: {comps['ngs_share_in_hp_observed_sets']*100:.2f}%
- NGS team-comps found among ladder-observed comps: {comps['ngs_share_in_ladder_observed']*100:.2f}%
- JS divergence (NGS vs ladder comp distribution): {comps['js_divergence_ngs_vs_ladder_bits']:.4f} bits
- comp entropy: NGS {comps['ngs_comp_entropy_bits']:.2f} vs ladder {comps['ladder_comp_entropy_bits']:.2f} bits

## Bans
- NGS bans/game: {bans['ngs_bans_per_game']:.2f}
- top-10 ban set Jaccard overlap with ladder: {bans['jaccard_top10']:.2f}; top-20: {bans['jaccard_top20']:.2f}
- Spearman of ban rates: {bans['spearman_ban_rates']:.3f}

## Per-draft off-meta score (mean ladder-pick surprisal, bits)
- NGS mean: {out['offmeta_score_mean_ngs']:.3f} — ladder expected: {lad_expected:.3f}

## Verdict
NGS drafts {'ARE' if verdict['ngs_less_concentrated'] else 'are NOT'} less meta-concentrated than ladder drafts
(entropy {'higher' if dist['ngs']['entropy_bits']>dist['ladder']['entropy_bits'] else 'lower'}, top-10 concentration {'lower' if dist['ngs']['top10_share']<dist['ladder']['top10_share'] else 'higher'}).
"""
    with open(OUT_MD, "w") as f:
        f.write(md)
    print(md)
    print(f"Wrote {OUT_JSON} and {OUT_MD}")


if __name__ == "__main__":
    main()
