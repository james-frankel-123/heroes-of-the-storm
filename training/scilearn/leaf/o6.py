"""O6 search-behavior probes for an additive leafpack (FRAMING 1, O6 and G3).

  (a) degenerate-team rate of our team under greedy and under MCTS at each
      --sims budget, no role mask. Degenerate = shared.is_degenerate (paper 1:
      no healer, or no frontline, or 3+ of Tank / Melee Assassin / Support /
      Healer; Uther counts as frontline beside a second healer).
  (b) five-tank probe: P(5 tanks beat the standard team), paper-1 probe
      (experiment_synthetic_augmentation.DEGEN_COMPS / STANDARD, Cursed Hollow,
      mid). Flag at > 0.25 (G3).
  (c) no-healer probe: the paper-1 "No healer high WR" team vs the standard
      team, same setting, flag at > 0.25 (paper-1 all_degen_below_25). All six
      paper-1 degenerate probes are reported.
  (d) optimism O(W) = mean over W's own MCTS drafts of p_W - p_J, p = sigmoid(eta)
      on the draft's map and tier, J = --judge leafpack (heroes matched by name).
  (e) split-half pick agreement: paper 1's "divergence" definition
      (paper1_revision/crosseval.py): share of our pick steps, same
      configuration and same step, where two drafters chose the same hero.
      The two drafters are MCTS at --split-sims with search seeds 0 and 1 on W
      (opponent RNG shared, so drafts stay matched until a pick differs), or W
      vs --split-pack (a model fit on the other half of the data) if given.

Configurations: random.Random(--seed) draws (map, tier, our_team = i % 2) like
paper 1 (greedy_evals.py), over --maps (default: maps GD knows) and --tiers.

Usage (local synthetic smoke run, about 75 s per pack on 4 cores):
  nice -n 19 taskset -c 60-63 python3 training/scilearn/leaf/o6.py \
      --pack /tmp/leaf/exploit --judge /tmp/leaf/judge --out /tmp/leaf/o6_exploit
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from leaf import LeafModel, sigmoid  # noqa: E402
from search import GD_MAPS, GDOpponent, _cfg_seed, greedy_draft, mcts_draft  # noqa: E402

# Paper-1 probes, copied from training/experiment_synthetic_augmentation.py
# (DEGEN_COMPS, STANDARD; evaluate_config scores them on Cursed Hollow, mid).
DEGEN_COMPS = {
    "5 tanks": ["Muradin", "Johanna", "Diablo", "E.T.C.", "Mal'Ganis"],
    "5 healers": ["Brightwing", "Malfurion", "Rehgar", "Uther", "Anduin"],
    "5 ranged assassins": ["Valla", "Jaina", "Li-Ming", "Falstad", "Raynor"],
    "5 melee assassins": ["Zeratul", "Illidan", "Kerrigan", "Malthael", "Qhira"],
    "3T 2H no damage": ["Muradin", "Johanna", "Diablo", "Brightwing", "Malfurion"],
    "No healer high WR": ["Muradin", "Johanna", "Valla", "Falstad", "Li-Ming"],
}
STANDARD = ["Muradin", "Brightwing", "Valla", "Sonya", "Jaina"]
PROBE_MAP, PROBE_TIER, PROBE_MAX = "Cursed Hollow", "mid", 0.25

_W = {}


def _init(args):
    _W["args"] = args
    _W["packs"] = {"W": LeafModel.load(args.pack)}
    if args.split_pack:
        _W["packs"]["S"] = LeafModel.load(args.split_pack)
    _W["gd"] = {k: GDOpponent(m, args.gd_dir, kind=args.opponent) for k, m in _W["packs"].items()}


def _run(task):
    kind, pack, sims, sseed, cfg = task
    a = _W["args"]
    i, gmap, tier, our = cfg
    L, gd = _W["packs"][pack], _W["gd"][pack]
    t0 = time.time()
    if kind == "greedy":
        r = greedy_draft(L, gd, gmap, tier, our, seed=_cfg_seed(a.seed, i, 7), kappa=a.kappa)
    else:
        r = mcts_draft(L, gd, gmap, tier, our, sims, search_seed=sseed,
                       opp_seed=_cfg_seed(a.seed, i), kappa=a.kappa, prior=a.prior,
                       prior_eps=a.prior_eps, gd_bans=a.gd_bans)
    r.update(cfg=i, run=f"{kind}{'' if kind == 'greedy' else sims}_{pack}_s{sseed}",
             secs=time.time() - t0)
    return r


def probes(L: LeafModel, kappa):
    out = {}
    for name, comp in DEGEN_COMPS.items():
        d = np.array(L.hero_ids(comp + STANDARD))[None]
        eta, sd, c = L.evaluate(d, PROBE_MAP, PROBE_TIER)
        top = sorted(((k, float(v[0])) for k, v in c.items()), key=lambda x: -abs(x[1]))[:3]
        out[name] = {"p": float(sigmoid(eta[0])), "leaf": float(sigmoid(eta[0] - kappa * sd[0])),
                     "sd": float(sd[0]), "top_terms": top}
    return out


def p_under(L: LeafModel, recs):
    """sigmoid(eta) of each record's draft under leafpack L (heroes by name)."""
    out = np.empty(len(recs))
    for j, r in enumerate(recs):
        d = np.array(L.hero_ids(r["our"] + r["opp"]))[None]
        out[j] = sigmoid(L.evaluate(d, r["map"], r["tier"])[0][0])
    return out


def mean_se(x):
    x = np.asarray(x, float)
    return [float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")]


def agreement(ra, rb, typ="pick"):
    """Paper-1 crosseval agreement: same config, same step, our steps of `typ`."""
    by = {r["cfg"]: r for r in rb}
    hit = tot = 0
    for r in ra:
        o = by.get(r["cfg"])
        if o is None:
            continue
        e = {s["step"]: s["hero"] for s in o["steps"] if s["ours"] and s["type"] == typ}
        for s in r["steps"]:
            if s["ours"] and s["type"] == typ and s["step"] in e:
                hit += s["hero"] == e[s["step"]]
                tot += 1
    return hit / tot if tot else float("nan"), tot


def agreement_same_state(ra, rb, typ="pick"):
    """Agreement only at our steps whose whole draft history is identical in both
    runs (pure search noise; the paper-1 version also counts diverged states)."""
    by = {r["cfg"]: r for r in rb}
    hit = tot = 0
    for r in ra:
        o = by.get(r["cfg"])
        if o is None:
            continue
        for s, t in zip(r["steps"], o["steps"]):
            if s["ours"] and s["type"] == typ:
                hit += s["hero"] == t["hero"]
                tot += 1
            if s["hero"] != t["hero"]:
                break
    return hit / tot if tot else float("nan"), tot


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", required=True, help="leafpack dir of the model W")
    ap.add_argument("--judge", required=True, help="leafpack dir of the independent judge J")
    ap.add_argument("--split-pack", default=None, help="optional half-B leafpack for (e)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-drafts", type=int, default=12, help="configurations per drafter")
    ap.add_argument("--sims", default="400,1000")
    ap.add_argument("--split-sims", type=int, default=1000)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--maps", default=None, help="comma list; default: pack maps GD knows")
    ap.add_argument("--tiers", default="low,mid,high")
    ap.add_argument("--kappa", type=float, default=None, help="default: pack json kappa")
    ap.add_argument("--prior", default="gd", choices=["gd", "uniform"])
    ap.add_argument("--prior-eps", type=float, default=0.1)
    ap.add_argument("--opponent", default="gd", choices=["gd", "uniform"])
    ap.add_argument("--gd-dir", default=str(Path(__file__).resolve().parents[2] / "rerun2026" / "models"))
    ap.add_argument("--no-greedy", action="store_true")
    ap.add_argument("--gd-bans", action="store_true",
                    help="our bans by GD argmax instead of search (default: searched, like the kernel)")
    ap.add_argument("--ref-degen", type=float, default=None, help="N's MCTS-1000 degenerate %% (G3)")
    ap.add_argument("--ref-optimism", type=float, default=None, help="N's optimism (G3)")
    a = ap.parse_args()

    W, J = LeafModel.load(a.pack), LeafModel.load(a.judge)
    if a.kappa is None:
        a.kappa = W.kappa
    W.export_check(n=500)
    J.export_check(n=500)
    maps = a.maps.split(",") if a.maps else [m for m in W.maps if m in GD_MAPS]
    tiers = a.tiers.split(",")
    rnd = random.Random(a.seed)
    cfgs = [(i, rnd.choice(maps), rnd.choice(tiers), i % 2) for i in range(a.n_drafts)]
    sims = [int(x) for x in a.sims.split(",") if x]
    tasks = [] if a.no_greedy else [("greedy", "W", 0, 0, c) for c in cfgs]
    tasks += [("mcts", "W", s, 0, c) for s in sims for c in cfgs]
    split_other = ("S", 0) if a.split_pack else ("W", 1)
    if a.split_sims not in sims:
        tasks += [("mcts", "W", a.split_sims, 0, c) for c in cfgs]
    tasks += [("mcts", split_other[0], a.split_sims, split_other[1], c) for c in cfgs]
    tasks.sort(key=lambda t: -t[2])        # long searches first for load balance

    t0 = time.time()
    if a.procs > 1:
        with mp.get_context("fork").Pool(a.procs, initializer=_init, initargs=(a,)) as pool:
            recs = pool.map(_run, tasks, chunksize=1)
    else:
        _init(a)
        recs = [_run(t) for t in tasks]
    wall = time.time() - t0

    runs = {}
    for r in recs:
        runs.setdefault(r["run"], []).append(r)
    for v in runs.values():
        v.sort(key=lambda r: r["cfg"])
    res = {"pack": W.model_id, "judge": J.model_id, "pack_dir": a.pack, "judge_dir": a.judge,
           "split_pack": a.split_pack, "n_drafts": a.n_drafts, "seed": a.seed, "kappa": a.kappa,
           "prior": a.prior, "prior_eps": a.prior_eps, "opponent": a.opponent, "gd_bans": a.gd_bans, "maps": maps,
           "tiers": tiers, "wall_secs": wall, "drafters": {}}
    for name, rr in sorted(runs.items()):
        pw, pj = np.array([r["p_self"] for r in rr]), p_under(J, rr)
        res["drafters"][name] = {
            "n": len(rr),
            "degenerate_pct": mean_se(100 * np.array([r["degenerate"] for r in rr])),
            "healer_pct": mean_se(100 * np.array([r["has_healer"] for r in rr])),
            "p_W": mean_se(pw), "p_J": mean_se(pj), "optimism": mean_se(pw - pj),
            "secs_per_draft": float(np.mean([r["secs"] for r in rr])),
        }
    res["probes_W"] = probes(W, a.kappa)
    res["probes_J"] = probes(J, 0.0)
    A = runs[f"mcts{a.split_sims}_W_s0"]
    B = runs[f"mcts{a.split_sims}_{split_other[0]}_s{split_other[1]}"]
    res["split_half"] = {"definition": ("W search seed 0 vs " + ("split pack" if a.split_pack else
                                                                 "W search seed 1")
                                        + f", MCTS-{a.split_sims}, paper-1 same-step agreement"),
                         "pick": agreement(A, B, "pick"), "ban": agreement(A, B, "ban"),
                         "pick_same_state": agreement_same_state(A, B, "pick"),
                         "ban_same_state": agreement_same_state(A, B, "ban")}
    m1000 = res["drafters"].get(f"mcts{max(sims)}_W_s0", {})
    g3 = {
        "mcts_degenerate_le_5pct": m1000.get("degenerate_pct", [None])[0] is not None
        and m1000["degenerate_pct"][0] <= 5.0,
        "five_tank_le_0.25": res["probes_W"]["5 tanks"]["p"] <= PROBE_MAX,
        "split_half_ge_40pct": res["split_half"]["pick"][0] >= 0.40,
    }
    if a.ref_degen is not None and m1000:
        g3["mcts_degenerate_le_N"] = m1000["degenerate_pct"][0] <= a.ref_degen
    if a.ref_optimism is not None and m1000:
        g3["optimism_le_half_N"] = m1000["optimism"][0] <= 0.5 * a.ref_optimism
    res["G3"] = g3
    res["flags"] = {"five_tank": res["probes_W"]["5 tanks"]["p"] > PROBE_MAX,
                    "no_healer": res["probes_W"]["No healer high WR"]["p"] > PROBE_MAX,
                    "any_probe": any(v["p"] > PROBE_MAX for v in res["probes_W"].values())}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "o6.json").write_text(json.dumps(res, indent=1))
    (out / "o6_drafts.json").write_text(json.dumps(recs))
    md = to_markdown(res)
    (out / "o6.md").write_text(md)
    print(md)


def to_markdown(res):
    f = lambda x: f"{x[0]:.1f} ± {x[1]:.1f}"             # noqa: E731
    g = lambda x: f"{x[0]:+.3f} ± {x[1]:.3f}"            # noqa: E731
    L = [f"# O6 search behavior: {res['pack']} (judge {res['judge']})", "",
         f"{res['n_drafts']} configurations per drafter, seed {res['seed']}, kappa {res['kappa']}, "
         f"prior {res['prior']} (eps {res['prior_eps']}), opponent {res['opponent']}, "
         f"bans {'GD argmax' if res.get('gd_bans') else 'searched'}, "
         f"wall {res['wall_secs']:.0f} s.", "",
         "| Drafter | n | Degenerate % | Healer % | p_W | p_J | Optimism p_W - p_J | s/draft |",
         "|---|---|---|---|---|---|---|---|"]
    for k, v in res["drafters"].items():
        L.append(f"| {k} | {v['n']} | {f(v['degenerate_pct'])} | {f(v['healer_pct'])} | "
                 f"{v['p_W'][0]:.3f} | {v['p_J'][0]:.3f} | {g(v['optimism'])} | {v['secs_per_draft']:.1f} |")
    L += ["", f"Probes vs the standard team ({PROBE_MAP}, {PROBE_TIER}); flag above {PROBE_MAX}:", "",
          "| Probe | P under W | P under J | Largest W terms |", "|---|---|---|---|"]
    for k, v in res["probes_W"].items():
        terms = ", ".join(f"{t} {x:+.2f}" for t, x in v["top_terms"])
        flag = " (flag)" if v["p"] > PROBE_MAX else ""
        L.append(f"| {k} | {v['p']:.3f}{flag} | {res['probes_J'][k]['p']:.3f} | {terms} |")
    sh = res["split_half"]
    L += ["", f"Split-half agreement ({sh['definition']}): picks {sh['pick'][0]:.1%} of {sh['pick'][1]}, "
              f"bans {sh['ban'][0]:.1%} of {sh['ban'][1]}. Same draft history only: picks "
              f"{sh['pick_same_state'][0]:.1%} of {sh['pick_same_state'][1]}, bans "
              f"{sh['ban_same_state'][0]:.1%} of {sh['ban_same_state'][1]}.", "",
          "G3: " + ", ".join(f"{k} {'pass' if v else 'FAIL'}" for k, v in res["G3"].items()), ""]
    return "\n".join(L)


if __name__ == "__main__":
    main()
