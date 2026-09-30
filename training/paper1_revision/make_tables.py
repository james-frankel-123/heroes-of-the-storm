"""
Emit LaTeX table bodies for draft_revision.tex from results/SUMMARY.json
(run compile_results.py first), so no manuscript number is transcribed by hand.
Output: results/tables.tex (one commented block per table) + stdout.
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from paper1_revision import core

S = json.load(open(os.path.join(core.RESULTS, "SUMMARY.json")))
OUT = []


def emit(title, lines):
    OUT.append(f"% ---- {title} ----")
    OUT.extend(lines)
    OUT.append("")


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


def table1():
    t = S["table1"]
    sub = S["submitted"] or {}
    rows = [("Naive", "naive", "naive", 197, "Multi-hot + map + tier"),
            ("Hero Str.", "herostrength", "herostrength", 209, "+ per-hero WR, team avg"),
            ("Enriched", "enriched", "enriched_256", 283, "+ role counts, pairwise, comp WR, meta")]
    L = []
    for lab, k, sk, dim, feat in rows:
        d = t[k]
        s = sub[sk]
        L.append(f"{lab} & {pct(s['test_hp']['acc'])} & {pct(s['NODRIFT_hp']['acc'])} & "
                 f"{s['NODRIFT_hp']['slope']:.2f} & {pct(d['test']['acc'])} & "
                 f"{pct(d['NODRIFT']['acc'])} & {d['NODRIFT']['slope']:.2f} & {dim} \\\\")
    d = t["enriched_leak"]
    L.append(f"Enr., in-sample stats & -- & -- & -- & {pct(d['test']['acc'])} & "
             f"{pct(d['NODRIFT']['acc'])} & {d['NODRIFT']['slope']:.2f} & 283 \\\\")
    emit("Table I", L)


def table_mcts():
    m = S["mcts"]["configs"]
    names = [("old:B_fullwp", "B\\_fullwp", 200, "sub"), ("old:F_400sim", "F\\_400sim", 400, "sub"),
             ("old:I_600sim", "I\\_600sim", 600, "sub"), ("old:J_800sim", "J\\_800sim", 800, "sub"),
             ("old:E_1M", "E\\_1M", "200/1M", "sub"), ("old:K_truebase", "K\\_truebase", 200, None),
             ("new:B_oof", "B\\_oof", 200, "rev"), ("new:F_oof", "F\\_oof", 400, "rev"),
             ("new:J_oof", "J\\_oof", 800, "rev")]
    L = []
    for T in ("T1", "T0"):
        L.append(f"% root {T}")
        for k, lab, sims, proxy in names:
            key = f"{k}|{T}"
            if key not in m:
                continue
            v = m[key]["mean"]
            se = m[key]["se"]
            own = (f"{v['proxy_sub']:.3f}" if proxy == "sub" else
                   f"{v['proxy_rev']:.3f}" if proxy == "rev" else "--")
            L.append(f"{lab} & {sims} & {own} & {v['gN']:.3f}{{\\scriptsize$\\pm$.{int(round(1000 * se['gN'])):03d}}} & "
                     f"{v['RN']:.3f} & {v['QM2026']:.3f} & {v['R17']:.3f} & {pct(v['degen'])} & "
                     f"{pct(v['healer'], 0)} & {v['syn_hp']:+.2f} & {m[key]['distinct']:.0f} \\\\")
    emit("Table VI (MCTS)", L)


def table_tournament():
    t = S["tournament"]
    if not t:
        return
    st = t["standings"]
    order = sorted(st, key=lambda s: -st[s]["consensus"]["mean"])
    LAB = {"constrained_mcts": "Constrained MCTS", "mcts": "MCTS", "constrained_greedy":
           "Constrained greedy", "enriched": "Enriched greedy", "enriched_aug": "Enr.+aug greedy",
           "k_truebase": "MCTS, no features (K)", "gourdeau": "G\\&A estimator",
           "gourdeau_disc": "G\\&A discriminator", "cql_naive_a1.0": "CQL naive",
           "cql_enr_a2.0": "CQL enriched", "gd": "GD", "mcq_t0.5": "MCQ"}
    L = []
    for s in order:
        e = st[s]
        L.append(f"{LAB.get(s, s)} & {e['consensus']['mean']:.3f} & {e['consensus']['wins']}/{e['n_pairs']} & "
                 f"{e['gN']['mean']:.3f} & {e['gN_naive']['mean']:.3f} & {e['RN']['mean']:.3f} & "
                 f"{e['QM2026']['mean']:.3f} & {e['QM2021']['mean']:.3f} & "
                 f"{100 * t['draft_metrics'][s]['degen']:.1f} & {t['draft_metrics'][s]['synergy']:+.2f} \\\\")
    emit("Table VII (tournament)", L)


def table_sweep():
    sw = S["sweep"]
    if not sw:
        return
    L = []
    for g, e in sorted(sw["main_effects"].items(), key=lambda kv: -kv[1]["nodrift_acc"]):
        L.append(f"{g.replace('_', chr(92) + '_')} & {100 * e['test_acc']:+.2f} & "
                 f"{100 * e['nodrift_acc']:+.2f} & {1000 * e['nodrift_ll']:+.2f} \\\\")
    emit("Sweep main effects", L)


def main():
    for f in (table1, table_mcts, table_tournament, table_sweep):
        try:
            f()
        except Exception as ex:          # partial data while runs are in flight
            OUT.append(f"% {f.__name__}: {ex!r}")
    txt = "\n".join(OUT)
    open(os.path.join(core.RESULTS, "tables.tex"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
