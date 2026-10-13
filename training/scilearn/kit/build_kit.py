"""Re-derive kit_2.57.json from the stored raw codings.

Inputs (all in this directory): codings/coder*_batch*.json (raw coder outputs), codings/run_log.json,
hid_map_PRIVATE.json, sources_2.57.json, blind_text/, RUBRIC.md, and src/lib/data/hero-roles.ts.
Per hero and axis: mean of the three coders, sample SD (n-1) across coders, flag if SD > 1.0.
The sha256 is taken over the canonical JSON (sorted keys, compact separators, UTF-8) of the whole
file minus the "sha256" field. The date is the finish time of the last coder run, so a rebuild
from the same raw codings gives the same hash.

Usage: python3 -I build_kit.py [--check]   (--check: rebuild and compare with the stored file)
"""
import glob
import hashlib
import json
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from kitdefs import AXES, AXIS_DEFINITIONS, AXIS_RANGE, IDENTITY_WEIGHTS, SD_FLAG  # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(HERE, "kit_2.57.json")
PATCH = "2.57"
OFFICIAL_257 = [
    "https://news.blizzard.com/en-us/article/24303007/ (Live patch notes, Sep 28 2026, includes the Oct 5 2026 hotfix)",
    "https://news.blizzard.com/en-us/article/24306095/ (PTR patch notes, Sep 14 2026)",
]
XAL_SOURCES = [
    "https://hotspatchnotes.com/heroes/xalatath/ (ability and talent text extracted from the game client, build 2.57.0.98348)",
    "https://news.blizzard.com/en-us/article/24303007/ (official live notes Sep 28 2026 and Oct 5 2026 hotfix; numbers match the client text)",
    "https://news.blizzard.com/en-us/article/24306095/ (official PTR notes Sep 14 2026, earlier numbers, not used for values)",
]


def sha_file(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def load_roles():
    ts = open(os.path.join(REPO, "src", "lib", "data", "hero-roles.ts"), encoding="utf-8").read()
    body = ts.split("HERO_ROLES", 1)[1].split("}", 1)[0]
    roles = {}
    for m in re.finditer(r"'((?:[^'\\]|\\.)+)'\s*:\s*'([^']+)'", body):
        roles[m.group(1).replace("\\'", "'")] = m.group(2)
    return roles


def build():
    hid = json.load(open(os.path.join(HERE, "hid_map_PRIVATE.json"), encoding="utf-8"))
    hero_of = {v: k for k, v in hid["hid"].items()}
    run_log = json.load(open(os.path.join(HERE, "codings", "run_log.json")))
    raw = {}  # coder -> code -> values
    runs = []
    for p in sorted(glob.glob(os.path.join(HERE, "codings", "coder*_batch*.json"))):
        r = json.load(open(p, encoding="utf-8"))
        runs.append({"coder": r["coder"], "batch": r["batch"], "prompt_file": r["prompt_file"],
                     "prompt_sha256": r["prompt_sha256"], "models_reported": r["models_reported"],
                     "started": r["started"], "finished": r["finished"], "attempts": len(r["attempts"]),
                     "raw_file_sha256": sha_file(p)})
        for code in r["codes"]:
            raw.setdefault(r["coder"], {})[code] = r["codes_out"][code]
    coders = sorted(raw)
    if len(coders) != 3 or any(len(raw[c]) != len(hero_of) for c in coders):
        raise SystemExit(f"incomplete codings: {[(c, len(raw[c])) for c in coders]}")
    roles = load_roles()
    heroes = {}
    for code, hero in sorted(hero_of.items(), key=lambda kv: kv[1]):
        per = {f"coder{c}": {a: raw[c][code][a] for a in AXES} for c in coders}
        mean = {a: round(statistics.mean(per[f"coder{c}"][a] for c in coders), 4) for a in AXES}
        sd = {a: round(statistics.stdev([per[f"coder{c}"][a] for c in coders]), 4) for a in AXES}
        low = {f"coder{c}": sorted(raw[c][code].get("low_confidence") or []) for c in coders}
        ident = {k: round(sum(w * mean[a] for a, w in ws.items()), 4) for k, ws in IDENTITY_WEIGHTS.items()}
        heroes[hero] = {"role": roles.get(hero), "values": mean, "coder_sd": sd,
                        "flagged_axes": [a for a in AXES if sd[a] > SD_FLAG], "identity": ident,
                        "raw": per, "low_confidence": low}
    axis_sd = {a: round(statistics.mean(h["coder_sd"][a] for h in heroes.values()), 4) for a in AXES}
    flag_counts = {a: sum(a in h["flagged_axes"] for h in heroes.values()) for a in AXES}
    src = json.load(open(os.path.join(HERE, "sources_2.57.json"), encoding="utf-8"))
    delta_urls = sorted({e["url"] for r in src["heroes"].values() for e in r["deltas"] if e.get("url")})
    weak = {
        "Xal'atath": "Not in the pinned reference. Kit text is the client-extracted 2.57.0.98348 text from "
                     "hotspatchnotes.com, cross-checked against the official live notes and Oct 5 hotfix. "
                     "No basic attack range or rate available (only 'ranged').",
        "Gall": "No basic attack of his own (shares a body with Cho); coded from abilities only.",
    }
    blind_hashes = {f: sha_file(os.path.join(HERE, "blind_text", f))
                    for f in sorted(os.listdir(os.path.join(HERE, "blind_text")))}
    kit = {
        "version": f"kit_{PATCH}",
        "patch": PATCH,
        "date": max(r["finished"] for r in runs),
        "n_heroes": len(heroes),
        "axes": AXES,
        "axis_ranges": {a: list(AXIS_RANGE[a]) for a in AXES},
        "axis_definitions": AXIS_DEFINITIONS,
        "rubric_file": "RUBRIC.md",
        "rubric_sha256": sha_file(os.path.join(HERE, "RUBRIC.md")),
        "identity_weights": IDENTITY_WEIGHTS,
        "identity_note": "Per-hero identity = sum(weight * mean axis value). Team identity = mean over its five heroes. Fixed, never fit.",
        "aggregation": f"values = mean of {len(coders)} coders; coder_sd = sample SD (n-1) across coders; "
                       f"flagged_axes = axes with coder_sd > {SD_FLAG}",
        "summary": {"mean_coder_sd_per_axis": axis_sd, "flagged_heroes_per_axis": flag_counts},
        "heroes": heroes,
        "provenance": {
            "pinned_reference": src["pinned_source"],
            "pinned_reference_note": "Ability and talent text from hero/*.json at this commit (data to 2.55.4.91368, Nov 2023); "
                                     "same commit as training/replays/build_reference.py.",
            "basic_attack_source": src.get("basic_attack_source"),
            "patch_note_deltas": {
                "method": "Live and hotfix entries after 2023-11-16 through build 2.57.0.98348 from the per-hero change "
                          "histories on hotspatchnotes.com (mirror of the official notes). PTR entries, bug fixes and "
                          "datamined blocks dropped. Stub entries resolved from the full patch pages.",
                "source_urls": delta_urls,
                "official_2_57_notes": OFFICIAL_257,
                "sources_file_sha256": sha_file(os.path.join(HERE, "sources_2.57.json")),
                "fetched": src["fetched"],
            },
            "xalatath_sources": XAL_SOURCES,
            "weak_source_text": weak,
            "blinding": {"script": "blind.py", "summon_names_sha256": sha_file(os.path.join(HERE, "summon_names.json")),
                         "blind_text_sha256": blind_hashes},
            "hid_map_PRIVATE": hid,
            "coding": {"model": run_log["model"], "cli": run_log["cli"], "coder_seeds": run_log["coder_seeds"],
                       "batch_size": run_log["batch_size"], "template_sha256": run_log["template_sha256"],
                       "rubric_sha256": run_log["rubric_sha256"], "runs": runs},
            "no_outcome_data": "No win rates, pick rates or database outcome data were used. The database was not queried.",
        },
    }
    kit["sha256"] = hashlib.sha256(canonical(kit).encode("utf-8")).hexdigest()
    return kit


def main():
    kit = build()
    if "--check" in sys.argv:
        old = json.load(open(OUT, encoding="utf-8"))
        same = old["sha256"] == kit["sha256"]
        body = {k: v for k, v in old.items() if k != "sha256"}
        ok = hashlib.sha256(canonical(body).encode("utf-8")).hexdigest() == old["sha256"]
        print(f"rebuild matches stored: {same}; stored hash self-consistent: {ok}")
        sys.exit(0 if same and ok else 1)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(kit, f, indent=1, ensure_ascii=False, sort_keys=True)
    print(f"wrote {OUT}  sha256={kit['sha256']}")
    for a, v in kit["summary"]["mean_coder_sd_per_axis"].items():
        print(f"  {a:22s} mean SD {v:.3f}  flagged heroes {kit['summary']['flagged_heroes_per_axis'][a]}")


if __name__ == "__main__":
    main()
