"""Make hero-blind kit texts from sources_2.57.json (FRAMING.md 10.1, WAVE0_BRIEF item 9).

Removes: the hero's own name, other heroes' names, the hero's ability and talent names
(replaced by slot tokens such as [Q], [R1], [T4.2]), and summon or unit names that identify the
hero (replaced by "summon A", "summon B", ...). Numbers are kept.

HIDs come from a random permutation drawn once (os.urandom seed) and stored in
hid_map_PRIVATE.json, which goes into the kit file's provenance. Rerunning with an existing
map reuses it.

Usage: python3 -I blind.py            writes blind_text/<HID>.txt and hid_map_PRIVATE.json
       python3 -I blind.py --scan     lists residual capitalised terms that occur in few texts
"""
import collections
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "sources_2.57.json")
OUT_DIR = os.path.join(HERE, "blind_text")
MAP = os.path.join(HERE, "hid_map_PRIVATE.json")

# Name variants per hero, used both for "this hero" (own text) and "another hero" (other texts).
NAME_VARIANTS = {
    "Cho": ["Cho'gall", "Chogall", "Cho"], "Gall": ["Gall"],
    "Kael'thas": ["Kael'thas", "Kael"], "Li-Ming": ["Li-Ming", "Li Ming"],
    "Lt. Morales": ["Lt. Morales", "Morales"], "Lúcio": ["Lúcio", "Lucio"],
    "The Butcher": ["The Butcher", "the Butcher", "Butcher"],
    "The Lost Vikings": ["The Lost Vikings", "the Lost Vikings", "Lost Vikings", "Lost Viking", "Vikings",
                         "Viking"],
    "Sgt. Hammer": ["Sgt. Hammer", "Sergeant Hammer"], "E.T.C.": ["E.T.C.", "E.T.C", "ETC"],
    "D.Va": ["D.Va", "DVa", "Hana"], "Deckard": ["Deckard Cain", "Deckard", "Cain"],
    "Anub'arak": ["Anub'arak", "Anubarak"], "Gul'dan": ["Gul'dan", "Guldan"],
    "Mal'Ganis": ["Mal'Ganis", "Malganis"], "Zul'jin": ["Zul'jin", "Zuljin"],
    "Kel'Thuzad": ["Kel'Thuzad", "Kel'thuzad", "Kelthuzad"], "Xal'atath": ["Xal'atath", "Xalatath"],
    "Li Li": ["Li Li"], "Alexstrasza": ["Alexstrasza", "Alex"],
}
# Single-word mechanic names left in place (not identifying, and needed to read the text).
GENERIC = {"Storm", "Earth", "Fire", "Melee", "Energy", "Fury", "Charge", "Taunt", "Teleport", "Blink",
           "Burrow", "Flight", "Warp", "Recall", "Leap", "Dive", "Portal", "Evasion", "Undying",
           "Resurrect", "Vanish", "Reload", "Hook", "Slam", "Cleave", "Regeneration", "Inspire", "Purge",
           "Shockwave", "Deflect", "Disengage", "Avoidance", "Parry", "Shove", "Sprint", "Stealth",
           "Mount", "Hearthstone", "Polymorph", "Bloodlust", "Ambush", "Snipe", "Isolation", "Taunting"}
# Summon, unit, form or companion names that identify a hero. Applied to that hero's text only.
SUMMONS = json.load(open(os.path.join(HERE, "summon_names.json"), encoding="utf-8")) \
    if os.path.exists(os.path.join(HERE, "summon_names.json")) else {}


def norm(s):
    return (s or "").replace("’", "'").replace("‘", "'").replace(" ", " ")


def variants(hero):
    return NAME_VARIANTS.get(hero, [hero])


def sub_words(text, words, repl, plural=False):
    """Replace whole-word occurrences. Multi-word names match case-insensitively; with plural=True
    a trailing s/es is absorbed into the token."""
    for w in sorted(set(words), key=len, reverse=True):
        if not w:
            continue
        core = re.escape(w).replace("'", "['\u2019]")
        pat = r"(?<![\w'\[])" + core + (r"(?:e?s)?" if plural else "") + r"(?![\w\]])"
        if w == "Blaze":
            pat = r"(?<!Consuming )" + pat
        flags = re.IGNORECASE if " " in w.strip() else 0
        text = re.sub(pat, lambda m: repl, text, flags=flags)
    return text


def ability_tokens(rec):
    toks = {}
    for gi, grp in enumerate(rec["abilities"]):
        seen = collections.Counter()
        heroic = 0
        for a in grp:
            hk = a.get("hotkey") or ("D" if a.get("trait") or a.get("type") == "trait" else (a.get("type") or "X"))
            if a.get("type") == "heroic" and gi == 0:
                heroic += 1
                base = f"R{heroic}"
            else:
                base = hk
            seen[base] += 1
            tok = base if seen[base] == 1 else f"{base}#{seen[base]}"
            if gi > 0:
                tok = f"alt-{tok}"
            a["_tok"] = f"[{tok}]"
            toks.setdefault(norm(a["name"]), a["_tok"])
    return toks


def talent_tokens(rec, ab_toks):
    toks = {}
    for lvl, items in sorted(rec["talents"].items(), key=lambda kv: int(kv[0])):
        for i, t in enumerate(items, 1):
            nm = norm(t["name"])
            t["_tok"] = ab_toks.get(nm) or f"[T{lvl}.{i}]"
            toks.setdefault(nm, t["_tok"])
    return toks


def fmt_num(x):
    if x is None:
        return None
    return str(int(x)) if float(x).is_integer() else str(x)


def compose(hero, rec):
    ab = ability_tokens(rec)
    tal = talent_tokens(rec, ab)
    lines = []
    ba = rec.get("basic_attack")
    if ba and ba["weapons"]:
        parts = [f"range {fmt_num(w['range'])}, one attack every {fmt_num(w['period_s'])} s, "
                 f"{fmt_num(w['damage'])} damage per attack" for w in ba["weapons"]]
        lines.append("Basic attack (level 1 values; several entries mean alternate forms, units or "
                     "attack modes): " + "; ".join(parts) + ".")
    elif ba and not ba["weapons"]:
        lines.append("Basic attack: this hero has no basic attack of its own.")
    else:
        lines.append(f"Basic attack: {(rec.get('client_attack_type_2_57') or 'unknown').lower()} "
                     "(range and rate not available).")
    lines.append("")
    lines.append("ABILITIES")
    for gi, grp in enumerate(rec["abilities"]):
        if gi > 0:
            lines.append("Alternate form or second unit abilities:")
        for a in grp:
            meta = []
            if a.get("cooldown"):
                meta.append(f"cooldown {fmt_num(a['cooldown'])} s")
            if a.get("manaCost"):
                meta.append(f"cost {fmt_num(a['manaCost'])}")
            kind = a.get("type") or ""
            m = f" ({', '.join(meta)})" if meta else ""
            lines.append(f"{a['_tok']} {kind}{m}: {norm(a['description'])}")
    lines.append("")
    lines.append("TALENTS (one is chosen per level)")
    for lvl, items in sorted(rec["talents"].items(), key=lambda kv: int(kv[0])):
        lines.append(f"Level {lvl}:")
        for t in items:
            cd = f" (cooldown {fmt_num(t['cooldown'])} s)" if t.get("cooldown") else ""
            mod = f" modifies {t['type']}" if t.get("type") else ""
            lines.append(f"  {t['_tok']}{mod}{cd}: {norm(t['description'])}")
    if rec["deltas"]:
        lines.append("")
        lines.append("LATER PATCH CHANGES (oldest first; they update the text above to the current version)")
        for e in rec["deltas"]:
            lines.append(f"Change set {e['date'][:7]}:")
            lines.append(norm(e["text"]))
    text = "\n".join(lines)

    # Talent hotkey labels such as "modifies W" are mechanical; keep. Replace names.
    names = {**tal, **ab}
    for nm in sorted(names, key=len, reverse=True):
        if nm in GENERIC or len(nm) < 3:
            continue
        text = sub_words(text, [nm.rstrip("!")], names[nm], plural=True)
    k = 0
    for item in sorted(SUMMONS.get(hero, []), key=lambda x: -len(x if isinstance(x, str) else x[0])):
        if isinstance(item, list):
            term, rep = item
        else:
            term, rep = item, f"term {chr(65 + k)}"
            k += 1
        text = sub_words(text, [norm(term)], rep, plural=True)
    if hero == "Cho":
        text = sub_words(text, variants("Gall"), "the partner hero")
    if hero == "Gall":
        text = sub_words(text, variants("Cho"), "the partner hero")
    text = sub_words(text, variants(hero), "this hero")
    others = [v for h in all_heroes if h != hero for v in variants(h)]
    text = sub_words(text, others, "another hero")
    text = re.sub(r"\[T(\d+)\.\d+\]\s*\[", "[", text)
    return text


all_heroes = []


def main():
    global all_heroes
    src = json.load(open(SRC, encoding="utf-8"))
    heroes = src["heroes"]
    all_heroes = list(heroes)
    if os.path.exists(MAP):
        hid = json.load(open(MAP))["hid"]
    else:
        seed = int.from_bytes(os.urandom(8), "big")
        rng = random.Random(seed)
        codes = rng.sample(range(100, 1000), len(all_heroes))
        order = all_heroes[:]
        rng.shuffle(order)
        hid = {h: f"K{c}" for h, c in zip(order, codes)}
        json.dump({"seed": seed, "note": "private: hero -> HID; do not show to coders or blinded arms",
                   "hid": hid}, open(MAP, "w"), indent=1, ensure_ascii=False)
    texts = {h: compose(h, rec) for h, rec in heroes.items()}
    if "--scan" in sys.argv:
        df = collections.Counter()
        per = {}
        for h, t in texts.items():
            terms = set(re.findall(r"\b[A-Z][a-z'\-]+(?:\s+[A-Z][a-z'\-]+)*", t))
            per[h] = terms
            df.update(terms)
        for h in all_heroes:
            rare = sorted(x for x in per[h] if df[x] <= 2)
            print(f"{h}: {rare}")
        return
    os.makedirs(OUT_DIR, exist_ok=True)
    for f in os.listdir(OUT_DIR):
        os.remove(os.path.join(OUT_DIR, f))
    for h, t in texts.items():
        with open(os.path.join(OUT_DIR, hid[h] + ".txt"), "w", encoding="utf-8") as f:
            f.write(f"Hero code: {hid[h]}\n{t}\n")
    print(f"wrote {len(texts)} texts to {OUT_DIR}")


if __name__ == "__main__":
    main()
