"""Assemble the source text for the kit table (FRAMING.md 10.1).

Base text: ability and talent text from heroespatchnotes/heroes-talents at the commit pinned in
training/replays/build_reference.py (data to 2.55.3/2.55.4.91368, Nov 2023).
Deltas: the official patch-note text of every Live or Hotfix patch after the pin, through
build 2.57.0.98348, taken from the per-hero change histories on hotspatchnotes.com (which
mirror the official notes). PTR entries, bug fixes and the site's "datamined" blocks are
dropped. Hotfix entries that only carry a tooltip diff keep the "after" tooltip.
Basic attack type (melee/ranged) is taken from the site's client-extracted hero stats.
Xal'atath is not in the pinned reference: her kit is the site's client-extracted 2.57 text.

Only the kit, stats-type and change-history blocks of each page are parsed. The pages also
show win rates, tiers and builds; those blocks are never read.

Usage: nice -n 19 python3 -I fetch_sources.py <download_dir>
Writes sources_2.57.json next to this script.
"""
import html
import json
import os
import re
import sys
import time
import urllib.request
from datetime import date, datetime, timezone

SHA = "2dcbbf65f81bc3ce85926a5f90b6d8b44d1e4027"
REF_BASE = f"https://raw.githubusercontent.com/heroespatchnotes/heroes-talents/{SHA}/hero/"
HPN = "https://hotspatchnotes.com/heroes/{}/"
PIN_DATE = date(2023, 11, 16)  # 2.55.4.91368 live, the build the pinned commit was patched to
LAST_BUILD = (2, 57, 0, 98348)
HTC_SHA = "2a2207d08bdb2bac8038e8bc0e10c13a952ee043"
HTC_URL = (f"https://raw.githubusercontent.com/HeroesToolChest/heroes-data/{HTC_SHA}"
           "/heroesdata/2.55.9.93640/data/herodata_93640_localized.json")
# DB name -> heroes-data key, where it differs from the name with punctuation removed.
HTC_KEYS = {"Anub'arak": "Anubarak", "Cho": "Cho", "D.Va": "DVa", "E.T.C.": "L90ETC",
            "Gul'dan": "Guldan", "Johanna": "Crusader", "Kael'thas": "Kaelthas",
            "Kel'Thuzad": "KelThuzad", "Kharazim": "Monk", "Li Li": "LiLi", "Li-Ming": "Wizard",
            "Lt. Morales": "Medic", "Lúcio": "Lucio", "Lunara": "Dryad", "Mal'Ganis": "MalGanis",
            "Mei": "MeiOW", "Nazeebo": "WitchDoctor", "Qhira": "NexusHunter", "Sgt. Hammer": "SgtHammer",
            "Sonya": "Barbarian", "The Butcher": "Butcher", "The Lost Vikings": "LostVikings",
            "Brightwing": "FaerieDragon", "Valla": "DemonHunter", "Gazlowe": "Tinker",
            "Xul": "Necromancer", "Zul'jin": "Zuljin", "Cassia": "Amazon", "Blaze": "Firebat",
            "Diablo": "Diablo", "Zagara": "Zagara"}


def basic_attack(htc, name):
    key = HTC_KEYS.get(name, re.sub(r"[^A-Za-z]", "", name))
    h = htc.get(key)
    if h is None:
        return None
    weps = list(h.get("weapons") or [])
    for u in h.get("heroUnits") or []:
        for v in u.values():
            weps += v.get("weapons") or []
    seen, out = set(), []
    for w in weps:
        if w.get("range") is None:
            continue
        t = (w["range"], w.get("period"), w.get("damage"))
        if t not in seen:
            seen.add(t)
            out.append({"range": w["range"], "period_s": w.get("period"), "damage": w.get("damage")})
    return {"key": key, "weapons": out}


UA = {"User-Agent": "Mozilla/5.0 (HotS research; kit table)"}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sources_2.57.json")

# DB name -> (pinned reference file, hotspatchnotes slug)
HEROES = {
    "Abathur": ("abathur", "abathur"), "Alarak": ("alarak", "alarak"),
    "Alexstrasza": ("alexstrasza", "alexstrasza"), "Ana": ("ana", "ana"),
    "Anduin": ("anduin", "anduin"), "Anub'arak": ("anubarak", "anubarak"),
    "Artanis": ("artanis", "artanis"), "Arthas": ("arthas", "arthas"),
    "Auriel": ("auriel", "auriel"), "Azmodan": ("azmodan", "azmodan"),
    "Blaze": ("blaze", "blaze"), "Brightwing": ("brightwing", "brightwing"),
    "Cassia": ("cassia", "cassia"), "Chen": ("chen", "chen"), "Cho": ("chogall", "cho"),
    "Chromie": ("chromie", "chromie"), "Deathwing": ("deathwing", "deathwing"),
    "Deckard": ("deckard", "deckard"), "Dehaka": ("dehaka", "dehaka"),
    "Diablo": ("diablo", "diablo"), "D.Va": ("dva", "dva"), "E.T.C.": ("etc", "etc"),
    "Falstad": ("falstad", "falstad"), "Fenix": ("fenix", "fenix"), "Gall": ("gall", "gall"),
    "Garrosh": ("garrosh", "garrosh"), "Gazlowe": ("gazlowe", "gazlowe"),
    "Genji": ("genji", "genji"), "Greymane": ("greymane", "greymane"),
    "Gul'dan": ("guldan", "guldan"), "Hanzo": ("hanzo", "hanzo"), "Hogger": ("hogger", "hogger"),
    "Illidan": ("illidan", "illidan"), "Imperius": ("imperius", "imperius"),
    "Jaina": ("jaina", "jaina"), "Johanna": ("johanna", "johanna"),
    "Junkrat": ("junkrat", "junkrat"), "Kael'thas": ("kaelthas", "kaelthas"),
    "Kel'Thuzad": ("kelthuzad", "kelthuzad"), "Kerrigan": ("kerrigan", "kerrigan"),
    "Kharazim": ("kharazim", "kharazim"), "Leoric": ("leoric", "leoric"),
    "Li Li": ("lili", "lili"), "Li-Ming": ("liming", "liming"),
    "Lt. Morales": ("ltmorales", "ltmorales"), "Lúcio": ("lucio", "lucio"),
    "Lunara": ("lunara", "lunara"), "Maiev": ("maiev", "maiev"),
    "Malfurion": ("malfurion", "malfurion"), "Mal'Ganis": ("malganis", "malganis"),
    "Malthael": ("malthael", "malthael"), "Medivh": ("medivh", "medivh"), "Mei": ("mei", "mei"),
    "Mephisto": ("mephisto", "mephisto"), "Muradin": ("muradin", "muradin"),
    "Murky": ("murky", "murky"), "Nazeebo": ("nazeebo", "nazeebo"), "Nova": ("nova", "nova"),
    "Orphea": ("orphea", "orphea"), "Probius": ("probius", "probius"),
    "Qhira": ("qhira", "qhira"), "Ragnaros": ("ragnaros", "ragnaros"),
    "Raynor": ("raynor", "raynor"), "Rehgar": ("rehgar", "rehgar"),
    "Rexxar": ("rexxar", "rexxar"), "Samuro": ("samuro", "samuro"),
    "Sgt. Hammer": ("sgthammer", "sgthammer"), "Sonya": ("sonya", "sonya"),
    "Stitches": ("stitches", "stitches"), "Stukov": ("stukov", "stukov"),
    "Sylvanas": ("sylvanas", "sylvanas"), "Tassadar": ("tassadar", "tassadar"),
    "The Butcher": ("thebutcher", "thebutcher"),
    "The Lost Vikings": ("lostvikings", "thelostvikings"), "Thrall": ("thrall", "thrall"),
    "Tracer": ("tracer", "tracer"), "Tychus": ("tychus", "tychus"),
    "Tyrael": ("tyrael", "tyrael"), "Tyrande": ("tyrande", "tyrande"),
    "Uther": ("uther", "uther"), "Valeera": ("valeera", "valeera"), "Valla": ("valla", "valla"),
    "Varian": ("varian", "varian"), "Whitemane": ("whitemane", "whitemane"),
    "Xal'atath": (None, "xalatath"), "Xul": ("xul", "xul"), "Yrel": ("yrel", "yrel"),
    "Zagara": ("zagara", "zagara"), "Zarya": ("zarya", "zarya"),
    "Zeratul": ("zeratul", "zeratul"), "Zul'jin": ("zuljin", "zuljin"),
}


def fetch(url, path):
    if not os.path.exists(path):
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
        with open(path, "wb") as f:
            f.write(data)
        time.sleep(1.0)
    with open(path, encoding="utf-8") as f:
        return f.read()


def to_text(fragment):
    s = re.sub(r"<s>.*?</s>", "", fragment, flags=re.S)  # struck-through old values
    s = re.sub(r"<script.*?</script>|<style.*?</style>", "", s, flags=re.S)
    s = re.sub(r"<(br|/li|/p|/div|/h\d|li|p|h\d)[^>]*>", "\n", s)
    s = re.sub(r"<[^>]+>", " ", s)
    s = html.unescape(s).replace("\xa0", " ")
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in s.split("\n")]
    return "\n".join(ln for ln in lines if ln)


def parse_build(v):
    m = re.match(r"(\d+)\.(\d+)\.(\d+)\.(\d+)", v or "")
    return tuple(int(x) for x in m.groups()) if m else None


def parse_history(page):
    a = page.find('id="history-h"')
    b = page.find('id="meta-h"', a)
    sec = page[a:b if b > 0 else None]
    out = []
    for m in re.finditer(r'<section class="change-entry"[^>]*data-hpn-type="([a-z]+)"[^>]*>(.*?)</section>',
                         sec, flags=re.S):
        kind, body = m.group(1), m.group(2)
        d = re.search(r'class="change-date">([^<]+)<', body)
        v = re.search(r'class="ver"[^>]*>([^<]+)<', body)
        link = re.search(r'class="change-go" href="([^"]+)"', body)
        when = datetime.strptime(d.group(1).strip(), "%B %d, %Y").date()
        build = v.group(1).strip() if v else None
        content = body.split('<div class="section-html">', 1)[-1]
        content = re.sub(r'<a class="change-go".*?</a>', "", content, flags=re.S)
        content = content.split('<div class="dm-inline">')[0]
        content = re.split(r"<h3>\s*Bug Fixes\s*</h3>|<p><strong>\s*Bug Fixes\s*</strong></p>", content)[0]
        # Tooltip diffs: keep only the "after" text.
        tips = []
        for t in re.finditer(r'<div class="dm-change[^"]*"><div class="dm-name">(.*?)</div>(.*?)</div></div>',
                             content, flags=re.S):
            new = re.search(r'dm-new"><span class="dm-tag">After</span><span class="dm-val">(.*?)</span>',
                            t.group(2), flags=re.S)
            if new:
                tips.append(f"{to_text(t.group(1))}: tooltip now reads: {to_text(new.group(1))}")
        content = re.sub(r'<div class="dm-change.*?</div></div>', "", content, flags=re.S)
        text = to_text(content)
        if tips:
            text = (text + "\n" if text else "") + "\n".join(tips)
        out.append({"date": when.isoformat(), "type": kind, "build": build,
                    "url": link.group(1) if link else None, "text": text})
    return out


def parse_type(page):
    m = re.search(r"Type\s*</[^>]+>\s*(?:<[^>]+>\s*)*(Melee|Ranged)", page)
    return m.group(1) if m else None


def parse_kit(page):
    """Structured client-extracted kit (abilities and talents) from the hero page."""
    a = page.find('id="kit-h"')
    b = page.find('id="history-h"', a)
    sec = page[a:b]
    clean = lambda x: re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", x))).strip()
    abilities = []
    for m in re.finditer(r'<span class="htc-slot[^"]*">([^<]+)</span>.*?<span class="htc-ability-name">(.*?)</span>'
                         r'<span class="htc-ability-desc">(.*?)</span>(?:<span class="htc-ability-meta">(.*?)</span>)?',
                         sec, flags=re.S):
        meta = clean(m.group(4) or "")
        cd = re.search(r"Cooldown: ([\d.]+)", meta)
        mana = re.search(r"Mana: ([\d.]+)", meta)
        hk = clean(m.group(1))
        abilities.append({"name": clean(m.group(2)), "description": clean(m.group(3)), "hotkey": hk,
                          "cooldown": float(cd.group(1)) if cd else None,
                          "manaCost": float(mana.group(1)) if mana else None,
                          "type": "trait" if hk == "D" else ("heroic" if hk == "R" else "basic"),
                          "trait": hk == "D" or None})
    talents = {}
    for tier in re.finditer(r'<div class="htc-tier-lvl">Level (\d+)</div>(.*?)(?=<div class="htc-tier-lvl">|$)',
                            sec, flags=re.S):
        items = []
        for t in re.finditer(r'<span class="htc-talent-name">(.*?)</span>(?:</span>)?<span class="htc-tslot[^"]*"[^>]*>(.*?)</span>'
                             r'.*?<span class="htc-talent-desc">(.*?)</span>', tier.group(2), flags=re.S):
            items.append({"name": clean(re.sub(r"<[^>]+>", " ", t.group(1))).rstrip(" !"),
                          "description": clean(t.group(3)), "type": clean(t.group(2)), "cooldown": None})
        talents[tier.group(1)] = items
    return [abilities], talents


def in_window(entry):
    if entry["type"] not in ("live", "hotfix"):
        return False
    if date.fromisoformat(entry["date"]) <= PIN_DATE:
        return False
    b = parse_build(entry["build"])
    return b is None or b <= LAST_BUILD


def patch_section(page, slug):
    """The hero's non-bug-fix blocks on a full patch page (used when the hero page has a stub)."""
    blocks = page.split('<div class="section-block heroes-section')
    for blk in blocks[1:]:
        if f'href="/heroes/{slug}/"' not in blk.split('<div class="section-html">')[0]:
            continue
        body = blk.split('<div class="section-html">', 1)[-1]
        parts = re.split(r"(?=<div>\s*<h3>)", body)
        keep = [pt for pt in parts if not re.search(r"<h3>\s*Bug Fixes", pt)]
        return to_text("".join(keep))
    return ""


def main():
    dl = os.path.abspath(sys.argv[1])
    os.makedirs(os.path.join(dl, "ref"), exist_ok=True)
    os.makedirs(os.path.join(dl, "hpn"), exist_ok=True)
    os.makedirs(os.path.join(dl, "patch"), exist_ok=True)
    out = {"pinned_source": f"heroespatchnotes/heroes-talents@{SHA} (MIT)",
           "delta_source": "hotspatchnotes.com per-hero change history (mirrors official notes)",
           "pin_date": PIN_DATE.isoformat(), "last_build": ".".join(map(str, LAST_BUILD)),
           "fetched": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "heroes": {}}
    htc = json.loads(fetch(HTC_URL, os.path.join(dl, "herodata_93640.json")))
    out["basic_attack_source"] = f"HeroesToolChest/heroes-data@{HTC_SHA} build 2.55.9.93640 (MIT)"
    for name, (ref, slug) in HEROES.items():
        page = fetch(HPN.format(slug), os.path.join(dl, "hpn", slug + ".html"))
        rec = {"page_url": HPN.format(slug), "client_attack_type_2_57": parse_type(page),
               "basic_attack": basic_attack(htc, name)}
        if ref:
            rec["ref_url"] = REF_BASE + ref + ".json"
            d = json.loads(fetch(rec["ref_url"], os.path.join(dl, "ref", ref + ".json")))
            rec["abilities"] = [[{k: a.get(k) for k in ("name", "description", "hotkey", "cooldown",
                                                          "manaCost", "type", "trait")} for a in grp]
                                for grp in d["abilities"].values()]
            rec["talents"] = {lvl: [{k: t.get(k) for k in ("name", "description", "type", "cooldown")}
                                    for t in items] for lvl, items in d["talents"].items()}
        else:
            rec["abilities"], rec["talents"] = parse_kit(page)
            rec["kit_source"] = "hotspatchnotes.com client-extracted kit, build 2.57.0.98348"
        hist = [e for e in parse_history(page) if in_window(e)]
        for e in hist:
            if not e["text"].strip() and e["url"]:
                fn = re.sub(r"[^a-z0-9-]", "", e["url"].rstrip("/").rsplit("/", 1)[-1]) + ".html"
                e["text"] = patch_section(fetch(e["url"], os.path.join(dl, "patch", fn)), slug)
                e["from_patch_page"] = True
        rec["deltas"] = [e for e in hist if e["text"].strip()]
        rec["deltas"].sort(key=lambda e: e["date"])
        out["heroes"][name] = rec
        ba = rec["basic_attack"]["weapons"] if rec["basic_attack"] else None
        print(f"{name}: ba={ba} deltas={len(rec['deltas'])}", flush=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
