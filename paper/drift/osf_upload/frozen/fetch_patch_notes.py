#!/usr/bin/env python3
"""Build HotS 2.55.x patch-notes ground truth: heroes with balance changes per patch,
mapped to replay-corpus builds.

Sources
-------
1. PRIMARY (hero extraction): official Blizzard patch-notes articles.
   - Index: https://news.blizzard.com/en-us/api/feed/heroes-of-the-storm?offset=N
     (JSON, paginated; ~15 items/page; 'newsUrl' per item).
   - Article pages are server-rendered HTML; the notes live in <section class="blog">.
   - Hero balance changes appear as <h4>HeroName</h4> under "Heroes" /
     "Balance Update(s)" / "Hero Updates" sections; hero bug fixes appear as
     <li><strong>HeroName</strong> items under the "Bug Fixes" section.
2. SECONDARY (build-number mapping + maintenance-era hotfixes that were never
   posted on news.blizzard.com): https://hotspatchnotes.com/patches/
   Its index rows carry publish date, patch type (Live/PTR/Hotfix) and the exact
   build number (e.g. 2.55.2.87990); per-patch pages list changed heroes as
   entity cards + per-hero <h2> sections with <h3>Updates/Bug Fixes</h3>.

Output: patch_notes_ground_truth.json (same directory).
Stdlib only; rerun with: python3 fetch_patch_notes.py
"""
import html
import json
import re
import time
import unicodedata
import urllib.request
from datetime import date, datetime
from pathlib import Path

OUT = Path(__file__).with_name("patch_notes_ground_truth.json")
UA = {"User-Agent": "Mozilla/5.0 (HotS drift research; max@segan.com)"}
FEED = "https://news.blizzard.com/en-us/api/feed/heroes-of-the-storm?offset={}"
HPN_INDEX = "https://hotspatchnotes.com/patches/"
DATE_MIN, DATE_MAX = date(2021, 11, 25), date(2026, 6, 30)

# Replay-corpus builds -> first-game date (approximates release date).
CORPUS_BUILDS = {
    "2.55.0.86938": "2021-12-07", "2.55.1.87306": "2022-02-01",
    "2.55.2.87774": "2022-03-29", "2.55.2.87990": "2022-04-27",
    "2.55.2.88122": "2022-05-11", "2.55.3.88481": "2022-07-12",
    "2.55.3.88936": "2022-10-11", "2.55.3.89566": "2023-01-10",
    "2.55.3.89754": "2023-02-07", "2.55.3.90670": "2023-07-24",
    "2.55.3.91020": "2023-09-20", "2.55.3.91081": "2023-09-22",
    "2.55.3.91093": "2023-09-25", "2.55.4.91368": "2023-11-16",
    "2.55.4.91418": "2023-11-21", "2.55.4.91769": "2024-02-06",
    "2.55.5.92264": "2024-05-21", "2.55.6.92665": "2024-08-12",
    "2.55.7.93009": "2024-10-14", "2.55.7.93054": "2024-10-21",
    "2.55.7.93151": "2024-11-14", "2.55.8.93357": "2024-12-09",
    "2.55.8.93382": "2024-12-12", "2.55.9.93565": "2025-01-27",
    "2.55.9.93613": "2025-01-28", "2.55.9.93640": "2025-02-03",
    "2.55.10.93810": "2025-03-12", "2.55.10.94189": "2025-05-15",
    "2.55.10.94387": "2025-06-02", "2.55.10.94470": "2025-06-15",
    "2.55.12.94714": "2025-07-29", "2.55.12.94786": "2025-08-04",
    "2.55.13.95213": "2025-09-30", "2.55.13.95301": "2025-10-06",
    "2.55.14.95774": "2025-12-01", "2.55.14.95817": "2025-12-05",
    "2.55.14.95883": "2025-12-12", "2.55.14.95918": "2025-12-16",
    "2.55.15.96370": "2026-02-10", "2.55.15.96443": "2026-02-19",
    "2.55.15.96477": "2026-02-20", "2.55.16.96846": "2026-04-19",
    "2.55.16.96870": "2026-04-21", "2.55.16.96881": "2026-04-21",
    "2.55.16.97039": "2026-05-11",
}
BUILDNUM_TO_CORPUS = {v.rsplit(".", 1)[1]: v for v in CORPUS_BUILDS}

HEROES_90 = [
    "Abathur", "Alarak", "Alexstrasza", "Ana", "Anduin", "Anub'arak", "Artanis",
    "Arthas", "Auriel", "Azmodan", "Blaze", "Brightwing", "Cassia", "Chen", "Cho",
    "Chromie", "D.Va", "Deathwing", "Deckard", "Dehaka", "Diablo", "E.T.C.",
    "Falstad", "Fenix", "Gall", "Garrosh", "Gazlowe", "Genji", "Greymane",
    "Gul'dan", "Hanzo", "Hogger", "Illidan", "Imperius", "Jaina", "Johanna",
    "Junkrat", "Kael'thas", "Kel'Thuzad", "Kerrigan", "Kharazim", "Leoric",
    "Li Li", "Li-Ming", "Lt. Morales", "Lunara", "Lúcio", "Maiev", "Mal'Ganis",
    "Malfurion", "Malthael", "Medivh", "Mei", "Mephisto", "Muradin", "Murky",
    "Nazeebo", "Nova", "Orphea", "Probius", "Qhira", "Ragnaros", "Raynor",
    "Rehgar", "Rexxar", "Samuro", "Sgt. Hammer", "Sonya", "Stitches", "Stukov",
    "Sylvanas", "Tassadar", "The Butcher", "The Lost Vikings", "Thrall", "Tracer",
    "Tychus", "Tyrael", "Tyrande", "Uther", "Valeera", "Valla", "Varian",
    "Whitemane", "Xul", "Yrel", "Zagara", "Zarya", "Zeratul", "Zul'jin",
]


def _norm(s: str) -> str:
    s = html.unescape(s)
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", s.lower())


# normalized string -> list of canonical hero names
HERO_ALIASES = {_norm(h): [h] for h in HEROES_90}
HERO_ALIASES.update({
    "chogall": ["Cho", "Gall"],          # notes write "Cho'Gall"
    "butcher": ["The Butcher"],
    "lostvikings": ["The Lost Vikings"],
    "nazebo": ["Nazeebo"],               # typo in Nov 2023 official notes
    "morales": ["Lt. Morales"],
    "sergeanthammer": ["Sgt. Hammer"],
})

# Headings that mark a (sub)section rather than a hero.
SECTION_BALANCE = {"heroes", "balanceupdate", "balanceupdates", "heroupdates",
                   "heroupdate", "balance"}
SECTION_BUGFIX = {"bugfixes", "bugfix", "bugs"}
SECTION_ARAM = {"aram"}
SECTION_OTHER = {
    "quicknavigation", "general", "maps", "mapupdates", "mapupdate",
    "battlegrounds", "battleground", "collection", "shop", "mounts", "art",
    "userinterface", "ui", "sound", "design", "gameplay", "newhero",
    "storeupdates", "miscellaneous",
}
STRUCTURAL = {
    "tank", "bruiser", "healer", "support", "rangedassassin", "meleeassassin",
    "assassin", "talents", "abilities", "stats", "base", "trait", "updates",
    "developercomment", "developercomments", "returntotop", "",
} | {f"level{n}" for n in (1, 4, 7, 10, 13, 16, 20)}


def fetch(url: str, retries: int = 3) -> str:
    for i in range(retries):
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, headers=UA), timeout=45
            ) as r:
                return r.read().decode("utf-8", "replace")
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(2 * (i + 1))


def fetch_json(url: str):
    return json.loads(fetch(url))


# ---------------------------------------------------------------- Blizzard news
def blizzard_article_index():
    """All patch-notes articles from the official news feed, oldest first."""
    items, offset = [], 0
    while True:
        d = fetch_json(FEED.format(offset))
        ci = d.get("contentItems", [])
        if not ci:
            break
        for it in ci:
            p = it["properties"]
            items.append({"title": p.get("title") or "",
                          "url": p.get("newsUrl"),
                          "lastUpdated": p.get("lastUpdated")})
        offset += len(ci)
        oldest = items[-1]["lastUpdated"] or "9999"
        if not d.get("pagination", {}).get("hasNextPage") or oldest < "2021-09-01":
            break
        time.sleep(0.3)

    out = []
    for it in items:
        if "patch notes" not in it["title"].lower():
            continue
        m = re.search(r"([A-Z][a-z]+ \d{1,2}, \d{4})", it["title"])
        if not m:
            continue
        pub = datetime.strptime(m.group(1), "%B %d, %Y").date()
        if not (DATE_MIN <= pub <= DATE_MAX):
            continue
        out.append({"title": it["title"], "url": it["url"], "published": pub,
                    "type": "PTR" if "PTR" in it["title"] else "Live"})
    out.sort(key=lambda e: e["published"])
    return out


def _headings(seg: str):
    for m in re.finditer(r"<h([2-6])[^>]*>(.*?)</h\1>", seg, re.S):
        txt = re.sub(r"<[^>]+>", "", m.group(2))
        txt = re.sub(r"\s+", " ", html.unescape(txt)).strip()
        yield int(m.group(1)), txt, m.start(), m.end()


def parse_blizzard_article(page: str):
    """Return dict with balance/bugfix/aram hero sets + unmatched strings."""
    m = re.search(r'<section class="blog">(.*?)</section>', page, re.S)
    body = m.group(1) if m else page

    balance, bugfix, aram, unmatched = set(), set(), set(), set()
    heads = list(_headings(body))

    # Pass 1: walk headings, maintaining an outline stack of section markers so
    # that e.g. "Heroes" nested under "Bug Fixes" stays a bug-fix region.
    # markers: list of (effective_type, heading_index, content_start)
    markers, stack = [], []  # stack entries: (level, effective_type)

    def raw_type(key):
        if key in SECTION_BUGFIX:
            return "bugfix"
        if key in SECTION_ARAM:
            return "aram"
        if key in SECTION_BALANCE or "rework" in key:
            return "balance"
        if key in SECTION_OTHER:
            return "other"
        return None

    for i, (lvl, txt, start, end) in enumerate(heads):
        key = _norm(txt)
        rt = raw_type(key)
        if rt is None:
            continue
        while stack and stack[-1][0] >= lvl:
            stack.pop()
        ancestors = {t for _l, t in stack}
        if "bugfix" in ancestors:
            eff = "bugfix" if rt == "balance" else "other"
        elif "aram" in ancestors:
            eff = "aram" if rt == "balance" else "other"
        else:
            eff = rt
        stack.append((lvl, eff))
        markers.append((eff, i, end))
        if "rework" in key and eff == "balance":
            for alias, canon in HERO_ALIASES.items():
                if alias and alias in key:
                    balance.update(canon)

    tgt_by_type = {"balance": balance, "bugfix": bugfix, "aram": aram}

    # Pass 2: each region runs from one marker to the next marker (hero headings
    # are not markers). Within balance/bugfix/aram regions, collect heroes from
    # sub-headings AND from exact hero-name <strong> items (the 2023-10..2024-02
    # era and all Bug Fixes sections use <li><strong>Hero</strong> instead of
    # <h4>Hero</h4>).
    for m_idx, (eff, i, content_start) in enumerate(markers):
        region_end = markers[m_idx + 1][2] if m_idx + 1 < len(markers) else len(body)
        # trim to the next marker's heading start, not its content end
        if m_idx + 1 < len(markers):
            nxt_head = heads[markers[m_idx + 1][1]]
            region_end = nxt_head[2]
        tgt = tgt_by_type.get(eff)
        if tgt is None:
            continue
        for lvl, txt, _s, _e in heads[i + 1:]:
            if _s >= region_end:
                break
            key = _norm(txt)
            canon = HERO_ALIASES.get(key)
            if canon:
                tgt.update(canon)
            elif lvl >= 4 and eff == "balance" and key not in STRUCTURAL:
                unmatched.add(txt)
        for sm in re.finditer(
            r"<strong[^>]*>(.*?)</strong>", body[content_start:region_end], re.S
        ):
            t = re.sub(r"<[^>]+>", "", sm.group(1))
            t = re.sub(r"\s+", " ", html.unescape(t)).strip()
            canon = HERO_ALIASES.get(_norm(t))
            if canon:
                tgt.update(canon)

    return {"balance": balance, "bugfix": bugfix, "aram": aram,
            "unmatched": unmatched}


# ------------------------------------------------------------- hotspatchnotes
def hpn_index():
    """{(published, type_lower): {url, build}} from hotspatchnotes.com/patches/."""
    idx_html = fetch(HPN_INDEX)
    out = {}
    for m in re.finditer(
        r'<a[^>]+href="(https://hotspatchnotes\.com/patches/[^"]+)"[^>]*>(.*?)</a>',
        idx_html, re.S,
    ):
        url, body = m.group(1), re.sub(r"<[^>]+>", " ", m.group(2))
        body = re.sub(r"\s+", " ", html.unescape(body)).strip()
        dm = re.search(r"([A-Z][a-z]+ \d{1,2}, \d{4})", body)
        if not dm:
            continue
        pub = datetime.strptime(dm.group(1), "%B %d, %Y").date()
        ptype = "ptr" if "PTR" in body else ("hotfix" if "Hotfix" in body else "live")
        bm = re.search(r"\b(\d+\.\d+\.\d+\.\d+)\b", body)
        out[(pub, ptype)] = {"url": url, "build": bm.group(1) if bm else None}
    return out


def parse_hpn_page(page: str):
    """Hero sets from a hotspatchnotes patch page (used for hotfixes only).

    heroes = 'Heroes changed this patch' entity cards; bugfix_only = heroes whose
    per-hero section contains no <h3> other than 'Bug Fixes'.
    """
    heroes, bugfix_only = set(), set()
    for m in re.finditer(
        r'entity-card--hero"\s+href="https://hotspatchnotes\.com/heroes/([a-z0-9-]+)/"',
        page,
    ):
        canon = HERO_ALIASES.get(m.group(1).replace("-", ""))
        if canon:
            heroes.update(canon)
    h2s = [(m.start(), re.sub(r"<[^>]+>", "", m.group(1)).strip())
           for m in re.finditer(r"<h2[^>]*>(.*?)</h2>", page, re.S)]
    for i, (pos, txt) in enumerate(h2s):
        canon = HERO_ALIASES.get(_norm(txt))
        if not canon:
            continue
        end = h2s[i + 1][0] if i + 1 < len(h2s) else len(page)
        h3s = [re.sub(r"<[^>]+>", "", t).strip()
               for t in re.findall(r"<h3[^>]*>(.*?)</h3>", page[pos:end], re.S)]
        if h3s and all(_norm(t) in SECTION_BUGFIX for t in h3s):
            bugfix_only.update(canon)
        heroes.update(canon)
    return heroes, bugfix_only


# -------------------------------------------------------------------- mapping
def map_build(published: date, ptype: str, build: str | None):
    if build:
        num = build.rsplit(".", 1)[1]
        if num in BUILDNUM_TO_CORPUS:
            return BUILDNUM_TO_CORPUS[num], "high"
    best, best_d = None, 999
    for b, ds in CORPUS_BUILDS.items():
        d = abs((date.fromisoformat(ds) - published).days)
        if d < best_d:
            best, best_d = b, d
    if ptype == "ptr":
        # PTR notes describe a PTR build; only claim a corpus build on an exact
        # date collision (the corpus contains a few PTR-era builds).
        return (best, "low") if best_d <= 1 else (None, None)
    if best_d <= 3:
        return best, "medium"
    if best_d <= 10:
        return best, "low"
    return None, None


# ----------------------------------------------------------------------- main
def main():
    articles = blizzard_article_index()
    hpn = hpn_index()
    print(f"Blizzard patch-notes articles in range: {len(articles)}; "
          f"hotspatchnotes index entries: {len(hpn)}")

    patches = []
    covered = set()  # (published, hpn_type) consumed by a Blizzard article

    for art in articles:
        page = fetch(art["url"])
        p = parse_blizzard_article(page)
        ptype = art["type"].lower()
        hp = hpn.get((art["published"], "ptr" if ptype == "ptr" else "live")) \
            or hpn.get((art["published"], "hotfix"))
        for t in ("ptr", "live", "hotfix"):
            if (art["published"], t) in hpn and (
                t == ("ptr" if ptype == "ptr" else "live")
            ):
                covered.add((art["published"], t))
        build = hp["build"] if hp else None
        mapped, conf = map_build(art["published"], ptype, build)
        changed = sorted(p["balance"] | p["bugfix"])
        patches.append({
            "published": art["published"].isoformat(),
            "title": art["title"],
            "url": art["url"],
            "patch_type": "PTR" if ptype == "ptr" else "Live",
            "extraction_source": "blizzard_news",
            "build_per_source": build,
            "mapped_build": mapped,
            "mapping_confidence": conf,
            "heroes_changed": changed,
            "heroes_bugfix_only": sorted(p["bugfix"] - p["balance"]),
            "heroes_aram_only": sorted(p["aram"] - p["balance"] - p["bugfix"]),
            "raw_hero_strings_unmatched": sorted(p["unmatched"]),
        })
        print(f"{art['published']} {ptype:6s} build={build} -> {mapped} ({conf}) "
              f"balance={len(p['balance'])} bugfix_only="
              f"{len(p['bugfix'] - p['balance'])} unmatched={sorted(p['unmatched'])}")
        time.sleep(0.4)

    # hotspatchnotes-only entries (maintenance hotfixes, extra PTR pushes)
    for (pub, ptype), hp in sorted(hpn.items()):
        if not (DATE_MIN <= pub <= DATE_MAX) or (pub, ptype) in covered:
            continue
        page = fetch(hp["url"])
        heroes, bugfix_only = parse_hpn_page(page)
        mapped, conf = map_build(pub, ptype, hp["build"])
        patches.append({
            "published": pub.isoformat(),
            "title": f"Heroes of the Storm {ptype.capitalize()} Patch - "
                     f"{pub.strftime('%B %d, %Y').replace(' 0', ' ')}",
            "url": hp["url"],
            "patch_type": ptype.capitalize() if ptype != "ptr" else "PTR",
            "extraction_source": "hotspatchnotes",
            "build_per_source": hp["build"],
            "mapped_build": mapped,
            "mapping_confidence": conf,
            "heroes_changed": sorted(heroes),
            "heroes_bugfix_only": sorted(bugfix_only),
            "heroes_aram_only": [],
            "raw_hero_strings_unmatched": [],
        })
        print(f"{pub} {ptype:6s} build={hp['build']} -> {mapped} ({conf}) "
              f"heroes={len(heroes)} [hotspatchnotes]")
        time.sleep(0.4)

    patches.sort(key=lambda p: (p["published"], p["patch_type"]))
    mapped_builds = {p["mapped_build"] for p in patches if p["mapped_build"]}
    unmapped = sorted((b for b in CORPUS_BUILDS if b not in mapped_builds),
                      key=lambda b: CORPUS_BUILDS[b])

    out = {
        "source": (
            "Official Blizzard patch notes via news.blizzard.com "
            "(index: https://news.blizzard.com/en-us/api/feed/heroes-of-the-storm, "
            "server-rendered article HTML) for hero extraction; "
            "https://hotspatchnotes.com/patches/ for official build numbers per patch "
            "and for maintenance-era hotfix notes never posted on Blizzard news "
            "(2022-04-27, 2022-05-11, and small hotfixes)."
        ),
        "scriptable": True,
        "fetch_method": (
            "Plain HTTP GET, stdlib urllib. Blizzard: JSON feed API paginated with "
            "?offset=N, then each article's HTML parsed (hero balance sections = "
            "<h4>Hero</h4> under Heroes/Balance Update/Hero Updates headings; hero bug "
            "fixes = <li><strong>Hero</strong> under the Bug Fixes heading; ARAM-only "
            "sections excluded). hotspatchnotes.com: patch index anchors carry date/"
            "type/build; hotfix pages parsed via 'Heroes changed this patch' cards."
        ),
        "generated": date.today().isoformat(),
        "notes": (
            "heroes_changed = heroes with a dedicated balance/rework section PLUS "
            "heroes with hero-attributed entries under the notes' Bug Fixes section; "
            "the latter subset is duplicated in heroes_bugfix_only so bug-fix-level "
            "changes can be excluded. Heroes changed only for the ARAM mode are "
            "excluded from heroes_changed and listed in heroes_aram_only. Heroes "
            "mentioned only in free text of general bug fixes/map notes are excluded "
            "by construction. Cho'gall counts as both Cho and Gall; known official "
            "typos/aliases handled (Nazebo->Nazeebo, Butcher->The Butcher, "
            "Lucio->Lúcio). mapping_confidence: high = exact build-number match via "
            "hotspatchnotes; medium = publish date within 3 days of the corpus "
            "build's first game; low = within 10 days, or a PTR note whose publish "
            "date exactly matches a corpus build (only 2024-11-14 PTR -> "
            "2.55.7.93151; the notes list PTR build 2.55.8.93193, so that corpus "
            "build may instead be an unannounced live hotfix - treat with care). "
            "PTR notes otherwise map to null (their content ships in the following "
            "Live patch, which is mapped separately - using both double-counts). "
            "Corpus builds with NO published patch notes found anywhere "
            "(maintenance/infrastructure patches): "
            + ", ".join(f"{b} ({CORPUS_BUILDS[b]})" for b in unmapped) + ". "
            "hotspatchnotes labels builds 94189/94387/94470 as version 2.55.11.x "
            "while the corpus labels them 2.55.10.x; mapping uses the trailing build "
            "number, which is unambiguous."
        ),
        "corpus_builds_without_patch_notes": unmapped,
        "patches": patches,
    }
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"\nWrote {OUT} ({len(patches)} patches; {len(mapped_builds)} corpus "
          f"builds mapped; {len(unmapped)} corpus builds without notes)")


if __name__ == "__main__":
    main()
