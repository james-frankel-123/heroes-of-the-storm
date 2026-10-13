"""Build ref/heroes_talents_ref.json: internal hero unit IDs and talent tree IDs mapped to
display names, from heroespatchnotes/heroes-talents (MIT) at a pinned commit. That data stops
at 2.55.3 (2023); newer talents and heroes fall back to name matching in parse_replay.py."""
import json
import os
import urllib.request

SHA = "2dcbbf65f81bc3ce85926a5f90b6d8b44d1e4027"
BASE = f"https://raw.githubusercontent.com/heroespatchnotes/heroes-talents/{SHA}/hero/"
LIST = f"https://api.github.com/repos/heroespatchnotes/heroes-talents/contents/hero?ref={SHA}"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref", "heroes_talents_ref.json")


def get(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def main():
    heroes, talents = {}, {}
    for f in get(LIST):
        if not f["name"].endswith(".json"):
            continue
        d = get(BASE + f["name"])
        heroes[d["cUnitId"]] = d["name"]
        for level, items in d["talents"].items():
            for t in items:
                talents[t["talentTreeId"]] = {"name": t["name"], "hero": d["name"], "level": int(level)}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({"source": f"heroespatchnotes/heroes-talents@{SHA} (MIT)",
                   "heroes": heroes, "talents": talents}, f, indent=0, sort_keys=True)
    print(f"{len(heroes)} heroes, {len(talents)} talents -> {OUT}")


if __name__ == "__main__":
    main()
